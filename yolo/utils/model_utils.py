import os
from copy import deepcopy
from math import exp
from pathlib import Path
from typing import List, Optional, Type, Union

import torch
import torch.distributed as dist
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from omegaconf import ListConfig
from torch import Tensor, no_grad
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, _LRScheduler

from yolo.config.config import IDX_TO_ID, NMSConfig, OptimizerConfig, SchedulerConfig
from yolo.model.yolo import YOLO
from yolo.utils.bounding_box_utils import Anc2Box, Vec2Box, bbox_nms, transform_bbox
from yolo.utils.logger import logger

import torch.nn.functional as F
from yolo.utils.ops import crop_mask


def lerp(start: float, end: float, step: Union[int, float], total: int = 1):
    """
    Linearly interpolates between start and end values.

    start * (1 - step) + end * step

    Parameters:
        start (float): The starting value.
        end (float): The ending value.
        step (int): The current step in the interpolation process.
        total (int): The total number of steps.

    Returns:
        float: The interpolated value.
    """
    return start + (end - start) * step / total


class EMA(Callback):
    def __init__(self, decay: float = 0.9999, tau: float = 2000):
        super().__init__()
        logger.info(":chart_with_upwards_trend: Enable Model EMA")
        self.decay = decay
        self.tau = tau
        self.step = 0
        self.ema_state_dict = None

    def setup(self, trainer, pl_module, stage):
        pl_module.ema = deepcopy(pl_module.model)
        self.tau /= trainer.world_size # type: ignore
        for param in pl_module.ema.parameters():
            param.requires_grad = False
        
        # Mover la inicialización de state_dict aquí
        model_state_dict = pl_module.model.state_dict()
        self.ema_state_dict = {k: v.clone().to(pl_module.device) for k, v in model_state_dict.items()}
        
    def on_validation_start(self, trainer: "Trainer", pl_module: "LightningModule"):
        # Ya no necesitamos comprobar si es None, solo cargar el estado actual
        if self.ema_state_dict: # Asegurarse de que no sea None (aunque setup ya lo hizo)
            pl_module.ema.load_state_dict(self.ema_state_dict)

    @no_grad()
    def on_train_batch_end(self, trainer: "Trainer", pl_module: "LightningModule", *args, **kwargs) -> None:
        self.step += 1
        decay_factor = self.decay * (1 - exp(-self.step / self.tau))
        
        # Asegurarse de que param esté en el mismo dispositivo que ema_state_dict
        # (Aunque deberían estarlo si pl_module.device es correcto)
        for key, param in pl_module.model.state_dict().items():
            if key in self.ema_state_dict:
                param_detached = param.detach()
                # Asegurar que ambos estén en el mismo dispositivo antes de lerp
                if param_detached.device != self.ema_state_dict[key].device:
                    param_detached = param_detached.to(self.ema_state_dict[key].device)
                
                self.ema_state_dict[key] = lerp(param_detached, self.ema_state_dict[key], decay_factor) # type: ignore
            else:
                logger.warning(f"Clave {key} no encontrada en EMA state_dict. Omitiendo.")


def create_optimizer(model: YOLO, optim_cfg: OptimizerConfig) -> Optimizer:
    """Create an optimizer for the given model parameters based on the configuration.

    Returns:
        An instance of the optimizer configured according to the provided settings.
    """
    optimizer_class: Type[Optimizer] = getattr(torch.optim, optim_cfg.type)

    bias_params = [p for name, p in model.named_parameters() if "bias" in name]
    norm_params = [p for name, p in model.named_parameters() if "weight" in name and "bn" in name]
    conv_params = [p for name, p in model.named_parameters() if "weight" in name and "bn" not in name]

    model_parameters = [
        {"params": bias_params, "momentum": 0.937, "weight_decay": 0},
        {"params": conv_params, "momentum": 0.937},
        {"params": norm_params, "momentum": 0.937, "weight_decay": 0},
    ]

    def next_epoch(self, batch_num, epoch_idx):
        self.min_lr = self.max_lr
        self.max_lr = [param["lr"] for param in self.param_groups]
        # TODO: load momentum from config instead a fix number
        #       0.937: Start Momentum
        #       0.8  : Normal Momemtum
        #       3    : The warm up epoch num
        self.min_mom = lerp(0.8, 0.937, min(epoch_idx, 3), 3)
        self.max_mom = lerp(0.8, 0.937, min(epoch_idx + 1, 3), 3)
        self.batch_num = batch_num
        self.batch_idx = 0

    def next_batch(self):
        self.batch_idx += 1
        lr_dict = dict()
        for lr_idx, param_group in enumerate(self.param_groups):
            min_lr, max_lr = self.min_lr[lr_idx], self.max_lr[lr_idx]
            param_group["lr"] = lerp(min_lr, max_lr, self.batch_idx, self.batch_num)
            param_group["momentum"] = lerp(self.min_mom, self.max_mom, self.batch_idx, self.batch_num)
            lr_dict[f"LR/{lr_idx}"] = param_group["lr"]
            lr_dict[f"momentum/{lr_idx}"] = param_group["momentum"]
        return lr_dict

    optimizer_class.next_batch = next_batch
    optimizer_class.next_epoch = next_epoch

    optimizer = optimizer_class(model_parameters, **optim_cfg.args)
    optimizer.max_lr = [0.1, 0, 0]
    return optimizer


def create_scheduler(optimizer: Optimizer, schedule_cfg: SchedulerConfig) -> _LRScheduler:
    """Create a learning rate scheduler for the given optimizer based on the configuration.

    Returns:
        An instance of the scheduler configured according to the provided settings.
    """
    scheduler_class: Type[_LRScheduler] = getattr(torch.optim.lr_scheduler, schedule_cfg.type)
    schedule = scheduler_class(optimizer, **schedule_cfg.args)
    if hasattr(schedule_cfg, "warmup"):
        wepoch = schedule_cfg.warmup.epochs
        lambda1 = lambda epoch: (epoch + 1) / wepoch if epoch < wepoch else 1
        lambda2 = lambda epoch: 10 - 9 * ((epoch + 1) / wepoch) if epoch < wepoch else 1
        warmup_schedule = LambdaLR(optimizer, lr_lambda=[lambda2, lambda1, lambda1])
        schedule = SequentialLR(optimizer, schedulers=[warmup_schedule, schedule], milestones=[wepoch - 1])
    return schedule


def initialize_distributed() -> None:
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    logger.info(f"🔢 Initialized process group; rank: {rank}, size: {world_size}")
    return local_rank


def get_device(device_spec: Union[str, int, List[int]]) -> torch.device:
    ddp_flag = False
    if isinstance(device_spec, (list, ListConfig)):
        ddp_flag = True
        device_spec = initialize_distributed()
    if torch.cuda.is_available() and "cuda" in str(device_spec):
        return torch.device(device_spec), ddp_flag
    if not torch.cuda.is_available():
        if device_spec != "cpu":
            logger.warning(f"❎ Device spec: {device_spec} not support, Choosing CPU instead")
        return torch.device("cpu"), False

    device = torch.device(device_spec)
    return device, ddp_flag


class PostProcess:
    """
    Realiza el post-procesamiento para detección y segmentación.
    - Decodifica las salidas raw del modelo.
    - Realiza NMS en las bounding boxes.
    - Si es un modelo de segmentación, reconstruye las máscaras para las
        predicciones que superan el NMS.
    """

    def __init__(self, converter: Union[Vec2Box, Anc2Box], nms_cfg: NMSConfig) -> None:
        self.converter = converter
        self.nms = nms_cfg
        # Asumimos que el converter sabe si es DFL (reg_max)
        self.reg_max = getattr(converter, "reg_max", 16) 
        # Tamaño de los prototipos (M_h, M_w) - Asumimos 1/4 del tamaño de imagen
        # Esto podría necesitar ajustarse si el modelo es diferente.
        self.mask_stride = 4

    def _reconstruct_masks(self, 
                            pred_coeffs: Tensor, # [N_post_nms, M_c]
                            pred_boxes_xyxy: Tensor, # [N_post_nms, 4]
                            proto: Tensor, # [1, M_c, M_h, M_w] (solo para esta imagen)
                            img_shape_hw: Tuple[int, int], # (H, W)
                            orig_shape_hw: Tuple[int, int] # (H_orig, W_orig)
                        ) -> Tensor:
        """Reconstruye, recorta y reescala las máscaras."""
        
        H_img, W_img = img_shape_hw
        H_orig, W_orig = orig_shape_hw
        M_h, M_w = proto.shape[-2:]

        if pred_coeffs.numel() == 0:
            return torch.empty(0, H_orig, W_orig, dtype=torch.bool, device=proto.device)

        # 1. Reconstruir (MatMul): [N, M_c] @ [M_c, M_h*M_w] -> [N, M_h*M_w]
        pred_masks_logits = (pred_coeffs @ proto.view(proto.shape[1], -1)).view(-1, M_h, M_w)

        # 2. Recortar (Crop)
        # Normalizar las cajas predichas al tamaño de la *imagen de entrada* (ej. 640x640)
        # Esto es necesario para 'crop_mask'
        boxes_norm_img = pred_boxes_xyxy / torch.tensor([W_img, H_img, W_img, H_img], device=proto.device)
        
        # crop_mask espera (Masks[N, M_h, M_w], Boxes[N, 4] normalizadas 0-1)
        masks_cropped = crop_mask(pred_masks_logits, boxes_norm_img) # [N, M_h, M_w]
        
        # 3. Interpolar (Upsample) a tamaño de imagen original y aplicar Sigmoid
        # Usamos F.interpolate para reescalar a H_orig, W_orig
        # masks_cropped.unsqueeze(1) -> [N, 1, M_h, M_w]
        masks_upscaled = F.interpolate(
            masks_cropped.unsqueeze(1),
            size=orig_shape_hw,
            mode='bilinear',
            align_corners=False
        ).squeeze(1) # [N, H_orig, W_orig]
        
        # 4. Binarizar
        # Aplicamos sigmoide (porque teníamos logits) y un umbral
        return (masks_upscaled.sigmoid() > self.nms.min_confidence) # [N, H_orig, W_orig] (Bool)
    
    def __call__(
        self, 
        predict: Dict[str, Any], # Salida del modelo (dict con 'Main', 'AUX')
        rev_tensor: Optional[Tensor] = None, 
        image_size: Optional[List[int]] = None # [W, H]
    ) -> List[Dict[str, Tensor]]: # Cambiado: Devuelve Lista de Dicts
        """
        Procesa la salida del modelo para obtener predicciones finales.

        Returns:
            Una lista de diccionarios (uno por imagen). Cada dict contiene:
            'boxes': Tensor [N, 4] (xyxy)
            'labels': Tensor [N]
            'scores': Tensor [N]
            'masks': Tensor [N, H_orig, W_orig] (Bool) (si es segm.)
        """
        
        if image_size is not None:
            self.converter.update(image_size) # [W, H]
        
        H_img, W_img = self.converter.image_size[1], self.converter.image_size[0]

        # --- 1. Extraer Salidas del Modelo ---
        # predict["Main"] es una TUPLA: (detection_outputs, segmentation_outputs)
        try:
            main_detect_raw = predict["Main"][0]   # Lista de tuplas (cls, dist, box)
            main_seg_list = predict["Main"][1]     # Lista de [coeffs..., proto]
            
            proto = main_seg_list[-1]              # [B, M_c, M_h, M_w]
            main_coeffs_raw = main_seg_list[:-1]   # Lista de [B, M_c, H_i, W_i]
            
            is_segmentation = True
        except (IndexError, TypeError):
            # Fallback para modelos solo de detección
            main_detect_raw = predict["Main"][0]
            is_segmentation = False
            proto = None
            main_coeffs_raw = None

        # --- 2. Decodificar Detecciones (BBoxes y Clases) ---
        # prediction = (preds_cls, preds_box_dist, preds_box_xyxy)
        preds_cls, _, preds_box_xyxy = self.converter(main_detect_raw)
        # preds_cls [B, A_total, C], preds_box_xyxy [B, A_total, 4]

        # --- 3. Decodificar Coeficientes de Máscara (si aplica) ---
        if is_segmentation and main_coeffs_raw is not None:
            batch_size = preds_cls.shape[0]
            all_raw_coeffs = []
            for raw_coeffs in main_coeffs_raw:
                # raw_coeffs [B, M_c, H, W] -> [B, H*W, M_c]
                all_raw_coeffs.append(raw_coeffs.reshape(batch_size, raw_coeffs.shape[1], -1).permute(0, 2, 1))
            all_raw_coeffs = torch.cat(all_raw_coeffs, dim=1) # [B, A_total, M_c]
        else:
            all_raw_coeffs = None

        # --- 4. Iterar por Imagen (NMS y Reconstrucción) ---
        batch_size = preds_cls.shape[0]
        results_list = []

        for i in range(batch_size):
            # --- 4a. Preparar datos para esta imagen ---
            img_preds_cls = preds_cls[i]       # [A_total, C]
            img_preds_box = preds_box_xyxy[i]  # [A_total, 4]
            
            # Obtener puntuaciones (confianza * prob_clase)
            scores, labels = img_preds_cls.sigmoid().max(1) # [A_total], [A_total]
            
            # Filtrar por confianza mínima ANTES de NMS
            keep = scores > self.nms.min_confidence
            
            boxes_pre_nms = img_preds_box[keep] # [K, 4]
            scores_pre_nms = scores[keep]     # [K]
            labels_pre_nms = labels[keep]     # [K]
            
            # --- 4b. Aplicar NMS (Batched NMS) ---
            # nms_idx son los índices *relativos* a 'boxes_pre_nms'
            nms_idx = batched_nms(
                boxes_pre_nms,
                scores_pre_nms,
                labels_pre_nms,
                self.nms.min_iou
            )
            
            # Limitar al máximo número de bboxes
            if nms_idx.shape[0] > self.nms.max_bbox:
                nms_idx = nms_idx[:self.nms.max_bbox]

            # Seleccionar las predicciones finales
            final_boxes = boxes_pre_nms[nms_idx]   # [N_final, 4]
            final_scores = scores_pre_nms[nms_idx]  # [N_final]
            final_labels = labels_pre_nms[nms_idx]  # [N_final]

            # --- 4c. Reconstruir Máscaras (si aplica) ---
            final_masks = torch.empty(0, device=preds_cls.device) # Placeholder
            
            if is_segmentation and all_raw_coeffs is not None and proto is not None:
                # Coeficientes para esta imagen [A_total, M_c]
                img_coeffs = all_raw_coeffs[i] 
                # Seleccionar coeficientes pre-NMS [K, M_c]
                coeffs_pre_nms = img_coeffs[keep]
                # Seleccionar coeficientes post-NMS [N_final, M_c]
                final_coeffs = coeffs_pre_nms[nms_idx]

                # Prototipo para esta imagen [1, M_c, M_h, M_w]
                img_proto = proto[i].unsqueeze(0) 
                
                # Definir formas de imagen
                img_shape_hw = (H_img, W_img)
                
                # Obtener forma original (si rev_tensor está disponible)
                if rev_tensor is not None:
                    # rev_tensor [B, 5] (scale, pad_x, pad_y, W_orig, H_orig)
                    # Asumiendo que rev_tensor tiene [scale, padW, padH, orig_W, orig_H]
                    # ¡¡CUIDADO!! El rev_tensor de tu data_loader puede ser diferente.
                    # Asumamos por ahora que la forma original es la misma que la de entrada
                    # TODO: Corregir esto si el data_loader escala y centra (letterbox)
                    H_orig, W_orig = int(H_img), int(W_img) # TODO: Usar rev_tensor
                else:
                    H_orig, W_orig = int(H_img), int(W_img)

                orig_shape_hw = (H_orig, W_orig)

                # Reconstruir las máscaras
                final_masks = self._reconstruct_masks(
                    final_coeffs,
                    final_boxes,
                    img_proto,
                    img_shape_hw,
                    orig_shape_hw
                ) # [N_final, H_orig, W_orig] (Bool)
                
                # TODO: Escalar 'final_boxes' si usamos rev_tensor
                if rev_tensor is not None:
                     # (pred_bbox - rev_tensor[:, None, 1:]) / rev_tensor[:, 0:1, None]
                     scale = rev_tensor[i, 0]
                     pad_x = rev_tensor[i, 1]
                     pad_y = rev_tensor[i, 1] # Asumiendo padding simétrico
                     # Esta lógica de re-escalado de tu PostProcess original era para V7
                     # Necesitamos adaptarla.
                     # Por ahora, nos centramos en las máscaras.
                     pass


            # --- 4d. Guardar resultados para esta imagen ---
            results_dict = {
                "boxes": final_boxes,
                "labels": final_labels,
                "scores": final_scores,
            }
            if is_segmentation:
                results_dict["masks"] = final_masks

            results_list.append(results_dict)

        # Devolvemos una lista de diccionarios
        return results_list


def collect_prediction(predict_json: List, local_rank: int) -> List:
    """
    Collects predictions from all distributed processes and gathers them on the main process (rank 0).

    Args:
        predict_json (List): The prediction data (can be of any type) generated by the current process.
        local_rank (int): The rank of the current process. Typically, rank 0 is the main process.

    Returns:
        List: The combined list of predictions from all processes if on rank 0, otherwise predict_json.
    """
    if dist.is_initialized() and local_rank == 0:
        all_predictions = [None for _ in range(dist.get_world_size())]
        dist.gather_object(predict_json, all_predictions, dst=0)
        predict_json = [item for sublist in all_predictions for item in sublist]
    elif dist.is_initialized():
        dist.gather_object(predict_json, None, dst=0)
    return predict_json


def predicts_to_json(img_paths, predicts, rev_tensor):
    """
    TODO: function document
    turn a batch of imagepath and predicts(n x 6 for each image) to a List of diction(Detection output)
    """
    batch_json = []
    for img_path, bboxes, box_reverse in zip(img_paths, predicts, rev_tensor):
        scale, shift = box_reverse.split([1, 4])
        bboxes = bboxes.clone()
        bboxes[:, 1:5] = (bboxes[:, 1:5] - shift[None]) / scale[None]
        bboxes[:, 1:5] = transform_bbox(bboxes[:, 1:5], "xyxy -> xywh")
        for cls, *pos, conf in bboxes:
            bbox = {
                "image_id": int(Path(img_path).stem),
                "category_id": IDX_TO_ID[int(cls)],
                "bbox": [float(p) for p in pos],
                "score": float(conf),
            }
            batch_json.append(bbox)
    return batch_json
