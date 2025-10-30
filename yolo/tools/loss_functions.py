from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss

import cv2
import numpy as np

from yolo.config.config import Config, LossConfig, TrainConfig
from yolo.utils.bounding_box_utils import BoxMatcher, Vec2Box, calculate_iou, transform_bbox
from yolo.utils.logger import logger
from yolo.utils.ops import crop_mask


class BCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = BCEWithLogitsLoss(reduction="none")

    def forward(self, predicts_cls: Tensor, targets_cls: Tensor, cls_norm: Tensor) -> Any:
        return self.bce(predicts_cls, targets_cls).sum() / cls_norm


class BoxLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, predicts_bbox: Tensor, targets_bbox: Tensor, valid_masks: Tensor, box_norm: Tensor, cls_norm: Tensor
    ) -> Any:
        # predicts_bbox y targets_bbox ya están en formato xyxy
        valid_bbox_mask = valid_masks[..., None].expand(-1, -1, 4) # Crear máscara [B, A, 4]
        # Seleccionar solo las cajas válidas
        picked_predict = predicts_bbox[valid_masks] # [N_valid, 4]
        picked_targets = targets_bbox[valid_masks] # [N_valid, 4]

        if picked_predict.numel() == 0:
            return torch.tensor(0.0, device=predicts_bbox.device)

        # Calcular IoU diagonalmente (cada predicción con su target asignado)
        iou = calculate_iou(picked_predict.unsqueeze(1), picked_targets.unsqueeze(0), "ciou") # [N_valid, N_valid]
        iou_diag = torch.diag(iou).clamp(0, 1) # Asegurar que esté en [0, 1]

        loss_iou = 1.0 - iou_diag

        # box_norm debería tener tamaño [N_valid] si se calcula correctamente
        if box_norm.shape[0] != loss_iou.shape[0]:
            logger.warning(f"Box norm shape mismatch: {box_norm.shape} vs loss_iou shape: {loss_iou.shape}. Using mean.")
            # Fallback si box_norm no tiene el tamaño correcto
            loss_iou_weighted = loss_iou.mean()
        else:
             loss_iou_weighted = (loss_iou * box_norm).sum() / cls_norm

        return loss_iou_weighted


class DFLoss(nn.Module):
    def __init__(self, vec2box: Vec2Box, reg_max: int) -> None:
        super().__init__()
        # Usamos anchor_grid y scaler directamente de vec2box
        self.vec2box = vec2box
        self.reg_max = reg_max

    def forward(
        self, predicts_dist: Tensor, targets_bbox: Tensor, valid_masks: Tensor, box_norm: Tensor, cls_norm: Tensor
    ) -> Any:
        # targets_bbox está en formato xyxy
        # Necesitamos calcular las distancias ltrb del target respecto al anchor
        
        # Obtener anchors y scalers correspondientes a las predicciones válidas
        # Necesitamos la grid de anchors [A, 2] y el scaler [A]
        anchor_grid = self.vec2box.anchor_grid # [A, 2]
        scaler = self.vec2box.scaler # [A]

        # Expandir anchors y scalers al tamaño del batch y seleccionar válidos
        # anchor_grid [A, 2] -> [1, A, 2] -> [B, A, 2]
        # scaler [A] -> [1, A, 1] -> [B, A, 1]
        batch_anchors = anchor_grid.unsqueeze(0).expand(valid_masks.shape[0], -1, -1)[valid_masks] # [N_valid, 2]
        batch_scalers = scaler.unsqueeze(0).unsqueeze(-1).expand(valid_masks.shape[0], -1, -1)[valid_masks] # [N_valid, 1]

        picked_targets = targets_bbox[valid_masks] # [N_valid, 4] (xyxy)
        picked_predict = predicts_dist[valid_masks] # [N_valid, 4 * reg_max]

        if picked_predict.numel() == 0:
            return torch.tensor(0.0, device=predicts_dist.device)

        # Calcular distancias ltrb del target
        t_lt = batch_anchors - picked_targets[..., :2] # Distancia a top-left [N_valid, 2]
        t_rb = picked_targets[..., 2:] - batch_anchors # Distancia a bottom-right [N_valid, 2]
        # Concatenar y escalar
        targets_dist = torch.cat((t_lt, t_rb), dim=-1) / batch_scalers # [N_valid, 4]
        # Clampear y aplanar
        targets_dist = targets_dist.clamp(0, self.reg_max - 1.01).view(-1) # [N_valid * 4]

        # Reformatear predicción para cross_entropy [N_valid * 4, reg_max]
        picked_predict = picked_predict.view(-1, self.reg_max)

        # Calcular DFL
        label_left = targets_dist.long()
        label_right = label_left + 1
        weight_left = label_right.float() - targets_dist
        weight_right = 1.0 - weight_left

        loss_left = F.cross_entropy(picked_predict, label_left, reduction="none")
        loss_right = F.cross_entropy(picked_predict, label_right, reduction="none")
        loss_dfl = (loss_left * weight_left + loss_right * weight_right)

        # Reformatear a [N_valid, 4], promediar sobre las 4 direcciones, ponderar y normalizar
        loss_dfl = loss_dfl.view(-1, 4).mean(-1) # [N_valid]

        if box_norm.shape[0] != loss_dfl.shape[0]:
            logger.warning(f"DFL Box norm shape mismatch: {box_norm.shape} vs loss_dfl shape: {loss_dfl.shape}. Using mean.")
            loss_dfl_weighted = loss_dfl.mean()
        else:
             loss_dfl_weighted = (loss_dfl * box_norm).sum() / cls_norm

        return loss_dfl_weighted

# Rasterización de los polígonos (máscaras)
def polygons_to_masks(polygons: List[np.ndarray], height: int, width: int) -> Tensor:
    """Convierte una lista de polígonos (formato xy normalizado) a un tensor de máscaras binarias."""
    masks = []
    for polygon in polygons:
        mask = np.zeros((height, width), dtype=np.uint8)
        # Desnormalizar coordenadas
        poly_pixels = (polygon * np.array([width, height])).astype(np.int32)
        cv2.fillPoly(mask, [poly_pixels], 1)
        masks.append(torch.from_numpy(mask))
    if not masks:
        return torch.empty(0, height, width)
    return torch.stack(masks)

# YOLOSegmentationLoss -- Action plan like Ultralytics.
class YOLOSegmentationLoss:
    def __init__(self, loss_cfg: LossConfig, vec2box: Vec2Box, class_num: int = 80, reg_max: int = 16) -> None:
        self.class_num = class_num
        self.vec2box = vec2box
        self.reg_max = reg_max
        
        # Items de pérdidas
        self.cls = BCELoss()
        self.dfl = DFLoss(vec2box, reg_max)
        self.iou = BoxLoss()
        self.bce_mask = BCEWithLogitsLoss(reduction='none') # BinaryCrossEntropy with Logits
        
        # Asignador
        self.matcher = BoxMatcher(loss_cfg.matcher, self.class_num, vec2box, reg_max)
    
    def __call__(self, detect_raw_list: List[Tuple], coeffs_raw_list: List[Tensor], proto: Tensor, targets: Dict) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Calcula las pérdidas de detección y segmentación.

        Args:
            predicts_raw (Tuple): Salidas raw de una cabeza (raw_cls, raw_anc/dist, raw_box/coeffs).
                                    Para DFL: (raw_cls, raw_dist, raw_coeffs)
            proto (Tensor): Prototipos de máscara [B, M_channels, M_h, M_w].
            targets (Dict): Diccionario con 'bboxes' [N, 6] y 'segments' [N].

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: loss_iou, loss_dfl, loss_cls, loss_mask
        """

        device = proto.device
        # raw_cls, raw_dist, raw_coeffs = predicts_raw # desempaquetar la salida
        batch_size = proto.shape[0]
        
        # Concatenar salidas de detección (cls, dist)
        # Asumimos que raw_cls y raw_dist vienen de MultiheadDetection -> Detection -> Anchor2Vec
        # Donde raw_cls es [B, C, H, W] y raw_dist es [B, 4*reg_max, H, W]
        # Y raw_coeffs es [B, M_c, H, W]

        # 1. Llamar a Vec2Box con la LISTA de tuplas raw (detect_raw_list)
        # Vec2Box se encarga internamente de iterar, rearrange y concatenar
        predicts_cls, _, predicts_box_xyxy = self.vec2box(detect_raw_list) 
        # predicts_cls es [B, A_total, C], predicts_box_xyxy es [B, A_total, 4]

        # 2. Concatenar los coeficientes (coeffs_raw_list) manualmente
        all_raw_coeffs = []
        for raw_coeffs in coeffs_raw_list:
            # raw_coeffs [B, M_c, H, W] -> [B, H*W, M_c]
            all_raw_coeffs.append(raw_coeffs.reshape(batch_size, raw_coeffs.shape[1], -1).permute(0, 2, 1)) # USAR RESHAPE
        all_raw_coeffs = torch.cat(all_raw_coeffs, dim=1) # [B, A_total, M_c]

        # 3. Concatenar los raw_dist (anchor_x) para DFLoss
        all_raw_dist = []
        for (_, raw_dist, _) in detect_raw_list: # Extraer el raw_dist de la lista
            # raw_dist [B, 4*reg, H, W] -> [B, H*W, 4*reg]
            all_raw_dist.append(raw_dist.reshape(batch_size, 4 * self.reg_max, -1).permute(0, 2, 1)) # USAR RESHAPE
        all_raw_dist = torch.cat(all_raw_dist, dim=1) # [B, A_total, 4*reg_max]
        target_bboxes_flat = targets['bboxes'].to(device)
        target_list = []
        max_targets_in_batch = 0
        for i in range(batch_size):
            batch_mask = (target_bboxes_flat[:, 0] == i)
            targets_in_image = target_bboxes_flat[batch_mask][:, 1:]
            target_list.append(targets_in_image)
            if targets_in_image.shape[0] > max_targets_in_batch:
                max_targets_in_batch = targets_in_image.shape[0]
        
        if max_targets_in_batch > 0:
            padded_targets = torch.full((batch_size, max_targets_in_batch, 5), -1.0, device=device)
            for i, t in enumerate(target_list):
                if t.numel() > 0:
                    padded_targets[i, :t.shape[0], :] = t
        else:
            padded_targets = torch.empty((batch_size, 0, 5), device=device)
        
        # --- INICIO DEBUG: Verificar padded_targets ---
        # logger.debug("--- DEBUG: Verificando padded_targets antes del Matcher ---")
        # logger.debug(f"batch_size={batch_size}, padded_targets.shape={padded_targets.shape}")
        # if padded_targets.numel() > 0:
        #     target_cls_values = padded_targets[..., 0]  # Extraer solo la columna de clase
        #     # Filtrar el padding (-1) antes de buscar min/max
        #     valid_cls_mask = target_cls_values != -1.0
        #     if valid_cls_mask.any():
        #         valid_classes = target_cls_values[valid_cls_mask]
        #         logger.debug(f"classes min/max: {valid_classes.min().item()}/{valid_classes.max().item()}")
        #         # Comprobar si hay clases fuera del rango esperado [0, class_num-1]
        #         if valid_classes.min() < 0 or valid_classes.max() >= self.class_num:
        #             logger.warning(
        #                 f"¡¡¡ERROR!!! Clases fuera del rango [0, {self.class_num-1}] detectadas."
        #             )
        #     else:
        #         logger.debug("No valid class entries in padded_targets (only padding).")
        # else:
        #     logger.debug("padded_targets está vacío.")
        # logger.debug("--- FIN DEBUG ---")
        
        # FALLO AL LLAMAR AL MATCHER SI HAY CLASES INVÁLIDAS.
        align_targets, valid_masks, gt_indices = self.matcher(
            padded_targets, (predicts_cls.detach(), predicts_box_xyxy.detach())
        )
        # DEBUG: información del matcher
        try:
            logger.debug(
                f"After matcher: valid_masks.any={valid_masks.any().item()}, total_valid={valid_masks.sum().item()}, gt_indices_unique={torch.unique(gt_indices).tolist()[:10]}"
            )
        except Exception:
            logger.debug("After matcher: Could not compute debug stats for valid_masks/gt_indices")
        targets_cls, targets_bbox = align_targets.split((self.class_num, 4), dim=-1)

        cls_norm = max(targets_cls.sum(), 1)
        box_norm = targets_cls.sum(-1)[valid_masks]

        # 2. Calcular Pérdidas de Detección
        loss_cls = self.cls(predicts_cls, targets_cls, cls_norm)
        loss_iou = self.iou(predicts_box_xyxy, targets_bbox, valid_masks, box_norm, cls_norm)
        # Usamos all_raw_dist concatenado para DFLoss
        loss_dfl = self.dfl(all_raw_dist, targets_bbox, valid_masks, box_norm, cls_norm)

        # 3. Calcular Pérdida de Máscara
        loss_mask = torch.tensor(0.0, device=device)
        if valid_masks.any():
            mask_h, mask_w = proto.shape[-2:]
            
            pos_indices_flat = torch.where(valid_masks)
            # Usamos all_raw_coeffs concatenado
            pos_coeffs = all_raw_coeffs[pos_indices_flat] # [N_valid, M_c]
            pos_gt_indices_flat = gt_indices[valid_masks]

            img_indices_in_flat_targets = target_bboxes_flat[:, 0].long()
            start_indices = torch.cat([torch.tensor([0], device=device), torch.cumsum(torch.bincount(img_indices_in_flat_targets), dim=0)[:-1]])
            batch_idx_for_valid = pos_indices_flat[0]
            global_gt_indices = start_indices[batch_idx_for_valid] + pos_gt_indices_flat

            pos_gt_segments = [targets['segments'][idx] for idx in global_gt_indices.tolist()]
            pos_gt_bboxes_xyxy = padded_targets[pos_indices_flat[0], pos_gt_indices_flat.long()][:, 1:]

            if pos_gt_segments:
                gt_masks_tensor = polygons_to_masks(pos_gt_segments, mask_h, mask_w).to(device).float()
                pos_proto = proto[batch_idx_for_valid]
                pos_proto_flat = pos_proto.view(pos_proto.shape[0], pos_proto.shape[1], -1)
                pred_masks_logits = torch.bmm(pos_coeffs.unsqueeze(1), pos_proto_flat).squeeze(1).view(-1, mask_h, mask_w)
                mask_loss_unweighted = self.bce_mask(pred_masks_logits, gt_masks_tensor)
                
                # Normalizar bboxes GT a tamaño de imagen (no tamaño de máscara)
                # self.vec2box.image_size es [W, H], necesitamos [W, H, W, H]
                image_size_tensor = torch.tensor(self.vec2box.image_size * 2, device=device)
                pos_gt_bboxes_norm_img = pos_gt_bboxes_xyxy / image_size_tensor

                # Recortar (crop_mask espera bboxes normalizadas a [0, 1])
                mask_loss_cropped = crop_mask(mask_loss_unweighted, pos_gt_bboxes_norm_img)

                loss_mask_per_instance = mask_loss_cropped.mean(dim=(1, 2))

                if box_norm.shape[0] != loss_mask_per_instance.shape[0]:
                    logger.warning(f"Mask Box norm shape mismatch: {box_norm.shape} vs loss_mask shape: {loss_mask_per_instance.shape}. Using mean.")
                    loss_mask = loss_mask_per_instance.mean()
                else:
                    loss_mask = (loss_mask_per_instance * box_norm).sum() / cls_norm

        return loss_iou, loss_dfl, loss_cls, loss_mask

class DualLoss:
    def __init__(self, cfg: Config, vec2box) -> None:
        loss_cfg = cfg.task.loss # type: ignore
        self.is_segment = 'seg' in cfg.model.name.lower() # type: ignore

        if self.is_segment:
            logger.info(":art: Loading Segmentation Loss")
            # Pasamos los parámetros necesarios a YOLOSegmentationLoss
            self.loss = YOLOSegmentationLoss(loss_cfg, vec2box, class_num=cfg.dataset.class_num, reg_max=cfg.model.anchor.reg_max) # type: ignore
            self.mask_rate = loss_cfg.objective.get("MaskLoss", 7.5)
        else:
            logger.error("Modo detección no implementado en esta versión de DualLoss.")
            raise NotImplementedError("Modo detección no implementado.")
            # self.loss = YOLOLoss(...) # Si tuvieras YOLOLoss
            # self.mask_rate = 0.0

        self.aux_rate = loss_cfg.aux
        self.iou_rate = loss_cfg.objective["BoxLoss"]
        self.dfl_rate = loss_cfg.objective["DFLoss"]
        self.cls_rate = loss_cfg.objective["BCELoss"]

    def __call__(self, 
        aux_predicts: Tuple[List[Tuple], List[Tensor]], # (detect_list, coeffs_list)
        main_predicts: Tuple[List[Tuple], List[Tensor]], # (detect_list, coeffs_list)
        proto_main: Tensor, # Prototipo de Main
        proto_aux: Tensor,  # Prototipo de Aux
        targets: Dict
    ) -> Tuple[Tensor, Dict[str, float]]:
        
        if self.is_segment:
            # Desempaquetar las tuplas
            aux_detect_raw, aux_coeffs_raw = aux_predicts
            main_detect_raw, main_coeffs_raw = main_predicts

            # Pasar las listas separadas a YOLOSegmentationLoss
            # Pasar el prototipo AUX a la pérdida AUX
            aux_iou, aux_dfl, aux_cls, aux_mask = self.loss(aux_detect_raw, aux_coeffs_raw, proto_aux, targets) # type: ignore
            # Pasar el prototipo MAIN a la pérdida MAIN
            main_iou, main_dfl, main_cls, main_mask = self.loss(main_detect_raw, main_coeffs_raw, proto_main, targets) # type: ignore
            total_loss = [
                self.iou_rate * (aux_iou * self.aux_rate + main_iou),
                self.dfl_rate * (aux_dfl * self.aux_rate + main_dfl),
                self.cls_rate * (aux_cls * self.aux_rate + main_cls),
                self.mask_rate * (aux_mask * self.aux_rate + main_mask), 
            ]
            loss_names = ["Box", "DFL", "BCE", "Mask"]
        else:
            logger.error("Modo detección no implementado en esta versión de DualLoss.")
            raise NotImplementedError("Modo detección no implementado.")
        
        loss_dict = {
            f"Loss/{name}Loss": value.detach().item() for name, value in zip(loss_names, total_loss)
        }
        return sum(total_loss), loss_dict # type: ignore


def create_loss_function(cfg: Config, vec2box) -> DualLoss:
    # Pasamos train_cfg directamente a DualLoss
    loss_function = DualLoss(cfg, vec2box)
    logger.info(":white_check_mark: Success load loss function")
    return loss_function
