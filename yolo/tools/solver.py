from math import ceil
from pathlib import Path
import torch

from lightning import LightningModule
from torchmetrics.detection import MeanAveragePrecision

from yolo.utils.logger import logger
from yolo.config.config import Config
from yolo.model.yolo import create_model
from yolo.tools.data_loader import create_dataloader
from yolo.tools.drawer import draw_bboxes
from yolo.tools.loss_functions import create_loss_function
from yolo.utils.bounding_box_utils import create_converter, to_metrics_format
from yolo.utils.model_utils import PostProcess, create_optimizer, create_scheduler
from yolo.tools.loss_functions import polygons_to_masks
from yolo.utils.solver_utils import make_ap_table
import numpy as np
from rich import print


class BaseModel(LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.model = create_model(cfg.model, class_num=cfg.dataset.class_num, weight_path=cfg.weight)

    def forward(self, x):
        return self.model(x)


class ValidateModel(BaseModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        if self.cfg.task.task == "validation":
            self.validation_cfg = self.cfg.task
        else:
            self.validation_cfg = self.cfg.task.validation # type: ignore
        self.metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy", backend="faster_coco_eval")
        self.metric.warn_on_many_detections = False
        
        self.val_loader = create_dataloader(self.validation_cfg.data, self.cfg.dataset, self.validation_cfg.task)
        
        # self.ema = self.model

    def setup(self, stage):
        self.vec2box = create_converter(
            self.cfg.model.name, self.model, self.cfg.model.anchor, self.cfg.image_size, self.device # type: ignore
        )
        self.post_process = PostProcess(self.vec2box, self.validation_cfg.nms)

        # Inicializar EMA aquí
        self.ema = self.model
        
        # Determinar iou_type (preparando el Paso 5 -- Validación)
        if 'seg' in self.cfg.model.name.lower():
            self.iou_type = "segm"
        else:
            self.iou_type = "bbox"

        # Inicializar la métrica aquí
        logger.info(f"Initializing MeanAveragePrecision with iou_type='{self.iou_type}'")
        self.metric = MeanAveragePrecision(iou_type=self.iou_type, box_format="xyxy", backend="faster_coco_eval")
        self.metric.warn_on_many_detections = False
    
    # --- AÑADE ESTA FUNCIÓN ENTERA ---
    def on_validation_epoch_start(self):
        """Se llama al inicio de cada época de validación."""
        # Inicializa el contador de predicciones en el dispositivo correcto.
        self.preds_found_in_epoch = torch.tensor(0, dtype=torch.long, device=self.device)
    # --- FIN DE LA NUEVA FUNCIÓN ---
        
    def val_dataloader(self):
        return self.val_loader

    def validation_step(self, batch, batch_idx):
        # collate_fn returns (images, targets_dict, rev_tensor, img_paths)
        images, targets, rev_tensor, img_paths = batch
        H, W = images.shape[2:]
        batch_size = images.shape[0]
        
        try:
            # Obtener y analizar la salida del modelo
            model_output = self.ema(images)
            
            # Post-proceso para obtener predicciones
            predicts = self.post_process(model_output, image_size=[W, H])
            
            # Preparar predicciones para métricas
            metrics_pred = []
            for predict in predicts:
                mp = to_metrics_format(predict)
                
                ### NUEVO: DESNORMALIZAR PREDICCIONES ###
                # Si las cajas están normalizadas (valores pequeños <= 1.0), las pasamos a píxeles (0-640)
                if mp['boxes'].numel() > 0 and mp['boxes'].max() <= 1.01:
                     mp['boxes'][:, [0, 2]] *= W
                     mp['boxes'][:, [1, 3]] *= H
                metrics_pred.append(mp)
                                ### FIN NUEVO ###
            
            # Preparar targets para métricas
            target_bboxes_flat = targets['bboxes'].to(self.device)
            target_segments_list = targets['segments']
                        
            # Obtener índices de inicio para segmentos
            img_indices_in_flat_targets = target_bboxes_flat[:, 0].long()
            counts = torch.bincount(img_indices_in_flat_targets.cpu(), minlength=batch_size)
            start_indices = torch.cat([torch.tensor([0]), torch.cumsum(counts, dim=0)[:-1]]).tolist()

            # Construir métricas target por imagen
            metrics_target = []
            for i in range(batch_size):
                mask = (target_bboxes_flat[:, 0] == i)
                bboxes_for_image = target_bboxes_flat[mask][:, 1:]
                num_gts = bboxes_for_image.shape[0]
                
                # Obtener y rasterizar segmentos
                start_idx = start_indices[i]
                segments_for_image = target_segments_list[start_idx : start_idx + num_gts]
                gt_masks_tensor = polygons_to_masks(segments_for_image, H, W).to(self.device) if num_gts > 0 else torch.empty(0, H, W, device=self.device)
                
                ### NUEVO: DESNORMALIZAR GROUND TRUTH (TARGETS) ###
                gt_boxes = bboxes_for_image[:, 1:].clone() # Copiamos para no afectar a otros procesos
                # Si tus TXT tienen valores entre 0 y 1, esto los pasará a 0-640
                if gt_boxes.numel() > 0 and gt_boxes.max() <= 1.01:
                    gt_boxes[:, [0, 2]] *= W
                    gt_boxes[:, [1, 3]] *= H
                
                # Construir diccionario target
                target_dict = {
                    "boxes": gt_boxes,
                    "labels": bboxes_for_image[:, 0].int(),
                    "masks": gt_masks_tensor.bool()
                }
                metrics_target.append(target_dict)
                
                # --- INICIO DE CORRECCIÓN DE DDP ---
                # Contar cuántas predicciones reales se hicieron en este batch
                num_preds = sum(p['boxes'].shape[0] for p in metrics_pred)
                if num_preds > 0:
                    self.preds_found_in_epoch += num_preds
                # --- FIN DE CORRECCIÓN ---

            # Actualizar métricas
            # --- DEBUG DE VALIDACION --- #
            # for i, (pred, tgt) in enumerate(zip(metrics_pred, metrics_target)):
            #     logger.info(f"--- IMAGE {i} ---")
            #     # Preds
            #     logger.info(
            #         f"PRED boxes: {pred['boxes'].shape}, min={pred['boxes'].min().item() if pred['boxes'].numel()>0 else None}, "
            #         f"max={pred['boxes'].max().item() if pred['boxes'].numel()>0 else None}, "
            #         f"labels: {pred['labels'].cpu().numpy()[:8] if pred['labels'].numel()>0 else None}, "
            #         f"scores: {pred['scores'].cpu().numpy()[:8] if 'scores' in pred and pred['scores'].numel()>0 else None}"
            #     )
            #     msk = pred.get('masks', None)
            #     if msk is not None and msk.numel() > 0:
            #         logger.info(f"PRED masks: {msk.shape}, dtype={msk.dtype}, unique={torch.unique(msk).cpu().numpy()}")
            #     # GT
            #     logger.info(
            #         f"GT boxes: {tgt['boxes'].shape}, labels: {tgt['labels'].cpu().numpy()[:8] if tgt['labels'].numel()>0 else None}"
            #     )
            #     msk_gt = tgt.get('masks', None)
            #     if msk_gt is not None and msk_gt.numel() > 0:
            #         logger.info(f"GT masks: {msk_gt.shape}, dtype={msk_gt.dtype}, unique={torch.unique(msk_gt).cpu().numpy()}")
            # # --- FIN DEBUG --- #
            mAP = self.metric(metrics_pred, metrics_target)
            
        except Exception as e:
            logger.error(f"Error in validation step: {str(e)}")
            raise

    def on_validation_epoch_end(self):
        
        if self.preds_found_in_epoch == 0:
            logger.info("No se encontraron predicciones en ningún batch de validación. Omitiendo cálculo de mAP.")
            self.log_dict({"map": 0.0, "map_50": 0.0}, prog_bar=True, sync_dist=True, rank_zero_only=True)
            self.metric.reset()
            self.preds_found_in_epoch = 0 # Reiniciar para la próxima época
            return # Salir de la función
        logger.info("Calculando metricas de final de epoca")
        epoch_metrics = self.metric.compute()
        logger.info("Metricas computadas")
        logger.info(f"Epoch {self.current_epoch} Validation Metrics Computed: {epoch_metrics}")
        
        score = [
            epoch_metrics.get('map', 0.0),
            epoch_metrics.get('map_50', 0.0),
            epoch_metrics.get('map_75', 0.0),
            epoch_metrics.get('map_small', 0.0),
            epoch_metrics.get('map_medium', 0.0),
            epoch_metrics.get('map_large', 0.0),
            epoch_metrics.get('mar_1', 0.0),
            epoch_metrics.get('mar_10', 0.0),
            epoch_metrics.get('mar_100', 0.0),
            epoch_metrics.get('mar_small', 0.0),
            epoch_metrics.get('mar_medium', 0.0),
            epoch_metrics.get('mar_large', 0.0),
        ]
        
        # Crear e imprimir la tabla
        max_result = np.zeros(12)
        ap_table, _ = make_ap_table(score, max_result=max_result, epoch=self.current_epoch)
        logger.info(f"Resultados de Validación Época {self.current_epoch}:")
        print(ap_table)
        del epoch_metrics["classes"]
        self.log_dict(epoch_metrics, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log_dict(
            {"PyCOCO/AP @ .5:.95": epoch_metrics["map"], "PyCOCO/AP @ .5": epoch_metrics["map_50"]},
            sync_dist=True,
            rank_zero_only=True,
        )
        self.metric.reset()
        self.preds_found_in_epoch = 0


class TrainModel(ValidateModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        # NO hay conversión manual de self.cfg.task aquí
        self.train_loader = create_dataloader(self.cfg.task.data, self.cfg.dataset, self.cfg.task.task) # type: ignore
        # Diagnostic flag to avoid repeating heavy logs
        self._diagnostics_logged = False

    def setup(self, stage):
        super().setup(stage)
        # NO hay conversión manual de self.cfg.task aquí
        # Vuelve a la llamada original
        self.loss_fn = create_loss_function(self.cfg, self.vec2box) # type: ignore
    def train_dataloader(self):
        return self.train_loader

    def on_train_epoch_start(self):
        # self.trainer.optimizers[0].next_epoch( # type: ignore
        #     ceil(len(self.train_loader) / self.trainer.world_size), self.current_epoch
        # )
        self.vec2box.update(self.cfg.image_size)

    # def on_train_start(self):
    #     """Log optimizer info at the start of training (diagnostic)."""
    #     try:
    #         optimizers = getattr(self.trainer, "optimizers", None)
    #         if optimizers and len(optimizers) > 0:
    #             opt = optimizers[0]
    #             pg = opt.param_groups[0]
    #             msg = f"TRAIN START: Optimizer={opt.__class__.__name__}, lr={pg.get('lr')}, weight_decay={pg.get('weight_decay')}, amsgrad={pg.get('amsgrad', None)}"
    #             logger.info(msg)
    #             # Ensure visible on consoles that may not capture logger
    #             print(msg)
    #         else:
    #             logger.info("TRAIN START: No optimizer found on trainer yet.")
    #             print("TRAIN START: No optimizer found on trainer yet.")
    #     except Exception:
    #         logger.exception("Error logging optimizer info on_train_start")
    #         print("Error logging optimizer info on_train_start")

    # def on_after_backward(self):
    #     """Light-weight gradient diagnostics after backward (diagnostic)."""
    #     try:
    #         total_params = 0
    #         params_with_grad = 0
    #         max_grad = 0.0
    #         mean_grad = 0.0
    #         cnt = 0
    #         for p in self.model.parameters():
    #             total_params += 1
    #             if p.grad is not None:
    #                 params_with_grad += 1
    #                 g = p.grad.detach()
    #                 try:
    #                     max_g = float(g.abs().max().item())
    #                     mean_g = float(g.abs().mean().item())
    #                 except Exception:
    #                     max_g = 0.0
    #                     mean_g = 0.0
    #                 if max_g > max_grad:
    #                     max_grad = max_g
    #                 mean_grad += mean_g
    #                 cnt += 1
    #         mean_grad = (mean_grad / cnt) if cnt > 0 else 0.0
    #         msg = f"AFTER_BACKWARD: params_with_grad={params_with_grad}/{total_params}, max_grad={max_grad:.6g}, mean_grad={mean_grad:.6g}"
    #         logger.info(msg)
    #         print(msg)
    #     except Exception:
    #         logger.exception("Error computing gradient stats in on_after_backward")
    #         print("Error computing gradient stats in on_after_backward")

    def training_step(self, batch, batch_idx):
        # lr_dict = self.trainer.optimizers[0].next_batch() # type: ignore
        # Ahora batch = (images, targets_dict, rev_tensor, img_paths)
        images, targets, *_ = batch 
        batch_size = images.shape[0]
        # Fallback diagnostic: log optimizer info on first batch if on_train_start didn't show up
        if not getattr(self, '_diagnostics_logged', False):
            try:
                optimizers = getattr(self.trainer, 'optimizers', None)
                if optimizers and len(optimizers) > 0:
                    opt = optimizers[0]
                    pg = opt.param_groups[0]
                    msg = f"FIRST_BATCH DIAG: Optimizer={opt.__class__.__name__}, lr={pg.get('lr')}, weight_decay={pg.get('weight_decay')}"
                    logger.info(msg)
                    print(msg)
                else:
                    logger.info("FIRST_BATCH DIAG: No optimizer found on trainer yet.")
                    print("FIRST_BATCH DIAG: No optimizer found on trainer yet.")
            except Exception:
                logger.exception("Error logging optimizer info in training_step first batch")
                print("Error logging optimizer info in training_step first batch")
            self._diagnostics_logged = True
        # targets es un dict {'bboxes': [N, 6], 'segments': [N]}
        predicts = self(images) # Salida del modelo (diccionario)
        
        # Extracción de salidas y llamada a loss_fn ---
        try:
            # predicts["Main"] es una TUPLA: (detection_outputs, segmentation_outputs)
            aux_outputs: Tuple[List[Tuple], List[Tensor]] = predicts["AUX"] 
            main_outputs: Tuple[List[Tuple], List[Tensor]] = predicts["Main"]

            # 1. Extraer Salidas de Detección
            aux_detect_raw = aux_outputs[0]  # Esto es Item 0: la lista de tuplas (cls, dist, box)
            main_detect_raw = main_outputs[0] # Esto es Item 0: la lista de tuplas (cls, dist, box)
            
            # 2. Extraer Salidas de Segmentación
            aux_seg_list = aux_outputs[1]    # Esto es Item 1: la lista de [coeffs..., proto]
            main_seg_list = main_outputs[1]  # Esto es Item 1: la lista de [coeffs..., proto]

            # 3. Extraer AMBOS Prototipos
            if isinstance(main_seg_list, list) and len(main_seg_list) > 0 and isinstance(main_seg_list[-1], torch.Tensor):
                proto_main = main_seg_list[-1]
                main_coeffs_raw = main_seg_list[:-1]
            else:
                raise ValueError(f"No se pudieron encontrar los prototipos 'Main'.")
            
            if isinstance(aux_seg_list, list) and len(aux_seg_list) > 0 and isinstance(aux_seg_list[-1], torch.Tensor):
                proto_aux = aux_seg_list[-1]
                aux_coeffs_raw = aux_seg_list[:-1]
            else:
                raise ValueError(f"No se pudieron encontrar los prototipos 'AUX'.")

            # 5. Llamar a la función de pérdida con ambos prototipos
            loss, loss_item = self.loss_fn(
                (aux_detect_raw, aux_coeffs_raw),   # Tupla Aux
                (main_detect_raw, main_coeffs_raw), # Tupla Main
                proto_main, # Prototipos Main
                proto_aux,  # Prototipos Aux
                targets  # type: ignore
            )
        except KeyError as e:
            logger.error(f"Error: Clave esperada '{e}' no encontrada en la salida del modelo 'predicts'. Claves disponibles: {list(predicts.keys())}")
            logger.error("Asegúrate de que las capas con 'output: True' en tu YAML tengan las 'tags' correctas ('AUX', 'Main').")
            # Podrías añadir un breakpoint() aquí para inspeccionar predicts
            # breakpoint()
            raise
        except (TypeError, IndexError, ValueError) as e:
            logger.error(f"Error al procesar la salida del modelo o al llamar a loss_fn: {e}")
            logger.error("Verifica la estructura devuelta por 'MultiheadSegmentation' y la esperada por 'YOLOSegmentationLoss.__call__'.")
            # breakpoint()
            raise
        
        # log_dict de pérdidas.
        self.log_dict(
            loss_item,
            prog_bar=True,
            on_epoch=True,
            on_step=True, # ver las métricas por pasos
            batch_size=batch_size,
            rank_zero_only=True,
        )
        # Si todas las componentes de la pérdida son cero, registrar información de depuración
        try:
            if all(float(v) == 0.0 for v in loss_item.values()):
                num_bboxes = targets.get("bboxes").shape[0] if isinstance(targets, dict) and "bboxes" in targets else -1
                num_segments = len(targets.get("segments", [])) if isinstance(targets, dict) else -1
                logger.warning(
                    f"All loss items are zero at batch {batch_idx}. bboxes={num_bboxes}, segments={num_segments}"
                )
        except Exception:
            # No fallar por problemas de logging
            pass
        #self.log_dict(lr_dict, prog_bar=False, logger=True, on_epoch=False, rank_zero_only=True)
        return loss # No multiplicar por batch_size si la pérdida ya está normalizada por lote

    def configure_optimizers(self):
        optimizer = create_optimizer(self.model, self.cfg.task.optimizer)
        scheduler = create_scheduler(optimizer, self.cfg.task.scheduler)
        return [optimizer], [scheduler]


class InferenceModel(BaseModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        # TODO: Add FastModel
        self.predict_loader = create_dataloader(cfg.task.data, cfg.dataset, cfg.task.task)

    def setup(self, stage):
        self.vec2box = create_converter(
            self.cfg.model.name, self.model, self.cfg.model.anchor, self.cfg.image_size, self.device
        )
        self.post_process = PostProcess(self.vec2box, self.cfg.task.nms)

    def predict_dataloader(self):
        return self.predict_loader

    def predict_step(self, batch, batch_idx):
        images, rev_tensor, origin_frame = batch
        predicts = self.post_process(self(images), rev_tensor=rev_tensor)
        img = draw_bboxes(origin_frame, predicts, idx2label=self.cfg.dataset.class_list)
        if getattr(self.predict_loader, "is_stream", None):
            fps = self._display_stream(img)
        else:
            fps = None
        if getattr(self.cfg.task, "save_predict", None):
            self._save_image(img, batch_idx)
        return img, fps

    def _save_image(self, img, batch_idx):
        save_image_path = Path(self.trainer.default_root_dir) / f"frame{batch_idx:03d}.png"
        img.save(save_image_path)
        print(f"💾 Saved visualize image at {save_image_path}")
