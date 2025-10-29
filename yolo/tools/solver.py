from math import ceil
from pathlib import Path
import torch

from lightning import LightningModule
from torchmetrics.detection import MeanAveragePrecision

from yolo.utils.logger import logger
from yolo.config.config import Config, TrainConfig 
from omegaconf import OmegaConf
from yolo.config.config import Config
from yolo.model.yolo import create_model
from yolo.tools.data_loader import create_dataloader
from yolo.tools.drawer import draw_bboxes
from yolo.tools.loss_functions import create_loss_function
from yolo.utils.bounding_box_utils import create_converter, to_metrics_format
from yolo.utils.model_utils import PostProcess, create_optimizer, create_scheduler


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
        self.ema = self.model

    def setup(self, stage):
        self.vec2box = create_converter(
            self.cfg.model.name, self.model, self.cfg.model.anchor, self.cfg.image_size, self.device # type: ignore
        )
        self.post_process = PostProcess(self.vec2box, self.validation_cfg.nms)

    def val_dataloader(self):
        return self.val_loader

    def validation_step(self, batch, batch_idx):
        batch_size, images, targets, rev_tensor, img_paths = batch
        H, W = images.shape[2:]
        predicts = self.post_process(self.ema(images), image_size=[W, H])
        mAP = self.metric(
            [to_metrics_format(predict) for predict in predicts], [to_metrics_format(target) for target in targets]
        )
        return predicts, mAP

    def on_validation_epoch_end(self):
        epoch_metrics = self.metric.compute()
        del epoch_metrics["classes"]
        self.log_dict(epoch_metrics, prog_bar=True, sync_dist=True, rank_zero_only=True)
        self.log_dict(
            {"PyCOCO/AP @ .5:.95": epoch_metrics["map"], "PyCOCO/AP @ .5": epoch_metrics["map_50"]},
            sync_dist=True,
            rank_zero_only=True,
        )
        self.metric.reset()


class TrainModel(ValidateModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        # NO hay conversión manual de self.cfg.task aquí
        self.train_loader = create_dataloader(self.cfg.task.data, self.cfg.dataset, self.cfg.task.task) # type: ignore

    def setup(self, stage):
        super().setup(stage)
        # NO hay conversión manual de self.cfg.task aquí
        # Vuelve a la llamada original
        self.loss_fn = create_loss_function(self.cfg, self.vec2box) # type: ignore
    def train_dataloader(self):
        return self.train_loader

    def on_train_epoch_start(self):
        self.trainer.optimizers[0].next_epoch( # type: ignore
            ceil(len(self.train_loader) / self.trainer.world_size), self.current_epoch
        )
        self.vec2box.update(self.cfg.image_size)

    def training_step(self, batch, batch_idx):
        lr_dict = self.trainer.optimizers[0].next_batch() # type: ignore
        # Ahora batch = (images, targets_dict, rev_tensor, img_paths)
        images, targets, *_ = batch 
        batch_size = images.shape[0]
        # targets es un dict {'bboxes': [N, 6], 'segments': [N]}
        predicts = self(images) # Salida del modelo (diccionario)
        
        # DEBUG: Inspeccionar salida del modelo ---
        if batch_idx == 0: 
            print("\n--- DEBUG: Salida del Modelo (predicts) ---")
            print(f"Claves encontradas: {list(predicts.keys())}")

            def print_nested_shapes(item, indent=""):
                """Función recursiva para imprimir formas de tensores anidados."""
                if isinstance(item, torch.Tensor):
                    print(f"{indent}Tensor{item.shape}")
                elif isinstance(item, (list, tuple)):
                    # Si es lista/tupla vacía, indicarlo
                    if not item:
                         print(f"{indent}Empty List/Tuple []")
                         return 
                    # Si no está vacía, imprimir tipo y longitud, luego iterar
                    print(f"{indent}{type(item).__name__} con {len(item)} elementos:")
                    for i, sub_item in enumerate(item):
                        print(f"{indent}  Item {i}:")
                        print_nested_shapes(sub_item, indent + "    ")
                else:
                    print(f"{indent}Tipo: {type(item)}")

            for key, value in predicts.items():
                print(f"  '{key}':")
                print_nested_shapes(value, indent="    ") # Llamar a la función recursiva

            print("--- FIN DEBUG ---")
        # ---------------------------------------------------------
        
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

            # 3. Extraer Prototipos
            # El DEBUG muestra que Item 1 (main_seg_list) tiene 4 tensores. El último es el proto.
            if isinstance(main_seg_list, list) and len(main_seg_list) > 0 and isinstance(main_seg_list[-1], torch.Tensor):
                 proto = main_seg_list[-1]         # El último tensor es el prototipo
                 main_coeffs_raw = main_seg_list[:-1] # Todos menos el último son coeficientes
            else:
                 # Este es el error que estabas viendo:
                 raise ValueError(f"No se pudieron encontrar los prototipos en 'predicts[\"Main\"][1]'. Estructura encontrada: {main_seg_list}")

            # 4. Extraer Coeficientes Auxiliares
            if isinstance(aux_seg_list, list) and len(aux_seg_list) > 0:
                 aux_coeffs_raw = aux_seg_list[:-1] # Asumimos que también tiene protos al final
            else:
                 aux_coeffs_raw = [] # O manejar error si se esperan siempre

            # 5. Llamar a la función de pérdida
            loss, loss_item = self.loss_fn(
                (aux_detect_raw, aux_coeffs_raw), # Tupla (Detección_Aux, Coeficientes_Aux)
                (main_detect_raw, main_coeffs_raw), # Tupla (Detección_Main, Coeficientes_Main)
                proto, 
                targets
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
        
        # Logging (sin cambios)
        self.log_dict(
            loss_item,
            prog_bar=True,
            on_epoch=True,
            batch_size=batch_size,
            rank_zero_only=True,
        )
        self.log_dict(lr_dict, prog_bar=False, logger=True, on_epoch=False, rank_zero_only=True)
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
