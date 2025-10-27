# from math import ceil
# from pathlib import Path

# from lightning import LightningModule
# from torchmetrics.detection import MeanAveragePrecision

from yolo.config.config import Config
# from yolo.model.yolo import create_model
# from yolo.tools.data_loader import create_dataloader
# from yolo.tools.drawer import draw_bboxes
# from yolo.tools.loss_functions import create_loss_function
# from yolo.utils.bounding_box_utils import create_converter, to_metrics_format
# from yolo.utils.model_utils import PostProcess, create_optimizer, create_scheduler
# import torch

import torch # For general tensor operations
import torch.nn.functional as F # For functions like interpolate
from einops import rearrange # For tensor reshaping
# from torchvision.ops import batched_nms # Potentially needed for NMS implementation in validation_step
# import numpy as np # Potentially needed for GT mask generation in validation_step
# import cv2 # Potentially needed for GT mask generation in validation_step

# Keep existing imports like:
from math import ceil
from pathlib import Path
from lightning import LightningModule
from torchmetrics.detection import MeanAveragePrecision
from yolo.config.config import Config
from yolo.model.yolo import create_model
from yolo.tools.data_loader import create_dataloader
# from yolo.tools.drawer import draw_bboxes # Likely not needed for segmentation training/validation loop itself
from yolo.tools.loss_functions import create_loss_function
from yolo.utils.bounding_box_utils import (
    create_converter,
    to_metrics_format,
    bbox_nms, # Make sure this import is present
)
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
            self.validation_cfg = self.cfg.task.validation
        # self.metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy", backend="faster_coco_eval")
        # --- MODIFIED: Use segmentation metric ---
        # Requires predictions to include 'masks' tensor later
        self.metric = MeanAveragePrecision(iou_type="segm", box_format="xyxy", backend="faster_coco_eval")
        self.metric.warn_on_many_detections = False
        self.val_loader = create_dataloader(self.validation_cfg.data, self.cfg.dataset, self.validation_cfg.task)
        self.ema = self.model

    def setup(self, stage):
        self.vec2box = create_converter(
            self.cfg.model.name, self.model, self.cfg.model.anchor, self.cfg.image_size, self.device
        )
        self.post_process = PostProcess(self.vec2box, self.validation_cfg.nms)

    def val_dataloader(self):
        return self.val_loader

    # def validation_step(self, batch, batch_idx):
    #     batch_size, images, targets, rev_tensor, img_paths = batch
    #     H, W = images.shape[2:]
    #     predicts = self.post_process(self.ema(images), image_size=[W, H])
    #     mAP = self.metric(
    #         [to_metrics_format(predict) for predict in predicts], [to_metrics_format(target) for target in targets]
    #     )
    #     return predicts, mAP
    
    def validation_step(self, batch, batch_idx):
        # --- MODIFIED: Unpack 6 items ---
        batch_size, images, targets_bboxes, targets_segments, rev_tensor, img_paths = batch
        images = images.to(self.device)
        targets_bboxes = targets_bboxes.to(self.device)
        # targets_segments remain on CPU for potential GT mask generation if needed by metric format
        H, W = images.shape[2:]
        # --- Get Raw Model Output ---
        with torch.no_grad():
            predicts = self.ema(images) # predicts = (detection_outputs, segmentation_outputs)

        # --- MODIFIED: Segmentation Post-processing ---
        # This needs to:
        # 1. Process detection_outputs (apply NMS to get final boxes, scores, labels)
        # 2. Reconstruct masks corresponding ONLY to the final boxes after NMS
        # 3. Format output for self.metric(iou_type="segm")

        # --- Placeholder for complex post-processing ---
        # Example structure (needs actual implementation)
        
        # a) Process detection part (similar to detection validation, get boxes after NMS)
        detection_outputs, segmentation_outputs = predicts['Main'] # Assuming 'Main' key
        preds_cls, _, preds_box_dist = [], [], [] # Collect raw detection outputs
        all_pred_mask_coeffs = [] # Collect raw coefficients
        num_mask_heads = len(segmentation_outputs) - 1
        
        # Unpack raw outputs (similar to loss function)
        for i in range(len(detection_outputs)):
             pred_cls, pred_anc, pred_box = detection_outputs[i]
             preds_cls.append(rearrange(pred_cls, "B C h w -> B (h w) C"))
             # We might not need pred_anc here if DFLoss isn't used in postproc
             preds_box_dist.append(rearrange(pred_box, "B X h w -> B (h w) X"))
             # Get corresponding coefficients
             if i < num_mask_heads:
                 pred_mask_coeff = segmentation_outputs[i]
                 all_pred_mask_coeffs.append(rearrange(pred_mask_coeff, "B M h w -> B (h w) M"))

        preds_cls = torch.cat(preds_cls, dim=1).sigmoid() # Apply sigmoid for probabilities
        preds_box_dist = torch.cat(preds_box_dist, dim=1)
        preds_mask_coeffs = torch.cat(all_pred_mask_coeffs, dim=1) #[B, Num_Total_Preds, Num_Mask_Coeffs]
        protos = segmentation_outputs[-1] #[B, Num_Mask_Coeffs, Mask_H, Mask_W]

        # Convert box distributions to bboxes (xyxy)
        vec2box = self.vec2box # vec2box should be available via __init__
        pred_LTRB = preds_box_dist * vec2box.scaler.view(1, -1, 1)
        lt, rb = pred_LTRB.chunk(2, dim=-1)
        preds_box = torch.cat([vec2box.anchor_grid - lt, vec2box.anchor_grid + rb], dim=-1) # [B, Num_Total_Preds, 4]
        
        # b) Apply NMS to Boxes/Scores (using bbox_nms utility)
        # Note: bbox_nms needs raw class logits/sigmoid probs, not just max confidence
        # We might need to adapt bbox_nms or use torchvision.ops.batched_nms directly
        # Example using a simplified NMS approach for now:
        final_preds_for_metric = []
        for i in range(batch_size): # Process each image in the batch
             
             # Perform NMS on boxes and scores for this image
             # (Needs implementation - e.g., using torchvision.ops.batched_nms)
             # Let's assume nms_indices contains the indices of predictions kept after NMS
             # scores, boxes, classes = ... apply nms(preds_box[i], preds_cls[i], ...)
             # nms_indices = indices_kept_by_nms 
             
             # --- Placeholder ---
             # For now, let's just take top K predictions as a placeholder for NMS output
             conf, _ = preds_cls[i].max(1)
             _, topk_indices = torch.topk(conf, k=100) # Keep top 100 for now
             nms_indices = topk_indices 
             # --- End Placeholder ---

             if nms_indices.numel() == 0:
                  # Handle case with no detections after NMS
                  final_preds_for_metric.append({
                       "boxes": torch.empty(0, 4, device=self.device),
                       "scores": torch.empty(0, device=self.device),
                       "labels": torch.empty(0, dtype=torch.long, device=self.device),
                       "masks": torch.empty(0, H, W, dtype=torch.bool, device=self.device) # Expected mask format
                  })
                  continue

             # c) Reconstruct Masks ONLY for NMS survivors
             coeffs_nms = preds_mask_coeffs[i, nms_indices] # [N_nms, Num_Coeffs]
             protos_img = protos[i] # [Num_Coeffs, Mask_H, Mask_W]

             # Matrix multiply: [N_nms, C] @ [C, H*W] -> [N_nms, H*W]
             masks_low_res_flat = coeffs_nms @ protos_img.view(protos_img.shape[0], -1) 
             masks_low_res = masks_low_res_flat.view(-1, protos_img.shape[-2], protos_img.shape[-1]) #[N_nms, Mask_H, Mask_W]

             # d) Upsample masks to original image size (H, W from batch)
             # NOTE: Check if metric expects logits or sigmoid masks
             masks_full_res = F.interpolate(masks_low_res.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)
             
             # e) Format for metric (boxes, scores, labels, masks)
             # Apply sigmoid and thresholding for boolean masks if required by metric
             final_masks = (masks_full_res.sigmoid() > 0.5) # Example thresholding to get boolean masks [N_nms, H, W]
             
             final_scores, final_labels = preds_cls[i, nms_indices].max(1)
             final_boxes = preds_box[i, nms_indices]
             
             final_preds_for_metric.append({
                  "boxes": final_boxes,
                  "scores": final_scores,
                  "labels": final_labels,
                  "masks": final_masks # Needs to be boolean [N, H, W] for torchmetrics mAP segm
             })

        # --- Format Ground Truth for Metric ---
        # Needs to convert targets_bboxes + targets_segments into the format expected by the metric
        # Usually: list of dicts, each with "boxes", "labels", "masks" (boolean [N_gt, H, W])
        target_list_for_metric = []
        # --- Placeholder ---
        # You need a function similar to _process_segments_to_masks but outputting boolean masks
        # of size [N_gt, H, W] for each image, aligned with targets_bboxes
        # Example structure:
        for i in range(batch_size):
            # gt_masks_i = generate_boolean_gt_masks(targets_segments[i], targets_bboxes[i], H, W) 
            # valid_gt = targets_bboxes[i, :, 0] != -1
            # target_list_for_metric.append({
            #      "boxes": targets_bboxes[i, valid_gt, 1:],
            #      "labels": targets_bboxes[i, valid_gt, 0].long(),
            #      "masks": gt_masks_i # Boolean [N_gt, H, W]
            # })
             # --- Simplified Placeholder ---
             valid_gt = targets_bboxes[i, :, 0] != -1
             num_gt = valid_gt.sum()
             target_list_for_metric.append({
                  "boxes": targets_bboxes[i, valid_gt, 1:],
                  "labels": targets_bboxes[i, valid_gt, 0].long(),
                  "masks": torch.zeros(num_gt, H, W, dtype=torch.bool, device=self.device) # Empty masks for now
             })
        # --- End Placeholder ---
             
        # --- Calculate Metric ---
        # self.metric update expects list[preds], list[targets]
        self.metric.update(final_preds_for_metric, target_list_for_metric)

        # We don't need to return predicts/mAP per step for validation like before
        # Logging happens in on_validation_epoch_end
        # Return None or necessary info if callbacks need it
        return final_preds_for_metric # Or return something if needed by other parts of Lightning

    def on_validation_epoch_end(self):
        epoch_metrics = self.metric.compute()
        # Clean up metric state
        self.metric.reset() 
        
        # Remove class-specific metrics if they exist
        if "classes" in epoch_metrics:
            del epoch_metrics["classes"]
            
        # Log the main mAP scores (names like 'map', 'map_50' are standard for torchmetrics)
        self.log_dict(epoch_metrics, prog_bar=True, sync_dist=True, rank_zero_only=True)
        
        # Log specific keys if needed elsewhere (optional, adjust names if desired)
        self.log_dict(
            {"PyCOCO/mAP_Seg": epoch_metrics["map"], "PyCOCO/mAP50_Seg": epoch_metrics["map_50"]},
            prog_bar=False, # Don't show these duplicates on prog bar
            logger=True, 
            sync_dist=True,
            rank_zero_only=True,
        )


class TrainModel(ValidateModel):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.train_loader = create_dataloader(self.cfg.task.data, self.cfg.dataset, self.cfg.task.task)

    def setup(self, stage):
        super().setup(stage)
        # Renamed for clarity, create_loss_function now handles seg vs det
        self.loss_fn = create_loss_function(self.cfg, self.vec2box)

    def train_dataloader(self):
        return self.train_loader

    def on_train_epoch_start(self):
        self.trainer.optimizers[0].next_epoch(
            ceil(len(self.train_loader) / self.trainer.world_size), self.current_epoch
        )
        self.vec2box.update(self.cfg.image_size)

    # def training_step(self, batch, batch_idx):
    #     lr_dict = self.trainer.optimizers[0].next_batch()
    #     # batch_size, images, targets, *_ = batch
    #     batch_size, images, targets_bboxes, targets_segments, *_ = batch ## MODIFICACIÓN PARA SEGM
    #     predicts = self(images) # contiene las salidas de detecc y seg
    #     ## --- ELIMINACIÓN DE CONVERSIÓN A BBOX ---## 
    #     # aux_predicts = self.vec2box(predicts["AUX"])
    #     # main_predicts = self.vec2box(predicts["Main"])
    #     # ---
    #     # Necesitamos una NUEVA función de pérdida para segmentación.
    #     # Esta función necesitará:
    #     #   - predicts["Main"] (salida del modelo)
    #     #   - predicts["AUX"] (salida auxiliar del modelo)
    #     #   - targets_bboxes (cajas de verdad)
    #     #   - targets_segments (polígonos de verdad)

    #     # loss, loss_item = self.loss_fn(aux_predicts, main_predicts, targets)
    #     # LLAMADA A LA NUEVA FUNCIÓN DE PÉRDIDA (aún por crear)
    #     # loss, loss_item_seg = self.segmentation_loss_fn(
    #     #     predicts, targets_bboxes, targets_segments
    #     # )
    #     self.log_dict(
    #         loss_item_seg, # Deberá contener BoxLoss, ClsLoss, MaskLoss # ?? Porque lo dice gemini
    #         prog_bar=True,
    #         on_epoch=True,
    #         batch_size=batch_size,
    #         rank_zero_only=True,
    #     )
    #     self.log_dict(lr_dict, prog_bar=False, logger=True, on_epoch=False, rank_zero_only=True)
    #     return loss * batch_size

    
    def training_step(self, batch, batch_idx):
        lr_dict = self.trainer.optimizers[0].next_batch()
        
        # --- MODIFIED: Unpack 6 items ---
        batch_size, images, targets_bboxes, targets_segments, *_ = batch 
        
        # --- MODIFIED: Ensure data is on the correct device ---
        images = images.to(self.device)
        targets_bboxes = targets_bboxes.to(self.device)
        # targets_segments is a list, remains on CPU for now (used in loss_fn)
        
        # Call the model (v9-c-seg)
        predicts = self(images) # 'predicts' = (detection_outputs, segmentation_outputs)
        
        # --- MODIFIED: Call the appropriate loss function ---
        # self.loss_fn is now YOLOSegmentationLoss (or DualLoss if detection)
        # Pass raw predicts, bboxes, and segments
        loss, loss_item = self.loss_fn(predicts, targets_bboxes, targets_segments) 
        
        # Log the loss components (Box, Cls, DFL, Mask)
        self.log_dict(
            loss_item, # Dictionary returned by the loss function
            prog_bar=True,
            on_epoch=True,
            batch_size=batch_size,
            sync_dist=True, # Ensure logging across GPUs if applicable
            rank_zero_only=True,
        )
        self.log_dict(lr_dict, prog_bar=False, logger=True, on_epoch=False, rank_zero_only=True)
        
        # Return loss * batch_size (standard practice in some frameworks)
        # Check if loss needs scaling or if optimizer handles it
        return loss # Or potentially loss * batch_size if required by Lightning/optimizer setup
    
    
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
