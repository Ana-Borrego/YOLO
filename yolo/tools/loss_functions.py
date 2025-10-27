from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import BCEWithLogitsLoss

from yolo.config.config import Config, LossConfig
from yolo.utils.bounding_box_utils import BoxMatcher, Vec2Box, calculate_iou
from yolo.utils.logger import logger

import cv2 # For polygon rasterization (pip install opencv-python-headless)
import numpy as np 
import segmentation_models_pytorch as smp # (pip install segmentation-models-pytorch)
from einops import rearrange

class BCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # TODO: Refactor the device, should be assign by config
        # TODO: origin v9 assing pos_weight == 1?
        self.bce = BCEWithLogitsLoss(reduction="none")

    def forward(self, predicts_cls: Tensor, targets_cls: Tensor, cls_norm: Tensor) -> Any:
        return self.bce(predicts_cls, targets_cls).sum() / cls_norm


class BoxLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, predicts_bbox: Tensor, targets_bbox: Tensor, valid_masks: Tensor, box_norm: Tensor, cls_norm: Tensor
    ) -> Any:
        valid_bbox = valid_masks[..., None].expand(-1, -1, 4)
        picked_predict = predicts_bbox[valid_bbox].view(-1, 4)
        picked_targets = targets_bbox[valid_bbox].view(-1, 4)

        iou = calculate_iou(picked_predict, picked_targets, "ciou").diag()
        loss_iou = 1.0 - iou
        loss_iou = (loss_iou * box_norm).sum() / cls_norm
        return loss_iou


class DFLoss(nn.Module):
    def __init__(self, vec2box: Vec2Box, reg_max: int) -> None:
        super().__init__()
        self.anchors_norm = (vec2box.anchor_grid / vec2box.scaler[:, None])[None]
        self.reg_max = reg_max

    def forward(
        self, predicts_anc: Tensor, targets_bbox: Tensor, valid_masks: Tensor, box_norm: Tensor, cls_norm: Tensor
    ) -> Any:
        valid_bbox = valid_masks[..., None].expand(-1, -1, 4)
        bbox_lt, bbox_rb = targets_bbox.chunk(2, -1)
        targets_dist = torch.cat(((self.anchors_norm - bbox_lt), (bbox_rb - self.anchors_norm)), -1).clamp(
            0, self.reg_max - 1.01
        )
        picked_targets = targets_dist[valid_bbox].view(-1)
        picked_predict = predicts_anc[valid_bbox].view(-1, self.reg_max)

        label_left, label_right = picked_targets.floor(), picked_targets.floor() + 1
        weight_left, weight_right = label_right - picked_targets, picked_targets - label_left

        loss_left = F.cross_entropy(picked_predict, label_left.to(torch.long), reduction="none")
        loss_right = F.cross_entropy(picked_predict, label_right.to(torch.long), reduction="none")
        loss_dfl = loss_left * weight_left + loss_right * weight_right
        loss_dfl = loss_dfl.view(-1, 4).mean(-1)
        loss_dfl = (loss_dfl * box_norm).sum() / cls_norm
        return loss_dfl


class YOLOLoss:
    def __init__(self, loss_cfg: LossConfig, vec2box: Vec2Box, class_num: int = 80, reg_max: int = 16) -> None:
        self.class_num = class_num
        self.vec2box = vec2box

        self.cls = BCELoss()
        self.dfl = DFLoss(vec2box, reg_max)
        self.iou = BoxLoss()

        self.matcher = BoxMatcher(loss_cfg.matcher, self.class_num, vec2box, reg_max)

    def separate_anchor(self, anchors):
        """
        separate anchor and bbouding box
        """
        anchors_cls, anchors_box = torch.split(anchors, (self.class_num, 4), dim=-1)
        anchors_box = anchors_box / self.vec2box.scaler[None, :, None]
        return anchors_cls, anchors_box

    def __call__(self, predicts: List[Tensor], targets: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        predicts_cls, predicts_anc, predicts_box = predicts
        # For each predicted targets, assign a best suitable ground truth box.
        align_targets, valid_masks = self.matcher(targets, (predicts_cls.detach(), predicts_box.detach()))

        targets_cls, targets_bbox = self.separate_anchor(align_targets)
        predicts_box = predicts_box / self.vec2box.scaler[None, :, None]

        cls_norm = max(targets_cls.sum(), 1)
        box_norm = targets_cls.sum(-1)[valid_masks]

        ## -- CLS -- ##
        loss_cls = self.cls(predicts_cls, targets_cls, cls_norm)
        ## -- IOU -- ##
        loss_iou = self.iou(predicts_box, targets_bbox, valid_masks, box_norm, cls_norm)
        ## -- DFL -- ##
        loss_dfl = self.dfl(predicts_anc, targets_bbox, valid_masks, box_norm, cls_norm)

        return loss_iou, loss_dfl, loss_cls


class DualLoss:
    def __init__(self, cfg: Config, vec2box) -> None:
        loss_cfg = cfg.task.loss
        self.loss = YOLOLoss(loss_cfg, vec2box, class_num=cfg.dataset.class_num, reg_max=cfg.model.anchor.reg_max)

        self.aux_rate = loss_cfg.aux

        self.iou_rate = loss_cfg.objective["BoxLoss"]
        self.dfl_rate = loss_cfg.objective["DFLoss"]
        self.cls_rate = loss_cfg.objective["BCELoss"]

    def __call__(
        self, aux_predicts: List[Tensor], main_predicts: List[Tensor], targets: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        # TODO: Need Refactor this region, make it flexible!
        aux_iou, aux_dfl, aux_cls = self.loss(aux_predicts, targets)
        main_iou, main_dfl, main_cls = self.loss(main_predicts, targets)

        total_loss = [
            self.iou_rate * (aux_iou * self.aux_rate + main_iou),
            self.dfl_rate * (aux_dfl * self.aux_rate + main_dfl),
            self.cls_rate * (aux_cls * self.aux_rate + main_cls),
        ]
        loss_dict = {
            f"Loss/{name}Loss": value.detach().item() for name, value in zip(["Box", "DFL", "BCE"], total_loss)
        }
        return sum(total_loss), loss_dict


# def create_loss_function(cfg: Config, vec2box) -> DualLoss:
#     # TODO: make it flexible, if cfg doesn't contain aux, only use SingleLoss
#     loss_function = DualLoss(cfg, vec2box)
#     logger.info(":white_check_mark: Success load loss function")
#     return loss_function

### CREAR LA FUNCIÓN DE PÉRDIDA PARA SEGMENTACIÓN. ###
# Clase YoloSegmentationLoss
# 3 componentes principales: 
    # Pérdida de Bounding Box (BoxLoss): Puedes reutilizar BoxLoss o una similar (ej. CIoU Loss) para comparar las cajas predichas por el cabezal de detección (predicts['Main'][0], [1], [2]) con targets_bboxes.
    # Pérdida de Clasificación (ClsLoss): Puedes usar BCELoss o FocalLoss para comparar las clases predichas por el cabezal de detección con targets_bboxes.
    # Pérdida de Máscara (MaskLoss): Esta es la parte nueva. Debes comparar los prototipos de máscara generados por el modelo (predicts['Main'][3]) y los coeficientes de máscara predichos (predicts['Main'][0], [1], [2]) 
    #                                con los polígonos de verdad (targets_segments). Comúnmente se usa una combinación de BCEWithLogitsLoss aplicada a las máscaras reconstruidas.

# Combinar las 3 pérdidas: loss = weight_box * loss_box + weight_cls * loss_cls + weight_mask * loss_mask
# Actualizar create_loss_function para que devuelva tu nueva YOLOSegmentationLoss cuando el modelo sea de segmentación.

# Add this class definition in loss_functions.py

class YOLOSegmentationLoss:
    def __init__(self, cfg: Config, vec2box: Vec2Box):
        loss_cfg = cfg.task.loss
        self.class_num = cfg.dataset.class_num
        self.reg_max = cfg.model.anchor.reg_max
        self.image_size = cfg.image_size # Needed for mask generation H, W
        self.mask_output_stride = 4 # Assumes final proto mask is 1/4th of input size (e.g., 160x160 for 640x640)

        # --- Initialize Sub-Losses ---
        self.bce_cls = BCEWithLogitsLoss(reduction="none")
        self.box_loss = BoxLoss() # Reuses existing BoxLoss logic
        self.dfl_loss = DFLoss(vec2box, self.reg_max)

        # Mask Losses (using SMP and PyTorch)
        self.bce_mask = BCEWithLogitsLoss(reduction='mean')
        self.dice_mask = smp.losses.DiceLoss(mode='binary', from_logits=True)

        # --- Matcher ---
        # Reuses the existing matcher for box/class assignment
        self.matcher = BoxMatcher(loss_cfg.matcher, self.class_num, vec2box, self.reg_max)

        # --- Loss Weights (from train.yaml) ---
        self.box_weight = loss_cfg.objective.get("BoxLoss", 7.5)
        self.cls_weight = loss_cfg.objective.get("BCELoss", 0.5)
        self.dfl_weight = loss_cfg.objective.get("DFLoss", 1.5)
        self.mask_weight = loss_cfg.objective.get("MaskLoss", 1.0) # Add a weight for MaskLoss

    def _process_segments_to_masks(
        self, 
        segments_batch, # target_segments que tenemos lista de listas
        batch_bboxes,  # target_bboxs tensor (shape [Batch, Max_Instances, 5]) se necesita para marcar qué polígono se relaciona con el bbox ground truth
        mask_h, 
        mask_w, # son la altura y anchura que se necesita para el mapeo, para mantener las proporciones.
        device # device donde se debe crear el output mask tensor (CPU o GPU)
        ):
        """Generates binary masks from polygon segments.
        
        targets_segments es una lista de listas, cada lista contiene todas las etiquetas que están representadas como lista de [class_id, x1_norm, y1_norm, x2_norm, y2_norm, ...]
        Las pérdidas de segmentación estándar como BCEWithLogitsLoss or DiceLoss trabajan píxel a píxel. Necesitan una máscara binaria (grid de 0 y 1) que represente el ground truth shape del objeto. 
        El bug es: The loss function cannot directly compare the model's predicted pixel mask (logits) with a list of polygon coordinates.
        """
        batch_masks = [] # almacenará las mascaras generadas y 
        num_instances_per_image = [] # la cuenta de instancias válidas por imagenes
        for i, segments in enumerate(segments_batch): # Iterate through images in batch
            img_h, img_w = self.image_size
            instance_masks = []
            bboxes_in_image = batch_bboxes[i] # Get bboxes for this image -- GT
            valid_bbox_mask = bboxes_in_image[:, 0] != -1 # Filter out padding bboxes
            valid_bboxes = bboxes_in_image[valid_bbox_mask]
            
            # Match segments to valid bboxes (using center point heuristic for simplicity)
            # A more robust approach might use IoU between segment bbox and target bbox
            # LIMITATION: This is a simplification. A more robust method would calculate the IoU between the bounding box of the polygon and the valid_bboxes. However, the center heuristic is often sufficient if annotations are clean.
            if valid_bboxes.numel() > 0: # Only proceed if there are valid bboxes
                 centers_x = (valid_bboxes[:, 1] + valid_bboxes[:, 3]) / 2 / img_w
                 centers_y = (valid_bboxes[:, 2] + valid_bboxes[:, 4]) / 2 / img_h
            else: # Handle case with no valid bboxes
                 centers_x, centers_y = torch.tensor([], device=device), torch.tensor([], device=device) # Use device

            # Initialize: -1 means no segment matched, inf means infinite distance
            matched_segment_indices = [-1] * len(valid_bboxes)
            min_distances = [float('inf')] * len(valid_bboxes)

            if segments and len(valid_bboxes) > 0: # Only match if both segments and bboxes exist
                for seg_idx, segment in enumerate(segments):
                    poly = np.array(segment[1:]).reshape(-1, 2)
                    if poly.size == 0: continue # Skip empty polygons

                    seg_center_x, seg_center_y = poly.mean(axis=0)

                    # Calculate distances from THIS segment to ALL valid bbox centers (ensure tensors are on same device)
                    distances_tensor = (centers_x.cpu() - seg_center_x)**2 + (centers_y.cpu() - seg_center_y)**2 # Perform calculation on CPU if centers might be GPU
                    
                    if distances_tensor.numel() == 0: continue # Skip if no bboxes to compare against

                    # Find the bbox closest to THIS segment
                    best_match_bbox_idx = distances_tensor.argmin()
                    current_distance = distances_tensor[best_match_bbox_idx].item() # Get distance as float

                    # --- Corrected Assignment Logic ---
                    # If this segment is closer than any previous segment for this bbox
                    if current_distance < min_distances[best_match_bbox_idx]:
                        min_distances[best_match_bbox_idx] = current_distance
                        matched_segment_indices[best_match_bbox_idx] = seg_idx
            
            # --- Rasterize Matched Polygons ---
            num_valid_instances = 0
            for bbox_idx in range(len(valid_bboxes)):
                # Create an empty mask for this instance
                mask = torch.zeros(mask_h, mask_w, device=device, dtype=torch.float32)
                # Get the segment index matched to this bbox
                seg_idx = matched_segment_indices[bbox_idx]
                
                if seg_idx != -1: # If a segment was matched to this bbox SUCCESFULLY
                    segment = segments[seg_idx]
                    poly = np.array(segment[1:]).reshape(-1, 2)
                    
                    # Un-normalize polygon to mask dimensions
                    poly_unnorm = poly * np.array([mask_w, mask_h]) 
                    
                    # Rasterize polygon using OpenCV
                    # Draw the polygon (filled with 1.0) onto the mask tensor
                    # Needs conversion to numpy int32 for cv2.fillPoly
                    cv2.fillPoly(mask.cpu().numpy(), [poly_unnorm.astype(np.int32)], 1.0)
                    num_valid_instances += 1
                instance_masks.append(mask)

            if not instance_masks: # Handle images with no valid bboxes/segments
                 instance_masks.append(torch.zeros(mask_h, mask_w, device=device, dtype=torch.float32))
                 num_instances_per_image.append(0)
            else:
                 num_instances_per_image.append(num_valid_instances)

            batch_masks.append(torch.stack(instance_masks)) # [Num_Valid_BBoxes, H, W]

        # Pad masks within the batch
        # Find the max number of instances in any image within this batch
        max_instances = max(m.shape[0] for m in batch_masks) if batch_masks else 0
        # Create the final padded tensor for the whole batch
        padded_masks = torch.zeros(len(batch_masks), max_instances, mask_h, mask_w, device=device)
        if max_instances > 0:
            for i, masks in enumerate(batch_masks):
                if masks.shape[0] > 0: # Check if there are masks for this image
                    #Copy the instance masks into the padded tensor
                    padded_masks[i, :masks.shape[0]] = masks
        
        return padded_masks, num_instances_per_image # [B, Max_Inst, H, W]

    def __call__(self, predicts, targets_bboxes, targets_segments):
        """
        Calculates the segmentation loss.
        Args:
            predicts (Dict): Output from the YOLO segmentation model. Keys: 'Main', 'AUX'.
                             Values are lists: [det_head1, det_head2, det_head3, proto_head]
            targets_bboxes (Tensor): Ground truth bounding boxes [B, Max_Inst, 5].
            targets_segments (List[List]): Ground truth segments for each image.
        """
        device = targets_bboxes.device
        total_box_loss = torch.tensor(0.0, device=device)
        total_cls_loss = torch.tensor(0.0, device=device)
        total_dfl_loss = torch.tensor(0.0, device=device)
        total_mask_loss = torch.tensor(0.0, device=device)

        # --- Process Predictions ---
        # predicts['Main'] AHORA es una tupla: (detection_outputs, segmentation_outputs)
        detection_outputs, segmentation_outputs = predicts['Main']
        
        # 1. Procesar Salidas de Detección
        all_pred_cls, all_pred_anc, all_pred_box = [], [], []
        # detection_outputs es una lista de tuplas [(cls, anc, box), ...]
        for pred_cls, pred_anc, pred_box in detection_outputs:
            all_pred_cls.append(rearrange(pred_cls, "B C h w -> B (h w) C"))
            all_pred_anc.append(rearrange(pred_anc, "B A R h w -> B (h w) R A"))
            all_pred_box.append(rearrange(pred_box, "B X h w -> B (h w) X"))

        preds_cls = torch.cat(all_pred_cls, dim=1)
        preds_anc = torch.cat(all_pred_anc, dim=1)
        preds_box_dist = torch.cat(all_pred_box, dim=1) # Box distributions (LTRB distances)

        # 2. Procesar Salidas de Segmentación
        # segmentation_outputs es una lista [coeffs1, coeffs2, coeffs3, protos]
        all_pred_mask_coeffs = []
        num_mask_heads = len(segmentation_outputs) - 1
        for i in range(num_mask_heads):
            # Asume que los coeficientes tienen shape [B, Num_Mask_Coeffs, h, w]
            pred_mask_coeff = segmentation_outputs[i] 
            all_pred_mask_coeffs.append(rearrange(pred_mask_coeff, "B M h w -> B (h w) M"))
        
        preds_mask_coeffs = torch.cat(all_pred_mask_coeffs, dim=1) # [B, Num_Total_Preds, Num_Mask_Coeffs]
        protos = segmentation_outputs[-1] # [B, Num_Mask_Coeffs, Mask_H, Mask_W]
        print(f"   DEBUG: Type of protos (segmentation_outputs[-1]): {type(protos)}")
        if not isinstance(protos, torch.Tensor):
            print(f"   DEBUG: Content of protos: {protos}") # See what's inside if it's not a tensor
            raise TypeError(f"Expected protos to be a Tensor, but got {type(protos)}")
        # Convertir distribuciones de caja a bboxes (xyxy) - (como antes)
        vec2box = self.matcher.vec2box 
        pred_LTRB = preds_box_dist * vec2box.scaler.view(1, -1, 1)
        lt, rb = pred_LTRB.chunk(2, dim=-1)
        preds_box = torch.cat([vec2box.anchor_grid - lt, vec2box.anchor_grid + rb], dim=-1) # [B, Num_Total_Preds, 4]
        
        # --- Match Predictions to Ground Truth BBoxes/Classes ---
        print("   Running BoxMatcher...") 
        # Now unpack THREE items from the matcher
        align_targets, valid_masks, unique_indices = self.matcher(targets_bboxes, (preds_cls.detach(), preds_box.detach()))
        print("   BoxMatcher finished.")

        # --- Prepare Targets based on Matcher Output ---
        # Check if align_targets was actually created
        if 'align_targets' not in locals():
             raise RuntimeError("Matcher did not assign align_targets!")
             
        # Now split the targets
        targets_cls_aligned, targets_bbox_aligned_raw = torch.split(align_targets, (self.class_num, 4), dim=-1) 

        # 2. Scale the bounding boxes (replicating the scaling from original separate_anchor)
        #    BoxLoss and DFLoss expect target boxes scaled relative to the anchor grid stride
        vec2box = self.matcher.vec2box # Access vec2box via the matcher instance
        targets_bbox_aligned = targets_bbox_aligned_raw / vec2box.scaler[None, :, None] 

        # Calculate normalization factors AFTER splitting targets
        cls_norm = max(targets_cls_aligned.sum(), 1) 
        # Ensure box_norm is calculated correctly even if valid_masks is empty
        box_norm = torch.zeros_like(valid_masks, dtype=torch.float32) # Placeholder
        if valid_masks.any():
            box_norm = targets_cls_aligned.sum(-1)[valid_masks] # Calculate only for valid matches


        # --- Calculate Box, DFL, and Class Losses ---
        print("   Calculating Box/DFL/Class losses...") # Add print
        # Only calculate if there are actual valid matches found by the matcher
        if valid_masks.any(): 
             total_box_loss = self.box_loss(preds_box / vec2box.scaler[None, :, None], targets_bbox_aligned, valid_masks, box_norm, cls_norm)
             total_dfl_loss = self.dfl_loss(preds_anc, targets_bbox_aligned, valid_masks, box_norm, cls_norm)
             # Use the ALIGNED class targets for BCE loss
             # Calculate raw BCE loss (element-wise because reduction='none')
             loss_cls_unreduced = self.bce_cls(preds_cls, targets_cls_aligned) # [B, Anchors, N_Classes]

             # IMPORTANT: Apply valid_masks to only consider losses for matched anchors/predictions
             # Also, multiply by the target class values (targets_cls_aligned is one-hot or similar)
             # This ensures we only sum losses where the target class was positive
             loss_cls_masked = loss_cls_unreduced * targets_cls_aligned
             loss_cls_masked = loss_cls_masked[valid_masks] # Select only losses for valid matches

             # Perform manual reduction (sum) and normalization
             total_cls_loss = loss_cls_masked.sum() / cls_norm
             print("   Box/DFL/Class losses calculated.") # Confirm calculation
        else:
            print("   No valid matches found by BoxMatcher. Skipping Box/DFL/Class loss calculation.")


        # --- Prepare Mask Inputs ---
        # protos = predicts['Main'][-1] # [B, Num_Mask_Coeffs, Mask_H, Mask_W] e.g., [B, 32, 160, 160]
        protos = segmentation_outputs[-1]
        b, num_coeffs, mask_h, mask_w = protos.shape
        print(f"   Protos shape: b={b}, num_coeffs={num_coeffs}, mask_h={mask_h}, mask_w={mask_w}")
        
        # 1. Generate True Masks from segments
        #    Rasterize polygons based on ALIGNED ground truth bboxes
        print("   Generating true masks from segments...") # Add print
        true_masks, num_instances = self._process_segments_to_masks(targets_segments, targets_bboxes, mask_h, mask_w, device) # [B, Max_Inst, H, W]
        print(f"   Generated true_masks shape: {true_masks.shape}") # Add print
        
        # 2. Reconstruct Predicted Masks
        #    Only reconstruct masks for the predictions that were matched by the BoxMatcher
        print("   Reconstructing predicted masks for matched predictions...") # Add print
        matched_pred_indices = torch.where(valid_masks) # Indices (batch_idx, anchor_idx) of matched predictions
        if matched_pred_indices[0].numel() > 0: # If there are any matches
            matched_coeffs = preds_mask_coeffs[matched_pred_indices] # [Num_Matches, Num_Mask_Coeffs]
            batch_indices_for_matches = matched_pred_indices[0] # Which image each match belongs to [Num_Matches]

            # Reshape protos for batch matrix multiplication
            protos_reshaped = protos.view(b, num_coeffs, mask_h * mask_w) # [B, C, H*W]
            
            # Select protos corresponding to the batch index of each match
            protos_matched = protos_reshaped[batch_indices_for_matches] # [Num_Matches, C, H*W]

            # Reconstruct masks: Coeffs @ Protos -> [Num_Matches, H*W]
            pred_masks_flat = torch.bmm(matched_coeffs.unsqueeze(1), protos_matched).squeeze(1)
            pred_masks_reconstructed = pred_masks_flat.view(-1, mask_h, mask_w) # [Num_Matches, H, W] - These are logits
            print(f"   Reconstructed pred_masks shape: {pred_masks_reconstructed.shape}") # Add print

            # --- Match Reconstructed Masks to True Masks ---
            # Get the indices of the GT instances assigned to each matched prediction
            # Use unique_indices (shape [B, Num_Anchors, 1]) filtered by valid_masks
            print("   Matching predicted masks to true masks...") # Add print
            # unique_indices contains the GT index (0..N-1) for EACH anchor
            # We select only the indices corresponding to the VALID matches
            gt_indices_for_matches = unique_indices[matched_pred_indices].squeeze(-1) # Shape [Num_Matches]
            print(f"   Obtained gt_indices_for_matches shape: {gt_indices_for_matches.shape}, Min: {gt_indices_for_matches.min()}, Max: {gt_indices_for_matches.max()}") # Add print debug

            # Select the corresponding true masks using the correct batch and GT indices
            true_masks_matched = true_masks[batch_indices_for_matches, gt_indices_for_matches] # Shape [Num_Matches, H, W]
            print(f"   Selected true_masks_matched shape: {true_masks_matched.shape}") # Add print

            # --- Calculate Mask Loss ---
            print("   Calculating Mask loss...") # Add print
            mask_loss_bce = self.bce_mask(pred_masks_reconstructed, true_masks_matched)
            mask_loss_dice = self.dice_mask(pred_masks_reconstructed, true_masks_matched)
            total_mask_loss = mask_loss_bce + mask_loss_dice
            print(f"   Mask loss calculated: BCE={mask_loss_bce.item()}, Dice={mask_loss_dice.item()}") # Add print
        else:
             print("   No valid matches found by BoxMatcher. Skipping Mask loss calculation.")

        # --- Combine Losses ---
        total_loss = (self.box_weight * total_box_loss +
                      self.cls_weight * total_cls_loss +
                      self.dfl_weight * total_dfl_loss +
                      self.mask_weight * total_mask_loss) # Add mask loss component

        loss_dict = {
            "Loss/BoxLoss": (self.box_weight * total_box_loss).detach().item(),
            "Loss/DFLLoss": (self.dfl_weight * total_dfl_loss).detach().item(),
            "Loss/BCELoss": (self.cls_weight * total_cls_loss).detach().item(),
            "Loss/MaskLoss": (self.mask_weight * total_mask_loss).detach().item(), # Log mask loss
        }

        # Handle potential Auxiliary Head Loss (Needs similar logic if AUX head also predicts masks)
        # For now, let's ignore AUX head for simplicity, assuming it's only for detection pre-training
        # or needs its own segmentation loss calculation.
        
        return total_loss, loss_dict

def create_loss_function(cfg: Config, vec2box): # El tipo de retorno depende del modelo
    # Comprueba si el modelo es de segmentación
    # Inferimos esto buscando 'MultiheadSegmentation' en la sección 'detection'
    is_segmentation_model = False
    # Comprobación más segura por si falta la clave 'detection'
    if 'model' in cfg and 'model' in cfg.model and 'detection' in cfg.model.model:
         for layer_spec in cfg.model.model['detection']:
             if 'MultiheadSegmentation' in layer_spec:
                 is_segmentation_model = True
                 break

    if is_segmentation_model:
        logger.info("Tipo de modelo: Segmentación. Usando YOLOSegmentationLoss.")
        # Nota: YOLOSegmentationLoss actualmente no maneja AUX de forma separada como DualLoss.
        loss_function = YOLOSegmentationLoss(cfg, vec2box)
    else:
        logger.info("Tipo de modelo: Detección. Usando DualLoss (YOLOLoss).")
        # Lógica existente para modelos de detección
        loss_function = DualLoss(cfg, vec2box)

    logger.info(":white_check_mark: Success load loss function")
    return loss_function