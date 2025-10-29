import math
from typing import Dict, List, Optional, Tuple, Union

import torch
from einops import rearrange
from torch import Tensor, tensor
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import batched_nms

from yolo.config.config import AnchorConfig, MatcherConfig, NMSConfig
from yolo.model.yolo import YOLO
from yolo.utils.logger import logger


def calculate_iou(bbox1, bbox2, metrics="iou") -> Tensor:
    metrics = metrics.lower()
    EPS = 1e-7
    dtype = bbox1.dtype
    bbox1 = bbox1.to(torch.float32)
    bbox2 = bbox2.to(torch.float32)

    # Expand dimensions if necessary
    if bbox1.ndim == 2 and bbox2.ndim == 2:
        bbox1 = bbox1.unsqueeze(1)  # (Ax4) -> (Ax1x4)
        bbox2 = bbox2.unsqueeze(0)  # (Bx4) -> (1xBx4)
    elif bbox1.ndim == 3 and bbox2.ndim == 3:
        bbox1 = bbox1.unsqueeze(2)  # (BZxAx4) -> (BZxAx1x4)
        bbox2 = bbox2.unsqueeze(1)  # (BZxBx4) -> (BZx1xBx4)

    # Calculate intersection coordinates
    xmin_inter = torch.max(bbox1[..., 0], bbox2[..., 0])
    ymin_inter = torch.max(bbox1[..., 1], bbox2[..., 1])
    xmax_inter = torch.min(bbox1[..., 2], bbox2[..., 2])
    ymax_inter = torch.min(bbox1[..., 3], bbox2[..., 3])

    # Calculate intersection area
    intersection_area = torch.clamp(xmax_inter - xmin_inter, min=0) * torch.clamp(ymax_inter - ymin_inter, min=0)

    # Calculate area of each bbox
    area_bbox1 = (bbox1[..., 2] - bbox1[..., 0]) * (bbox1[..., 3] - bbox1[..., 1])
    area_bbox2 = (bbox2[..., 2] - bbox2[..., 0]) * (bbox2[..., 3] - bbox2[..., 1])

    # Calculate union area
    union_area = area_bbox1 + area_bbox2 - intersection_area

    # Calculate IoU
    iou = intersection_area / (union_area + EPS)
    if metrics == "iou":
        return iou.to(dtype)

    # Calculate centroid distance
    cx1 = (bbox1[..., 2] + bbox1[..., 0]) / 2
    cy1 = (bbox1[..., 3] + bbox1[..., 1]) / 2
    cx2 = (bbox2[..., 2] + bbox2[..., 0]) / 2
    cy2 = (bbox2[..., 3] + bbox2[..., 1]) / 2
    cent_dis = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # Calculate diagonal length of the smallest enclosing box
    c_x = torch.max(bbox1[..., 2], bbox2[..., 2]) - torch.min(bbox1[..., 0], bbox2[..., 0])
    c_y = torch.max(bbox1[..., 3], bbox2[..., 3]) - torch.min(bbox1[..., 1], bbox2[..., 1])
    diag_dis = c_x**2 + c_y**2 + EPS

    diou = iou - (cent_dis / diag_dis)
    if metrics == "diou":
        return diou.to(dtype)

    # Compute aspect ratio penalty term
    arctan = torch.atan((bbox1[..., 2] - bbox1[..., 0]) / (bbox1[..., 3] - bbox1[..., 1] + EPS)) - torch.atan(
        (bbox2[..., 2] - bbox2[..., 0]) / (bbox2[..., 3] - bbox2[..., 1] + EPS)
    )
    v = (4 / (math.pi**2)) * (arctan**2)
    with torch.no_grad():
        alpha = v / (v - iou + 1 + EPS)
    # Compute CIoU
    ciou = diou - alpha * v
    return ciou.to(dtype)


def transform_bbox(bbox: Tensor, indicator="xywh -> xyxy"):
    data_type = bbox.dtype
    in_type, out_type = indicator.replace(" ", "").split("->")

    if in_type not in ["xyxy", "xywh", "xycwh"] or out_type not in ["xyxy", "xywh", "xycwh"]:
        raise ValueError("Invalid input or output format")

    if in_type == "xywh":
        x_min = bbox[..., 0]
        y_min = bbox[..., 1]
        x_max = bbox[..., 0] + bbox[..., 2]
        y_max = bbox[..., 1] + bbox[..., 3]
    elif in_type == "xyxy":
        x_min = bbox[..., 0]
        y_min = bbox[..., 1]
        x_max = bbox[..., 2]
        y_max = bbox[..., 3]
    elif in_type == "xycwh":
        x_min = bbox[..., 0] - bbox[..., 2] / 2
        y_min = bbox[..., 1] - bbox[..., 3] / 2
        x_max = bbox[..., 0] + bbox[..., 2] / 2
        y_max = bbox[..., 1] + bbox[..., 3] / 2

    if out_type == "xywh":
        bbox = torch.stack([x_min, y_min, x_max - x_min, y_max - y_min], dim=-1) # type:ignore
    elif out_type == "xyxy":
        bbox = torch.stack([x_min, y_min, x_max, y_max], dim=-1)
    elif out_type == "xycwh":
        bbox = torch.stack([(x_min + x_max) / 2, (y_min + y_max) / 2, x_max - x_min, y_max - y_min], dim=-1) # type:ignore

    return bbox.to(dtype=data_type)


def generate_anchors(image_size: List[int], strides: List[int]):
    """
    Find the anchor maps for each w, h.

    Args:
        image_size List: the image size of augmented image size
        strides List[8, 16, 32, ...]: the stride size for each predicted layer

    Returns:
        all_anchors [HW x 2]:
        all_scalers [HW]: The index of the best targets for each anchors
    """
    W, H = image_size
    anchors = []
    scaler = []
    for stride in strides:
        anchor_num = W // stride * H // stride
        scaler.append(torch.full((anchor_num,), stride))
        shift = stride // 2
        h = torch.arange(0, H, stride) + shift
        w = torch.arange(0, W, stride) + shift
        if torch.__version__ >= "2.3.0":
            anchor_h, anchor_w = torch.meshgrid(h, w, indexing="ij")
        else:
            anchor_h, anchor_w = torch.meshgrid(h, w)
        anchor = torch.stack([anchor_w.flatten(), anchor_h.flatten()], dim=-1)
        anchors.append(anchor)
    all_anchors = torch.cat(anchors, dim=0)
    all_scalers = torch.cat(scaler, dim=0)
    return all_anchors, all_scalers


class BoxMatcher:
    def __init__(self, cfg: MatcherConfig, class_num: int, vec2box, reg_max: int) -> None:
        self.class_num = class_num
        self.vec2box = vec2box
        self.reg_max = reg_max
        
        self.iou = getattr(cfg, "iou", "iou")
        self.topk = getattr(cfg, "topk", 10)
        self.factor = getattr(cfg, "factor", {"iou": 1.0, "cls": 1.0})

    def get_valid_matrix(self, target_bbox: Tensor):
        """
        Get a boolean mask that indicates whether each target bounding box overlaps with each anchor
        and is able to correctly predict it with the available reg_max value.

        Args:
            target_bbox [batch x targets x 4]: The bounding box of each target.
        Returns:
            [batch x targets x anchors]: A boolean tensor indicates if target bounding box overlaps
            with the anchors, and the anchor is able to predict the target.
        """
        x_min, y_min, x_max, y_max = target_bbox[:, :, None].unbind(3)
        anchors = self.vec2box.anchor_grid[None, None]  # add a axis at first, second dimension
        anchors_x, anchors_y = anchors.unbind(dim=3)
        x_min_dist, x_max_dist = anchors_x - x_min, x_max - anchors_x
        y_min_dist, y_max_dist = anchors_y - y_min, y_max - anchors_y
        targets_dist = torch.stack((x_min_dist, y_min_dist, x_max_dist, y_max_dist), dim=-1)
        targets_dist /= self.vec2box.scaler[None, None, :, None]  # (1, 1, anchors, 1)
        min_reg_dist, max_reg_dist = targets_dist.amin(dim=-1), targets_dist.amax(dim=-1)
        target_on_anchor = min_reg_dist >= 0
        target_in_reg_max = max_reg_dist <= self.reg_max - 1.01
        return target_on_anchor & target_in_reg_max

    def get_cls_matrix(self, predict_cls: Tensor, target_cls: Tensor) -> Tensor:
        """
        Get the (predicted class' probabilities) corresponding to the target classes across all anchors

        Args:
            predict_cls [batch x anchors x class]: The predicted probabilities for each class across each anchor.
            target_cls [batch x targets]: The class index for each target.

        Returns:
            [batch x targets x anchors]: The probabilities from `pred_cls` corresponding to the class indices specified in `target_cls`.
        """
        # predict_cls es [B, A, C], target_cls es [B, T]
        B, A, C = predict_cls.shape
        T = target_cls.shape[1]
        # Expandir target_cls para indexar predict_cls
        # target_cls [B, T] -> [B, T, 1] -> [B, T, A]
        # Asegurarse de que target_cls sea Long y clampear los índices negativos a 0
        # El padding -1.0 se convierte en índice -1 (long), lo clampeamos a 0
        idx = target_cls.long().clamp_(min=0)
        # .clamp_(min=0): Cambia in situ todos los valores negativos (-1) a 0. Ahora, todos los índices en idx son >= 0.
        
        # Usar gather para seleccionar las probabilidades correctas
        # predict_cls [B, A, C] -> [B, C, A] para poder usar gather en dim=1 (clases)
        cls_probabilities = torch.gather(predict_cls.transpose(1, 2), 1, idx)
        return cls_probabilities

    def get_iou_matrix(self, predict_bbox, target_bbox) -> Tensor:
        """
        Get the IoU between each target bounding box and each predicted bounding box.

        Args:
            predict_bbox [batch x predicts x 4]: Bounding box with [x1, y1, x2, y2].
            target_bbox [batch x targets x 4]: Bounding box with [x1, y1, x2, y2].
        Returns:
            [batch x targets x predicts]: The IoU scores between each target and predicted.
        """
        # target_bbox: [B, T, 4], predict_bbox: [B, A, 4]
        # Salida esperada: [B, T, A]
        return calculate_iou(target_bbox, predict_bbox, self.iou).clamp(0, 1)

    def filter_topk(self, target_matrix: Tensor, grid_mask: Tensor, topk: int = 10) -> Tuple[Tensor, Tensor]:
        """
        Filter the top-k suitability of targets for each anchor.

        Args:
            target_matrix [batch x targets x anchors]: The suitability for each targets-anchors
            grid_mask [batch x targets x anchors]: The match validity for each target to anchors
            topk (int, optional): Number of top scores to retain per anchor.

        Returns:
            topk_targets [batch x targets x anchors]: Only leave the topk targets for each anchor
            topk_mask [batch x targets x anchors]: A boolean mask indicating the top-k scores' positions.
        """
        # target_matrix, grid_mask: [B, T, A]
        masked_target_matrix = torch.where(grid_mask, target_matrix, torch.tensor(0.0, device=target_matrix.device))
        
        # Queremos top T anchors para cada GT, así que usamos topk en la dim A (-1)
        topk = min(topk, masked_target_matrix.shape[-1]) # Asegurar que topk no sea mayor que el número de anchors
        values, indices = masked_target_matrix.topk(topk, dim=-1)
        
        topk_targets = torch.zeros_like(target_matrix, device=target_matrix.device)
        topk_targets.scatter_(dim=-1, index=indices, src=values)
        topk_mask = topk_targets > 1e-9 # Usar un umbral pequeño en lugar de > 0
        return topk_targets, topk_mask

    def ensure_one_anchor(self, target_matrix: Tensor, topk_mask: tensor) -> Tensor: # type:ignore
        """
        Ensures each valid target gets at least one anchor matched based on the unmasked target matrix,
        which enables an otherwise invalid match. This enables too small or too large targets to be
        learned as well, even if they can't be predicted perfectly.

        Args:
            target_matrix [batch x targets x anchors]: The suitability for each targets-anchors
            topk_mask [batch x targets x anchors]: A boolean mask indicating the top-k scores' positions.

        Returns:
            topk_mask [batch x targets x anchors]: A boolean mask indicating the updated top-k scores' positions.
        """
        # Encontrar el mejor anchor para cada target (GT)
        values, indices = target_matrix.max(dim=-1, keepdim=True) # indices shape [B, T, 1]
        
        best_anchor_mask = torch.zeros_like(target_matrix, dtype=torch.bool)
        # Usar scatter_ para marcar el mejor anchor para cada GT
        best_anchor_mask.scatter_(-1, index=indices, value=True) 

        # Cuántos anchors tiene asignado cada GT después del topk?
        matched_anchor_num_per_gt = torch.sum(topk_mask, dim=-1) # shape [B, T]
        
        # Identificar GTs que son válidos (tienen alguna puntuación > 0) pero no tienen anchors asignados
        gt_is_valid = values.squeeze(-1) > 1e-9 # shape [B, T]
        target_without_anchor = (matched_anchor_num_per_gt == 0) & gt_is_valid # shape [B, T]
        
        # Para esos GTs, forzar la asignación a su mejor anchor
        # Expandir target_without_anchor para que tenga la misma forma que topk_mask
        force_assign_mask = target_without_anchor.unsqueeze(-1).expand_as(topk_mask) # [B, T, A]

        # Combinar la máscara original con la máscara de asignación forzada
        # Solo forzamos la asignación donde best_anchor_mask es True
        updated_topk_mask = topk_mask | (force_assign_mask & best_anchor_mask)

        return updated_topk_mask

    def filter_duplicates(self, iou_mat: Tensor, topk_mask: Tensor):
        """
        Filter the maximum suitability target index of each anchor based on IoU.

        Args:
            iou_mat [batch x targets x anchors]: The IoU for each targets-anchors
            topk_mask [batch x targets x anchors]: A boolean mask indicating the top-k scores' positions.

        Returns:
            unique_indices [batch x anchors x 1]: The index of the best targets for each anchors
            valid_mask [batch x anchors]: Mask indicating the validity of each anchor
            topk_mask [batch x targets x anchors]: A boolean mask indicating the updated top-k scores' positions.
        """
        # iou_mat, topk_mask: [B, T, A]
        
        # Cuántos GTs están asignados a cada anchor?
        num_gts_per_anchor = topk_mask.sum(dim=1, keepdim=True) # shape [B, 1, A]
        
        # Marcar anchors que tienen más de un GT asignado
        duplicates = (num_gts_per_anchor > 1) # shape [B, 1, A]
        
        # Poner a 0 los IoUs que no están en topk_mask
        masked_iou_mat = torch.where(topk_mask, iou_mat, torch.tensor(0.0, device=iou_mat.device))
        
        # Encontrar el GT con el IoU más alto para CADA anchor
        # masked_iou_mat [B, T, A] -> argmax(dim=1) -> [B, A]
        best_gt_indices_for_anchor = masked_iou_mat.argmax(dim=1) # shape [B, A]
        
        # Crear una máscara que solo sea True para el mejor GT de cada anchor
        best_target_mask = torch.zeros_like(topk_mask, dtype=torch.bool)
        # Usamos scatter_ en la dimensión 1 (GTs)
        best_target_mask.scatter_(1, index=best_gt_indices_for_anchor.unsqueeze(1), value=True)

        # Si un anchor tenía duplicados, solo mantenemos la asignación al mejor GT (según IoU).
        # Si no tenía duplicados, mantenemos la asignación original.
        final_mask = torch.where(duplicates.expand_as(topk_mask), best_target_mask, topk_mask) # [B, T, A]

        # Encontrar el índice del GT asignado a cada anchor (será 0 si no hay ninguno asignado)
        # Esto funciona porque final_mask ahora tiene como máximo un True por columna (anchor)
        assigned_gt_indices = final_mask.to(torch.int8).argmax(dim=1) # shape [B, A]
        
        # Máscara de validez: True si el anchor tiene algún GT asignado
        valid_anchor_mask = final_mask.any(dim=1) # shape [B, A]

        return assigned_gt_indices, valid_anchor_mask, final_mask

    def __call__(self, target: Tensor, predict: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]: 
        """Matches each target to the most suitable anchor.
        1. For each anchor prediction, find the highest suitability targets.
        2. Match target to the best anchor.
        3. Noramlize the class probilities of targets.

        Args:
            target: The ground truth class and bounding box information
                as tensor of size [batch x targets x 5].
            predict: Tuple of predicted class and bounding box tensors.
                Class tensor is of size [batch x anchors x class]
                Bounding box tensor is of size [batch x anchors x 4].

        Returns:
            anchor_matched_targets: Tensor of size [batch x anchors x (class + 4)].
                A tensor assigning each target/gt to the best fitting anchor.
                The class probabilities are normalized.
            valid_mask: Bool tensor of shape [batch x anchors].
                True if a anchor has a target/gt assigned to it.
        """
        predict_cls, predict_bbox = predict # cls: [B, A, C], bbox: [B, A, 4] (xyxy)

        # return if target has no gt information.
        n_targets = target.shape[1]
        B, A, C = predict_cls.shape
        device = predict_bbox.device

        if n_targets == 0:
            align_cls = torch.zeros_like(predict_cls, device=device)
            align_bbox = torch.zeros_like(predict_bbox, device=device)
            valid_mask = torch.zeros((B, A), dtype=torch.bool, device=device)
            gt_indices = torch.full((B, A), -1, dtype=torch.long, device=device) # Devolver -1 para no asignados
            anchor_matched_targets = torch.cat([align_cls, align_bbox], dim=-1)
            # --- Devolver 3 elementos ---
            return anchor_matched_targets, valid_mask, gt_indices

        target_cls, target_bbox = target.split([1, 4], dim=-1)  # cls: [B, T, 1], bbox: [B, T, 4] (xyxy)
        target_cls = target_cls.long().squeeze(-1) # -> [B, T]

        # get valid matrix (each gt appear in which anchor grid)
        grid_mask = self.get_valid_matrix(target_bbox) # [B, T, A]

        # get iou matrix (iou with each gt bbox and each predict anchor)
        iou_mat = self.get_iou_matrix(predict_bbox, target_bbox) # [B, T, A]

        # get cls matrix (cls prob with each gt class and each predict class)
        cls_mat = self.get_cls_matrix(predict_cls.sigmoid(), target_cls) # [B, T, A]
        
        # La función get_cls_matrix usa torch.gather para seleccionar las probabilidades de clase predichas (predict_cls) 
        # usando los índices de las clases reales (target_cls). Si algún índice en target_cls es negativo o 
        # mayor o igual que el número de clases (self.class_num), la operación gather en la GPU fallará con 
        # este assert triggered. -- DEBUGGEO EN YOLOSegmentationLoss para inspeccionar padded_targets. (Error CUDA índice)

        target_matrix = (iou_mat ** self.factor["iou"]) * (cls_mat ** self.factor["cls"]) # [B, T, A]

        # choose topk anchors for each GT based on target_matrix
        _, topk_mask = self.filter_topk(target_matrix, grid_mask, topk=self.topk) # [B, T, A]

        # match best anchor to valid targets without valid anchors
        topk_mask = self.ensure_one_anchor(target_matrix, topk_mask) # [B, T, A]

        # Ensure each anchor maps to at most one GT
        # assigned_gt_indices: [B, A], valid_mask: [B, A], final_topk_mask: [B, T, A]
        assigned_gt_indices, valid_mask, final_topk_mask = self.filter_duplicates(iou_mat, topk_mask)

        # --- Obtener los targets alineados ---
        # Usar assigned_gt_indices para seleccionar las bboxes GT correctas
        # assigned_gt_indices [B, A] -> [B, A, 1] -> [B, A, 4]
        bbox_indices = assigned_gt_indices.unsqueeze(-1).expand(B, A, 4)
        align_bbox = torch.gather(target_bbox, 1, bbox_indices) # [B, A, 4]
        
        # Usar assigned_gt_indices para seleccionar las clases GT correctas
        # assigned_gt_indices [B, A] -> [B, A, 1]
        cls_indices = assigned_gt_indices.unsqueeze(-1) # [B, A, 1]
        align_cls_indices = torch.gather(target_cls.unsqueeze(-1), 1, cls_indices) # [B, A, 1]

        # Crear one-hot encoding para las clases alineadas
        align_cls_onehot = torch.zeros_like(predict_cls, device=device) # [B, A, C]
        # Solo rellenar donde valid_mask es True para evitar scatter con índice -1 si no hay asignación
        valid_cls_indices = align_cls_indices[valid_mask] # Obtener índices válidos
        if valid_cls_indices.numel() > 0:
             align_cls_onehot[valid_mask] = align_cls_onehot[valid_mask].scatter(-1, index=valid_cls_indices.long(), value=1.0) # Convertir a long

        # --- Calcular puntuación de normalización ---
        # Usar la máscara final (final_topk_mask) que indica las asignaciones válidas
        masked_target_matrix = target_matrix * final_topk_mask # [B, T, A]
        masked_iou_mat = iou_mat * final_topk_mask # [B, T, A]

        # Normalizar puntuación por GT (máxima puntuación que obtuvo ese GT)
        max_score_per_gt = masked_target_matrix.amax(dim=-1, keepdim=True) # [B, T, 1]
        max_score_per_gt = torch.clamp(max_score_per_gt, min=1e-9) # Evitar división por cero

        # Normalizar IoU por GT (máximo IoU que obtuvo ese GT)
        max_iou_per_gt = masked_iou_mat.amax(dim=-1, keepdim=True) # [B, T, 1]
        max_iou_per_gt = torch.clamp(max_iou_per_gt, min=1e-9)

        # Calcular término de normalización [B, T, A]
        normalize_term = (masked_target_matrix / max_score_per_gt) * max_iou_per_gt

        # Seleccionar el término de normalización para los anchors asignados [B, A]
        # Usamos assigned_gt_indices [B, A] para seleccionar de normalize_term [B, T, A]
        # normalize_term.transpose(1, 2) -> [B, A, T]
        # assigned_gt_indices.unsqueeze(-1) -> [B, A, 1]
        selected_norm_term = torch.gather(normalize_term.transpose(1, 2), 2, assigned_gt_indices.unsqueeze(-1)).squeeze(-1) # [B, A]

        # Aplicar normalización a las clases one-hot y aplicar valid_mask
        align_cls_normalized = align_cls_onehot * selected_norm_term.unsqueeze(-1) * valid_mask.unsqueeze(-1)

        # Combinar clases y bboxes
        anchor_matched_targets = torch.cat([align_cls_normalized, align_bbox], dim=-1) # [B, A, C+4]
        
        # Preparar gt_indices de salida (-1 para no asignados)
        gt_indices_output = torch.where(valid_mask, assigned_gt_indices, torch.tensor(-1, dtype=torch.long, device=device))

        # --- Devolver 3 elementos ---
        return anchor_matched_targets, valid_mask, gt_indices_output


class Vec2Box:
    def __init__(self, model: YOLO, anchor_cfg: AnchorConfig, image_size, device):
        self.device = device

        if hasattr(anchor_cfg, "strides"):
            logger.info(f":japanese_not_free_of_charge_button: Found stride of model {anchor_cfg.strides}")
            self.strides = anchor_cfg.strides
        else:
            logger.info(":teddy_bear: Found no stride of model, performed a dummy test for auto-anchor size")
            self.strides = self.create_auto_anchor(model, image_size)

        anchor_grid, scaler = generate_anchors(image_size, self.strides)
        self.image_size = image_size
        self.anchor_grid, self.scaler = anchor_grid.to(device), scaler.to(device)

    def create_auto_anchor(self, model: YOLO, image_size):
        W, H = image_size
        # TODO: need accelerate dummy test
        dummy_input = torch.zeros(1, 3, H, W)
        dummy_output = model(dummy_input)
        strides = []
        for predict_head in dummy_output["Main"]:
            _, _, *anchor_num = predict_head[2].shape
            strides.append(W // anchor_num[1])
        return strides

    def update(self, image_size):
        """
        image_size: W, H
        """
        if self.image_size == image_size:
            return
        anchor_grid, scaler = generate_anchors(image_size, self.strides)
        self.image_size = image_size
        self.anchor_grid, self.scaler = anchor_grid.to(self.device), scaler.to(self.device)

    def __call__(self, predicts: List[Tuple[Tensor, Tensor, Tensor]]): # Añadida anotación de tipo
        preds_cls, preds_box_dist, preds_box_vec = [], [], [] 
        
        for layer_output in predicts:
            # Ahora layer_output = (class_x, anchor_x_raw, vector_x), todos 4D
            pred_cls, pred_box_dist_raw, pred_box_vec_raw = layer_output 
            
            # pred_cls: [B, C, H, W] -> [B, H*W, C]
            preds_cls.append(rearrange(pred_cls, "B C h w -> B (h w) C"))
            
            # pred_box_dist_raw (original 4D): [B, 4*reg_max, H, W] -> [B, H*W, 4*reg_max]
            # Esta es la línea que causaba el EinopsError. Ahora SÍ es 4D.
            preds_box_dist.append(rearrange(pred_box_dist_raw, "B X h w -> B (h w) X")) 
            
            # pred_box_vec_raw (4D): [B, 4, H, W] -> [B, H*W, 4]
            preds_box_vec.append(rearrange(pred_box_vec_raw, "B X h w -> B (h w) X"))
            
        preds_cls = torch.concat(preds_cls, dim=1)
        preds_box_dist = torch.concat(preds_box_dist, dim=1) # [B, A_total, 4*reg_max] (Para DFLoss)
        preds_box_vec = torch.concat(preds_box_vec, dim=1) # [B, A_total, 4] (Decodificado por DFL)

        # Usar preds_box_vec (que es la salida directa de Anchor2Vec)
        pred_LTRB = preds_box_vec * self.scaler.view(1, -1, 1) # [B, A_total, 4]
        
        lt, rb = pred_LTRB.chunk(2, dim=-1)
        # self.anchor_grid debe expandirse al batch size ---
        # self.anchor_grid es [A_total, 2]
        # Necesitamos [B, A_total, 2]
        anchor_grid_batch = self.anchor_grid.unsqueeze(0).expand(preds_cls.shape[0], -1, -1)
        preds_box = torch.cat([anchor_grid_batch - lt, anchor_grid_batch + rb], dim=-1) # xyxy
        #
        
        # Devolvemos cls, dist (para DFLoss) y box_xyxy (para IoULoss y Matcher)
        return preds_cls, preds_box_dist, preds_box


class Anc2Box:
    def __init__(self, model: YOLO, anchor_cfg: AnchorConfig, image_size, device):
        self.device = device

        if hasattr(anchor_cfg, "strides"):
            logger.info(f":japanese_not_free_of_charge_button: Found stride of model {anchor_cfg.strides}")
            self.strides = anchor_cfg.strides
        else:
            logger.info(":teddy_bear: Found no stride of model, performed a dummy test for auto-anchor size")
            self.strides = self.create_auto_anchor(model, image_size)

        self.head_num = len(anchor_cfg.anchor)
        self.anchor_grids = self.generate_anchors(image_size)
        self.anchor_scale = tensor(anchor_cfg.anchor, device=device).view(self.head_num, 1, -1, 1, 1, 2)
        self.anchor_num = self.anchor_scale.size(2)
        self.class_num = model.num_classes

    def create_auto_anchor(self, model: YOLO, image_size):
        W, H = image_size
        dummy_input = torch.zeros(1, 3, H, W).to(self.device)
        dummy_output = model(dummy_input)
        strides = []
        for predict_head in dummy_output["Main"]:
            _, _, *anchor_num = predict_head.shape
            strides.append(W // anchor_num[1])
        return strides

    def generate_anchors(self, image_size: List[int]):
        anchor_grids = []
        for stride in self.strides:
            W, H = image_size[0] // stride, image_size[1] // stride
            anchor_h, anchor_w = torch.meshgrid([torch.arange(H), torch.arange(W)], indexing="ij")
            anchor_grid = torch.stack((anchor_w, anchor_h), 2).view((1, 1, H, W, 2)).float().to(self.device)
            anchor_grids.append(anchor_grid)
        return anchor_grids

    def update(self, image_size):
        self.anchor_grids = self.generate_anchors(image_size)

    def __call__(self, predicts: List[Tensor]):
        preds_box, preds_cls, preds_cnf = [], [], []
        for layer_idx, predict in enumerate(predicts):
            predict = rearrange(predict, "B (L C) h w -> B L h w C", L=self.anchor_num)
            pred_box, pred_cnf, pred_cls = predict.split((4, 1, self.class_num), dim=-1)
            pred_box = pred_box.sigmoid()
            pred_box[..., 0:2] = (pred_box[..., 0:2] * 2.0 - 0.5 + self.anchor_grids[layer_idx]) * self.strides[
                layer_idx
            ]
            pred_box[..., 2:4] = (pred_box[..., 2:4] * 2) ** 2 * self.anchor_scale[layer_idx]
            preds_box.append(rearrange(pred_box, "B L h w A -> B (L h w) A"))
            preds_cls.append(rearrange(pred_cls, "B L h w C -> B (L h w) C"))
            preds_cnf.append(rearrange(pred_cnf, "B L h w C -> B (L h w) C"))

        preds_box = torch.concat(preds_box, dim=1)
        preds_cls = torch.concat(preds_cls, dim=1)
        preds_cnf = torch.concat(preds_cnf, dim=1)

        preds_box = transform_bbox(preds_box, "xycwh -> xyxy")
        return preds_cls, None, preds_box, preds_cnf.sigmoid()


def create_converter(model_version: str = "v9-c", *args, **kwargs) -> Union[Anc2Box, Vec2Box]:
    if "v7" in model_version:  # check model if v7
        converter = Anc2Box(*args, **kwargs)
    else:
        converter = Vec2Box(*args, **kwargs)
    return converter


def bbox_nms(cls_dist: Tensor, bbox: Tensor, nms_cfg: NMSConfig, confidence: Optional[Tensor] = None):
    cls_dist = cls_dist.sigmoid() * (1 if confidence is None else confidence)

    batch_idx, valid_grid, valid_cls = torch.where(cls_dist > nms_cfg.min_confidence)
    valid_con = cls_dist[batch_idx, valid_grid, valid_cls]
    valid_box = bbox[batch_idx, valid_grid]

    nms_idx = batched_nms(valid_box, valid_con, batch_idx + valid_cls * bbox.size(0), nms_cfg.min_iou)
    predicts_nms = []
    for idx in range(cls_dist.size(0)):
        instance_idx = nms_idx[idx == batch_idx[nms_idx]]

        predict_nms = torch.cat(
            [valid_cls[instance_idx][:, None], valid_box[instance_idx], valid_con[instance_idx][:, None]], dim=-1
        )

        predicts_nms.append(predict_nms[: nms_cfg.max_bbox])
    return predicts_nms


def calculate_map(predictions, ground_truths) -> Dict[str, Tensor]:
    metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy")
    mAP = metric([to_metrics_format(predictions)], [to_metrics_format(ground_truths)])
    return mAP


def to_metrics_format(prediction: Tensor):
    prediction = prediction[prediction[:, 0] != -1]
    bbox = {"boxes": prediction[:, 1:5], "labels": prediction[:, 0].int()}
    if prediction.size(1) == 6:
        bbox["scores"] = prediction[:, 5]
    return bbox 

