import torch

def crop_mask(masks, boxes):
    """
    Recorta máscaras a las regiones de las bounding boxes.

    Args:
        masks (torch.Tensor): Máscaras con forma (N, H, W).
        boxes (torch.Tensor): Coordenadas de Bounding box (N, 4) en formato
                                xyxy relativo [0, 1].

    Returns:
        (torch.Tensor): Máscaras recortadas.
    """
    n, h, w = masks.shape
    # 'boxes' está normalizado (0-1). Necesitamos escalarlo a las 
    # dimensiones de la máscara (h, w) ANTES de usarlo.
    boxes_scaled = boxes.clone()
    boxes_scaled[:, 0] *= w  # Escalar x1
    boxes_scaled[:, 1] *= h  # Escalar y1
    boxes_scaled[:, 2] *= w  # Escalar x2
    boxes_scaled[:, 3] *= h  # Escalar y2
    if n < 50:  # faster for fewer masks (predict)
        # Usar 'boxes_scaled' aquí
        for i, (x1, y1, x2, y2) in enumerate(boxes_scaled.round().int()):
            # Clampear los valores para asegurarse de que están dentro de los límites (0, w) y (0, h)
            x1 = torch.clamp(x1, 0, w)
            y1 = torch.clamp(y1, 0, h)
            x2 = torch.clamp(x2, 0, w)
            y2 = torch.clamp(y2, 0, h)
            
            masks[i, :y1] = 0
            masks[i, y2:] = 0
            masks[i, :, :x1] = 0
            masks[i, :, x2:] = 0
        return masks
    else:  # faster for more masks (val)
        # Usar 'boxes_scaled' aquí
        x1, y1, x2, y2 = torch.chunk(boxes_scaled[:, :, None], 4, 1)  # x1 shape(n,1,1)
        
        r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
        c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)
        
        # La comparación ahora es Píxel vs Píxel (ej. 10 >= 32.6)
        # Asegurarse de clampear las coordenadas para que no salgan del rango 0-w o 0-h
        x1 = torch.clamp(x1, 0, w)
        y1 = torch.clamp(y1, 0, h)
        x2 = torch.clamp(x2, 0, w)
        y2 = torch.clamp(y2, 0, h)
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))