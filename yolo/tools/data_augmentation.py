from typing import List
import os

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF


class AugmentationComposer:
    """Composes several transforms together."""

    def __init__(self, transforms, image_size: int = [640, 640], base_size: int = 640): # type: ignore
        self.transforms = transforms
        # TODO: handle List of image_size [640, 640]
        self.pad_resize = PadAndResize(image_size)
        self.base_size = base_size

        self.get_more_data = None
        
        for transform in self.transforms:
            if hasattr(transform, "set_parent"):
                transform.set_parent(self)

    def __call__(self, image, boxes=torch.zeros(0, 5), segments:List=[]):
        """
        Applies transformations to image, boxes, and segments.

        Args:
            image (PIL.Image): Input image.
            boxes (torch.Tensor): Bounding boxes [N, 5] (cls, xmin, ymin, xmax, ymax).
            segments (List[np.array]): List of segments [N, K, 2] (x, y).

        Returns:
            Tuple[Tensor, Tensor, List[np.array], Tensor]:
                - Transformed image tensor.
                - Transformed boxes tensor.
                - Transformed segments list.
                - Reverse tensor for coordinate mapping.
        """
        for transform in self.transforms:
            # get segments to each transformations
            image, boxes, segments = transform(image, boxes, segments)
        # PadAndResize should transform the segments too
        image, boxes, segments, rev_tensor = self.pad_resize(image, boxes, segments)
        image = TF.to_tensor(image)
        return image, boxes, segments, rev_tensor

class PadAndResize:
    def __init__(self, image_size, background_color=(114, 114, 114)):
        """Initialize the object with the target image size."""
        self.target_width, self.target_height = image_size
        self.background_color = background_color

    def set_size(self, image_size: List[int]):
        self.target_width, self.target_height = image_size
 
    def __call__(self, image: Image, boxes, segments: List): # type: ignore
        img_width, img_height = image.size # type: ignore
        scale = min(self.target_width / img_width, self.target_height / img_height)
        new_width, new_height = int(img_width * scale), int(img_height * scale)

        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS) #type:ignore

        pad_left = (self.target_width - new_width) // 2
        pad_top = (self.target_height - new_height) // 2
        padded_image = Image.new("RGB", (self.target_width, self.target_height), self.background_color)
        padded_image.paste(resized_image, (pad_left, pad_top))

        # Transform segments
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] * new_width + pad_left) / self.target_width
        boxes[:, [2, 4]] = (boxes[:, [2, 4]] * new_height + pad_top) / self.target_height

        # Transform segments
        if segments: 
            segments_out = []
            for seg in segments:
                # seg is [K,2]
                seg_copy = seg.copy()
                seg_copy[:, 0] = (seg_copy[:, 0] * new_width + pad_left) / self.target_width # scale y pad X
                seg_copy[:, 1] = (seg_copy[:, 1] * new_height + pad_top) / self.target_height # scale y pad Y
                segments_out.append(seg_copy)
            segments = segments_out
        transform_info = torch.tensor([scale, pad_left, pad_top, pad_left, pad_top])
        return padded_image, boxes, segments, transform_info


class HorizontalFlip:
    """Randomly horizontally flips the image along with the bounding boxes and segments."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, boxes, segments:List=[]):
        if torch.rand(1) < self.prob:
            image = TF.hflip(image)
            # Flip bbox
            boxes[:, [1, 3]] = 1 - boxes[:, [3, 1]]
            # Flip segments
            if segments:
                for seg in segments:
                    seg[:,0] = 1.0 - seg[:, 0] # flip x coords
        return image, boxes, segments


class VerticalFlip:
    """Randomly vertically flips the image along with the bounding boxes and segments."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, boxes, segments:List=[]):
        if torch.rand(1) < self.prob:
            image = TF.vflip(image)
            # Flip bbox
            boxes[:, [2, 4]] = 1 - boxes[:, [4, 2]]
            #Flip segments
            if segments:
                for seg in segments:
                    seg[:, 1] = 1.0 - seg[:, 1] # flip y coordinates
        return image, boxes, segments

class MixUp:
    """Applies the MixUp augmentation to a pair of images, boxes, and segments."""

    def __init__(self, prob=0.5, alpha=1.0):
        self.alpha = alpha
        self.prob = prob
        self.parent = None

    def set_parent(self, parent):
        """Set the parent dataset object for accessing dataset methods."""
        self.parent = parent

    def __call__(self, image, boxes, segments:List=[]):
        if torch.rand(1) >= self.prob:
            return image, boxes, segments

        assert self.parent is not None, "Parent is not set. MixUp cannot retrieve additional data."
        assert hasattr(self.parent, "get_more_data"), "Parent object must have 'get_more_data' method."
        
        # Retrieve another image and its boxes randomly from the dataset
        image2, boxes2, segments2 = self.parent.get_more_data()[0]

        # Calculate the mixup lambda parameter
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 0.5

        # Mix images
        image1, image2 = TF.to_tensor(image), TF.to_tensor(image2)
        mixed_image = lam * image1 + (1 - lam) * image2

        # Merge bounding boxes and segments
        merged_boxes = torch.cat((boxes, boxes2))
        merged_segments = segments + segments2 # just concatenate the lists

        return TF.to_pil_image(mixed_image), merged_boxes, merged_segments

# Test script
if __name__ == "__main__":
    """ 
    Execute a simple test to verify the segmentation transforms. 
    """
    
    TEST_IMAGE_PATH = "test_data/image/non_perio_12-tif_0_png.rf.c8860e3e21c78e84efd8bca41479ad7f.jpg"
    TEST_LABEL_PATH = "test_data/label/non_perio_12-tif_0_png.rf.c8860e3e21c78e84efd8bca41479ad7f.txt"
    
    if not os.path.exists(TEST_IMAGE_PATH) or not os.path.exists(TEST_LABEL_PATH):
        print(f"Error: No se encontraron los archivos de prueba en el directorio actual.")
        print(f"Asegúrate de copiar '{TEST_IMAGE_PATH}' y '{TEST_LABEL_PATH}' a la carpeta donde ejecutas el script.")
    else:
        def load_real_label_data(txt_path):
            """Lee un archivo .txt de YOLO-segmentation y devuelve bboxes y segmentos."""
            bboxes = []
            segments = []
            with open(txt_path, 'r') as f:
                for line in f:
                    # El formato es (cls x1 y1 x2 y2 ...)
                    parts = line.strip().split()
                    if len(parts) < 3: # (cls + al menos 1 punto)
                        continue

                    cls_id = int(parts[0])
                    # Coordenadas (x, y)
                    coords = np.array(parts[1:], dtype=float).reshape(-1, 2)
                    
                    # Calcular BBox a partir de segmentos (en formato [xmin, ymin, xmax, ymax])
                    xmin, ymin = coords.min(axis=0)
                    xmax, ymax = coords.max(axis=0)
                    
                    # Guardamos la BBox en formato (cls, xmin, ymin, xmax, ymax)
                    bboxes.append([cls_id, xmin, ymin, xmax, ymax])
                    # Guardamos el segmento
                    segments.append(coords)
            return torch.tensor(bboxes, dtype=torch.float32), segments
        
        img_real = Image.open(TEST_IMAGE_PATH).convert("RGB")
        boxes_real, segments_real = load_real_label_data(TEST_LABEL_PATH)
        img_size_original = img_real.size
        target_size = [640, 640]
        
        def real_get_more_data(num=1):
            """Simula YoloDataset.get_more_data devolviendo los mismos datos reales."""
            # Usamos .copy() para evitar la modificación in-situ
            return [(img_real.copy(), boxes_real.clone(), [s.copy() for s in segments_real])]
        transforms_list = [
            HorizontalFlip(prob=1.0),
            VerticalFlip(prob=1.0),
            MixUp(prob=1.0) # Forzamos MixUp para probar la concatenación
        ]
        
        composer = AugmentationComposer(transforms_list, image_size=target_size, base_size=target_size[0])
        composer.get_more_data = real_get_more_data
        
        print(f"Datos reales cargados: {len(boxes_real)} cajas, {len(segments_real)} segmentos.")
        
        # 4. Ejecutar la cadena
        img_out, boxes_out, segs_out, rev_out = composer(
            img_real.copy(), 
            boxes_real.clone(), 
            [s.copy() for s in segments_real]
        )

        # 5. Verificar resultados
        print("\n--- PRUEBAS UNITARIAS (Datos Reales) ---")
        
        # Verificar MixUp (debería duplicar el número de instancias)
        expected_count = len(boxes_real) * 2
        assert len(boxes_out) == expected_count
        assert len(segs_out) == expected_count
        print(f"✅ MixUp (con segmentos) funcional (Salida: {len(boxes_out)} cajas, {len(segs_out)} segmentos).")

        # Verificar Flips (solo en el primer segmento)
        # Coordenadas del primer punto del primer segmento [cite: 1]
        original_seg_point_x = segments_real[0][0, 0] # 0.47446... 
        original_seg_point_y = segments_real[0][0, 1] # 0.5390...
        
        # Punto después de HFlip(1.0) y VFlip(1.0)
        # HFlip: 1.0 - 0.47446...
        # VFlip: 1.0 - 0.5390...
        expected_x_flipped = 1.0 - original_seg_point_x
        expected_y_flipped = 1.0 - original_seg_point_y
        
        # NOTA: PadAndResize se aplica DESPUÉS. Debemos revertir su transformación para la prueba.
        # (new_x * target_w - pad_left) / new_width = flipped_x
        # (new_y * target_h - pad_top) / new_height = flipped_y
        
        # Extraer info de PadAndResize
        scale, pad_left, pad_top = rev_out[0], rev_out[1], rev_out[2]
        new_width = int(img_size_original[0] * scale)
        new_height = int(img_size_original[1] * scale)

        # Coordenadas X e Y del primer punto del primer segmento en la SALIDA
        final_seg_point_x = segs_out[0][0, 0] 
        final_seg_point_y = segs_out[0][0, 1]
        
        # Revertir la transformación de PadAndResize
        reverted_x = (final_seg_point_x * target_size[0] - pad_left) / new_width
        reverted_y = (final_seg_point_y * target_size[1] - pad_top) / new_height

        assert np.isclose(reverted_x, expected_x_flipped)
        assert np.isclose(reverted_y, expected_y_flipped)
        print(f"✅ HorizontalFlip / VerticalFlip (Segmentos) funcionales (verificado tras Pad&Resize).")

        print(f"✅ PadAndResize (con segmentos) funcional (rev_tensor: {rev_out}).")
        print("\n--- ¡'data_augmentation.py' modificado y verificado con datos reales! ---")