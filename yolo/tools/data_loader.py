from pathlib import Path
from queue import Empty, Queue
from statistics import mean
from threading import Event, Thread
from typing import Generator, List, Tuple, Union

import numpy as np
import torch
from PIL import Image
from rich.progress import track
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from yolo.config.config import DataConfig, DatasetConfig
from yolo.tools.data_augmentation import *
from yolo.tools.data_augmentation import AugmentationComposer
from yolo.tools.dataset_preparation import prepare_dataset
from yolo.utils.dataset_utils import (
    create_image_metadata,
    locate_label_paths,
    scale_segmentation,
)
from yolo.utils.logger import logger


class YoloDataset(Dataset):
    def __init__(self, data_cfg: DataConfig, dataset_cfg: DatasetConfig, phase: str = "train2017"):
        augment_cfg = data_cfg.data_augment
        self.image_size = data_cfg.image_size
        phase_name = getattr(dataset_cfg, phase, phase) #type:ignore
        self.batch_size = data_cfg.batch_size
        self.dynamic_shape = getattr(data_cfg, "dynamic_shape", False)
        self.base_size = mean(self.image_size)

        transforms = [eval(aug)(prob) for aug, prob in augment_cfg.items()]
        self.transform = AugmentationComposer(transforms, self.image_size, self.base_size) #type:ignore
        self.transform.get_more_data = self.get_more_data
        data = self.load_data(Path(dataset_cfg.path), phase_name)
        
        # Get data
        try:
            img_paths, bboxes, segments, ratios = zip(*data)
        except ValueError as e:
            logger.error(
                ":rotating_light: Error al descomprimir datos. Esto puede ser causado por un caché antiguo."
                "\n:rotating_light: Por favor, elimina el archivo .pache (o .cache) e inténtalo de nuevo."
            )
            raise e

        self.img_paths = list(img_paths)
        self.bboxes = list(bboxes)      # list of tensors
        self.segments = list(segments)  # list of arrays
        self.ratios = list(ratios)

    def load_data(self, dataset_path: Path, phase_name: str):
        """
        Loads data from a cache or generates a new cache for a specific dataset phase.

        Parameters:
            dataset_path (Path): The root path to the dataset directory.
            phase_name (str): The specific phase of the dataset (e.g., 'train', 'test') to load or generate data for.

        Returns:
            dict: The loaded data from the cache for the specified phase.
        """
        cache_path = Path("/kaggle/working/") / f"{dataset_path.name}-{phase_name}.cache"

        if not cache_path.exists():
            logger.info(f":factory: Generating {phase_name} cache")
            data = self.filter_data(dataset_path, phase_name, self.dynamic_shape)
            torch.save(data, cache_path)
        else:
            try:
                data = torch.load(cache_path, weights_only=False)
            except Exception as e:
                logger.error(
                    f":rotating_light: Failed to load the cache at '{cache_path}'.\n"
                    ":rotating_light: This may be caused by using cache from different other YOLO.\n"
                    ":rotating_light: Please clean the cache and try running again."
                )
                raise e
            logger.info(f":package: Loaded {phase_name} cache, there are {len(data)} data in total.")
        return data

    def filter_data(self, dataset_path: Path, phase_name: str, sort_image: bool = False) -> list:
        """
        Filters and collects dataset information by pairing images with their corresponding labels.

        Parameters:
            images_path (Path): Path to the directory containing image files.
            labels_path (str): Path to the directory containing label files.
            sort_image (bool): If True, sorts the dataset by the width-to-height ratio of images in descending order.

        Returns:
            list: A list of tuples, each containing the path to an image file and its associated segmentation as a tensor.
        """
        images_path = dataset_path / "images" / phase_name
        labels_path, data_type = locate_label_paths(dataset_path, phase_name)
        file_list, adjust_path = dataset_path / f"{phase_name}.txt", False
        if file_list.exists():
            data_type, adjust_path = "txt", True
            with open(file_list, "r") as file:
                images_list = [dataset_path / line.rstrip() for line in file]
            labels_list = [
                Path(str(image_path).replace("images", "labels")).with_suffix(".txt") for image_path in images_list
            ]
        else:
            images_list = sorted([p.name for p in Path(images_path).iterdir() if p.is_file()])

        if data_type == "json":
            annotations_index, image_info_dict = create_image_metadata(labels_path) #type:ignore

        data = []
        valid_inputs = 0
        for idx, image_name in enumerate(track(images_list, description="Filtering data")):
            if not adjust_path and not str(image_name).lower().endswith((".jpg", ".jpeg", ".png")): #
                continue
            image_id = Path(image_name).stem
            if data_type == "json":
                image_info = image_info_dict.get(image_id, None)
                if image_info is None:
                    continue
                annotations = annotations_index.get(image_info["id"], [])
                image_seg_annotations = scale_segmentation(annotations, image_info)
            elif data_type == "txt":
                label_path = labels_list[idx] if adjust_path else labels_path / f"{image_id}.txt" # type:ignore
                if not label_path.is_file():
                    image_seg_annotations = []
                else:
                    with open(label_path, "r") as file:
                        image_seg_annotations = [list(map(float, line.strip().split())) for line in file]
            else:
                image_seg_annotations = []

            labels_tuple = self.load_valid_labels(image_id, image_seg_annotations)  # type:ignore
            img_path = image_name if adjust_path else images_path / image_name
            if sort_image:
                with Image.open(img_path) as img:
                    width, height = img.size
            else:
                width, height = 0, 1
            
            data.append((img_path, labels_tuple[0], labels_tuple[1], width / height)) # type:ignore
            if len(image_seg_annotations) != 0: # type:ignore
                valid_inputs += 1

        if sort_image: # Solo ordenar si se solicita
            data = sorted(data, key=lambda x: x[3], reverse=True)
        logger.info(f"Recorded {valid_inputs}/{len(images_list)} valid inputs")
        return data

    def load_valid_labels(self, label_path: str, seg_data_one_img: List) -> Tuple[Tensor, List]:
        """
        Loads valid COCO style segmentation data (values between [0, 1]) and converts it to bounding box coordinates
        by finding the minimum and maximum x and y values.

        Parameters:
            label_path (str): The filepath to the label file containing annotation data.
            seg_data_one_img (list): The actual list of annotations (in segmentation format)

        Returns:
            Tensor or None: A tensor of all valid bounding boxes if any are found; otherwise, None.
        """
        bboxes = []
        segments = []
        for seg_data in seg_data_one_img:
            cls = seg_data[0]
            points = np.array(seg_data[1:]).reshape(-1, 2).clip(0, 1)
            valid_points = points[(points >= 0) & (points <= 1)].reshape(-1, 2)
            if valid_points.size > 1:
                bbox = torch.tensor([cls, *valid_points.min(axis=0), *valid_points.max(axis=0)])
                bboxes.append(bbox)
                segments.append(valid_points)

        if bboxes:
            return torch.stack(bboxes), segments
        else:
            logger.warning(f"No valid BBox in {label_path}")
            return torch.zeros((0, 5)), []

    def get_data(self, idx):
        img_path, bboxes = self.img_paths[idx], self.bboxes[idx].clone()
        segments = [s.copy() for s in self.segments[idx]]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
        return img, bboxes, segments, img_path

    def get_more_data(self, num: int = 1):
        indices = torch.randint(0, len(self), (num,))
        return [self.get_data(idx)[:3] for idx in indices]

    def _update_image_size(self, idx: int) -> None:
        """Update image size based on dynamic shape and batch settings."""
        batch_start_idx = (idx // self.batch_size) * self.batch_size
        image_ratio = self.ratios[batch_start_idx].clip(1 / 3, 3)
        shift = ((self.base_size / 32 * (image_ratio - 1)) // (image_ratio + 1)) * 32

        self.image_size = [int(self.base_size + shift), int(self.base_size - shift)]
        self.transform.pad_resize.set_size(self.image_size)

    def __getitem__(self, idx) -> Tuple[Image.Image, Tensor, List, Tensor, str]:
        img, bboxes, segments, img_path = self.get_data(idx)

        if self.dynamic_shape:
            self._update_image_size(idx)

        img, bboxes, segments, rev_tensor = self.transform(img, bboxes, segments)
        
        return img, bboxes, segments, rev_tensor, img_path # type:ignore

    def __len__(self) -> int:
        return len(self.bboxes)


def collate_fn(batch: List[Tuple[Tensor, Tensor, List, Tensor, str]]):
    """
    A collate function to handle batching of images and their corresponding targets.

    Args:
        batch (list of tuples): Each tuple contains:
            - image (Tensor): The image tensor.
            - labels (Tensor): The tensor of labels (bboxes) for the image.
            - segments (List[np.array]): The list of segments for the image.
            - rev_tensor (Tensor): The reverse transform tensor.
            - img_path (str): The path to the image.

    Returns:
        Tuple: A tuple containing:
            - A tensor of batched images.
            - A tensor of all bboxes [N, 6] (batch_idx, cls, xmin, ymin, xmax, ymax).
            - A list of all segments [N].
            - (Deprecado, se devuelve bboxes)
            - A tensor of batched reverse tensors.
            - A list of image paths.
    """
    batch_images = []
    batch_bboxes_list = []
    batch_segments_list = []
    batch_reverse_list = []
    batch_path_list = []
    for i, (img, bboxes, segments, rev_tensor, img_path) in enumerate(batch):
        batch_images.append(img)
        batch_reverse_list.append(rev_tensor)
        batch_path_list.append(img_path)
        
        if bboxes.numel() > 0:
            # Creamos el batch_idx y lo añadimos como primera columna
            batch_idx = torch.full((bboxes.size(0), 1), i, dtype=torch.float32)
            # bboxes ahora es [N, 6] -> (batch_idx, cls, xmin, ymin, xmax, ymax)
            bboxes_with_idx = torch.cat([batch_idx, bboxes], dim=1)
            
            batch_bboxes_list.append(bboxes_with_idx)
            batch_segments_list.extend(segments) # Añadimos la lista de segmentos

    # Apilamos imágenes y tensores de reversa
    stacked_images = torch.stack(batch_images, 0)
    stacked_reverse = torch.stack(batch_reverse_list, 0)

    # Concatenamos bboxes si hay alguna
    if batch_bboxes_list:
        cat_bboxes = torch.cat(batch_bboxes_list, 0)
    else:
        # Creamos un tensor vacío con la forma correcta [0, 6]
        cat_bboxes = torch.empty((0, 6), dtype=torch.float32)

    # El lote (batch) que se pasará a training_step será una tupla
    # batch_idx (índice 0) -> stacked_images
    # batch_idx (índice 1) -> targets (que contiene bboxes y segmentos)
    
    targets = {
        "bboxes": cat_bboxes,
        "segments": batch_segments_list
    }

    # Devolvemos (imágenes, objetivos, tensor_reversa, caminos)
    return stacked_images, targets, stacked_reverse, batch_path_list


def create_dataloader(data_cfg: DataConfig, dataset_cfg: DatasetConfig, task: str = "train"):
    if task == "inference":
        return StreamDataLoader(data_cfg)

    if getattr(dataset_cfg, "auto_download", False):
        prepare_dataset(dataset_cfg, task)
    dataset = YoloDataset(data_cfg, dataset_cfg, task)

    return DataLoader(
        dataset,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.cpu_num,
        pin_memory=data_cfg.pin_memory,
        collate_fn=collate_fn,
        shuffle=(task == "train")
    )


class StreamDataLoader:
    def __init__(self, data_cfg: DataConfig):
        self.source = data_cfg.source
        self.running = True
        self.is_stream = isinstance(self.source, int) or str(self.source).lower().startswith("rtmp://")

        self.transform = AugmentationComposer([], data_cfg.image_size)
        self.stop_event = Event()

        if self.is_stream:
            import cv2

            self.cap = cv2.VideoCapture(self.source)
        else:
            self.source = Path(self.source)
            self.queue = Queue()
            self.thread = Thread(target=self.load_source)
            self.thread.start()

    def load_source(self):
        if self.source.is_dir():  # image folder
            self.load_image_folder(self.source)
        elif any(self.source.suffix.lower().endswith(ext) for ext in [".mp4", ".avi", ".mkv"]):  # Video file
            self.load_video_file(self.source)
        else:  # Single image
            self.process_image(self.source)

    def load_image_folder(self, folder):
        folder_path = Path(folder)
        for file_path in folder_path.rglob("*"):
            if self.stop_event.is_set():
                break
            if file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                self.process_image(file_path)

    def process_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        if image is None:
            raise ValueError(f"Error loading image: {image_path}")
        self.process_frame(image)

    def load_video_file(self, video_path):
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            self.process_frame(frame)
        cap.release()

    def process_frame(self, frame):
        if isinstance(frame, np.ndarray):
            # TODO: we don't need cv2
            import cv2

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
        origin_frame = frame
        frame, _, rev_tensor = self.transform(frame, torch.zeros(0, 5))
        frame = frame[None]
        rev_tensor = rev_tensor[None]
        if not self.is_stream:
            self.queue.put((frame, rev_tensor, origin_frame))
        else:
            self.current_frame = (frame, rev_tensor, origin_frame)

    def __iter__(self) -> Generator[Tensor, None, None]:
        return self

    def __next__(self) -> Tensor:
        if self.is_stream:
            ret, frame = self.cap.read()
            if not ret:
                self.stop()
                raise StopIteration
            self.process_frame(frame)
            return self.current_frame
        else:
            try:
                frame = self.queue.get(timeout=1)
                return frame
            except Empty:
                raise StopIteration

    def stop(self):
        self.running = False
        if self.is_stream:
            self.cap.release()
        else:
            self.thread.join(timeout=1)

    def __len__(self):
        return self.queue.qsize() if not self.is_stream else 0

if __name__ == "__main__":
    """ 
    Ejecutar testeo de script
    """
    
    import os
    import yaml
    from yolo.config.config import DataConfig, DatasetConfig
    
    config_path = "yolo/config/dataset/dental_dataset.yaml"
    if not os.path.exists(config_path):
        print(f"Error: No se encontró el archivo de configuración en '{config_path}'.")
        print("Asegúrate de ejecutar este script desde el directorio raíz 'YOLO'.")
    else:
        try:
            # 2. Cargar el DatasetConfig desde el YAML
            with open(config_path, 'r') as f:
                dataset_config_data = yaml.safe_load(f)
            
            # Asumimos que el YAML contiene los campos para DatasetConfig
            # (path, train, val, nc, names, etc.)
            dataset_cfg = DatasetConfig(
                path=dataset_config_data.get('path', ''),
                class_num=dataset_config_data.get('class_num', 0),
                class_list=dataset_config_data.get('class_list', []),
                # Asumimos que auto_download no está presente o es None
                auto_download=dataset_config_data.get('auto_download', None) 
            )
            
            if 'train' in dataset_config_data:
                setattr(dataset_cfg, 'train', dataset_config_data['train'])
            if 'validation' in dataset_config_data:
                 # Cambiamos 'validation' a 'val' si YoloDataset lo espera así
                 # Basado en, usa 'phase', así que mantenemos 'validation' por ahora
                setattr(dataset_cfg, 'validation', dataset_config_data['validation'])
            
            # 3. Crear un DataConfig 'mock' (mínimo necesario para el test)
            # Usamos los valores por defecto que YoloDataset espera
            mock_data_cfg = DataConfig(
                image_size=[640, 640],
                batch_size=2,  # Probamos con un batch_size pequeño
                data_augment={}, # Desactivamos aumentaciones para la prueba
                dynamic_shape=False,
                cpu_num=0,       # 0 es a menudo más rápido para depurar
                pin_memory=True,
                source="",      # No es relevante para 'train'
                shuffle=False
            )

            print(f"Configuración de dataset '{config_path}' cargada.")
            print(f"Creando DataLoader para 'train'...")

            # 4. Llamar a create_dataloader (esto prueba YoloDataset y collate_fn)
            train_loader = create_dataloader(
                data_cfg=mock_data_cfg,
                dataset_cfg=dataset_cfg,
                task="train"
            )
            
            # 5. Intentar obtener un lote
            print("Intentando obtener un lote (batch)...")
            # El dataloader devuelve: (stacked_images, targets, stacked_reverse, batch_path_list)
            images, targets, _, paths = next(iter(train_loader))
            
            # 6. Verificar el lote
            print("\n--- ¡Prueba exitosa! ---")
            print(f"Forma del lote de imágenes: {images.shape}")
            print(f"Tipo de objetivos (targets): {type(targets)}")
            
            # targets es un dict: {'bboxes': tensor, 'segments': list}
            bboxes = targets['bboxes']
            segments = targets['segments']
            
            print(f"Forma de BBoxes concatenadas: {bboxes.shape}")
            print(f"Total de segmentos en el lote: {len(segments)}")
            
            # Verificamos que la estructura sea la esperada
            # bboxes debe ser [N, 6] (batch_idx, cls, xmin, ymin, xmax, ymax)
            assert images.shape[0] == mock_data_cfg.batch_size
            assert isinstance(targets, dict)
            assert bboxes.shape[1] == 6
            assert len(segments) == bboxes.shape[0]
            
            print("\n✅ 'data_loader.py' (incl. YoloDataset y collate_fn) parece funcionar correctamente.")

        except Exception as e:
            print(f"\n--- PRUEBA FALLIDA ---")
            print(f"Se produjo un error: {e}")
            import traceback
            traceback.print_exc()