from pathlib import Path
import cv2
import torch
import pickle
import numpy as np
import albumentations as A

from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

class ImagesDataset(Dataset):
    def __init__(self, args, data_type, phase='train'):
        self.args = args
        self.phase = phase
        self.data_type = data_type
        self._read_path_label()
        self._setup_transformation(self.phase)
        self._get_label_list()

    def _read_path_label(self):
        if not Path(self.args.annot_file).is_file():
            raise FileNotFoundError(f'Annotation not found: {self.args.annot_file}. Run 000_make_Dataset.py first.')
        # Load only annotations generated locally or from a trusted source.
        with open(self.args.annot_file, 'rb') as handle:
            pkl = pickle.load(handle)

        if self.data_type is not None:
            pkl = pkl[self.data_type]
        if self.phase == 'train':
            self.data = pkl['Training_Set']
        elif self.phase == 'val':
            self.data = pkl['Validating_Set']
        elif self.phase == 'test':
            self.data = pkl['Testing_Set']
        else:
            raise ValueError("train mode must be in : Train or Validation")
        self.dataset_size = len(self.data)
        if not self.dataset_size:
            raise ValueError(f'Empty {self.phase} split in {self.args.annot_file}.')
        for split in ('Training_Set', 'Validating_Set', 'Testing_Set'):
            for item in pkl[split]:
                if not isinstance(item['label'], (int, np.integer)) or not 0 <= item['label'] < self.args.classes:
                    raise ValueError(f'Label {item["label"]} is outside [0, {self.args.classes}). Set --classes correctly.')
        self._get_mean_std(pkl['Training_Set'])

    def _get_mean_std(self, data):
        dataset_size = len(data)
        if dataset_size == 0:
            raise ValueError("Training_Set is empty; normalization requires training images.")
        self.mean = np.zeros(1)
        self.std = np.zeros(1)
        for idx in range(dataset_size):
            img = self._read_image(data[idx]['path'], flags=0)
            self.mean += np.mean(img)
            self.std += np.std(img)
        self.mean = self.mean / dataset_size / 255.
        self.std = self.std / dataset_size / 255.
        if not np.all(np.isfinite(self.std)) or np.any(self.std <= 0):
            raise ValueError("Training images must have a finite, nonzero mean per-image standard deviation.")
        self.mean = self.mean.repeat(3)
        self.std = self.std.repeat(3)
            
    def _read_image(self, path, flags=cv2.IMREAD_COLOR):
        path = Path(path)
        root = getattr(self.args, "data_root", None)
        if root is not None and not path.is_absolute():
            path = Path(root) / path
        image = cv2.imread(str(path), flags)
        if image is None:
            raise ValueError(f"Cannot read image: {path}. Check annotation paths and --data_root.")
        return image

    def _get_label_list(self):
        self.label_list = [data['label'] for data in self.data]
    
    def _setup_transformation(self, phase):
        self.phase = phase
        if self.phase == 'train':
            self.transforms = A.Compose([
                A.PadIfNeeded(min_height=self.args.pad_height_width, min_width=self.args.pad_height_width, border_mode=0),
                A.Resize(self.args.img_size, self.args.img_size), 
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(),
            ])

        elif self.phase == 'test' or self.phase == 'val':
            self.transforms = A.Compose([
                A.PadIfNeeded(min_height=self.args.pad_height_width, min_width=self.args.pad_height_width, border_mode=0),
                A.Resize(self.args.img_size, self.args.img_size), 
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2(),
            ])

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        img = self._read_image(self.data[idx]['path'])
        img = self.transforms(image=img)['image'] 
        label = self.data[idx]['label']
        label = torch.tensor(label, dtype=torch.long)
        return img, label


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None
        self.preload()

    def preload(self):
        try:
            self.next_image, self.next_label = next(self.loader)
        except StopIteration:
            self.next_image = None
            self.next_label = None
            return

        with torch.cuda.stream(self.stream):
            self.next_image = self.next_image.cuda(non_blocking=True)
            self.next_label = self.next_label.cuda(non_blocking=True)
    
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        image = self.next_image
        label = self.next_label
        if image is not None:
            image.record_stream(torch.cuda.current_stream())
            label.record_stream(torch.cuda.current_stream())
        self.preload()
        return image, label
