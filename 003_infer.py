"""Extract a prediction and embedding from a user-provided CUDA checkpoint."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2

from datasets import ImagesDataset
from models import CLCNN
from utils import configure_cuda, load_checkpoint


def get_argument(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', default=None, help='Optional output JSON file.')
    parser.add_argument('--trusted_checkpoint', action='store_true')
    parser.add_argument('--annot_file', default=None, help='Trusted training annotation for legacy checkpoints without preprocessing metadata.')
    parser.add_argument('--data_root', default=None)
    parser.add_argument('--data_type', choices=['LED', 'LASER'], default=None)
    return parser.parse_args(argv)


def main(args):
    configure_cuda(args)
    weights = load_checkpoint(args.checkpoint, args.device, args.trusted_checkpoint)
    metadata = weights.get('preprocessing')
    if metadata is None:
        if args.annot_file is None:
            raise ValueError('This legacy checkpoint has no preprocessing metadata. Supply --annot_file from its original training run.')
        saved = weights.get('args', {})
        config = vars(saved).copy() if isinstance(saved, argparse.Namespace) else dict(saved)
        required = ('classes', 'img_size', 'pad_height_width')
        if any(key not in config for key in required):
            raise ValueError('Legacy checkpoint needs saved classes, img_size and pad_height_width to reconstruct preprocessing.')
        modality = args.data_type
        if config.get('datasets') == 'PLUSVein' and modality is None:
            raise ValueError('For a legacy PLUSVein checkpoint, specify --data_type LED or LASER.')
        config.update(annot_file=args.annot_file, data_root=args.data_root)
        dataset = ImagesDataset(SimpleNamespace(**config), modality, phase='train')
        metadata = {key: config[key] for key in required}
        metadata.update(mean=dataset.mean.tolist(), std=dataset.std.tolist())
    std = np.asarray(metadata['std'])
    if not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError('Invalid checkpoint normalization standard deviation.')
    model = CLCNN(num_classes=metadata['classes']).to(args.device)
    model.load_state_dict(weights['model_state_dict'])
    model.eval()
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f'Cannot read image: {args.image}')
    transform = A.Compose([
        A.PadIfNeeded(min_height=metadata['pad_height_width'], min_width=metadata['pad_height_width'], border_mode=0),
        A.Resize(metadata['img_size'], metadata['img_size']),
        A.Normalize(mean=metadata['mean'], std=metadata['std']),
        ToTensorV2(),
    ])
    inputs = transform(image=image)['image'].unsqueeze(0).to(args.device)
    with torch.inference_mode():
        logits, features = model(inputs)
        embedding = F.normalize(features, dim=1)
    result = {'image': args.image, 'class_index': int(logits.argmax(dim=1).item()),
              'logits': logits[0].cpu().tolist(), 'embedding': embedding[0].cpu().tolist()}
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + '\n', encoding='utf-8')
    print(payload)


if __name__ == '__main__':
    main(get_argument())
