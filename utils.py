import torch
import shutil

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(filename, state, is_bestF1, is_bestAcc, is_bestLoss):
    torch.save(state, filename)

    if is_bestF1:
        shutil.copyfile(filename, filename.replace('pth.tar', 'bestF1.pth.tar'))
        print('Best F1 score model saved !')
    if is_bestAcc:
        shutil.copyfile(filename, filename.replace('pth.tar', 'bestAcc.pth.tar'))
        print('Best Accuracy model saved !')
    if is_bestLoss:
        shutil.copyfile(filename, filename.replace('pth.tar', 'bestLoss.pth.tar'))
        print('Best Loss model saved !')

def configure_cuda(args):
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('These experiments require an NVIDIA GPU and CUDA-enabled PyTorch. See README installation instructions.')
    torch.cuda.set_device(device)
    for key in ('classes', 'img_size', 'pad_height_width', 'batch_size', 'epochs'):
        value = getattr(args, key, None)
        if value is not None and value <= 0:
            raise ValueError(f'{key} must be positive.')
    if getattr(args, 'workers', 0) < 0:
        raise ValueError('workers must be nonnegative.')
    return device


def seed_everything(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint(path, device, trusted=False):
    from pathlib import Path
    import pickle
    if not Path(path).is_file():
        raise FileNotFoundError(f'Checkpoint not found: {path}. Train first or supply your own checkpoint; no pretrained weights are bundled.')
    try:
        return torch.load(path, map_location=device, weights_only=not trusted)
    except pickle.UnpicklingError as exc:
        if trusted:
            raise
        raise ValueError('This checkpoint is not weights-only compatible. Use --trusted_checkpoint only for a trusted legacy file you created or verified.') from exc
