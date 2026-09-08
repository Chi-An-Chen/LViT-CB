import os
import torch
import random
import datasets
import argparse
import numpy as np
import sklearn.metrics as skm
import torch.nn.functional as F

from utils import *
from tqdm import tqdm
from tqdm.contrib import tzip
from sklearn.model_selection import KFold

from models.model import CLCNN

def get_argument(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets',   type=str, default="FVUSM", choices=['FVUSM', 'PLUSVein'])
    parser.add_argument('--device',     type=str, default='cuda:0')
    parser.add_argument('--img_size',   type=int, default=112)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers',    type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--annot_file', type=str, default=None)
    parser.add_argument('--root_model', type=str, default=None)
    parser.add_argument('--data_root', type=str, default=None,
                        help='Optional base for relative paths inside the annotation.')
    parser.add_argument('--classes', type=int, default=None)
    parser.add_argument('--pad_height_width', type=int, default=None)
    parser.add_argument('--data_type', choices=['LED', 'LASER'], default=None,
                        help='Select one PLUSVein modality; default runs both.')
    parser.add_argument('--checkpoint', type=str, default=None, help='Evaluate one checkpoint; default compares bestF1/bestAcc/bestLoss.')
    parser.add_argument('--trusted_checkpoint', action='store_true', help='Allow pickle loading of your trusted legacy checkpoint.')
    return parser.parse_args(argv)

def getDatasetParams(args):
    overrides = {key: getattr(args, key) for key in ('classes', 'pad_height_width', 'root_model', 'annot_file')}
    modality = args.data_type
    if args.datasets == 'FVUSM':
        args.classes = 492
        args.pad_height_width = 300
        args.data_type  = [None]
        args.init_type  = 'xavier_normal'
        args.root_model = './checkpoint_woComp/FV-USM'
        args.annot_file = './datasets/annotations_fvusm.pkl'
    
    elif args.datasets == 'PLUSVein':
        args.classes = 360
        args.pad_height_width = 736
        args.data_type  = ['LED', 'LASER']
        args.init_type  = 'xavier_uniform'
        args.root_model = './checkpoint_woComp/PLUSV-FV3'
        args.annot_file = './datasets/annotations_plusvein.pkl'
    for key, value in overrides.items():
        if value is not None:
            setattr(args, key, value)
    if modality is not None:
        if args.datasets != 'PLUSVein':
            raise ValueError('--data_type is only valid for PLUSVein.')
        args.data_type = [modality]
    return args

def _get_model(args):
    model = CLCNN(num_classes=args.classes).to(args.device)
    return model

def calculate_metrics(distances, labels, threshold, dist_type='cosine'):
    if dist_type == 'cosine':
        preds = np.greater(distances, threshold)
    elif dist_type == 'euclidean':
        preds = np.less(distances, threshold)
    else:
        raise ValueError('dist_type must be cosine or euclidean.')
    tn, fp, fn, tp = skm.confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    if tn + fp == 0 or tp + fn == 0:
        raise ValueError('Each evaluation fold needs both genuine and impostor pairs.')
    return (float(tp + tn) / distances.size * 100,
            float(fp) / (tn + fp) * 100,
            float(fn) / (tp + fn) * 100)


def calculate_average_metrics(dists, labels, num_folds=5, seed=42, dist_type='cosine'):
    dists, labels = np.asarray(dists), np.asarray(labels)
    if dists.ndim != 1 or labels.shape != dists.shape or len(dists) < num_folds:
        raise ValueError('Expected matching nonempty 1-D scores and labels, with at least num_folds pairs.')
    if not np.all(np.isfinite(dists)) or set(np.unique(labels)) != {0, 1}:
        raise ValueError('Scores must be finite and labels must contain both 0 and 1.')
    # Keep the original 0.01 grid; include both acceptance endpoints.
    upper = 1.0 if dist_type == 'cosine' else float(np.ceil(np.max(dists)))
    thresholds = np.arange(-0.01, upper + 0.02, 0.01)
    eer_list, acc_list = [], []
    folds = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    for train_set, test_set in folds.split(labels):
        calibration = [calculate_metrics(dists[train_set], labels[train_set], t, dist_type)
                       for t in thresholds]
        best_threshold = thresholds[np.argmax([m[0] for m in calibration])]
        acc, _, _ = calculate_metrics(dists[test_set], labels[test_set], best_threshold, dist_type)
        # EER describes the held-out fold ROC, not a deployment threshold.
        held_out = np.array([calculate_metrics(dists[test_set], labels[test_set], t, dist_type)
                             for t in thresholds])
        eer_idx = np.argmin(np.abs(held_out[:, 1] - held_out[:, 2]))
        eer_list.append(float((held_out[eer_idx, 1] + held_out[eer_idx, 2]) / 2))
        acc_list.append(acc)
    return np.mean(acc_list), np.mean(eer_list)


def evaluate(*kwargs):
    args = kwargs[0]
    model = kwargs[1]
    test_DataLoader = kwargs[2]
    valid_DataLoader = kwargs[3]
    dist_type = kwargs[4]
    
    dists = []
    labels = []
    embeds_list = []
    targets_list = []
    outputs_list = []
    model.eval()

    for inputs, targets in tqdm(test_DataLoader):
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)
        with torch.set_grad_enabled(False):
            outputs, feature = model(inputs)
            embeds = F.normalize(feature, dim=1)
        
        outputs = F.sigmoid(outputs)
        _, outputs = torch.max(outputs, 1)
        outputs_list.extend(outputs.cpu().detach().numpy())
        embeds_list.extend(embeds.cpu().detach().numpy())
        targets_list.extend(targets.cpu().detach().numpy())
    
    for inputs, targets in tqdm(valid_DataLoader):
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)
        with torch.set_grad_enabled(False):
            outputs, feature = model(inputs)
            embeds = F.normalize(feature, dim=1)
        
        outputs = F.sigmoid(outputs)
        _, outputs = torch.max(outputs, 1)
        outputs_list.extend(outputs.cpu().detach().numpy())
        embeds_list.extend(embeds.cpu().detach().numpy())
        targets_list.extend(targets.cpu().detach().numpy())

    outputs_list = np.array(outputs_list)
    embeds_list = np.array(embeds_list)
    targets_list = np.array(targets_list)

    for i, (embed_A, target_A) in enumerate(tzip(embeds_list, targets_list)):
        for embed_B, target_B in zip(embeds_list[i + 1:], targets_list[i + 1:]):
            if dist_type == 'cosine':
                dist = np.dot(embed_A, embed_B) / (np.linalg.norm(embed_A) * np.linalg.norm(embed_B))
                dist = np.clip((dist + 1) / 2, 0.0, 1.0)
            elif dist_type == 'euclidean':
                dist = np.sum((embed_A - embed_B) ** 2) ** 0.5

            label = int(target_A == target_B)
            dists.append(dist)
            labels.append(label)
    dists = np.array(dists, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    dists_1 = dists[labels == 1]
    dists_2 = dists[labels == 0]
    if not len(dists_1) or len(dists_2) < len(dists_1):
        raise ValueError('Need genuine pairs and at least as many impostor pairs. Provide multiple classes with multiple evaluation images.')
    rng = np.random.default_rng(args.seed)
    dists_2 = rng.choice(dists_2, size=len(dists_1), replace=False)
    dists = np.hstack([dists_1, dists_2])
    labels = np.hstack([np.ones(len(dists_1)), np.zeros(len(dists_2))])
    avg_acc, eer = calculate_average_metrics(dists, labels, seed=args.seed, dist_type=dist_type)
    acc = skm.accuracy_score(targets_list, outputs_list) * 100
    return eer, acc, avg_acc

def main(args, database_results=None):
    if database_results is None:
        database_results = {}
    if args.checkpoint and len(args.data_type) != 1:
        raise ValueError("Select --data_type LED or LASER when using --checkpoint with PLUSVein.")
    print(f'Using {args.datasets} dataset for testing ...')
    for data_type in args.data_type:
        print(f'Testing data type : {data_type} ...')
        path = os.path.join(args.root_model, str(data_type))
        candidates = [('provided', args.checkpoint)] if args.checkpoint else [
            (metric, os.path.join(path, f'Backbone_ckpt.best{metric}.pth.tar'))
            for metric in ('F1', 'Acc', 'Loss')]
        for _, filename in candidates:
            if not os.path.isfile(filename):
                raise FileNotFoundError(f'Checkpoint not found: {filename}. Train first or use --checkpoint.')
        test_dataset = datasets.ImagesDataset(args=args, data_type=data_type, phase='test')
        test_DataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, persistent_workers=args.workers > 0, pin_memory=True)
        valid_dataset = datasets.ImagesDataset(args=args, data_type=data_type, phase='val')
        valid_DataLoader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, persistent_workers=args.workers > 0, pin_memory=True)
        
        model = _get_model(args)
        model.eval()

        best_eer = float('inf')
        for metrics, filename in candidates:
            weights = load_checkpoint(filename, args.device, args.trusted_checkpoint)
            model.load_state_dict(weights['model_state_dict'])

            dist_type='cosine'
            eer, acc, avg_acc = evaluate(args, model, test_DataLoader, valid_DataLoader, dist_type)

            is_best = best_eer > eer
            best_eer = min(best_eer, eer)
            if is_best:
                database_results[f'{args.datasets}_{data_type}'] = {
                    'acc':f'{acc:.4f}',
                    'avg_acc':f'{avg_acc:.4f}',
                    'eer':f'{eer:.4f}'
                }
            print(f'Database: {args.datasets}, data_type: {data_type}, Metrics: {metrics}')
            print(f'Accuracy: {acc:.4f}')
            print(f'Avg Accuracy: {avg_acc:.4f}')
            print(f'EER: {eer:.4f}')
            print('-'*100)
        print('='*100)
    
    return database_results

if __name__ == '__main__':
    database_results = {}
    args = get_argument()
    args = getDatasetParams(args)
    configure_cuda(args)
    seed_everything(args.seed)
    database_results = main(args)
    print(database_results)