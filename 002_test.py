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

def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets',   type=str, default="FVUSM", choices=['FVUSM', 'PLUSVein'])
    parser.add_argument('--device',     type=str, default='cuda:0')
    parser.add_argument('--img_size',   type=int, default=112)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers',    type=int, default=2)
    return parser.parse_args([])

def getDatasetParams(args):
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
    return args

def _get_model(args):
    model = CLCNN(num_classes=args.classes).to(args.device)
    return model

def calculate_metrics(distances, labels, threshold, dist_type='cosine'):
    if dist_type == 'cosine':
        preds = np.greater(distances, threshold)
    elif dist_type == 'euclidean':
        preds = np.less(distances, threshold)

    tn, fp, fn, tp = skm.confusion_matrix(labels, preds).ravel()
    fpr = float(fp) / (tn + fp) * 100
    fnr = float(fn) / (tp + fn) * 100
    acc = float(tp + tn) / distances.size * 100
    return acc, fpr, fnr

def calculate_average_metrics(dists, labels, num_folds=5):
    dist_min, dist_max = np.min(dists), np.max(dists)
    thresholds = np.arange(0, np.ceil(dist_max), 0.01)
    print(f'Distance Min: {dist_min} Max: {dist_max}')
    eer_list = []
    acc_list = []
    folds = KFold(n_splits=num_folds, shuffle=True)
    for train_set, test_set in folds.split(labels):
        _acc_fold = []
        _fpr_fold = []
        _fnr_fold = []
        for threshold in thresholds:
            acc, fpr, fnr = calculate_metrics(dists[train_set], labels[train_set], threshold)
            _acc_fold.append(acc)
            _fpr_fold.append(fpr)
            _fnr_fold.append(fnr)
        eer_idx = np.nanargmin(np.absolute((np.array(_fnr_fold) - np.array(_fpr_fold))))
        eer = (_fpr_fold[eer_idx] + _fnr_fold[eer_idx]) / 2

        best_threshold = thresholds[np.argmax(_acc_fold)]
        acc, fpr, fnr = calculate_metrics(dists[test_set], labels[test_set], best_threshold)

        eer_list.append(eer)
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

    for embed_A, target_A in tzip(embeds_list, targets_list):
        for embed_B, target_B in zip(embeds_list, targets_list):
            if dist_type == 'cosine':
                dist = np.dot(embed_A, embed_B) / (np.linalg.norm(embed_A) * np.linalg.norm(embed_B))
                dist = (dist + 1) / 2
            elif dist_type == 'euclidean':
                dist = np.sum((embed_A - embed_B) ** 2) ** 0.5

            label = int(target_A == target_B)
            dists.append(dist)
            labels.append(label)
    dists = np.array(dists, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    dists_1 = dists[labels==1]
    dists_2 = dists[labels==0]
    random.shuffle(dists_2)
    dists_2 = dists_2[:len(dists_1)]
    dists = np.hstack([dists_1, dists_2])

    labels_1 = labels[labels==1]
    labels_2 = labels[labels==0]
    random.shuffle(labels_2)
    labels_2 = labels_2[:len(labels_1)]
    labels = np.hstack([labels_1, labels_2])

    avg_acc, eer = calculate_average_metrics(dists, labels)
    acc = skm.accuracy_score(targets_list, outputs_list) * 100
    return eer, acc, avg_acc

def main(args, database_results={}):
    print(f'Using {args.datasets} dataset for testing ...')
    for data_type in args.data_type:
        print(f'Testing data type : {data_type} ...')
        test_dataset = datasets.ImagesDataset(args=args, data_type=data_type, phase='test')
        test_DataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, persistent_workers=True, pin_memory=True)
        valid_dataset = datasets.ImagesDataset(args=args, data_type=data_type, phase='val')
        valid_DataLoader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, persistent_workers=True, pin_memory=True)
        
        model = _get_model(args)
        model.eval()

        best_eer = float('inf')
        for metrics in ['F1', 'Acc', 'Loss']:
            path = os.path.join(args.root_model, str(data_type)) 
            weights = torch.load(os.path.join(path, f"Backbone_ckpt.best{metrics}.pth.tar"), weights_only=False)
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
    database_results = main(args)
    print(database_results)