import os
import time
import torch
import datasets
import argparse
import sklearn.metrics as skm 
import torch.nn.functional as F

from utils import *
from tqdm import tqdm
from torch.nn import init

from models.model import CLCNN
from models.loss import CombineLoss


def get_argument(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets',   type=str, default="FVUSM", choices=['FVUSM', 'PLUSVein'])
    parser.add_argument('--device',     type=str, default='cuda:0')
    parser.add_argument('--optim',      type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument('--scheduler',  type=str, default='Cosine', choices=['Cosine', 'ReduceLROnPlateau', ''])
    parser.add_argument('--workers',    type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size',   type=int, default=112)
    parser.add_argument('--epochs',     type=int, default=500)
    
    parser.add_argument('--arcface_weight', type=float, default=0.1)
    parser.add_argument('--center_weight',  type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--annot_file', type=str, default=None)
    parser.add_argument('--root_model', type=str, default=None)
    parser.add_argument('--data_root', type=str, default=None,
                        help='Optional base for relative paths inside the annotation.')
    parser.add_argument('--classes', type=int, default=None)
    parser.add_argument('--pad_height_width', type=int, default=None)
    parser.add_argument('--data_type', choices=['LED', 'LASER'], default=None,
                        help='Select one PLUSVein modality; default runs both.')
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

def getOptimParams(args):
    if args.optim == 'adamw':
        args.lr = 2e-4
        args.weight_decay = 1e-4
    if args.optim == 'sgd':
        args.lr = 1e-1
        args.momentum = 0.9
        args.weight_decay = 2e-4
        
    if args.scheduler == 'Cosine':
        args.T_max = 32
        args.eta_min = 1e-6
    if args.scheduler == 'ReduceLROnPlateau':
        args.factor = 0.9
        args.patience = 10
        args.verbose = True
    return args


def _get_optimizer(args, model):
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.scheduler == 'Cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.T_max, eta_min=args.eta_min)
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=args.factor, patience=args.patience)
    else:
        scheduler = None

    print(f'Using {args.optim} as optimizer, {args.scheduler} as scheduler')
    return optimizer, scheduler


def init_weights(net, init_type='normal', gain=1.0):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)
            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    print('Initialize network with %s' % init_type)
    net.apply(init_func)


def _get_model(args):
    model = CLCNN(num_classes=args.classes).to(args.device)
    print('Building model ...')
    init_weights(model, args.init_type)
    return model


def Learning(*kwargs):
    all_preds   = []
    all_targets = []
    losses = AverageMeter('Loss', ':.4e')

    args          = kwargs[0]
    model         = kwargs[1]
    DataLoader    = kwargs[2]
    criterion_cls = kwargs[3]
    optimizer     = kwargs[4]
    train_infos   = kwargs[-1]

    loop = tqdm(range(args.max_iteration_train if args.phase == 'train' else args.max_iteration_val),
                desc=f"Epoch [{args.epoch}/{args.epochs}]")

    model.train() if args.phase == "train" else model.eval()

    for _ in loop:
        inputs, targets = DataLoader.next()
        if inputs is None or targets is None:
            break
        with torch.set_grad_enabled(args.phase=="train"):
            outputs, features = model(inputs)
            loss = criterion_cls(outputs, targets, features)
            if args.phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        outputs = F.sigmoid(outputs)
        _, pred = torch.max(outputs, 1)

        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        losses.update(loss.item(), inputs.size(0))

    acc = skm.accuracy_score(all_targets, all_preds) * 100
    f1  = skm.f1_score(all_targets, all_preds, average='macro') * 100

    if args.phase == "train": 
        train_infos += f"\nTrain Loss     : {losses.avg:.3f}"
        train_infos += f"\nTrain Accuracy : {acc:.2f}"
        return args, model, train_infos
    
    val_infos  = f'Val Loss       : {losses.avg:.4f}'
    val_infos += f'\nVal Accuracy   : {acc:.2f}'
    val_infos += f'\nVal F1-Score   : {f1:.2f}'
    return losses.avg, acc, f1


def main(args):
    if args.batch_size < 2:
        raise ValueError("Training batch_size must be at least 2 for BatchNorm.")
    for data_type in args.data_type:
        train_dataset = datasets.ImagesDataset(args=args, data_type=data_type, phase='train')
        val_dataset   = datasets.ImagesDataset(args=args, data_type=data_type, phase='val')

        Train_DataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, persistent_workers=args.workers > 0, pin_memory=True, drop_last=True)
        Val_DataLoader   = torch.utils.data.DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=True, num_workers=args.workers, persistent_workers=args.workers > 0, pin_memory=True)
        
        args.max_iteration_train = len(Train_DataLoader)
        args.max_iteration_val   = len(Val_DataLoader)
        if not args.max_iteration_train:
            raise ValueError('Training set is smaller than batch_size; lower --batch_size (at least 2).')

        best_loss = float(' inf')
        best_acc  = float('-inf')
        best_f1   = float('-inf')

        model = _get_model(args)

        criterion = CombineLoss(args.classes, arcface_weight=args.arcface_weight, center_weight=args.center_weight).to(args.device)
        optimizer, scheduler = _get_optimizer(args, model)
        data_type_show = '' if data_type is None else data_type

        print(f'-------------------- Start Training with {args.datasets} : {data_type_show} --------------------')
        start_time = time.time()
        print(f'Start time : {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))}')
        loss_history = []

        for epoch in range(args.epochs):
            args.epoch = epoch + 1
            TrainDataLoaderPrefetcher = datasets.data_prefetcher(Train_DataLoader)
            ValDataLoaderPrefetcher   = datasets.data_prefetcher(Val_DataLoader)

            train_infos = f"Epoch: [{args.epoch:03d}/{args.epochs:03d}]"
            args.phase = 'train'
            args, model, train_infos = Learning(args, model, TrainDataLoaderPrefetcher, criterion, optimizer, train_infos)
            args.phase = 'val'
            loss, acc, f1 = Learning(args, model, ValDataLoaderPrefetcher, criterion, optimizer, train_infos)
            loss_history.append(loss)

            if args.scheduler == 'ReduceLROnPlateau':
                scheduler.step(loss)
            elif scheduler is not None:
                scheduler.step()

            is_bestLoss = loss < best_loss
            is_bestF1   = f1 > best_f1
            is_bestAcc  = acc > best_acc

            best_loss = min(loss, best_loss)
            best_f1   = max(f1, best_f1)
            best_acc  = max(acc, best_acc)

            save_path = os.path.join(args.root_model, str(data_type)) 
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            save_checkpoint(os.path.join(save_path, f"Backbone_ckpt.pth.tar"), {
                'args': vars(args).copy(),
                'preprocessing': {
                    'mean': train_dataset.mean.tolist(), 'std': train_dataset.std.tolist(),
                    'img_size': args.img_size, 'pad_height_width': args.pad_height_width,
                    'classes': args.classes, 'datasets': args.datasets, 'data_type': data_type,
                },
                'Accuracy': float(acc),
                'F1-Score': float(f1),
                'model_state_dict': model.state_dict(),
            }, is_bestF1, is_bestAcc, is_bestLoss)
            print('='*100)
            print(f'Best Loss     : {best_loss:.6f}')
            print(f'Best F1-Score : {best_f1:.4f}')
            print(f'Best Accuracy : {best_acc:.4f}')
            print('='*100)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours   = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        print(f'Using time: {hours:02d}:{minutes:02d}:{seconds:02d}')
        print(f'-------------------- End Training  --------------------')


if __name__ == '__main__':
    args = get_argument()
    args = getOptimParams(args)
    args = getDatasetParams(args)
    configure_cuda(args)
    seed_everything(args.seed)
    main(args)