"""
Author: Chi-An Chen
Date: 2025-04-01
Version: 8.0
"""
import torch
import models 
import argparse

from utils import *
from thop import profile
from pytorch_model_summary import summary
from ptflops import get_model_complexity_info

def get_argument(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default="PLUSVein", choices=["FVUSM", "PLUSVein"])
    parser.add_argument('--img_size', type=int, default=112)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--classes', type=int, default=None)
    return parser.parse_args(argv)

def getDatasetParams(args):
    classes = args.classes
    if args.datasets == 'FVUSM':
        args.classes = 492
        args.pad_height_width = 300
        args.data_type = [None]
        args.root_model = './checkpoint/FV-USM'
        args.annot_file = './datasets/annotations_fvusm.pkl'
    elif args.datasets == 'PLUSVein':
        args.classes = 360
        args.pad_height_width = 736
        args.data_type = ['LED', 'LASER']
        args.root_model = './checkpoint/PLUSV-FV3'
        args.annot_file = './datasets/annotations_plusvein.pkl'
    if classes is not None:
        args.classes = classes
    return args

def _get_model(args):
    model = models.CLCNN(num_classes=args.classes).to(args.device)
    return model

def main(args):
    model = _get_model(args)
    model.eval()
    # print(model)
    
    print(f'Model: CLCNN (LViT-CB), classes={args.classes}, input=3x{args.img_size}x{args.img_size}')
    print('Model Summary:')
    test_input = torch.zeros([1, 3, args.img_size, args.img_size], device=args.device)
    print(summary(model, test_input, show_input=False))
    
    macs_thop, params = profile(model, inputs=(test_input,))
    print(f"THOP operations (MACs convention): {macs_thop / 1e9:.4f} G")
    print(f"THOP counted params: {params / 1e6:.4f} M")
    print(f"All registered params: {sum(p.numel() for p in model.parameters()) / 1e6:.4f} M")
    
    macs, _ = get_model_complexity_info(model, (3, args.img_size, args.img_size), as_strings=True, print_per_layer_stat=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))


if __name__ == '__main__':
    database_results = {}
    args = get_argument()
    args = getDatasetParams(args)
    configure_cuda(args)
    main(args)