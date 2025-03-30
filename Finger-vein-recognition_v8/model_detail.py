import warnings
warnings.filterwarnings("ignore")

import torch
import models 
import argparse

from utils import *
from thop import profile
import torch.nn.functional as F
from pytorch_model_summary import summary
from ptflops import get_model_complexity_info

def get_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default="PLUSVein")
    parser.add_argument('--img_size', type=int, default=112)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    return parser.parse_args([])

def getDatasetParams(args):
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
    return args

def _get_model(args):
    model = models.LightWeightedModel(num_classes=args.classes).to(args.device)
    return model

def main(args):
    model = _get_model(args)
    model.eval()
    # print(model)
    input1 = torch.zeros([1, 3, args.img_size, args.img_size], device=args.device)
    print(summary(model, input1, show_input=False))
    flops, params = profile(model, inputs=(input1,))
    print(f"FLOPs: {flops / 1e9:.4f} GFLOPs")
    print(f"Params: {params / 1e6:.4f} M")
    
    macs, _ = get_model_complexity_info(model, (3, args.img_size, args.img_size), as_strings=True, print_per_layer_stat=False)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))


if __name__ == '__main__':
    database_results = {}
    args = get_argument()
    args = getDatasetParams(args)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args)