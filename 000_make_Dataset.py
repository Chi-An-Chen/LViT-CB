import os
import glob
import random
import pickle
import argparse
import traceback
from pathlib import Path
from tqdm import tqdm

def get_argument(argv=None):
	parser = argparse.ArgumentParser()
	parser.add_argument('--FVUSM_split',              type=int, default=6)
	parser.add_argument('--PLUSVein_split',           type=int, default=5)
	parser.add_argument('--FVUSM_root',               type=str, default=r'./data/FV-USM')
	parser.add_argument('--FVUSM_annotation_file',    type=str, default=r'./datasets/annotations_fvusm.pkl')
	parser.add_argument('--PLUSVein_root',            type=str, default=r'./data/PLUSVein-FV3/PLUSVein-FV3-ROI_combined/ROI')
	parser.add_argument('--PLUSVein_annotation_file', type=str, default=r'./datasets/annotations_plusvein.pkl')
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--datasets', choices=['FVUSM', 'PLUSVein', 'all'], default='all')
	return parser.parse_args(argv)


def create_FVUSM_annotation(args):
	if args.FVUSM_split < 3:
		raise ValueError('FVUSM_split must be at least 3.')
	random.seed(args.seed)
	Path(args.FVUSM_annotation_file).parent.mkdir(parents=True, exist_ok=True)
	def iter(root, trainingSamples=[], validatingSamples=[], testingSamples=[], sub2classes={}):
		for sub in tqdm(os.listdir(root)):
			if not os.path.isdir(os.path.join(root, sub)):
				continue
			paths = glob.glob(os.path.join(root, sub, '*.jpg'))
			random.shuffle(paths)
			breakpoints = len(paths) // args.FVUSM_split
			if breakpoints == 0:
				raise ValueError(f"Not enough .jpg images for splitting: {root}/{sub}")
			forTest     = paths[:breakpoints]
			forValidate = paths[breakpoints:breakpoints*2]
			if sub not in sub2classes:
				sub2classes[sub] = len(sub2classes)
		
			for path in paths:
				if path in forValidate:
					validatingSamples.append({'path':path, 'label':sub2classes[sub]})
				elif path in forTest:
					testingSamples.append({'path':path, 'label':sub2classes[sub]})
				else:
					trainingSamples.append({'path':path, 'label':sub2classes[sub]})
		return trainingSamples, validatingSamples, testingSamples, sub2classes
	
	trainingSamples, validatingSamples, testingSamples, sub2classes = iter(os.path.join(args.FVUSM_root, '1st_session', 'extractedvein'))
	trainingSamples, validatingSamples, testingSamples, sub2classes = iter(os.path.join(args.FVUSM_root, '2nd_session', 'extractedvein'), trainingSamples, validatingSamples, testingSamples, sub2classes)
	if not trainingSamples or not validatingSamples or not testingSamples:
		raise ValueError("No complete FVUSM splits found; check the input directory layout.")
	pickle.dump({
		'Training_Set'  :trainingSamples, 
		'Validating_Set':validatingSamples, 
		'Testing_Set'   :testingSamples,
	}, open(args.FVUSM_annotation_file, 'wb'))

def create_PLUSVein_annotation(args):
	if args.PLUSVein_split < 3:
		raise ValueError('PLUSVein_split must be at least 3.')
	random.seed(args.seed)
	Path(args.PLUSVein_annotation_file).parent.mkdir(parents=True, exist_ok=True)
	def iter(root, Where):
		sub2classes = {}
		trainingSamples, validatingSamples, testingSamples = [], [], []
		data_path = os.path.join(root, Where)

		for folder in tqdm(os.listdir(data_path)):
			if not os.path.isdir(os.path.join(data_path, folder)):
				continue
			paths = glob.glob(os.path.join(data_path, folder, '*.png'))
			for idx in ['02', '03', '04', '07', '08', '09']:
				identity = f'{folder}_{idx}'
				filter_paths = [path for path in paths if identity in path]
				random.shuffle(filter_paths)
				breakpoints = len(filter_paths) // args.PLUSVein_split
				if breakpoints == 0:
					raise ValueError(f"Not enough .png images for identity {identity} in {data_path}")
				forTest = filter_paths[:breakpoints]
				forValidate = filter_paths[breakpoints:breakpoints*2]

				if not identity in sub2classes:
					sub2classes[identity] = len(sub2classes)
			
				for path in filter_paths:
					if path in forValidate:
						validatingSamples.append({'path':path, 'label':sub2classes[identity]})
					elif path in forTest:
						testingSamples.append({'path':path, 'label':sub2classes[identity]})
					else:
						trainingSamples.append({'path':path, 'label':sub2classes[identity]})
		return trainingSamples, validatingSamples, testingSamples
	
	trainingSamples_LED, validatingSamples_LED, testingSamples_LED = iter(
		args.PLUSVein_root, os.path.join('PLUS-FV3-LED', 'PALMAR', '01'))
	trainingSamples_LASER, validatingSamples_LASER, testingSamples_LASER = iter(
		args.PLUSVein_root, os.path.join('PLUS-FV3-Laser', 'PALMAR', '01'))
	if not all((trainingSamples_LED, validatingSamples_LED, testingSamples_LED, trainingSamples_LASER, validatingSamples_LASER, testingSamples_LASER)):
		raise ValueError('No complete PLUSVein splits found; check ROI paths and filename conventions.')
	pickle.dump({
		'LED':{
			'Training_Set'  :trainingSamples_LED, 
			'Validating_Set':validatingSamples_LED, 
			'Testing_Set'   :testingSamples_LED,
		},
		'LASER':{
			'Training_Set'  :trainingSamples_LASER, 
			'Validating_Set':validatingSamples_LASER, 
			'Testing_Set'   :testingSamples_LASER,
		}
	}, open(args.PLUSVein_annotation_file, 'wb'))


if __name__ == '__main__':
    args = get_argument()
    failed = False
    for name, prepare in [('FVUSM', create_FVUSM_annotation), ('PLUSVein', create_PLUSVein_annotation)]:
        if args.datasets not in ('all', name):
            continue
        print(f'Create {name} Annotation ...')
        try:
            prepare(args)
            print('Done!')
        except Exception:
            traceback.print_exc()
            failed = True
    raise SystemExit(1 if failed else 0)
