# Competitive Lightweight Convolutional Neural Network for Finger Vein Recognition 

## Environments  
```
conda env create -f environment.yaml
```
```
conda activate fingervein
```

## 1. Prepare dataset for Training and Testing
Get a .pkl file contain data's path 
```
python 000_make_Dataset.py
```  

## 2. Training
Start training the model 
```
python 001_train.py
```

## 4. Testing 
Start evaluating the model
```
python 002_test.py
```

## 5. See model's structure and details
```
python model_detail.py
```
## Experiment Results  
```
==================================
FV-USM
Accuracy               : 99.93
EER (Equal Error Rate) :  0.07
----------------------------------
PLUSVein-FV3 [LED]
Accuracy               : 99.17
EER (Equal Error Rate) :  1.19
----------------------------------
PLUSVein-FV3 [Laser]
Accuracy               : 98.92
EER (Equal Error Rate) :  1.09
==================================
```  

## 6. Hardware
The model architectures proposed in this study are implemented using the PyTorchDL framework, and training is conducted on hardware featuring an Intel® Core™ i7-12700 CPU and Nvidia RTX 4080 graphics processing unit (GPU).

# References： 
> * 柯良頴, 夏至賢, 許良亦, 陳麒安, "Virtual Try-on Based on Composable Sequential Appearance Flow," ITAC, 2024.
> [[Virtual Try-on Based on Composable Sequential Appearance Flow]](https://github.com/Anguschen1011/VirtualTryOn-VITON-V1)
> * P. Li, Y. Xu, Y. Wei and Y. Yang, "Self-Correction for Human Parsing," _IEEE Transactions on Pattern Analysis and Machine Intelligence_, vol. 44, no. 6, pp. 3260-3271, 2022.
>[[Self Correction Human Parsing]](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing)  
> * N. Ravi, et al., "Sam 2: Segment anything in images and videos," _ArXiv_, 2024.
>[[StyleGAN-Human]](https://github.com/facebookresearch/segment-anything-2)

