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
> * C.-H. Hsia, L.-Y. Ke, and S.-T. Chen, “Improved lightweight convolutional neural network for finger vein recognition system,” _Bioengineering_, vol. 10, no. 8, pp. 919, 2023.
> * [[ILCNN]](https://github.com/liangying-Ke/ILCNN)
> * T.F. Zhang, et al., "Cas-vit: Convolutional additive self-attention vision transformers for efficient mobile applications," _ArXiv_ ,arXiv:2408.03703, 2024.
> * [[CAS-ViT]](https://github.com/Tianfang-Zhang/CAS-ViT)
> * Z. Yang, et al., "Comprehensive Competition Mechanism in Palmprint Recognition," _IEEE Transactions on Information Forensics and Security_, vol. 18, pp. 5160-5170, 2023.
>[[CCNet]](https://github.com/Zi-YuanYang/CCNet)

