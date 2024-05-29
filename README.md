


 <img src="https://github.com/ifiaposto/Distance_Aware_Bottleneck/assets/11561732/250606f3-542e-4167-8bd0-3860520d4c23" alt="dab" width="1000px" height="300px">

# Distance Aware Bottleneck

## Install

### Cuda Envirnoment

Below, we provide the exact cuda environment for our tensorflow version:

```
   nvidia-cublas-cu11==11.11.3.6
   nvidia-cuda-cupti-cu11==11.8.87
   nvidia-cuda-nvcc-cu11==11.8.89
   nvidia-cuda-runtime-cu11==11.8.89
   nvidia-cudnn-cu11==8.6.0.163
   nvidia-cufft-cu11==10.9.0.58
   nvidia-curand-cu11==10.3.0.86
   nvidia-cusolver-cu11==11.4.1.48
   nvidia-cusparse-cu11==11.7.5.86
```


### Dependecies

DAB is built in Python 3.9.18  using Tensorflow 2.12.0. The cuda version is 11.8. 
Use the following command to install the requirements:
```
pip install -r requirements.txt
``` 

## Datasets
All datasets are placed in ~/tensorflow_datasets

* Download from [Tensorflow Datasets](https://www.tensorflow.org/datasets) for cifar10, cifar100, SVHN.
* Use [cifar10_corrupted](https://www.tensorflow.org/datasets/catalog/cifar10_corrupted) for the cifar10 noise corruptions.
* For preparing ImageNet for use with tensorflow_datasets, these [instructions](https://github.com/leondgarse/keras_cv_attention_models/discussions/9) might be useful.
* For ImageNetO, please download the serialized dataset from [here](https://drive.google.com/file/d/1D3lfSmd4cv7cSqw1Kj65Dn6jQuBccRJb/view?usp=sharing), and place it in the ~/tensorflow_datasets directory.

## Train DAB

### Synthetic Regression

 <details><summary>Figure 3a in the paper.   </summary>
 
```
python synthetic_regression.py --example=1 --codebook_size=1
```  
</details>

 <details><summary>Figure 3b in the paper.   </summary>

```
python synthetic_regression.py --example=2 --codebook_size=2
```  
</details>

### Cifar10

### ImageNet-1K

## Evaluate DAB

### Trained Models

You can download the trained DAB models presented in the paper [here](https://drive.google.com/file/d/1Ql1pJV3xFgIgLabqWegeNW74WCdwpmNL/view?usp=drive_link).

### Commands

#### Cifar10

#### ImageNet-1K

## Citation

## Contact

For any questions or problems using the library, please don't hesitate to contact me at ifiaposto@gmail.com.


