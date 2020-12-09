# targeted-black-box-attack

Code of Targeted Black Box Adversarial Attack Method for Image Classification Models, published on IJCNN 2019. 

## 1. Files

>* MNIST.py: train a model on MNIST dataset. 

>* AdvMNIST.py: train an adversarial attack model on MNIST dataset. 

>* CIFAR10.py: train a model on CIFAR10 dataset. 

>* AdvCIFAR10.py: train an adversarial attack model on CIFAR10 dataset.  

>* CIFAR100.py: train a model on CIFAR100 dataset. 

>* AdvCIFAR100.py: train an adversarial attack model on CIFAR100 dataset.  

>* CIFAR10.py: train a model on CIFAR10 dataset. 

>* AdvCIFAR10.py: train a adversarial attack model on CIFAR10 dataset.  

>* AdvMNIST_sk.py: train an adversarial model on MNIST dataset to attack Naive Bayes, Decision Tree, and Random Forest et al. 

>* FashionMNIST.py: train a model on MNIST dataset. 

>* AdvFashionMNIST.py: train an adversarial attack model on MNIST dataset. 

## 2. Datasets

>* The datasets can be downloaded from https://1drv.ms/u/s!AgmtMXmpucYFgRbUnDWq7sM460St?e=QpwRpE

>* You can dump MNIST, FashionMNIST, CIFAR10, and CIFAR100 datasets into h5 files rather than download the files. (MNIST.h5, FashionMNIST.h5, CIFAR10.h5, CIFAR100.h5)

## 3. Test environment: 

>* CPU: Intel E5-2430 X 2, GPU: NVIDIA GTX1070Ti

## 4. Notes: 

>* You can change the net body to try different networks. 

>* SimpleV1C, SimpleV3, and SimpleV7 in the code correspond to SmallNet, SimpleNet, and ConcatNet in the paper respectively. 