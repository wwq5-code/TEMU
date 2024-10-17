#

# Towards Evaluating Machine Unlearning
## Overview
This repository is the official implementation of TEMU, and the corresponding paper is under review.


## Prerequisites

```
python = 3.10.10
torch==2.0.0
torchvision==0.15.1
matplotlib==3.7.1
numpy==1.23.5
```

We also show the requirements packages in requirements.txt

Here, we demonstrate the overall evaluations, which are also the main achievement claimed in the paper. We will explain the results and demonstrate how to achieve these results using the script and corresponding parameters.

Evaluated on NVIDIA Quadro RTX 6000 GPUs,
### TABLE I: General Evaluation Results on MNIST, CIFAR10, and CelebA:

On MNIST, ESS=1

| On MNIST             | MIB         | EMU (CT) | EMU (AT)   | 
| --------             | --------    | --------  | -------- |  
| Model Utility (Acc.) | 98.31%      | 99.23%    | 99.31%   |   
| Rec. Sim.            | -           | 0.441     |   0.967  |  
| Verifiability        | 0.00%       | 87.93%    | 99.33%   | 
| Running time (s)     | 639         | 135       |  145     |   
 
On CIFAR10, ESS=1

| On CIFAR10           | MIB         | EMU (CT) |   EMU (AT)   | 
| --------             | --------    | --------  | -------- |   
| Model Utility (Acc.) | 79.45%      | 81.34%    | 81.44%   |   
| Rec. Sim.            | -           | 0.895     | 0.975    |  
| Verifiability        | 0.00%       | 0.00%     | 95.64%   |  
| Running time (s)     | 673         | 136       |  137     | 

In this table, we can achieve these metric values by running corresponding python files.


1. To run the EMU on MNIST, we can run
```
python /MUV_Reconstruciton/On_MNIST/Our_method/MUV_on_MNIST_unl_multi.py
```

2. To run the EMU on CIFAR10, we can run

```
python /MUV_Reconstruciton/On_CIFAR10/Our_method/MUV_on_CIFAR10_unl_multi.py
```


3. To run the EMU on CelebA, we can run

```
python /MUV_Reconstruciton/On_CelebA/Our_method/MUV_on_CelebA_unl_multi.py
```
Note that, to sucessfully run the program on CelebA, we need first prepare the CelebA dataset, which can be downloaded from: 
(https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg)