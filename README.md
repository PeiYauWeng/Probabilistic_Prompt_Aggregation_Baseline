## Implemented Algorithms

1. FedAvg
2. FedProx
3. FedOPT
4. Scaffold
5. FedProx-GMMclustering
6. pFedPG

## Installation

```
git clone https://github.com/PeiYauWeng/Probabilistic_Prompt_Aggregation_Baseline
cd PersonalizedFL
```
Install required package
```
pip3 install -r requirements.txt
```

## Dataset
Support the following dataset:
* CIFAR10
* CIFAR100

## Usage

1. Modify .sh file if you need
2. `bash run_cifar10.sh`
3. You can assign which GPU to run.<br>`bash run_fedavg_cifar100.sh n`  In which, n means n th GPU.<br>For example, if GPU:1 is available, then<br>`bash run_fedavg_cifar100.sh 1`