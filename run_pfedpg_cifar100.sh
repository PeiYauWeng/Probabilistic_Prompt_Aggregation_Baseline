python3 -u main.py --alg pfedpg --dataset cifar100 --device cuda:$1 --comms 400 --alpha 0.5
python3 -u main.py --alg pfedpg --dataset cifar100 --device cuda:$1 --comms 400 --alpha 0.4
python3 -u main.py --alg pfedpg --dataset cifar100 --device cuda:$1 --comms 400 --alpha 0.3
python3 -u main.py --alg pfedpg --dataset cifar100 --device cuda:$1 --comms 400 --alpha 0.2
python3 -u main.py --alg pfedpg --dataset cifar100 --device cuda:$1 --comms 400 --alpha 0.1
python3 -u main.py --alg pfedpg --dataset cifar100 --device cuda:$1 --comms 400 --data_distribution manual_extreme_heterogeneity --n_dominated_class 10