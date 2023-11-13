#Dirichlet 0.5
python3 main.py --alg fedavg --dataset cifar100 --comms 100 --alpha 0.5
python3 main.py --alg fedprox --dataset cifar100 --comms 100 --alpha 0.5
python3 main.py --alg fedopt --dataset cifar100 --comms 120 --alpha 0.5
python3 main.py --alg scaffold --dataset cifar100 --comms 100 --alpha 0.5
python3 main.py --alg fedprox_gmm --dataset cifar100 --comms 100 --alpha 0.5
python3 main.py --alg pfedpg --dataset cifar100 --comms 400 --alpha 0.5
#Dirichlet 0.4
python3 main.py --alg fedavg --dataset cifar100 --comms 100 --alpha 0.4
python3 main.py --alg fedprox --dataset cifar100 --comms 100 --alpha 0.4
python3 main.py --alg fedopt --dataset cifar100 --comms 120 --alpha 0.4
python3 main.py --alg scaffold --dataset cifar100 --comms 100 --alpha 0.4
python3 main.py --alg fedprox_gmm --dataset cifar100 --comms 100 --alpha 0.4
python3 main.py --alg pfedpg --dataset cifar100 --comms 400 --alpha 0.4
#Dirichlet 0.3
python3 main.py --alg fedavg --dataset cifar100 --comms 100 --alpha 0.3
python3 main.py --alg fedprox --dataset cifar100 --comms 100 --alpha 0.3
python3 main.py --alg fedopt --dataset cifar100 --comms 120 --alpha 0.3
python3 main.py --alg scaffold --dataset cifar100 --comms 100 --alpha 0.3
python3 main.py --alg fedprox_gmm --dataset cifar100 --comms 100 --alpha 0.3
python3 main.py --alg pfedpg --dataset cifar100 --comms 400 --alpha 0.3
#Dirichlet 0.2
python3 main.py --alg fedavg --dataset cifar100 --comms 100 --alpha 0.2
python3 main.py --alg fedprox --dataset cifar100 --comms 100 --alpha 0.2
python3 main.py --alg fedopt --dataset cifar100 --comms 120 --alpha 0.2
python3 main.py --alg scaffold --dataset cifar100 --comms 100 --alpha 0.2
python3 main.py --alg fedprox_gmm --dataset cifar100 --comms 100 --alpha 0.2
python3 main.py --alg pfedpg --dataset cifar100 --comms 400 --alpha 0.2
#Dirichlet 0.5
python3 main.py --alg fedavg --dataset cifar100 --comms 100 --alpha 0.1
python3 main.py --alg fedprox --dataset cifar100 --comms 100 --alpha 0.1
python3 main.py --alg fedopt --dataset cifar100 --comms 120 --alpha 0.1
python3 main.py --alg scaffold --dataset cifar100 --comms 100 --alpha 0.1
python3 main.py --alg fedprox_gmm --dataset cifar100 --comms 100 --alpha 0.1
python3 main.py --alg pfedpg --dataset cifar100 --comms 400 --alpha 0.1
#Manual extreme heterogeneity
python3 main.py --alg fedavg --dataset cifar100 --comms 100 --data_distribution manual_extreme_heterogeneity --n_dominated_class 10
python3 main.py --alg fedprox --dataset cifar100 --comms 100 --data_distribution manual_extreme_heterogeneity --n_dominated_class 10
python3 main.py --alg fedopt --dataset cifar100 --comms 120 --data_distribution manual_extreme_heterogeneity --n_dominated_class 10
python3 main.py --alg scaffold --dataset cifar100 --comms 100 --data_distribution manual_extreme_heterogeneity --n_dominated_class 10
python3 main.py --alg fedprox_gmm --dataset cifar100 --comms 100 --data_distribution manual_extreme_heterogeneity --n_dominated_class 10
python3 main.py --alg pfedpg --dataset cifar100 --comms 400 --data_distribution manual_extreme_heterogeneity --n_dominated_class 10