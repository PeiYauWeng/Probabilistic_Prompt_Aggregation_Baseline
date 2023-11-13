#Dirichlet 0.5
python3 main.py --alg fedavg --comms 100 --alpha 0.5
python3 main.py --alg fedprox --comms 100 --alpha 0.5
python3 main.py --alg fedopt --comms 120 --alpha 0.5
python3 main.py --alg scaffold --comms 100 --alpha 0.5
python3 main.py --alg fedprox_gmm --comms 100 --alpha 0.5
python3 main.py --alg pfedpg --comms 400 --alpha 0.5
#Dirichlet 0.4
python3 main.py --alg fedavg --comms 100 --alpha 0.4
python3 main.py --alg fedprox --comms 100 --alpha 0.4
python3 main.py --alg fedopt --comms 120 --alpha 0.4
python3 main.py --alg scaffold --comms 100 --alpha 0.4
python3 main.py --alg fedprox_gmm --comms 100 --alpha 0.4
python3 main.py --alg pfedpg --comms 400 --alpha 0.4
#Dirichlet 0.3
python3 main.py --alg fedavg --comms 100 --alpha 0.3
python3 main.py --alg fedprox --comms 100 --alpha 0.3
python3 main.py --alg fedopt --comms 120 --alpha 0.3
python3 main.py --alg scaffold --comms 100 --alpha 0.3
python3 main.py --alg fedprox_gmm --comms 100 --alpha 0.3
python3 main.py --alg pfedpg --comms 400 --alpha 0.3
#Dirichlet 0.2
python3 main.py --alg fedavg --comms 100 --alpha 0.2
python3 main.py --alg fedprox --comms 100 --alpha 0.2
python3 main.py --alg fedopt --comms 120 --alpha 0.2
python3 main.py --alg scaffold --comms 100 --alpha 0.2
python3 main.py --alg fedprox_gmm --comms 100 --alpha 0.2
python3 main.py --alg pfedpg --comms 400 --alpha 0.2
#Dirichlet 0.5
python3 main.py --alg fedavg --comms 100 --alpha 0.1
python3 main.py --alg fedprox --comms 100 --alpha 0.1
python3 main.py --alg fedopt --comms 120 --alpha 0.1
python3 main.py --alg scaffold --comms 100 --alpha 0.1
python3 main.py --alg fedprox_gmm --comms 100 --alpha 0.1
python3 main.py --alg pfedpg --comms 400 --alpha 0.1
#Manual extreme heterogeneity
python3 main.py --alg fedavg --comms 100 --data_distribution manual_extreme_heterogeneity
python3 main.py --alg fedprox --comms 100 --data_distribution manual_extreme_heterogeneity
python3 main.py --alg fedopt --comms 120 --data_distribution manual_extreme_heterogeneity
python3 main.py --alg scaffold --comms 100 --data_distribution manual_extreme_heterogeneity
python3 main.py --alg fedprox_gmm --comms 100 --data_distribution manual_extreme_heterogeneity
python3 main.py --alg pfedpg --comms 400 --data_distribution manual_extreme_heterogeneity