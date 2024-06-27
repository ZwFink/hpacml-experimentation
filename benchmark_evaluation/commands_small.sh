#!/bin/bash

python3 driver.py --config config_small.yaml --benchmark bonds --output results/bonds_small.csv --local
python3 driver.py --config config_small.yaml --benchmark binomialoptions --output results/binomialoptions_small.csv --local
python3 driver.py --config config_small.yaml --benchmark minibude --output results/minibude_small.csv --local
python3 driver.py --config config_small.yaml --benchmark miniweather --output results/miniweather_small.csv --local
python3 driver.py --config config_small.yaml --benchmark particlefilter --output results/particlefilter_small.csv --local
