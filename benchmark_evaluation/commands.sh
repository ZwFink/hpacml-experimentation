#!/bin/bash

python3 driver.py --config config.yaml --benchmark bonds --output results/bonds.csv --local
python3 driver.py --config config.yaml --benchmark binomialoptions --output results/binomialoptions.csv --local
python3 driver.py --config config.yaml --benchmark minibude --output results/minibude.csv --local
python3 driver.py --config config.yaml --benchmark miniweather --output results/miniweather.csv --local
python3 driver.py --config config.yaml --benchmark particlefilter --output results/particlefilter.csv --local
