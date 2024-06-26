#!/bin/bash

python3 driver.py --config config_small.yaml --benchmark bonds --output bonds.csv --local
python3 driver.py --config config_small.yaml --benchmark binomialoptions --output binomial_options.csv --local
python3 driver.py --config config_small.yaml --benchmark minibude --output minibude_output.csv --local
python3 driver.py --config config_small.yaml --benchmark miniweather --output miniweather.csv --local
python3 driver.py --config config_small.yaml --benchmark particlefilter --output particlefilter.csv --local
