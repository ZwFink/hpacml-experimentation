#!/bin/bash

python3 driver.py --config config.yaml --benchmark bonds --output 2024-03-17_bonds.csv
python3 driver.py --config config.yaml --benchmark binomialoptions --output 2024-03-15_binomial_options.csv
python3 driver.py --config config.yaml --benchmark minibude --output 2024-03-28_minibude_output.csv
python3 driver.py --config config.yaml --benchmark miniweather --output 2024-03-19_miniweather.csv
python3 driver.py --config config.yaml --benchmark particlefilter --output 2024-03-14_particlefilter.csv

