#!/bin/bash

python3 driver.py --config config.yaml --benchmark bonds --output bonds.csv --local
python3 driver.py --config config.yaml --benchmark binomialoptions --output binomial_options.csv --local
python3 driver.py --config config.yaml --benchmark minibude --output minibude_output.csv --local
python3 driver.py --config config.yaml --benchmark miniweather --output miniweather.csv --local
python3 driver.py --config config.yaml --benchmark particlefilter --output particlefilter.csv --local
