#!/bin/bash
python3 bo_driver.py --benchmark bonds --config ./config.yaml --parsl_rundir 2024-03-16_bonds/hyper_rundir
python3 bo_driver.py --benchmark binomial_options --config ./config.yaml --parsl_rundir 2024-03-16_binomial_options/hyper_logdir
python3 bo_driver.py --benchmark minibude --config ./config.yaml --parsl_rundir 2024-03-08_minibude/hyper_rundir --restart 2024-03-08_minibude
python3 bo_driver.py --benchmark miniweather --config ./config.yaml --parsl_rundir 2024-03-11_miniweather/hyper_rundir
python3 bo_driver.py --benchmark particlefilter --config ./config.yaml --parsl_rundir 2024-03-09_particlefilter/hyper_rundir
