#!/usr/bin/env python
# coding: utf-8



# particle filter using Bayesian Optimization to find a good model architecture
import sys
import yaml
from datetime import datetime
import os
import time
import struct
import numpy as np
import pandas as pd
import json

from ax.plot.contour import plot_contour, interact_contour
from ax.plot.trace import optimization_trace_single_method
from ax.plot.diagnostic import interact_cross_validation
from ax.modelbridge.cross_validation import cross_validate
from ax.service.managed_loop import optimize
from ax.metrics.branin import branin
import numpy as np
from dataclasses import dataclass
import click
from pathlib import Path
import sh
import tempfile
from botorch.acquisition import qExpectedImprovement
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import ModelRegistryBase, Models
from ax.service.ax_client import AxClient, ObjectiveProperties

TRIALS_ARCH=2
TRIALS_HYPERPARMS=3
class OutputManager:
    def __init__(self, directory_prefix, benchmark_name):
        self.benchmark_name = benchmark_name
        self.output_dir_name = f'{directory_prefix}_{benchmark_name}'
        self.output_dir_path = Path(self.output_dir_name)
        self.output_dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_datetime_prefix(cls):
        return datetime.now().strftime("%Y-%m-%d")

    def save_ax_client(self, optimization_step, ax_client, name="ax_client"):
        ax_client.save_to_json_file(f"{str(self.output_dir_path)}/{name}.json")
        dat = {'optimization_step': optimization_step}
        with open(f'{str(self.output_dir_path)}/{name}_optimization_step.json', 'w') as f:
            f.write(json.dumps(dat))

    def save_pareto_parameters(self, pareto_parameters, name="pareto_parameters"):
        with open(f'{str(self.output_dir_path)}/{name}.json', 'w') as f:
            f.write(pareto_parameters)

    def save_trial_results_df(self, trial_results_df, name="trial_results"):
        trial_results_df.to_csv(f"{str(self.output_dir_path)}/{name}.csv")


@dataclass
class BOParameterWrapper:
    parameter_space: list
    parameter_constraints: list
    objectives: dict
    tracking_metric_names: list

    def get_parameter_names(self):
        return [p['name'] for p in self.parameter_space]

def get_params(config):
    parm_space = config['parameter_space']
    if 'constraints' in config:
        constraints = config['parameter_constraints']
    else:
        constraints = []
    objectives = config['objectives']
    tracking_metric_names = config['tracking_metrics']

    objectives_l = dict()
    for c in objectives:
        if c['type'] == 'minimize':
            objectives_l[c['name']] = ObjectiveProperties(minimize=True)
        else:
            objectives_l[c['name']] = ObjectiveProperties(minimize=False)
    return BOParameterWrapper(parm_space, constraints, objectives_l, tracking_metric_names)


def get_trial_output_data(output_columns, data_dict):
    data = dict()
    for k in data_dict:
      if k in data_dict:
        try:
          data[k] = data_dict[k][0]
        except TypeError:
          data[k] = data_dict[k]
    return data

@dataclass
class EvalArgs:
    benchmark_name: str
    config_global: dict
    config_filename: str
    arch_parameters: dict
    ax_client_hypers: AxClient
    hyper_parameters: dict

def evaluate_hyperparameters(eval_args):
    config_filename = eval_args.config_filename
    arch_parameters = eval_args.arch_parameters
    hyper_parameters = eval_args.hyper_parameters

    input_config_tmp = tempfile.NamedTemporaryFile(mode='w', prefix='/tmp/', delete=False)
    output_config_tmp = tempfile.NamedTemporaryFile(mode='w', prefix='/tmp/', delete=False)
    # write arch_parameters and hyper_parameters to a yaml file
    print("Tmp file name is:", input_config_tmp.name)
    with open(input_config_tmp.name, 'w') as f:
        yaml.dump({'arch_parameters': arch_parameters, 'hyper_parameters': hyper_parameters}, f)

    command = sh.Command('python3').bake(eval_args.config_global['train_script'], 
    '--name', eval_args.benchmark_name,
    '--config', config_filename, 
    '--architecture_config', input_config_tmp.name, '--output', output_config_tmp.name
    )
    print(str(command))
    command(_out=sys.stdout, _err=sys.stderr)
    with open(output_config_tmp.name, 'r') as f:
        results = yaml.safe_load(f)
    error, inference_time = results['average_mse'], results['inference_time']
    return {"average_mse": (error, 0), 'inference_time': (inference_time, 0)}

def evaluate_architecture(eval_args):
    ax_client_hyperparams = eval_args.ax_client_hypers
    for i in range(TRIALS_HYPERPARMS):
        parameters_hyperparams, trial_index_hyperparams = ax_client_hyperparams.get_next_trial()
        eval_args.hyper_parameters= parameters_hyperparams
        print(parameters_hyperparams)
        ax_client_hyperparams.complete_trial(trial_index = trial_index_hyperparams, 
        raw_data=evaluate_hyperparameters(eval_args)
        ) 
    best_parameters = ax_client_hyperparams.get_best_parameters()
    print(best_parameters)
    error = best_parameters[1][0]['average_mse']
    inference_time = best_parameters[1][0]['inference_time']
    best_hypers = best_parameters[0]
    print("best hypers:", best_hypers)
    data = {"average_mse": (error, 0), 'inference_time': (inference_time, 0), 'learning_rate': (best_hypers['learning_rate'], 0),
    'weight_decay': (best_hypers['weight_decay'], 0), 'epochs': (best_hypers['epochs'], 0), 'batch_size': (best_hypers['batch_size'], 0)}
    return data, eval_args.arch_parameters


@click.command()
@click.option('--config', default='./config.yaml', help='Path to the config file', required=True)
@click.option('--benchmark', help='Name of the benchmark', required=True)
@click.option('--output_base', default='./', help='Path to the base output directory', required=False)
def main(config, benchmark, output_base):
  om = OutputManager(f'{output_base}/{OutputManager.get_datetime_prefix()}', benchmark)
  
  config_filename = config
  with open(config_filename, 'r') as file:
      config = yaml.safe_load(file)
      config_global = config.copy()[benchmark]
      config = config[benchmark]['bayesian_opt_driver_args']
  
  arch_search_params = get_params(config['architecture_config'])
  hyper_search_params = get_params(config['hyperparameter_config'])

  global TRIALS_ARCH
  global TRIALS_HYPERPARMS
  TRIALS_ARCH = config['architecture_config']['trials']
  TRIALS_HYPERPARMS = config['hyperparameter_config']['trials']
  print("Trials for architecture:", TRIALS_ARCH)
  print("Trials for hyperparameters:", TRIALS_HYPERPARMS)
  
  ax_client_hyperparams = AxClient()
  ax_client_architecture = AxClient()
  
  exp_arch = ax_client_architecture.create_experiment(name="Architecture_search",
  parameters=arch_search_params.parameter_space, objectives=arch_search_params.objectives,
  tracking_metric_names = arch_search_params.tracking_metric_names,
  parameter_constraints = arch_search_params.parameter_constraints)
  
  output_columns = arch_search_params.get_parameter_names() 
  output_columns += hyper_search_params.get_parameter_names()
  output_columns += arch_search_params.tracking_metric_names
  
  output_df = pd.DataFrame(columns=output_columns)
  
  eval_args = EvalArgs(benchmark, config_global, config_filename, None, ax_client_hyperparams, None)
  
  for i in range(TRIALS_ARCH):
      parameters, trial_index = ax_client_architecture.get_next_trial()
      eval_args.arch_parameters = parameters
  
      ax_client_hyperparams = AxClient()
      eval_args.ax_client_hypers = ax_client_hyperparams
  
      hyper_arch = ax_client_hyperparams.create_experiment(name="PF_hyperparameters",
          parameters=hyper_search_params.parameter_space, objectives=hyper_search_params.objectives,
          tracking_metric_names=hyper_search_params.tracking_metric_names,
          outcome_constraints=hyper_search_params.parameter_constraints)
      data_hyper, data_arch = evaluate_architecture(eval_args)
      ax_client_architecture.complete_trial(trial_index = trial_index, raw_data=data_hyper)
      data_hyper.update(data_arch)
      print(data_hyper)
      output_data = get_trial_output_data(output_columns, data_hyper)
      print("output_df", output_df)
      print("output data", output_data)
      new_row_df = pd.DataFrame([output_data])
      output_df = pd.concat([output_df, new_row_df], ignore_index=True)
  
      print(output_df)
  
      # print pareto optimal parameters 
      best_parameters = ax_client_architecture.get_pareto_optimal_parameters()
      print("pareto parameters:", best_parameters)
      om.save_ax_client(i, ax_client_architecture)
      om.save_trial_results_df(output_df)
      om.save_pareto_parameters(json.dumps(best_parameters))

if __name__ == "__main__":
    main()