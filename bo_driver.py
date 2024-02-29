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
import parsl
from parsl.app.app import python_app
from parsl.app.app import join_app
from parsl.app.app import bash_app
from parsl.config import Config
from parsl.providers import SlurmProvider
from parsl.launchers import SrunLauncher
from parsl.executors import HighThroughputExecutor
from parsl import set_stream_logger

TRIALS_ARCH = 2
TRIALS_HYPERPARMS = 3

class TrialFailureException(Exception):
    pass


parsl_config = Config(
    retries=2,
    executors=[
        HighThroughputExecutor(
            cores_per_worker=15,
            worker_debug=True,
            available_accelerators=1,
            label="GPU Executor",
            provider=SlurmProvider(
                partition="gpuA100x4",
                account="mzu-delta-gpu",
                scheduler_options="#SBATCH --gpus-per-task=1 --cpus-per-gpu=15 --nodes=1 --ntasks-per-node=1",
                worker_init='conda deactivate; source ~/activate.sh',
                nodes_per_block=1,
                max_blocks=9,
                init_blocks=1,
                exclusive=False,
                mem_per_node=35,
                walltime="0:4:00",
                cmd_timeout=500,
                launcher=SrunLauncher()
            )
        )
    ]
)

parsl.load(parsl_config)

class OutputManager:
    def __init__(self, directory_prefix, benchmark_name, append_benchmark_name=True):
        self.benchmark_name = benchmark_name
        if append_benchmark_name:
            self.output_dir_name = f'{directory_prefix}_{benchmark_name}'
        else:
            self.output_dir_name = f'{directory_prefix}'
        self.output_dir_path = Path(self.output_dir_name)
        self.output_dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_datetime_prefix(cls):
        return datetime.now().strftime("%Y-%m-%d")

    def save_optimization_state(self, optimization_step, ax_client, name="ax_client"):
        ax_client.save_to_json_file(f"{str(self.output_dir_path)}/{name}.json")
        dat = {'optimization_step': optimization_step}
        with open(f'{str(self.output_dir_path)}/{name}_optimization_step.json', 'w') as f:
            f.write(json.dumps(dat))

    def save_pareto_parameters(self, pareto_parameters, name="pareto_parameters"):
        with open(f'{str(self.output_dir_path)}/{name}.json', 'w') as f:
            f.write(pareto_parameters)

    def save_trial_results_df(self, trial_results_df, name="trial_results"):
        # print trail results index name
        print(trial_results_df.index.name)
        trial_results_df.to_csv(f"{str(self.output_dir_path)}/{name}.csv", index=True)


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


@python_app
def evaluate_hyperparameters(config_filename, arch_parameters,
                             hyper_parameters, config_global,
                             train_script, benchmark_name,
                             trial_index):
    import tempfile
    import sh
    import yaml
    import sys

    input_config_tmp = tempfile.NamedTemporaryFile(mode='w', prefix='/tmp/', delete=False)
    output_config_tmp = tempfile.NamedTemporaryFile(mode='w', prefix='/tmp/', delete=False)
    print("Tmp file name is:", input_config_tmp.name)
    
    with open(input_config_tmp.name, 'w') as f:
        yaml.dump({'arch_parameters': arch_parameters, 'hyper_parameters': hyper_parameters}, f)

    command = sh.Command('python3').bake(train_script, 
    '--name', benchmark_name,
    '--config', config_filename, 
    '--architecture_config', input_config_tmp.name, '--output', output_config_tmp.name
    )
    print(str(command))
    
    try:
        command(_out=sys.stdout, _err=sys.stderr)
    except sh.ErrorReturnCode as e:
        print("Error running command")
        print(e)
        return {"average_mse": (1e9, 0), 'inference_time': (1e9, 0)}

    with open(output_config_tmp.name, 'r') as f:
        results = yaml.safe_load(f)

    error, inference_time = results['average_mse'], results['inference_time']
    return trial_index, {"average_mse": (error, 0), 'inference_time': (inference_time, 0)}


def submit_parallel_trial(parameters_hyperparams, trial_index, eval_args):
    # This causes issues when submitting one job from another, see:
    # https://bugs.schedmd.com/show_bug.cgi?id=14298
    del os.environ['SLURM_CPU_BIND']
    results = evaluate_hyperparameters(
        config_filename=eval_args.config_filename,
        arch_parameters=eval_args.arch_parameters,
        hyper_parameters=parameters_hyperparams,
        config_global=eval_args.config_global,
        train_script=eval_args.config_global['train_script'],
        benchmark_name=eval_args.benchmark_name,
        trial_index=trial_index
    )
    return results


def evaluate_architecture(eval_args):
    ax_client_hyperparams = eval_args.ax_client_hypers
    max_parallelism = ax_client_hyperparams.get_max_parallelism()[-1][1]
    print("Max parallelism:", ax_client_hyperparams.get_max_parallelism())

    for i in range(0, TRIALS_HYPERPARMS, max_parallelism):
        job_futures = list()
        print(f"Running trials {i} to {i + max_parallelism}")
        tst = time.time()
        for j in range(max_parallelism):
            if not (i + j < TRIALS_HYPERPARMS):
                continue
            params, trial_index = ax_client_hyperparams.get_next_trial()
            job_futures.append(submit_parallel_trial(params,
                                                     trial_index,
                                                     eval_args))

        results = [f.result() for f in job_futures]
        tend = time.time()
        print(f'Finished running trials {i} to {i + max_parallelism} in {tend - tst} seconds')
        for res in results:
            trial_index, result = res
            process_result(trial_index, result, ax_client_hyperparams)

    best_parameters = ax_client_hyperparams.get_best_parameters()
    print(best_parameters)
    error = best_parameters[1][0]['average_mse']
    inference_time = best_parameters[1][0]['inference_time']
    best_hypers = best_parameters[0]
    print("best hypers:", best_hypers)
    result_data = {"average_mse": (error, 0), 'inference_time': (inference_time, 0)}
    data = {"average_mse": (error, 0), 'inference_time': (inference_time, 0), 'learning_rate': (best_hypers['learning_rate'], 0),
    'weight_decay': (best_hypers['weight_decay'], 0), 'epochs': (best_hypers['epochs'], 0), 'batch_size': (best_hypers['batch_size'], 0),
    'dropout': (best_hypers['dropout'], 0)}
    return result_data, data, eval_args.arch_parameters


def process_result(trial_index, result, ax_client_hyperparams):
    if result['average_mse'][0] == 1e9:
        ax_client_hyperparams.log_trial_failure(trial_index)
        raise TrialFailureException()
    else:
        ax_client_hyperparams.complete_trial(trial_index=trial_index,
                                             raw_data=result
                                            )


@click.command()
@click.option('--config', default='./config.yaml', help='Path to the config file', required=True)
@click.option('--benchmark', help='Name of the benchmark', required=True)
@click.option('--output_base', default='./', help='Path to the base output directory', required=False)
@click.option('--restart', default=None, help='Restart the optimization from the data in this directory', required=False)
@click.option('--output', default=None, help='Path to the output directory. Mutually exclusive with output_base.', required=False)
def main(config, benchmark, output_base, restart, output):
  # Give 'output' the highest precedence for creating the output directory
  if output:
    om = OutputManager(output, benchmark, append_benchmark_name=False)
  # if we don't have output, but we do have restart, then choose to continue from the restart directory
  elif restart:
    om = OutputManager(restart, benchmark, append_benchmark_name=False)
  # otherwise, create the output directory from the output_base
  elif not output and not restart:
    om = OutputManager(f'{output_base}/{OutputManager.get_datetime_prefix()}', benchmark)
  
  config_filename = config
  with open(config_filename, 'r') as file:
      config = yaml.safe_load(file)
      config_global = config.copy()[benchmark]
      config = config[benchmark]['bayesian_opt_driver_args']
  
  arch_search_params = get_params(config['architecture_config'])
  hyper_search_params = get_params(config['hyperparameter_config'])

  output_columns = ['trial']
  output_columns += arch_search_params.get_parameter_names() 
  output_columns += hyper_search_params.get_parameter_names()
  output_columns += arch_search_params.tracking_metric_names

  global TRIALS_ARCH
  global TRIALS_HYPERPARMS
  TRIALS_ARCH = config['architecture_config']['trials']
  TRIALS_HYPERPARMS = config['hyperparameter_config']['trials']
  print("Trials for architecture:", TRIALS_ARCH)
  print("Trials for hyperparameters:", TRIALS_HYPERPARMS)
  
  ax_client_hyperparams = AxClient()
  ax_client_architecture = AxClient()

  start_round = 0
  
  if restart is None:
    output_df = pd.DataFrame(columns=output_columns)
    output_df.set_index('trial', inplace=True)
    exp_arch = ax_client_architecture.create_experiment(name="Architecture_search",
    parameters=arch_search_params.parameter_space, objectives=arch_search_params.objectives,
    tracking_metric_names = arch_search_params.tracking_metric_names,
    parameter_constraints = arch_search_params.parameter_constraints)
  else:
    restart_file = f'{restart}/ax_client.json'
    step_file = f'{restart}/ax_client_optimization_step.json'
    ax_client_architecture = AxClient.load_from_json_file(restart_file)
    output_df = pd.read_csv(f'{restart}/trial_results.csv', index_col='trial')
    with open(step_file, 'r') as step_f:
      step = json.load(step_f)
      start_round = step['optimization_step'] + 1
      print(f'Restarting from step {start_round}')
  
  
  eval_args = EvalArgs(benchmark, config_global, config_filename, None, ax_client_hyperparams, None)
  
  for i in range(start_round, TRIALS_ARCH):
      parameters, trial_index = ax_client_architecture.get_next_trial()
      eval_args.arch_parameters = parameters
  
      ax_client_hyperparams = AxClient()
      eval_args.ax_client_hypers = ax_client_hyperparams
      n_success = 0
  
      hyper_arch = ax_client_hyperparams.create_experiment(name="PF_hyperparameters",
          parameters=hyper_search_params.parameter_space, objectives=hyper_search_params.objectives,
          tracking_metric_names=hyper_search_params.tracking_metric_names,
          outcome_constraints=hyper_search_params.parameter_constraints)
      try:
        data_objectives, data_hyper, data_arch = evaluate_architecture(eval_args)
        ax_client_architecture.complete_trial(trial_index=trial_index,
                                              raw_data=data_objectives
                                              )
        n_success += 1
      except TrialFailureException:
        data_hyper, data_arch = {}, eval_args.arch_parameters
        ax_client_architecture.log_trial_failure(trial_index)
      data_hyper.update(data_arch)
      print(data_hyper)
      output_data = get_trial_output_data(output_columns, data_hyper)
      output_data['trial'] = i  # Ensure this matches how you're tracking trial indices
      print("output_df", output_df)
      print("output data", output_data)
      new_row_df = pd.DataFrame([output_data]).set_index('trial')
      output_df = pd.concat([output_df, new_row_df], ignore_index=False)
      # set the name of the index column to trial
      output_df.index.name = 'trial'
  
      print(output_df)
  
      # print pareto optimal parameters 
      if n_success > 0:
        best_parameters = ax_client_architecture.get_pareto_optimal_parameters()
      else:
        best_parameters = {}
      print("pareto parameters:", best_parameters)
      om.save_optimization_state(i, ax_client_architecture)
      om.save_trial_results_df(output_df)
      om.save_pareto_parameters(json.dumps(best_parameters))

if __name__ == "__main__":
    main()