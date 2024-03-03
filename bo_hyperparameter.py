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
from parsl.providers import LocalProvider
from parsl.launchers import SingleNodeLauncher
from parsl.executors import HighThroughputExecutor
from parsl import set_stream_logger
from util import BOParameterWrapper, EvalArgs

TRIALS_HYPERPARMS = 3

class TrialFailureException(Exception):
    pass


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
    # inference_time is a list [inference_time, sem]
    return trial_index, {"average_mse": (error, 0), 'inference_time': inference_time}


def submit_parallel_trial(parameters_hyperparams, trial_index, eval_args):
    # This causes issues when submitting one job from another, see:
    # https://bugs.schedmd.com/show_bug.cgi?id=14298
    try:
        os.unsetenv('SLURM_CPU_BIND')
        os.unsetenv('SLURM_CPU_BIND_LIST')
        os.unsetenv('SLURM_CPUS_ON_NODE')
        os.unsetenv('SLURM_CPUS_PER_TASK')
        os.unsetenv('SLURM_CPU_BIND_TYPE')
    except KeyError:
        pass
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


def evaluate_architecture(ax_client, eval_args):
    ax_client_hyperparams = ax_client

    max_parallelism = ax_client_hyperparams.get_max_parallelism()[-1][1]
    print("Max parallelism:", ax_client_hyperparams.get_max_parallelism())
    trial_to_runtime_sem = dict()

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
            result['inference_time'] = tuple(result['inference_time'])
            trial_to_runtime_sem[trial_index] = result['inference_time'][1]
            process_result(trial_index, result, ax_client_hyperparams)

    best_index, best_parameters, results = ax_client_hyperparams.get_best_trial(use_model_predictions=False)
    best_parameters = (best_parameters, results)
    best_sem = trial_to_runtime_sem[best_index]
    print(best_parameters)
    error = best_parameters[1][0]['average_mse']
    inference_time = best_parameters[1][0]['inference_time']
    best_hypers = best_parameters[0]
    print("best hypers:", best_hypers)
    result_data = {"average_mse": [error, 0], 'inference_time': [inference_time, best_sem]}
    data = {"average_mse": [error, 0], 'inference_time': [inference_time, best_sem], 'learning_rate': [best_hypers['learning_rate'], 0],
    'weight_decay': [best_hypers['weight_decay'], 0], 'epochs': [best_hypers['epochs'], 0], 'batch_size': [best_hypers['batch_size'], 0],
    'dropout': [best_hypers['dropout'], 0]}
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
@click.option('--trial_index', help='Index of the trial', required=True, type=int)
@click.option('--architecture', help='Architecture parameter file', required=True)
@click.option('--benchmark', help='Name of the benchmark', required=True)
@click.option('--output', default=None, help='Path to the output directory. Mutually exclusive with output_base.', required=False)
@click.option('--parsl_rundir', default='./rundir', help='Path to the parsl run directory', required=False)
def main(config, trial_index, architecture, benchmark, output, parsl_rundir):

    config_filename = config
    local_provider = LocalProvider(
        init_blocks=1,
        max_blocks=1,
        parallelism=1
    )

    slurm_provider = SlurmProvider(
        partition="gpuA100x4",
        account="mzu-delta-gpu",
        scheduler_options="#SBATCH --gpus-per-task=4 --cpus-per-gpu=15 --nodes=1 --ntasks-per-node=1",
        worker_init='source ~/activate.sh',
        nodes_per_block=1,
        max_blocks=5,
        init_blocks=1,
        parallelism=1,
        exclusive=False,
        mem_per_node=150,
        walltime="9:45:00",
        cmd_timeout=500,
        launcher=SingleNodeLauncher()
    )

    parsl_config = Config(
        retries=2,
        run_dir=parsl_rundir,
        executors=[
            HighThroughputExecutor(
                cores_per_worker=15,
                available_accelerators=4,
                cpu_affinity='block',
                mem_per_worker=35,
                worker_debug=False,
                label="BO_Search_Exec",
                provider=slurm_provider
            )
        ]
    )

    parsl.load(parsl_config)
    output_columns = ['trial']
    global TRIALS_ARCH
    with open(config, 'r') as f:
        config_global = yaml.safe_load(f)
        config_driver = config_global[benchmark]['bayesian_opt_driver_args']
        config_global = config_global[benchmark]
  
    TRIALS_ARCH = config_driver['architecture_config']['trials']
    print("Trials for hyperparameters:", TRIALS_HYPERPARMS)
    hyper_search_params = get_params(config_driver['hyperparameter_config'])

    with open(architecture, 'r') as f:
        arch_config = yaml.safe_load(f)
    arch_params = arch_config['arch_params']
        
    ax_client_hyperparams = AxClient()
    hyper_arch = ax_client_hyperparams.create_experiment(name="PF_hyperparameters",
        parameters=hyper_search_params.parameter_space, 
        objectives=hyper_search_params.objectives,
        tracking_metric_names=hyper_search_params.tracking_metric_names,
        outcome_constraints=hyper_search_params.parameter_constraints)


    output_values = dict()
    eval_args = EvalArgs(benchmark, config_global,
                         config_filename, arch_params,
                         None
                         )
  
    try:
        eval_results = evaluate_architecture(ax_client_hyperparams, eval_args)
        output_values['success'] = True
    except TrialFailureException:
        output_values['success'] = False
    data_objectives, data_hyper, data_arch = eval_results
    trial_results = dict()
    trial_results['objectives'] = data_objectives
    data_hyper.update(data_arch)
    trial_results['hyperparameters'] = data_hyper
    output_values['results'] = trial_results
    output_values['trial_index'] = trial_index

    with open(output, 'w') as f:
        yaml.dump(output_values, f)

if __name__ == "__main__":
    main()
