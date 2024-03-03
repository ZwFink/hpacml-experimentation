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
from parsl.launchers import SrunLauncher
from parsl.data_provider.files import File as ParslFile
from parsl.executors import HighThroughputExecutor
from parsl import set_stream_logger
from util import BOParameterWrapper

# set_stream_logger()
TRIALS_ARCH = 2
TRIALS_HYPERPARMS = 3


def get_temp_file_located_here():
    return tempfile.NamedTemporaryFile(mode='w', delete=False, dir=os.getcwd())

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
        try:
            threshold = c['threshold']
        except KeyError:
            threshold = None
        if c['type'] == 'minimize':
            objectives_l[c['name']] = ObjectiveProperties(minimize=True,
                                                          threshold=threshold
                                                          )
        else:
            objectives_l[c['name']] = ObjectiveProperties(minimize=False,
                                                          threshold=threshold)
    return BOParameterWrapper(parm_space, constraints,
                              objectives_l, 
                              tracking_metric_names)


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


@bash_app
def architecture_driver(benchmark_name, parsl_rundir_name,
                        inputs=(), outputs=(),
                        stdout=parsl.AUTO_LOGNAME,
                        stderr=parsl.AUTO_LOGNAME
                        ):
    import yaml
    import sh
    import time
    print(f'Architecture driver started at {time.time()}')

    ipt_file = inputs[1]
    with open(ipt_file, 'r') as f:
        parameters = yaml.safe_load(f)

    input_config, input_params = inputs

    cmd = sh.Command('python3')
    cmd = cmd.bake('bo_hyperparameter.py')
    cmd = cmd.bake('--config', input_config,
                   '--trial_index', parameters['trial_index'],
                   '--architecture', input_params,
                   '--output', outputs[0],
                   '--parsl_rundir', parsl_rundir_name,
                   '--benchmark', benchmark_name
                   )
    f = open('debug_file', 'w')
    print("Running command: ", str(cmd), file=f)
    f.close()
    return str(cmd)


@click.command()
@click.option('--config', default='./config.yaml', help='Path to the config file', required=True)
@click.option('--benchmark', help='Name of the benchmark', required=True)
@click.option('--output_base', default='./', help='Path to the base output directory', required=False)
@click.option('--restart', default=None, help='Restart the optimization from the data in this directory', required=False)
@click.option('--output', default=None, help='Path to the output directory. Mutually exclusive with output_base.', required=False)
@click.option('--parsl_rundir', default='./rundir', help='Path to the parsl run directory', required=False)
def main(config, benchmark, output_base, restart, output, parsl_rundir):
    slurm_provider = SlurmProvider(
                    partition="cpu",
                    account="mzu-delta-cpu",
                    scheduler_options="#SBATCH --cpus-per-task=5 --nodes=1 --ntasks-per-node=1",
                    worker_init='source ~/activate.sh',
                    nodes_per_block=1,
                    max_blocks=9,
                    init_blocks=1,
                    exclusive=False,
                    mem_per_node=10,
                    walltime="00:20:00",
                    cmd_timeout=500,
                    launcher=SrunLauncher()
                )

    local_provider = LocalProvider(
        init_blocks=1,
        max_blocks=10,
        parallelism=1
    )

    parsl_config = Config(
        retries=2,
        run_dir=parsl_rundir,
        executors=[
            HighThroughputExecutor(
                cores_per_worker=1,
                worker_debug=False,
                label="CPU_Executor",
                provider=local_provider
            )
        ]
    )

    parsl.load(parsl_config)
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
  
    ax_client_architecture = AxClient()

    start_round = 0
  
    if restart is None:
      output_df = pd.DataFrame(columns=output_columns)
      output_df.set_index('trial', inplace=True)
      exp_arch = ax_client_architecture.create_experiment(name="Architecture_search",
      parameters=arch_search_params.parameter_space, objectives=arch_search_params.objectives,
      tracking_metric_names = arch_search_params.tracking_metric_names,
      parameter_constraints = arch_search_params.parameter_constraints
      )
    else:
      restart_file = f'{restart}/ax_client.json'
      step_file = f'{restart}/ax_client_optimization_step.json'
      ax_client_architecture = AxClient.load_from_json_file(restart_file)
      output_df = pd.read_csv(f'{restart}/trial_results.csv', index_col='trial')
      with open(step_file, 'r') as step_f:
        step = json.load(step_f)
        start_round = step['optimization_step'] + 1
        print(f'Restarting from step {start_round}')
  

    max_parallelism = ax_client_architecture.get_max_parallelism()[-1][1]
    for i in range(start_round, TRIALS_ARCH, max_parallelism):
        output_futures = list()
        files_to_cleanup = list()
        tst = time.time()
        for j in range(max_parallelism):
            if not (i + j < TRIALS_ARCH):
                continue

            parameters, trial_index = ax_client_architecture.get_next_trial()

            _input_file = get_temp_file_located_here().name
            _output_file = get_temp_file_located_here().name

            files_to_cleanup.append(_input_file)
            files_to_cleanup.append(_output_file)
            
            input_file = ParslFile(_input_file)
            output_file = ParslFile(_output_file)
            input_file_config = ParslFile(config_filename)

            with open(input_file, 'w') as f:
                params = dict()
                params['arch_params'] = parameters
                params['trial_index'] = trial_index
                yaml.dump(params, f)
  
            rundir_arch = f'{parsl_rundir}/architecture_{i + j}'
            n_success = 0
  
            output = architecture_driver(benchmark,
                                         rundir_arch,
                                         inputs=[input_file_config, input_file],
                                         outputs=[output_file]
                                         )
            output_futures.append(output)

        results = output_futures
        for j, res in enumerate(results):
            of = res.outputs[0].result()
            with open(of, 'r') as f:
                results = yaml.safe_load(f)
                trial_index = results['trial_index']
                if results['success']:
                    n_success += 1
                    objective_results = results['results']['objectives']
                    print(f"Trial {trial_index} succeeded with parameters {objective_results}")
                    # Change the list values in objective_results to tuple
                    for k in objective_results:
                        if isinstance(objective_results[k], list):
                            objective_results[k] = tuple(objective_results[k])
                    ax_client_architecture.complete_trial(trial_index=trial_index,
                                                          raw_data=objective_results)
                else:
                    print(f"Trial {results['trial_index']} failed")
                    ax_client_architecture.log_trial_failure(trial_index)
            
            output_data = get_trial_output_data(output_columns, results['results']['hyperparameters'])
            output_data['trial'] = i+j
            print("output_df", output_df)
            print("output data", output_data)
            new_row_df = pd.DataFrame([output_data]).set_index('trial')
            output_df = pd.concat([output_df, new_row_df], ignore_index=False)
            # set the name of the index column to trial
            output_df.index.name = 'trial'
  
            print(output_df)

        tend = time.time()
        print(f'Completed trials {i} to {i + max_parallelism - 1} in {tend - tst} seconds')
        for file in files_to_cleanup:
            os.remove(file)
  
        # print pareto optimal parameters 
        if n_success > 0:
          best_parameters = ax_client_architecture.get_pareto_optimal_parameters(use_model_predictions=False)
        else:
          best_parameters = {}
        print("pareto parameters:", best_parameters)
        om.save_optimization_state(i, ax_client_architecture)
        om.save_trial_results_df(output_df)
        om.save_pareto_parameters(json.dumps(best_parameters))

if __name__ == "__main__":
    main()
