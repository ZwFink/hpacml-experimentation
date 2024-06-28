import click
import os
import re
from evaluators import Evaluator
import yaml
import pandas as pd
import parsl
from parsl.app.app import python_app
from parsl.data_provider.files import File as ParslFile
from parsl.app.app import join_app
from parsl.app.app import bash_app
from parsl.config import Config
from parsl.providers import SlurmProvider
from parsl.providers import LocalProvider
from parsl.launchers import SingleNodeLauncher
from parsl.executors import HighThroughputExecutor
import random


@bash_app
def evaluate(benchmark, config_file, trial_num, outputs=(),
             stdout=parsl.AUTO_LOGNAME,
             stderr=parsl.AUTO_LOGNAME
             ):
    import yaml
    import re
    import os

    def get_model_for_trial(trial, models_dir):
        model_re = re.compile(f'model_{trial}_\d+.pth')
        matched_files = [f for f in os.listdir(models_dir) if model_re.match(f)]
    
        if len(matched_files) == 1:
            return os.path.join(models_dir, matched_files[0])
        elif len(matched_files) > 1:
            raise ValueError(f"More than one file matches the trial {trial} pattern.")
        else:
            return None
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    benchmark_config = config[benchmark]
    models_dir = benchmark_config['models_directory']
    # iterate of the index col of trials_df
    model_path = get_model_for_trial(trial_num, models_dir)
    cmd = ['python3', '/srgt/experimentation/benchmark_evaluation/evaluators.py',
           '--benchmark', benchmark,
           '--trial_num', str(trial_num),
           '--config', config_file,
           '--model_path', model_path,
           '--output', outputs[0].filepath
           ]

    print(cmd)
    print(f'Running trial {trial_num} for {benchmark} benchmark')
    return ' '.join(cmd)


def get_parsl_config(is_local, parsl_rundir):
    local_provider = LocalProvider(
        init_blocks=1,
        max_blocks=1,
        parallelism=1
    )

    slurm_provider = SlurmProvider(
        partition="gpuA100x4",
        account="mzu-delta-gpu",
        scheduler_options="#SBATCH --gpus-per-task=1 --nodes=1 --ntasks-per-node=1",
        worker_init='source ~/activate.sh',
        nodes_per_block=1,
        max_blocks=5,
        cores_per_node=15,
        init_blocks=1,
        parallelism=1,
        exclusive=True,
        mem_per_node=70,
        walltime="0:35:00",
        cmd_timeout=500,
        launcher=SingleNodeLauncher()
    )

    if is_local:
        provider = local_provider
    else:
        provider = slurm_provider

    parsl_config = Config(
        retries=2,
        run_dir=parsl_rundir,
        executors=[
            HighThroughputExecutor(
                cores_per_worker=15,
                available_accelerators=1,
                cpu_affinity='block',
                mem_per_worker=70,
                worker_debug=False,
                label="Benchmark_Evaluator",
                provider=provider
            )
        ]
    )

    return parsl_config


@click.command()
@click.option('--benchmark', type=click.Choice(['particlefilter', 'minibude', 'binomialoptions', 'bonds', 'miniweather']))
@click.option('--config', type=click.Path(exists=True))
@click.option('--output', type=click.Path())
@click.option('--local', is_flag=True, default=False)
def main(benchmark, config, output, local):
    parsl_rundir = f'parsl_rundir_{benchmark}'
    parsl_config = get_parsl_config(local, parsl_rundir)
    parsl.load(parsl_config)
    config_file = config
    with open(config, 'r') as f:
        config = yaml.safe_load(f)

    benchmark_config = config[benchmark]
    trials_file = benchmark_config['all_trials_file']
    trials_df = pd.read_csv(trials_file, index_col='trial')

    all_trials_df = pd.DataFrame()

    results = list()
    trials_todo = list(trials_df.index)
    random.shuffle(trials_todo)
    for trial_num in trials_todo:
        # if the value value for column 'inference_time' is nan
        if pd.isna(trials_df.loc[trial_num, 'inference_time']):
            print(f'Skipping trial {trial_num} for {benchmark} benchmark. ')
            print('(The model architecture was likely invalid, so the model did not train).')
            continue
        output_file = f'{output}_{benchmark}_{trial_num}.csv'
        parsl_output = ParslFile(output_file)
        trial_results = evaluate(benchmark, config_file,
                                 trial_num, outputs=[parsl_output]
                                 )
        results.append(trial_results)

    for trial_results in results:
        result_file = trial_results.outputs[0].result()
        print(f'Completed trial {trial_results}')
        trial_results = pd.read_csv(result_file)
        all_trials_df = pd.concat([all_trials_df, trial_results],
                                  ignore_index=True
                                  )
        os.remove(result_file)

    all_trials_df.to_csv(output, index=False)

if __name__ == '__main__':
    main()
