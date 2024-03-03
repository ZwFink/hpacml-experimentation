import sh
from nns import *
import tempfile
import click
import parsl
import yaml
import json
import pandas as pd
import os
from parsl.app.app import python_app
from parsl.app.app import join_app
from parsl.app.app import bash_app
from parsl.config import Config
from parsl.providers import SlurmProvider
from parsl.providers import LocalProvider
from parsl.launchers import SrunLauncher, SingleNodeLauncher
from parsl.data_provider.files import File as ParslFile
from parsl.executors import HighThroughputExecutor

@bash_app
def train_model(benchmark_name, config_file, 
                arch_index, out_model_file,
                outputs=()):
    import sh
    output_results_file = outputs[0]

    python = sh.Command('python3').bake('train.py')
    cmd = python.bake('--name', benchmark_name,
                      '--config', config_file,
                      '--architecture_index', arch_index,
                      '--output_model', out_model_file,
                      '--output', output_results_file
                      )
    print('Command to run:', cmd)
    return str(cmd)

def get_temp_file_located_here():
    return tempfile.NamedTemporaryFile(mode='w', delete=False, 
                                       dir=os.getcwd()
                                       )

class TempFileGenerator:
    def __init__(self):
        self.files = []
    def get_temp_file_located_here(self):
        f = get_temp_file_located_here()
        self.files.append(f)
        return f
    def __del__(self):
        for f in self.files:
            os.remove(f.name)

@click.command()
@click.option('--config', default='config', help='Name of the config file')
@click.option('--benchmark', help='Name of the benchmark to run')
def main(config, benchmark):
    slurm_provider = SlurmProvider(
        partition="gpuA100x4",
        account="mzu-delta-gpu",
        scheduler_options="#SBATCH --gpus-per-task=4 --cpus-per-gpu=15 --nodes=1 --ntasks-per-node=1",
        worker_init='source ~/activate.sh',
        nodes_per_block=1,
        max_blocks=3,
        init_blocks=1,
        parallelism=1,
        exclusive=False,
        mem_per_node=150,
        walltime="9:45:00",
        cmd_timeout=500,
        launcher=SingleNodeLauncher()
    )

    local_provider = LocalProvider(
        init_blocks=1,
        max_blocks=1,
        parallelism=1
    )

    parsl_rundir_ = tempfile.TemporaryDirectory(dir=os.getcwd())
    parsl_rundir = parsl_rundir_.name
                    
    # This should be a GPU accelerator
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
                label="Test_GPU_Executor",
                provider=slurm_provider
            )
        ]
    )

    parsl.load(parsl_config)

    configurations = yaml.safe_load(open(config))
    bm_config = configurations[benchmark]
    train_args = bm_config['train_args']
    data_args = bm_config['data_args']

    output_dir = data_args['output_models_directory']
    sh.mkdir('-p', output_dir)

    df = pd.read_csv(train_args['all_trials_file'])
    tf = TempFileGenerator()
    # iterate over the index
    results = []
    for index, row in df.iterrows():
        # train the model
        out_file = ParslFile(tf.get_temp_file_located_here().name)
        out_model_file = f'{output_dir}/model_{index}.pth'
        out_model_file = ParslFile(out_model_file)
        res = train_model(benchmark, config, index,
                          out_model_file,
                          outputs=[out_file]
                        )
        results.append(res)

    # this script is the one that needs to filter wrt pareto
    res_dict = dict()
    for r in results:
        with open(r.outputs[0].result(), 'r') as f:
            loaded = yaml.safe_load(f)
            index = loaded['index']
            average_mse = loaded['average_mse']
            inference_time = loaded['inference_time']
            res_dict[index] = {'average_mse': average_mse,
                               'inference_time': inference_time
                               }
            print(f'Trial {index} has average mse {average_mse} and inference time {inference_time}')
    # save res to a dataframe
    res_df = pd.DataFrame(res_dict).T
    res_df['benchmark'] = benchmark
    res_df.to_csv(f'{output_dir}/results.csv')
    del parsl_rundir_

if __name__ == '__main__':
    main()