import pandas as pd
import sh
import numpy as np
import h5py
import io
import os
import tempfile
from pathlib import Path
import glob
import re


DEFAULT_APPROX_H5 = f'/tmp/hpac_db_approx_{os.getpid()}.h5'
DEFAULT_EXACT_H5 = f'/tmp/hpac_db_exact_{os.getpid()}.h5'

def RMSE(ground_truth, approx):
    return np.sqrt(np.mean((ground_truth - approx)**2))

def MAPE(ground_truth, approx):
    epsilon = 1e-8  # small constant to avoid division by zero
    mape = np.mean(np.abs((ground_truth - approx) / (ground_truth + epsilon)))
    return mape*100

def get_loss_fn_from_str(loss_str):
    if loss_str == 'rmse':
        return RMSE
    elif loss_str == 'mape':
        return MAPE
    else:
        raise ValueError(f'Unknown loss function: {loss_str}')

class Evaluator:
    def __init__(self, benchmark, config):
        self.benchmark = benchmark
        self.config = config

    @classmethod
    def get_evaluator_class(cls, benchmark):
        if benchmark == 'particlefilter':
            return ParticleFilterEvaluator
        elif benchmark == 'minibude':
            return MiniBUDEEvaluator
        else:
            raise ValueError(f'Unknown benchmark: {benchmark}')

    def run_and_compute_error(self):
        benchmark_dir = self.config['benchmark_location']
        with tempfile.TemporaryDirectory() as tmpdirname:
            with sh.pushd(tmpdirname):
                self.tmpdirname = tmpdirname
                all_files = glob.glob(f'{benchmark_dir}/*')
                [sh.cp('-r', f, tmpdirname) for f in all_files]
                self.build_approx()
                approx_out = self.run_approx(self.get_run_command())
                approx_processed = self.process_raw_data(approx_out,
                                                         is_approx=True
                                                         )

        with tempfile.TemporaryDirectory() as tmpdirname:
            with sh.pushd(tmpdirname):
                self.tmpdirname = tmpdirname
                all_files = glob.glob(f'{benchmark_dir}/*')
                [sh.cp('-r', f, tmpdirname) for f in all_files]
                self.build_exact()
                exact_out = self.run_exact(self.get_run_command())
                exact_processed = self.process_raw_data(exact_out,
                                                        is_approx=False
                                                        )

        loss = self.config['comparison_args']['loss_fn']
        loss_fn = get_loss_fn_from_str(loss)
        error = self.get_error(exact_processed,
                               approx_processed,
                               loss_fn
                               )
        print(f'The error is {error}')
        speedup = self.get_speedup(exact_processed, 
                                   approx_processed
                                   )


        return self.combine_error_speedup(speedup, error)
    
    def get_run_command(self):
        return self.config['run_command']

    def run_approx(self, run_command):
        # TODO: This is not from the config: need to get the number
        #from somewhere else
        # because each run of this will have a different one
        os.environ['SURROGATE_MODEL'] = self.config['surrogate_model']
        os.environ['HPAC_DB_FILE'] = DEFAULT_APPROX_H5
        os.environ['CAPTURE_OUTPUT'] = '1'
        return self.run(run_command)

    def run_exact(self, run_command):
        os.environ['HPAC_DB_FILE'] = DEFAULT_EXACT_H5
        os.environ['CAPTURE_OUTPUT'] = '1'
        return self.run(run_command)

    def run(self, cmd_str):
        cmd = self.create_command(cmd_str)
        buf = io.StringIO()
        cmd(_out=buf)
        buf.seek(0)
        print(buf.read())
        buf.seek(0)
        return buf.read()

    def create_command(self, cmd_str):
        spl = cmd_str.split()
        return sh.Command(spl[0]).bake(*spl[1:])

    def build_approx(self):
        sh.make('clean')
        sh.make('-f', 'Makefile.approx', 'CAPTURE_OUTPUT=1')

    def build_exact(self):
        sh.make('clean')
        sh.make('CAPTURE_OUTPUT=1')

    def combine_error_speedup(self, speedup, error):
        raise NotImplementedError

    # remove the approx and exact h5 files
    # def __del__(self):
    #     try:
    #         os.remove(DEFAULT_APPROX_H5)
    #         os.remove(DEFAULT_EXACT_H5)
    #     except FileNotFoundError:
    #         pass

class ProcessedResultsWrapper:
    def __init__(self, speedup=None, error=None):
        self.speedup = speedup
        self.error = error

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self)

    def get_error(self):
        return self.error
    
    def get_speedup(self):
        return self.speedup


class ParticleFilterProcessedResultsWrapper(ProcessedResultsWrapper):
    def __init__(self, speedup=None, error=None):
        super().__init__(speedup, error)


class ParticleFilterEvaluator(Evaluator):
    def __init__(self, config):
        super().__init__('particlefilter', config)

    class ParticleFilterResultsWrapper:
        def __init__(self, df):
            self.result = df

    def combine_error_speedup(self, speedup, error):
        pfr = ParticleFilterProcessedResultsWrapper
        return pfr(speedup=speedup.speedup, error=error.error)

    def process_raw_data(self, data_str, is_approx=False):
        data = io.StringIO()
        my_re = re.compile('DATA:(\\S+)')
        # get all matches
        matches = my_re.findall(data_str)
        for match in matches:
            data.write(match + '\n')
        data.seek(0)
        df = pd.read_csv(data, index_col='count')
        return self.ParticleFilterResultsWrapper(df)

    def get_error(self, ground_truth, approx, loss):
        pf_exact_results = ground_truth.result
        pf_approx_results = approx.result
        # get the column 'x0' and 'y0' from the exact results
        x0, y0 = pf_approx_results['x0'], pf_approx_results['y0']
        pf_approx_xe = pf_exact_results['xe']
        pf_approx_ye = pf_exact_results['ye']
        approx_xe, approx_ye = pf_approx_results['xe'], pf_approx_results['ye']

        # combine x0, y0 into [x0, y0]
        ground_truth = np.array([x0, y0]).T
        pf_stacked = np.array([pf_approx_xe, pf_approx_ye]).T
        approx_stacked = np.array([approx_xe, approx_ye]).T

        pf_error = loss(ground_truth, pf_stacked)
        approx_error = loss(ground_truth, approx_stacked)
        res = ParticleFilterProcessedResultsWrapper(error=(pf_error,
                                                           approx_error)
                                                    )
        return res

    def get_speedup(self, ground_truth, approx):
        pf_exact_results = ground_truth.result
        pf_approx_results = approx.result

        exact_ot = pf_exact_results['offload_time'][1::]
        approx_ot = pf_approx_results['offload_time'][1::]
        avg_speedup = exact_ot.mean()/approx_ot.mean()
        return ParticleFilterProcessedResultsWrapper(speedup=avg_speedup)


class HDF5ResultsWrapper:
    def __init__(self, filename, group_name):
        self._result = h5py.File(filename, 'r')
        self._result = self._result[group_name]['output']
        print(f'The shape is {self._result.shape}')

    @classmethod
    def get_error(cls, ground_truth, approx, loss):
        gt_result = ground_truth.result
        approx_result = approx.result
        return loss(gt_result, approx_result)

    @property
    def result(self):
        return np.array(self._result)

class MiniBUDEEvaluator(Evaluator):
    def __init__(self, config):
        super().__init__('minibude', config)

    class MiniBUDEResultsWrapper(HDF5ResultsWrapper):
        def __init__(self, filename, group_name, stdout):
            super().__init__(filename, group_name)
            self.stdout = stdout

    def get_run_command(self):
        trial_num = self.get_trial_num(self.config['surrogate_model'])
        num_items = self.get_num_items_for_trial(trial_num)
        cmd_args = self.get_data_gen_command(self.config['dataset_gen_command'],
                                             num_items
                                        )
        return ' '.join(cmd_args)

    def get_trial_num(self, model_path):
        path = Path(model_path)
        model_re = re.compile('model_(\d+)_\d+')
        filename = path.stem
        trial_num = model_re.match(filename).group(1)
        return int(trial_num)

    def get_num_items_for_trial(self, trial_num):
        trials = self.config['all_trials_file']
        df = pd.read_csv(trials, index_col='trial')
        my_row = df.loc[trial_num]
        multiplier = int(my_row['multiplier'])
        return multiplier

    def get_data_gen_command(self, args, num_items):
        args += ['--ni', str(num_items)]
        return args

    def process_raw_data(self, data_str, is_approx=False):
        rgn_name = self.config['comparison_args']['region_name']
        if is_approx:
            fname = DEFAULT_APPROX_H5
        else:
            fname = DEFAULT_EXACT_H5

        return self.MiniBUDEResultsWrapper(fname,
                                           rgn_name,
                                           data_str
                                           )

    def get_error(self, ground_truth, approx, loss):
        return HDF5ResultsWrapper.get_error(ground_truth, approx, loss)
