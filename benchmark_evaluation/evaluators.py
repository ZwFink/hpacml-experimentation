import pandas as pd
import sh
import numpy as np
import h5py
import io
import os
import tempfile
import glob
import re


DEFAULT_APPROX_H5 = f'/tmp/hpac_db_approx_{os.getpid()}.h5'
DEFAULT_EXACT_H5 = f'/tmp/hpac_db_exact_{os.getpid()}.h5'

def RMSE(ground_truth, approx):
    return np.sqrt(np.mean((ground_truth - approx)**2))

class Evaluator:
    def __init__(self, benchmark, config):
        self.benchmark = benchmark
        self.config = config

    @classmethod
    def get_evaluator_class(cls, benchmark):
        if benchmark == 'particlefilter':
            return ParticleFilterEvaluator
        else:
            raise ValueError('Unknown benchmark')

    def run_and_compute_error(self):
        benchmark_dir = self.config['benchmark_location']
        with tempfile.TemporaryDirectory() as tmpdirname:
            with sh.pushd(tmpdirname):
                self.tmpdirname = tmpdirname
                all_files = glob.glob(f'{benchmark_dir}/*')
                [sh.cp('-r', f, tmpdirname) for f in all_files]
                self.build_approx()
                approx_out = self.run_approx(self.config['run_command'])
                approx_processed = self.process_raw_data(approx_out)

        with tempfile.TemporaryDirectory() as tmpdirname:
            with sh.pushd(tmpdirname):
                self.tmpdirname = tmpdirname
                all_files = glob.glob(f'{benchmark_dir}/*')
                [sh.cp('-r', f, tmpdirname) for f in all_files]
                self.build_exact()
                exact_out = self.run_exact(self.config['run_command'])
                exact_processed = self.process_raw_data(exact_out)

        loss = self.config['comparison_args']['loss_fn']
        return self.get_error(exact_processed,
                              approx_processed,
                              RMSE
                              )

    def run_approx(self, run_command):
        # TODO: This is not from the config: need to get the number
        #from somewhere else
        # because each run of this will have a different one
        os.environ['SURROGATE_MODEL'] = self.config['surrogate_model']
        os.environ['HPAC_DB_FILE'] = DEFAULT_APPROX_H5
        return self.run(run_command)

    def run_exact(self, run_command):
        os.environ['HPAC_DB_FILE'] = DEFAULT_EXACT_H5
        return self.run(run_command)

    def run(self, cmd_str):
        cmd = self.create_command(cmd_str)
        buf = io.StringIO()
        cmd(_out=buf)
        buf.seek(0)
        return buf.read()

    def create_command(self, cmd_str):
        spl = cmd_str.split()
        return sh.Command(spl[0]).bake(*spl[1:])

    def build_approx(self):
        sh.make('clean')
        sh.make('-f', 'Makefile.approx')

    def build_exact(self):
        sh.make('clean')
        sh.make()


class ParticleFilterEvaluator(Evaluator):
    def __init__(self, config):
        super().__init__('particlefilter', config)

    class ParticleFilterResultsWrapper:
        def __init__(self, df):
            self.result = df

    class ParticleFilterProcessedResultsWrapper:
        def __init__(self, speedup=None, error=None):
            self.speedup = speedup
            self.error = error

        def get_error(self):
            return self.error
    
        def get_speedup(self):
            return self.speedup

    def process_raw_data(self, data_str):
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
        res = self.ParticleFilterProcessedResultsWrapper(error=(pf_error,
                                                         approx_error)
                                                         )
        return res


