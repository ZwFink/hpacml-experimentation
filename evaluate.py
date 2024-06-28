import click
import sh
import os
import time

large_times = {
    'minibude': 240,
    'binomialoptions': 28,
    'bonds': 13,
    'miniweather': 30,
    'particlefilter': 50,
}

small_times = {
    'minibude': 23,
    'binomialoptions': 7,
    'bonds': 8,
    'miniweather': 2.5,
    'particlefilter': 15,
}

class EvalCommand:
    def __init__(self, benchmark, size):
        self.benchmark = benchmark
        self.size = size

    def _get_config_name(self):
        if self.size == 'small':
            return 'config_small.yaml'
        elif self.size == 'large':
            return 'config.yaml'

    def _get_output_fname(self):
        if self.size == 'small':
            return f'{self.benchmark}_small.csv'
        elif self.size == 'large':
            return f'{self.benchmark}.csv'


    def get_command(self):
        eval_path = 'benchmark_evaluation'
        result_path = f'{eval_path}/results'
        config_name  = self._get_config_name()
        output_fname = self._get_output_fname()
        cmd = [f'{eval_path}/driver.py',
               '--benchmark', self.benchmark,
               '--config', f'{eval_path}/{config_name}',
               '--output', f'{result_path}/{output_fname}',
               '--local']
        return sh.Command('python3').bake(cmd)


class PlotCommand:
    def __init__(self, benchmark, size, output_dir):
        self.benchmark = benchmark
        self.size = size
        self.output_dir = output_dir


    def get_command(self):
        plot_path = 'analysis'
        config_name = f'{plot_path}/config.yaml'
        args = ['--config', config_name,
                '--benchmark', self.benchmark,
                '--output', f'{self.output_dir}']
        if self.size == 'small':
            args.append('--small')
        return sh.Command('python3').bake([f'{plot_path}/plotting.py'] + args)



@click.command()
@click.option('--benchmark', type=click.Choice(['binomialoptions', 'minibude', 'miniweather', 'particlefilter', 'bonds']), required=True)
@click.option('--size', type=click.Choice(['small', 'large']), required=True)
@click.option('--output', type=click.Path(), help='Name of directory to write output plot to.', default='plots')
def main(benchmark, size, output):
    cmd = EvalCommand(benchmark, size).get_command()
    print(f'Running {size} evaluation for {benchmark} benchmark...')
    print(f'Expected runtime on A100 evaluation platform: {small_times[benchmark] if size == "small" else large_times[benchmark]} minutes')
    tst = time.time()
    cmd(_out=print, _err=print)
    print(cmd)
    tend = time.time()
    print(f'Evaluation completed in {(tend - tst)/60} minutes.')

    print(f'Creating plot for {benchmark} benchmark...')
    # change output to absolute
    output = os.path.abspath(output)
    sh.mkdir('-p', output)
    PlotCommand(benchmark, size, output).get_command()(_out=print, _err=print)
    print(f'Created plot for {benchmark} benchmark.')


if __name__ == '__main__':
    main()