import click
from evaluators import Evaluator
import yaml


@click.command()
@click.option('--benchmark', type=click.Choice(['particlefilter']))
@click.option('--config', type=click.Path(exists=True))
def main(benchmark, config):
    with open(config, 'r') as f:
        config = yaml.safe_load(f)

    benchmark_config = config[benchmark]

    evaluator = Evaluator.get_evaluator_class(benchmark)(benchmark_config)
    wrapped_result = evaluator.run_and_compute_error()
    print(wrapped_result.get_error())
    print(wrapped_result.get_speedup())


if __name__ == '__main__':
    main()
