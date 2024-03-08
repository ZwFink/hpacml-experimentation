import click
import os
import re
from evaluators import Evaluator
import yaml
import pandas as pd


def evaluate(benchmark, config, trial_num):
    import yaml
    from evaluators import Evaluator
    with open(config, 'r') as f:
        config = yaml.safe_load(f)

    benchmark_config = config[benchmark]
    trials_file = benchmark_config['all_trials_file']
    trials_df = pd.read_csv(trials_file, index_col='trial')
    models_dir = benchmark_config['models_directory']
    # iterate of the index col of trials_df
    model_path = get_model_for_trial(trial_num, models_dir)
    evaluator = Evaluator.get_evaluator_class(benchmark)(benchmark_config,
                                                          model_path
                                                          )
    wrapped_result, events = evaluator.run_and_compute_speedup_and_error(get_events=True)
    events['avg_speedup'] = wrapped_result.get_speedup()
    events['error'] = wrapped_result.get_error()
    events['error_metric'] = wrapped_result.get_error_metric()
    events['Trial'] = trial_num
    print(events)
    return events

def get_model_for_trial(trial, models_dir):
    model_re = re.compile(f'model_{trial}_\d+.pth')
    matched_files = [f for f in os.listdir(models_dir) if model_re.match(f)]

    if len(matched_files) == 1:
        return os.path.join(models_dir, matched_files[0])
    elif len(matched_files) > 1:
        raise ValueError(f"More than one file matches the trial {trial} pattern.")
    else:
        return None



@click.command()
@click.option('--benchmark', type=click.Choice(['particlefilter', 'minibude', 'binomialoptions']))
@click.option('--config', type=click.Path(exists=True))
@click.option('--output', type=click.Path())
def main(benchmark, config, output):
    config_file = config
    with open(config, 'r') as f:
        config = yaml.safe_load(f)

    benchmark_config = config[benchmark]
    trials_file = benchmark_config['all_trials_file']
    trials_df = pd.read_csv(trials_file, index_col='trial')

    all_trials_df = pd.DataFrame()

    for trial_num in trials_df.index:
        trial_results = evaluate(benchmark, config_file, trial_num)
        all_trials_df = pd.concat([all_trials_df, trial_results],
                                  ignore_index=True
                                  )

    all_trials_df.to_csv(output, index=False)

if __name__ == '__main__':
    main()
