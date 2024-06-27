import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import yaml
import click
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import statsmodels.api as sm
import scipy
# import seaborn as sns
import matplotlib.pylab as plt
import os
# import pingouin as pg
from plotnine import *
# from plotnine_prism import *
import plotnine as pn
import sqlite3
# import patchworklib as pw
pn.options.dpi=150

# IMPORT MY LATEX SO I CAN USE \TEXTSC
import matplotlib as mpl
# mpl.rc('text', **{'usetex':True})
import re
plt.rc( 'font', family = 'serif')
# sns.set_style("whitegrid")
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

bo_bude_bonds_aspect_ratio=0.4

def combine_dfs(descriptor_dict):
    # rename the 'trial' column to 'Architecture'
    descriptor_dict['training_df'] = descriptor_dict['training_df'].rename(columns={'trial':'Architecture'})
    joined = descriptor_dict['eval_df'].merge(descriptor_dict['training_df'], left_on='Architecture', right_on='Architecture', how='left')
    approx_only = joined[joined['mode'] == 'Approx']
    # Turn this on only forn particlefilter (yes I know that's bad design)
    # The first few trials for PF have way higher runtimes and need to warmpu
    # approx_only = approx_only[approx_only['Trial'] < 1]
    mean_vals = approx_only.groupby(['Architecture', 'mode', 'error_metric']).mean().reset_index()
    return mean_vals


def pareto_frontier(df):
    # Sort the dataframe by descending speedup
    sorted_df = df.sort_values('avg_speedup', ascending=False).reset_index(drop=True)

    # Initialize the Pareto frontier with the first point
    pareto_frontier = [sorted_df.iloc[0]]

    # Iterate over the sorted dataframe to identify the Pareto optimal points
    for _, row in sorted_df.iterrows():
        speedup = row['avg_speedup']
        error = row['error']

        # Check if the current point is dominated by any of the points in the Pareto frontier
        is_dominated = False
        for pf_point in pareto_frontier:
            # if pf_point['speedup'] >= speedup and pf_point['error'] <= error:
            if (False) or pf_point['error'] <= error:
                is_dominated = True
                break

        # If the point is not dominated, add it to the Pareto frontier
        if not is_dominated:
            pareto_frontier.append(row)

    # Convert the Pareto frontier list to a dataframe
    pareto_frontier_df = pd.DataFrame(pareto_frontier)

    return pareto_frontier_df

def savefig(filepath, fig):
    pdf = f'{filepath}.pdf'
    png = f'{filepath}.png'
    
    # fig.save(pdf, bbox_inches='tight', dpi=300)
    fig.save(png, bbox_inches='tight', dpi=300)


def plot_minibude(files):
    bude_data = files
    
    bude_data = {  
        'eval_df': pd.read_csv(bude_data['eval_file']),
        'training_df': pd.read_csv(bude_data['training_file'])
    }
    
    
    bude_data['training_df'].duplicated(subset=['feature_multiplier', 'multiplier', 'num_hidden_layers', 'hidden_1_features']).sum()
    bude_data['eval_df']
    
    first_df = combine_dfs(bude_data)
    first_df = first_df[first_df['mode'] == 'Approx']
    
    plot_df = first_df.copy()
    plot_df['num_params'] /= 1000
    plot_df['num_params'] = plot_df['num_params'].apply(lambda x: round(x))
    mult1 = plot_df[plot_df['multiplier'] == 1]
    plot_df = mult1
    fastest_model = plot_df['Forward Pass'].max()
    plot_df['relative_inference_time'] =   fastest_model / plot_df['Forward Pass']
    largest_model = plot_df['num_params'].min()
    plot_df['relative_num_params'] = plot_df['num_params'] / largest_model
    print(plot_df)
    p1=(
        ggplot(plot_df, aes(x='error', y='avg_speedup', label='num_params')) + theme_seaborn("whitegrid")
        + geom_point(aes(fill='relative_num_params'), stroke=0.2, color='black', size=12)#, size=8)
        + labs(fill='Relative Model Size', shape='Hidden Layers')
        + ylab('Speedup') + xlab('MAPE ($\%$)')
        + theme(axis_text=element_text(size=17), axis_title=element_text(size=20), legend_title=element_text(size=15), legend_text=element_text(size=14))
        + theme(legend_title_position=('top'), legend_direction=('horizontal'), legend_position=('top'), legend_background=element_rect(color='white'), legend_box_margin=3.2, legend_margin=0.3, legend_entry_spacing=0.1, legend_key_size=15)
        + scale_fill_continuous('viridis_r', breaks=[1, 200, 400], limits=(1,400))
        + theme(panel_grid_major=element_line(size=0.5, color='gray', linetype='dashed', alpha=0.3), panel_grid_minor=element_blank())
        + theme(aspect_ratio=bo_bude_bonds_aspect_ratio)
    )

    return p1


def plot_binomialoptions(files):
  bo_data = files
  
  bo_data = {  
      'eval_df': pd.read_csv(bo_data['eval_file']),
      'training_df': pd.read_csv(bo_data['training_file'])
  }
  bo_data ['eval_df'] = bo_data['eval_df'][bo_data['eval_df']['mode'] == 'Approx']
  bo_df = combine_dfs(bo_data)

  plot_df = bo_df.copy()
  
  plot_df = pareto_frontier(plot_df)
  plot_df['num_params'] = plot_df['num_params'].apply(lambda x: round(x))
  slowest_model = plot_df['Forward Pass'].max()
  plot_df['relative_inference_time'] =   slowest_model / plot_df['Forward Pass']
  smallest_model = plot_df['num_params'].min()
  plot_df['relative_num_params'] = plot_df['num_params'] / smallest_model
  p1=(
      ggplot(plot_df, aes(x='error', y='avg_speedup', label='num_params')) + theme_seaborn("whitegrid")
      + geom_point(aes(fill='relative_num_params'), stroke=0.2, color='black', size=12, alpha=0.75)
      # + geom_point(stroke=0.2, color='black')
      + labs(fill='Relative Model\nSize')#, size='Relative Model Size')
      + ylab('Speedup') + xlab('RMSE')
      # set the breaks for the y axis
      + theme(axis_text=element_text(size=17), axis_title=element_text(size=20), legend_title=element_text(size=15), legend_text=element_text(size=14))
      + theme(legend_title_position=('top'), legend_direction=('horizontal'), legend_position=('top'), legend_background=element_rect(color='white'), legend_box_margin=3.2, legend_margin=0.3, legend_entry_spacing=0.1, legend_key_size=15)
      + scale_fill_continuous('viridis_r', limits=(1,80), labels=[1, 40, 80], breaks = [1, 40, 80])
      + theme(panel_grid_major=element_line(size=0.5, color='gray', linetype='dashed', alpha=0.3), panel_grid_minor=element_blank())
      + theme(aspect_ratio=bo_bude_bonds_aspect_ratio)
  )
  return p1

def plot_particlefilter(files):
    pf_data = files
    pf_data = {  
        'eval_df': pd.read_csv(pf_data['eval_file']),
        'training_df': pd.read_csv(pf_data['training_file'])
    }
    pf_data ['eval_df'] = pf_data['eval_df'][pf_data['eval_df']['mode'] == 'Approx']
    pf_df = combine_dfs(pf_data)
    pf_df = pf_df[pf_df['mode'] == 'Approx']

    plot_df = pf_df.copy()
    plot_df = plot_df.query('error <= 1')

    plot_df['num_params'] = plot_df['num_params'].apply(lambda x: round(x))
    slowest_model = plot_df['Forward Pass'].max()
    plot_df['relative_inference_time'] =   slowest_model / plot_df['Forward Pass']
    smallest_model = plot_df['num_params'].min()
    plot_df['relative_num_params'] = plot_df['num_params'] / smallest_model
    # convert 'conv_stride' column to int
    plot_df['conv_stride'] = plot_df['conv_stride'].apply(lambda x: int(x))
    p1=(
        ggplot(plot_df, aes(x='error', y='avg_speedup', label='num_params')) + theme_seaborn("whitegrid")
        + geom_point(aes(fill='relative_num_params'), stroke=0.2, color='black', size=8, alpha=0.75)
        # + geom_point(stroke=0.2, color='black')
        + labs(fill='Relative Model Size', shape='Conv. Stride') #, size='Relative Model Size')
        + ylab('Speedup') + xlab('RMSE')
        + theme(axis_text=element_text(size=17), axis_title=element_text(size=20), legend_title=element_text(size=12), legend_text=element_text(size=12))
        + theme(legend_title_position=('top'), legend_direction=('horizontal'), legend_position=('top'), legend_background=element_rect(color='white'), legend_box_margin=3.2, legend_margin=0.3, legend_entry_spacing=0.1, legend_key_size=15)
        + scale_fill_continuous('viridis_r', labels=[1, 10, 20], breaks=[1, 10, 20]) #, breaks=[1, 2, 3], limits=[1,3])#, breaks=[1, 1.5, 2], labels=[1,1.5,2])
        + theme(panel_grid_major=element_line(size=0.5, color='gray', linetype='dashed', alpha=0.3), panel_grid_minor=element_blank())
        # I manually calculated this x-intercept because the original PF cannot run successfully for more than one trial
        # in the same process???
        + geom_vline(xintercept=0.4766649925261974, linetype='dashed', color='black', size=0.5)
        + theme(aspect_ratio=0.52)
    )

    return p1

def plot_bonds(files):
  bonds_data = files
  bonds_data = {  
      'eval_df': pd.read_csv(bonds_data['eval_file']),
      'training_df': pd.read_csv(bonds_data['training_file'])
  }
  bonds_data ['eval_df'] = bonds_data['eval_df'][bonds_data['eval_df']['mode'] == 'Approx']
  bonds_df = combine_dfs(bonds_data)
  unique_on_columns = ['multiplier', 'hidden1_features', 'hidden2_features']
  bonds_df = bonds_df.sort_values('error').drop_duplicates(subset=unique_on_columns, keep='first')

  plot_df = bonds_df.copy()
  plot_df = plot_df[plot_df['multiplier'] == 1]
  plot_df = plot_df[plot_df['error'] < 2]
  plot_df = plot_df[plot_df['num_params'] != plot_df['num_params'].max()]
  
  
  plot_df['num_params'] = plot_df['num_params'].apply(lambda x: round(x))
  slowest_model = plot_df['Forward Pass'].max()
  plot_df['relative_inference_time'] =   slowest_model / plot_df['Forward Pass']
  smallest_model = plot_df['num_params'].min()
  plot_df['relative_num_params'] = plot_df['num_params'] / smallest_model

  p1=(
      ggplot(plot_df, aes(x='error', y='avg_speedup', label='num_params')) + theme_seaborn("whitegrid")
      + geom_point(aes(fill='relative_num_params'), stroke=0.2, color='black', size=12, alpha=0.75)
      + ylab('Speedup') + xlab('RMSE')
      + theme(axis_text=element_text(size=17), axis_title=element_text(size=20), legend_title=element_text(size=13), legend_text=element_text(size=13))
      + theme(legend_direction=('horizontal'), legend_title_position=('top'), legend_position=('top'), legend_background=element_rect(color='white'), legend_box_margin=3.2, legend_margin=0.3, legend_entry_spacing=0.1, legend_key_size=15)
      + scale_fill_continuous('viridis_r', labels=[1, 100, 200, 300], breaks=[1, 100, 200, 300], limits=(1,300))#, breaks=[1, 2, 3], limits=[1,3])#, breaks=[1, 1.5, 2], labels=[1,1.5,2])
      + labs(fill='Relative Model Size')
      + theme(panel_grid_major=element_line(size=0.5, color='gray', linetype='dashed', alpha=0.3), panel_grid_minor=element_blank())
      + theme(aspect_ratio=bo_bude_bonds_aspect_ratio)
  )
  return p1


def plot_miniweather(files):
  mw_data = files  
  mw_data = {  
      'eval_df': pd.read_csv(mw_data['eval_file']),
      'training_df': pd.read_csv(mw_data['training_file'])
  }
  mw_data['training_df'] = mw_data['training_df'].drop(columns=['activation_function'])
  
  mw_data ['eval_df'] = mw_data['eval_df'][mw_data['eval_df']['mode'] == 'Approx']
  
  mw_df = combine_dfs(mw_data)
  mw_df = mw_df.drop(columns=['conv1_out_channels'])
  unique_on_columns = ['conv1_stride', 'conv1_kernel_size', 'conv2_kernel_size', 'batchnorm']
  mw_df = mw_df.sort_values('error').drop_duplicates(subset=unique_on_columns, keep='first')

  plot_df = mw_df.copy()
  
  plot_df = plot_df[plot_df['error'] != float('inf')]
  plot_df = plot_df[plot_df['error'] < 50]
  
  plot_df['num_params'] = plot_df['num_params'].apply(lambda x: round(x))
  smallest_model = plot_df['num_params'].min()
  plot_df['relative_num_params'] = plot_df['num_params'] / smallest_model
  p1=(
      ggplot(plot_df, aes(x='error', y='avg_speedup', label='num_params')) + theme_seaborn("whitegrid")
      + geom_point(aes(fill='relative_num_params'), stroke=0.2, color='black', size=8, alpha=0.75)
      + ylab('Speedup') + xlab('RMSE')
      + xlim(0, 50)
      + theme(axis_text=element_text(size=17), axis_title=element_text(size=20), legend_title=element_text(size=15), legend_text=element_text(size=14))
      + theme(legend_direction=('horizontal'), legend_title_position=('top'), legend_position=(0.322, 0.295), legend_background=element_rect(color='white'), legend_box_margin=3.2, legend_margin=0.3, legend_entry_spacing=0.1, legend_key_size=15)
      + scale_fill_continuous('viridis_r')#, labels=[1, 20, 40, 60], breaks=[1, 20, 40, 60])#, breaks=[1, 2, 3], limits=[1,3])#, breaks=[1, 1.5, 2], labels=[1,1.5,2])
      + theme(panel_grid_major=element_line(size=0.5, color='gray', linetype='dashed', alpha=0.3), panel_grid_minor=element_blank())
      + labs(fill='Relative Model Size')
  )
  return p1

@click.command()
@click.option('--config', default='config.yaml', help='Path to the configuration file')
@click.option('--benchmark', type=click.Choice(['minibude', 'particlefilter', 'binomialoptions', 'bonds', 'miniweather']))
@click.option('--small', is_flag=True, help='Use the small dataset')
def main(config, benchmark, small):
    config = yaml.load(open(config), Loader=yaml.FullLoader)
    config = config[benchmark]

    if small:
        files = config['small']
    else:
        files = config['large']

    if benchmark == 'minibude':
        plot = plot_minibude(files)
    elif benchmark == 'binomialoptions':
        plot = plot_binomialoptions(files)
    elif benchmark == 'particlefilter':
        plot = plot_particlefilter(files)
    elif benchmark == 'bonds':
        plot = plot_bonds(files)
    elif benchmark == 'miniweather':
        plot = plot_miniweather(files)

    savefig(f'plots/{benchmark}', plot)

if __name__ == '__main__':
    main()