# Model Evaluation
This step takes the models trained by the Neural Architecture search and tests them on the unseen validation data set.
It runs the application approximated with HPAC-ML and the original application, capturing the output values and computing speedup and error.

Each application can be run individually using the provided `driver.py` script.
An example use of this script is:
```python
python3 driver.py --config config.yaml --benchmark bonds --output bonds.csv --local
```

This uses the configuration specified for the `bonds` benchmark in the `config.yaml` file.
Each model located in the corresponding `models_directory` will be used for the above process.
Passing `--local` means Parsl will not submit parallel jobs to the Slurm cluster.
To avoid using `--local` and parallelize the runs for each model across different parallel jobs, you will need to edit the `SlurmProvider` configuration in `driver.py` for compatibility with your cluster.
However, the **SlurmProvider is not currently supported in the container**.


The `driver.py` script will run `evaluators.py` for each benchmark and model and combine the results into a single file that is used to analyze data and create plots.

The output file can be fed directly to the plot script.

## Output Format
The output contains one line for each of 20 trials run for the original (Exact) and approximated (Approx benchmark).
Entries in the file look like this:
```csv
HtoD,DtoH,Trial,mode,To Tensor,Wrap output Memory,Forward Pass,From Tensor,avg_speedup,error,error_metric,Architecture
126.447,5.24525,834.418,Exact,,,,,6.0559312726548065,0.41337475,rmse,17
125.728,3.17427,136.463,Approx,3.67766,0.007168,3.69254,0.113664,6.0559312726548065,0.41337475,rmse,17
```

All times are in ms.
The HtoD/DtoH colums are for time spent transferring application data between device and host.
The Trial column is the runtime of the entire trial.
Mode is whether the trial is for an exact or approximated run of the benchmark.
Forward Pass times NN inference.
To/From Tensor times copies between NN tensors and application memory.
Wrap output Memory considers only the time spent *wrapping* application output memory, e.g., the `from_blob` call, but not the memory copies.
Average speedup is the speedup computed when comparing the average `Trial` time between exact/approx, after dropping the first two instances of each for warmup.
Architecture is the NN architecture index.

The csv output can be fed directly to the plotting scripts to create the plots for a benchmark.


## Benchmark Runtimes
We offer two configurations for evaluation trained models in the application.
**Large configuration**: this includes all of the trained models for each benchmark.

**Small configuration**: this includes only the models found in the plots for each benchmark.
For Binomial Options and MiniBUDE, this includes the Pareto-optimal models.
For MiniBUDE, we run 8 trials of each configuration, rather than the 20 used for the paper to reduce time-to-result.
For Bonds, all models are included.
MiniWeather includes only those models whose RMSE was less than 50.
ParticleFilter includes models whose RMSE is less than 1.

The runtimes for each configuration are shown in the table below.
|     Benchmark    | Large (m) | Small (m) |
|:----------------:|:---------:|:---------:|
|     MiniBUDE     |    240    |     23    |
| Binomial Options |     28    |     7     |
|       Bonds      |     13    |     8     |
|    MiniWeather   |     30    |    2.5    |
|  ParticleFilter  |    100    |     30    |

For MiniWeather, this script does not perform the analysis shown in Figure 9(d).
Instead, it runs all trained MiniWeather models doing auto-regressive inference where the model is used for all timesteps.
This result is not shown in the paper. Instead, 
Figure 9(d) shows different configurations of interleaving model inference with timesteps of the original application, using a few of the most accurate models.
This interleaving substantially reduces error, but has not been automated and the process is left out of this repository.
We have, however, included the notebook `plotting.ipynb` in the directory `plotting` with the code that creates all MiniWeather plots.

## Skipped Trials

Note that some trials may be skipped with the message:
```bash
Skipping trial 14 for binomialoptions benchmark. 
(The model architecture was likely invalid, so the model did not train).
```
This means that the architecture $14$ proposed by the Bayesian Optimization search was invalid.
This could be caused by an invalid combination of architecture parameters or a large model that uses too much memory for instance.
This message does **not** mean that something has gone wrong in the current model evaluation step.

