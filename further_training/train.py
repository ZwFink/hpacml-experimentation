import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset
import click
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import h5py
from torch.utils.data import SubsetRandomSampler
import sys
import glob
import yaml
import os
import time
import sh
import numpy as np
import pandas as pd
import numpy as np
import tempfile
from dataclasses import dataclass
from pathlib import Path
import math
import scipy
from nns import *

DATATYPE = torch.float32
BenchSpec = None

class HDF5DataSet(Dataset):
    def __init__(self, file_path, group_name, ipt_dataset, opt_dataset, train_end_index = np.inf, target_type=None, load_entire = False):
        self.open_file = h5py.File(file_path, 'r')
        self.file_path = file_path
        self.target_type = target_type
        self.train_end_index = train_end_index
        self.load_entire = load_entire

        self.ipt_dataset, self.opt_dataset = self.get_datasets(group_name, ipt_dataset, opt_dataset)
        if target_type is None:
            self.target_dtype = torch.from_numpy(np.array([], dtype=self.opt_dataset.dtype)).dtype
        else:
            self.target_dtype = target_type
        assert len(self.ipt_dataset) == len(self.opt_dataset)
        self.length = len(self.ipt_dataset)
        if load_entire:
            tgt = self.target_type
            self.ipt_dataset = torch.tensor(self.ipt_dataset).to(tgt)
            self.opt_dataset = torch.tensor(self.opt_dataset).to(tgt)

    def get_datasets(self, group_name, ipt_dataset, opt_dataset):
        group = self.open_file[group_name]
        ipt_dataset = group[ipt_dataset]
        opt_dataset = group[opt_dataset]
        if (ipt_dataset.shape[0] == 1):
            print(f"WARNING: Found left dimension of 1 in shape {ipt_dataset.shape},"
                  f" assuming this is not necessary and removing it."
                  f" Reshaping to {ipt_dataset.shape[1:]}"
                  )
            ipt_dataset = ipt_dataset[0]
            opt_dataset = opt_dataset[0]

        if self.train_end_index == np.inf:
            self.train_end_index = ipt_dataset.shape[0]
        ipt_dataset = ipt_dataset[:self.train_end_index]
        opt_dataset = opt_dataset[:self.train_end_index]
        return ipt_dataset, opt_dataset

    def set_for_predicting_multiple_instances(self, n_instances):
        ipt_shape = self.ipt_dataset.shape
        opt_shape = self.opt_dataset.shape
        assert len(ipt_shape) == 2, "Input dataset must have 2 dimensions for compatibility with multiple instances"

        # Calculate the maximum number of full instances that can be formed
        max_full_ipt_instances = (ipt_shape[0] // n_instances) * n_instances
        max_full_opt_instances = (opt_shape[0] // n_instances) * n_instances

        # Slice the dataset to include only complete groups
        self.ipt_dataset = self.ipt_dataset[:max_full_ipt_instances].reshape((-1, ipt_shape[1] * n_instances))
        self.opt_dataset = self.opt_dataset[:max_full_opt_instances].reshape((-1, opt_shape[1] * n_instances))

        self.length = self.ipt_dataset.shape[0]

    def input_as_torch_tensor(self):
        if self.load_entire:
            return self.ipt_dataset.clone().detach()
        else:
            return torch.tensor(self.ipt_dataset).to(self.target_type)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.load_entire:
            return (self.ipt_dataset[idx].clone().detach().to(self.target_type),
                    self.opt_dataset[idx].clone().detach().to(self.target_type)
                )
        else:
            return (torch.tensor(self.ipt_dataset[idx]).to(self.target_type),
                    torch.tensor(self.opt_dataset[idx]).to(self.target_type))


class GeneratorHDF5DataSet(Dataset):
    def __init__(self, generation_command,
                 target_path, group_name,
                 ipt_dataset, opt_dataset, target_type=None
                 ):
        generation_command()
        super().__init__(target_path, group_name, 
                         ipt_dataset, opt_dataset, target_type
                         )


class DummyScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self, loss):
        pass

    def get_last_lr(self):
        return 0

def MAPE(actual, forecast):
    if actual.shape != forecast.shape:
        raise ValueError("The shape of actual and forecast tensors must be the same.")

    epsilon = 1e-8  # small constant to avoid division by zero
    mape = torch.mean(torch.abs((actual - forecast) / (actual + epsilon))) * 100
    return mape


class EarlyStopper:
    def __init__(self, patience=1, min_delta_percent=1):
        self.patience = patience
        self.min_delta_percent = min_delta_percent / 100.0  # Convert percentage to decimal
        self.counter = 0
        self.best_loss = None  # Change from float('inf') to None

    def early_stop(self, validation_loss):
        if self.best_loss is None:  # Initial case where best_loss is not set
            self.best_loss = validation_loss
            return False

        improvement = (self.best_loss - validation_loss) / self.best_loss
        if improvement >= self.min_delta_percent:  # Check if improvement meets threshold
            self.best_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



def train_loop(writer, dataloader, model, loss_fn, optimizer, scheduler, epoch):
    size = len(dataloader.dataset)
    model = model.to(device)
    test_loss_mape = 0
    for batch, dat in enumerate(dataloader):
        X = dat[0]
        y = dat[1]
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        test_loss_mape += MAPE(y, pred).item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch*len(X)
            # writer.add_scalar('training loss', loss, epoch*len(dataloader) + batch)
            # print(f"Epoch: {epoch}, loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
    print(f"(Training) Epoch: {epoch}, loss: {test_loss_mape/len(dataloader):>7f} [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss = 0
    test_loss_mape = 0
    num_batches = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            test_loss_mape += MAPE(y, pred).item()
            num_batches += 1
            if num_batches > dataloader.max_batches:
                break

    print(f"Test Error: \n Avg loss: {test_loss / num_batches:>8f}, Avg MAPE: {test_loss_mape / num_batches:>8f}")
    return test_loss / num_batches, test_loss_mape / num_batches


def infer_loop(model, dataloader, trials, writer=None):

    read_time = time.time()
    X = BenchSpec.get_infer_data_from_dl(dataloader)
    print("Read time: ", time.time() - read_time)
    X = X.to(device)
    print("Size of the data for inference:", X.shape)

    total_time = 0
    with torch.no_grad():
        with torch.jit.optimized_execution(True):
            model = model.to(DATATYPE)
            model.eval()
            traced_script_module = torch.jit.trace(model, X)
            model = traced_script_module
            model = torch.jit.freeze(model)
        model = model.to(DATATYPE)
        model.eval()
        model.to(device)

        X = X.to(device)

        times = np.zeros(trials, dtype=np.float64)
        warmup_time = time.time()
        # Do warmup
        pred = model(X)
        pred = model(X)
        torch.cuda.synchronize()
        print("Warmup time: ", time.time() - warmup_time)


        for i in range(trials):
            start = time.time()
            pred = model(X)
            torch.cuda.synchronize()
            end = time.time()
            times[i] = (end - start) * 1000
    sem = scipy.stats.sem(times)
    sem = sem.item()
    mean = np.mean(times).item()

    average_time = total_time / trials
    if writer:
      writer.add_scalar('inference time', average_time*1000, 0)
    return model, mean, sem

def train_test_infer_loop(nn_class, train_dl, test_dl, early_stopper, arch_params, hyper_params):
    learning_rate = hyper_params.get("learning_rate")
    epochs = int(hyper_params.get("epochs"))
    batch_size = int(hyper_params.get("batch_size"))
    weight_decay = hyper_params.get("weight_decay")

    if 'dropout' in hyper_params:
        arch_params['dropout'] = hyper_params['dropout']

    model = nn_class(arch_params).to(DATATYPE).to(device)
    model.calculate_and_save_normalization_parameters(train_dl)

    print(model)
    writer = None
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=learning_rate,
                                  weight_decay=weight_decay
                                  )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           'min'
                                                           )
    best_test_loss = np.inf
    model_epoch = 0
    best_model = None

    for t in range(model_epoch, epochs):
        epoch_start = time.time()
        tl_start = time.time()
        train_loop(writer, train_dl, model, loss_fn, optimizer, scheduler, t+1)
        tl_end = time.time()
        print(f"Training time: {tl_end - tl_start}")
        test_loop_start = time.time()
        test_loss, test_loss_mape = test_loop(test_dl, model, loss_fn)
        test_loop_end = time.time()
        scheduler.step(test_loss)
        print(f'Learning rate: {scheduler.get_last_lr()}')
        print(f"Test loop time: {(test_loop_end - test_loop_start)}")
        epoch_time = time.time() - epoch_start
        print(f"Epoch {t+1}\n-------------------------------")
        print(f"Test Error: \n Avg loss: {test_loss:>8f}, Time: {epoch_time:>8f}\n")

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            infer_start = time.time()
            model_optimized, infer_time, infer_sem = infer_loop(model, test_dl, 100, writer)
            infer_end = time.time()
            print(f"Inference time: {(infer_end - infer_start)}")
            best_model = model_optimized
        if early_stopper.early_stop(test_loss):
            print(f'Early stopping at epoch {t+1} after max patience reached.')
            break

    # YAML doesn't know how to handle tuples, so we return a list
    return best_model, best_test_loss, [infer_time, infer_sem]


class BenchmarkSpecifier:
    def __init__(self, name):
        self.name = name

    def get_nn_class(self):
        return None

    def get_target_type(self):
        return None

    def get_name(self):
        return self.name

    def get_infer_data_from_dl(self, dataloader):
        pass

    @classmethod
    def get_dataset_generator_class(cls):
        return BaseDatasetGenerator()

    @classmethod
    def get_specifier(cls, name):
        if name == 'particlefilter':
            return ParticleFilterSpecifier()
        elif name == 'miniweather':
            return MiniWeatherSpecifier()
        elif name == 'binomial_options':
            return BinomialOptionsSpecifier()
        elif name == 'bonds':
            return BondsOptionsSpecifier()
        elif name == 'minibude':
            data_gen = MiniBUDEOptionsSpecifier.get_dataset_generator_class()
            return MiniBUDEOptionsSpecifier(data_gen)


def get_all_infer_data(dataloader):
    data = list(dataloader)
    # concatenate everything
    item = torch.cat([x[0] for x in data])
    return item


class BaseDatasetGenerator:
    def __init__(self):
        pass

    def __call_(self):
        pass


class MiniBUDEDatasetGenerator(BaseDatasetGenerator):
    def __init__(self, datagen_args):
        super().__init__()
        self.args = datagen_args
        self.run_command = None
        self.db_file = None
        print("I got the args ", datagen_args)

    def __call__(self):
        return self.generate_dataset()

    def generate_dataset(self):
        # create a tmp directory for the build data
        args = self.args
        tmp_dir = tempfile.TemporaryDirectory()
        bm_dir = args['benchmark_location']
        files = map(lambda x: f'{x}/{bm_dir}', sh.ls(bm_dir))
        files = glob.glob(f'{bm_dir}/*')
        [sh.cp(file, tmp_dir.name) for file in files]
        with sh.pushd(tmp_dir.name):
            sh.make('-f', 'Makefile.approx', 'clean')
            sh.make('-f', 'Makefile.approx')
            cmd = self.create_run_command(tmp_dir.name, args)
            db_file = tempfile.NamedTemporaryFile()
            os.environ['HPAC_DB_FILE'] = db_file.name
            # Tie the lifetime of the db file
            # to our own
            self.db_file = db_file
            cmd(_out=print, _err=print)
        print(f'Generated dataset at {db_file.name}')
        return db_file.name

    def create_run_command(self, dir_name, arg_dict):
        dgen_cmd = arg_dict['dataset_gen_command']
        dgen_exe = dgen_cmd[0]
        dgen_args = dgen_cmd[1:]
        ni = arg_dict['multiplier']
        for i in range(len(dgen_args)):
            if dgen_args[i] == '--ni':
                dgen_args[i+1] = str(ni)
        exe_path = f'{dir_name}/{dgen_exe}'

        cmd = sh.Command(exe_path)
        cmd = cmd.bake(*dgen_args)
        print(cmd)
        return cmd



class MiniBUDEOptionsSpecifier(BenchmarkSpecifier):
    def __init__(self, data_gen):
        super().__init__('minibude')
        self.data_gen = data_gen

    def get_nn_class(self):
        return ConfigurableNumHiddenLayersMiniBUDENeuralNetwork

    def get_target_type(self):
        return None

    def get_infer_data_from_dl(self, dataloader):
        return self.get_infer_data_from_ds(dataloader.dataset)

    def get_infer_data_from_ds(self, dataset):
        return dataset.input_as_torch_tensor()

    @classmethod
    def get_dataset_generator_class(cls):
        return MiniBUDEDatasetGenerator


class MiniWeatherSpecifier(BenchmarkSpecifier):
    def __init__(self):
        super().__init__('miniweather')

    def get_nn_class(self):
        return MiniWeatherNeuralNetwork

    def get_target_type(self):
        return None

    def get_infer_data_from_dl(self, dataloader):
        return self.get_infer_data_from_ds(dataloader.dataset)

    def get_infer_data_from_ds(self, dataset):
        # return self.get_infer_data_from_ds(dataset)
        return dataset.input_as_torch_tensor()[0:32]


class ParticleFilterSpecifier(BenchmarkSpecifier):
    def __init__(self):
        super().__init__('particlefilter')

    def get_nn_class(self):
        return ParticleFilterNeuralNetwork

    def get_target_type(self):
        return torch.float32

    def get_infer_data_from_dl(self, dataloader):
        # Keeping this one the same for now, as PF has
        # made lots of progress
        return get_all_infer_data(dataloader)

    def get_infer_data_from_ds(self, dataset):
        return dataset.input_as_torch_tensor()


class BinomialOptionsSpecifier(BenchmarkSpecifier):
    def __init__(self):
        super().__init__('binomialoptions')

    def get_nn_class(self):
        return BinomialOptionsNeuralNetwork

    def get_target_type(self):
        return None

    def get_infer_data_from_dl(self, dataloader):
        return self.get_infer_data_from_ds(dataloader.dataset)
        
    def get_infer_data_from_ds(self, dataset):
        return dataset.input_as_torch_tensor()


class BondsOptionsSpecifier(BenchmarkSpecifier):
    def __init__(self):
        super().__init__('bonds')

    def get_nn_class(self):
        return BondsNeuralNetwork

    def get_target_type(self):
        return None

    def get_infer_data_from_dl(self, dataloader):
        return self.get_infer_data_from_ds(dataloader.dataset)

    def get_infer_data_from_ds(self, dataset):
        return dataset.input_as_torch_tensor()


def has_datagen_args(config):
    return 'datagen_args' in config

def get_params_from_file(file_path, idx):
    return pd.read_csv(file_path).iloc[idx]

@click.command()
@click.option('--name', help='Name of the benchmark')
@click.option('--config', default='config.yaml',
              help='Path to the configuration file')
@click.option('--architecture_index', default=None,
              help='Index of the architecture to use',
              type=int, required=True)
@click.option('--output', default='results.yaml')
@click.option('--output_model', default=None, required=True,
              help='Path to write the best model to')
def main(name, config, architecture_index, output, output_model):
    # TODO: This we need to get from the configuration file
    with open(config) as f:
      config = yaml.load(f, Loader=yaml.FullLoader)
    train_args = config[name]['train_args']
    data_args = config[name]['data_args']
    file_path = data_args['train_test_dataset']
    region_name = data_args['region_name']
    architecture_params_f = train_args['all_trials_file']

    # they're in the same dictionary
    arch_params = get_params_from_file(architecture_params_f, architecture_index)
    hyper_params = arch_params
    hyper_params['epochs'] = train_args['epochs']
    print(hyper_params)

    global DATATYPE
    # TODO: We need to get the region name and the input/output from the configuration file
    if 'train_end_index' in data_args:
      tei = data_args['train_end_index']
    else:
      tei = np.inf
    
    if 'test_split' in data_args:
        validation_split = data_args['test_split']
    else:
        validation_split = 0.2

    if 'max_test_batches' in data_args:
        max_test_batches = data_args['max_test_batches']
    else:
        max_test_batches = sys.maxsize

    global BenchSpec
    BenchSpec = BenchmarkSpecifier.get_specifier(name)
    target_type = BenchSpec.get_target_type()

    if has_datagen_args(data_args):
        datagen_args = data_args['datagen_args']
        if 'multiplier' in arch_params:
            datagen_args['multiplier'] = arch_params['multiplier']
        data_gen = BenchSpec.get_dataset_generator_class()
        data_gen = data_gen(datagen_args)
        file_path = data_gen()

    ds = HDF5DataSet(file_path, region_name, 'input', 'output',
                     train_end_index=tei, target_type=target_type,
                     load_entire=data_args['load_entire']
                     )

    # awkward
    # I don't have a good way to distinguish between the case when
    # we already to the multiple instances because we created the
    # data set and when we didn't
    if 'multiplier' in arch_params and not has_datagen_args(config['data_args']):
        ds.set_for_predicting_multiple_instances(arch_params['multiplier'])
    DATATYPE = ds.target_dtype
    batch_size = hyper_params.get("batch_size")
    shuffle_dataset = True
    random_seed = 42

    dataset_size = len(ds)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    batch_size = int(batch_size)
    train_dl = DataLoader(ds, batch_size=batch_size, sampler=train_sampler)
    test_dl = DataLoader(ds, batch_size=batch_size*2, sampler=valid_sampler)
    test_dl.max_batches = max_test_batches

    nn = BenchSpec.get_nn_class()
    early_stop_parms = train_args['early_stop_args']
    early_stopper = EarlyStopper(early_stop_parms['patience'], 
                                 early_stop_parms['min_delta_percent'])
    optimized_model, test_loss, runtime = train_test_infer_loop(nn, train_dl, test_dl, 
                                            early_stopper, arch_params, 
                                            hyper_params
                                            )
    torch.jit.save(optimized_model, output_model)
    runtime, runtime_sem = runtime
    results = {"average_mse": test_loss, 'inference_time': runtime,
               'inference_time_sem': runtime_sem
               }
    results['index'] = architecture_index
    print(results)
    results['model_location'] = output_model
    with open(output, 'w') as f:
        yaml.dump(results, f)

if __name__ == '__main__':
    main()
