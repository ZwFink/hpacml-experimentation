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
import yaml
import os
import time
import numpy as np
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import math

DATATYPE = torch.float32
BenchSpec = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


class MiniWeatherNeuralNetwork(nn.Module):
    def __init__(self, network_params):
        super(MiniWeatherNeuralNetwork, self).__init__()
        conv1_kernel_size = network_params.get("conv1_kernel_size")
        conv1_stride = network_params.get("conv1_stride")
        conv1_out_channels = network_params.get("conv1_out_channels")
        dropout = network_params.get("dropout")
        activ_fn_name = network_params.get("activation_function")
        conv2_kernel_size = network_params.get("conv2_kernel_size")

        if activ_fn_name == "relu":
            self.activ_fn = nn.ReLU()
        elif activ_fn_name == "leaky_relu":
            self.activ_fn = nn.LeakyReLU()
        elif activ_fn_name == "tanh":
            self.activ_fn = nn.Tanh()

        c1ks = conv1_kernel_size
        c1s = conv1_stride

        self.dropout = nn.Dropout(dropout)
        if conv2_kernel_size != 0:
            self.conv1 = nn.Conv2d(in_channels=4,
                                   out_channels=conv1_out_channels,
                                   kernel_size=(c1ks, c1ks), stride=(c1s, c1s),
                                   padding='same',
                                   )

            self.conv2 = nn.Conv2d(in_channels=conv1_out_channels,
                                   out_channels=4,
                                   kernel_size=(conv2_kernel_size,
                                                conv2_kernel_size),
                                   stride=(1, 1), padding='same'
                                   )
            self.fp = nn.Sequential(self.conv1, self.activ_fn, 
                                    self.dropout, self.conv2, self.activ_fn
                                    )
        else:
            # Here, we ignore Conv1 out channels
            self.conv1 = nn.Conv2d(in_channels=4, out_channels=4, 
                                   kernel_size=(c1ks, c1ks), stride=(c1s, c1s), 
                                   padding='same'
                                   )
            self.fp = nn.Sequential(self.conv1, self.activ_fn, self.dropout)

        self.register_buffer('min', torch.full((4, 1), torch.inf))
        self.register_buffer('max', torch.full((4, 1), -torch.inf))

    def forward(self, x):
        x = (x - self.min) / (self.max - self.min)

        x = self.fp(x)

        x = x * (self.max - self.min) + self.min

        return x

    def calculate_and_save_normalization_parameters(self, train_dl):
        for x, y in train_dl:
            x = x.to(device)  # Assuming x is of shape [N, C, H, W]
            y = y.to(device)
            # transpose to [C, N, H, W]
            x = x.transpose(0, 1)
            # reshape to [ C, N*H*W]
            x = x.reshape(x.shape[0], -1)
            # Compute min and max across the flattened spatial dimensions
            batch_min = x.min(dim=1, keepdim=True).values
            batch_max = x.max(dim=1, keepdim=True).values
            self.min = torch.min(self.min, batch_min)
            self.max = torch.max(self.max, batch_max)
        # Adjust the min and max shapes to [C, 1, 1] by adding an extra dimension
        self.min = self.min.unsqueeze(0)
        self.max = self.max.unsqueeze(0)
        self.min = self.min.unsqueeze(-1)
        self.max = self.max.unsqueeze(-1)
        print("Min shape ", self.min.shape)
        print("Max shape ", self.max.shape)
        # print the number of non-zeros in max
        print("Min is ", self.min)
        print("Max is ", self.max)


class ParticleFilterNeuralNetwork(nn.Module):
    def __init__(self, network_params):
        super(ParticleFilterNeuralNetwork, self).__init__()

        conv_kernel_size = network_params.get("conv_kernel_size")
        conv_stride = network_params.get("conv_stride")
        maxpool_kernel_size = network_params.get("maxpool_kernel_size")
        maxpool_stride = maxpool_kernel_size
        fc2_size = network_params.get("fc2_size")

        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=conv_kernel_size, stride=conv_stride, padding=1),
            nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=maxpool_stride, padding=0,dilation=1)
        ).to(device)
        input_size = 128

        # Calculate output size after conv and maxpool layers
        def conv_output_size_1d(dimension, kernel_size, stride, padding, dilation):
            return math.floor((dimension + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
        def conv_output_size(input_width, input_height, kernel_size, stride, dilation, padding):
            return (conv_output_size_1d(input_width, kernel_size, stride, padding, dilation), conv_output_size_1d(input_height, kernel_size, stride, padding, dilation))

        # Output size after convolution
        conv_output = conv_output_size(input_size, input_size, conv_kernel_size, conv_stride, 1, 1)

        # Output size after max pooling
        maxpool_output = conv_output_size(conv_output[0], conv_output[1], maxpool_kernel_size, maxpool_stride, 1, 0)

        # Overall output size for the linear layer
        output_size = maxpool_output[0] * maxpool_output[1]

        print("FC1 size is ", output_size)

        # Linear layers
        if fc2_size == 0:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(output_size, 2)
            )
        else:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(output_size, fc2_size),
                nn.ReLU(),
                nn.Linear(fc2_size, 2)
            )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.squeeze(-1)
        x = x / 255
        logits = self.conv_stack(x)
        logits = logits.flatten(1)
        logits = F.relu(logits)
        logits = self.linear_relu_stack(logits)
        logits = logits * 128
        logits = torch.clamp(logits, min=0, max=128)
        return logits

    def calculate_and_save_normalization_parameters(self, train_dl):
        return None


class BinomialOptionsNeuralNetwork(nn.Module):
    def __init__(self, network_params):
        super(BinomialOptionsNeuralNetwork, self).__init__()
        print("Network params are ", network_params)
        multiplier = network_params.get("multiplier")
        hidden1_features = network_params.get("hidden1_features")
        hidden2_features = network_params.get("hidden2_features")
        dropout = network_params.get("dropout")

        n_ipt_features = 5 * multiplier
        hidden1_features *= multiplier
        hidden2_features *= multiplier
        n_opt_features = 1 * multiplier

        if hidden2_features != 0:
            self.layers = nn.Sequential(
                nn.Linear(n_ipt_features, hidden1_features),
                nn.LeakyReLU(),
                nn.Linear(hidden1_features, hidden2_features),
                nn.Dropout(dropout),
                nn.LeakyReLU(),
                nn.Linear(hidden2_features, n_opt_features)
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(n_ipt_features, hidden1_features),
                nn.LeakyReLU(),
                nn.Linear(hidden1_features, n_opt_features)
            )
        self.register_buffer('ipt_min',
                             torch.full((1, 5*multiplier), torch.inf))
        self.register_buffer('ipt_max',
                             torch.full((1, 5*multiplier), -torch.inf))

        self.register_buffer('opt_min',
                             torch.full((1, multiplier), torch.inf))
        self.register_buffer('opt_max',
                             torch.full((1, multiplier), -torch.inf))

    def forward(self, x):
        x = (x - self.ipt_min) / (self.ipt_max - self.ipt_min)
        x = self.layers(x)
        x = torch.clamp(x, min=0)
        x = x * (self.opt_max - self.opt_min) + self.opt_min
        return x

    def calculate_and_save_normalization_parameters(self, train_dl):
        for x, y in train_dl:
            x = x.to(device)
            y = y.to(device)
            batch_min = x.min(dim=0, keepdim=True).values
            batch_max = x.max(dim=0, keepdim=True).values
            self.ipt_min = torch.min(self.ipt_min, batch_min)
            self.ipt_max = torch.max(self.ipt_max, batch_max)

            batch_min = y.min(dim=0, keepdim=True).values
            batch_max = y.max(dim=0, keepdim=True).values
            self.opt_min = torch.min(self.opt_min, batch_min)
            self.opt_max = torch.max(self.opt_max, batch_max)


class BondsNeuralNetwork(nn.Module):
    def __init__(self, network_params):
        super(BondsNeuralNetwork, self).__init__()
        print("Network params are ", network_params)
        multiplier = network_params.get("multiplier")
        hidden1_features = network_params.get("hidden1_features")
        hidden2_features = network_params.get("hidden2_features")
        dropout = network_params.get("dropout")

        n_ipt_features = 9 * multiplier
        hidden1_features *= multiplier
        hidden2_features *= multiplier
        n_opt_features = 1 * multiplier

        if hidden2_features != 0:
            self.layers = nn.Sequential(
                nn.Linear(n_ipt_features, hidden1_features),
                nn.LeakyReLU(),
                nn.Linear(hidden1_features, hidden2_features),
                nn.Dropout(dropout),
                nn.LeakyReLU(),
                nn.Linear(hidden2_features, n_opt_features)
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(n_ipt_features, hidden1_features),
                nn.LeakyReLU(),
                nn.Linear(hidden1_features, n_opt_features)
            )
        self.register_buffer('ipt_min',
                             torch.full((1, 9*multiplier), torch.inf))
        self.register_buffer('ipt_max',
                             torch.full((1, 9*multiplier), -torch.inf))
        self.register_buffer('opt_min',
                             torch.full((1, multiplier), torch.inf))
        self.register_buffer('opt_max',
                             torch.full((1, multiplier), -torch.inf))

    def forward(self, x):
        x = (x - self.ipt_min) / ((self.ipt_max - self.ipt_min))
        x = self.layers(x)
        x = torch.clamp(x, min=0)
        x = x * (self.opt_max - self.opt_min) + self.opt_min
        return x

    def calculate_and_save_normalization_parameters(self, train_dl):
        for x, y in train_dl:
            x = x.to(device)  
            y = y.to(device)
            batch_min = x.min(dim=0, keepdim=True).values
            batch_max = x.max(dim=0, keepdim=True).values
            self.ipt_min = torch.min(self.ipt_min, batch_min)
            self.ipt_max = torch.max(self.ipt_max, batch_max)

            batch_min = y.min(dim=0, keepdim=True).values
            batch_max = y.max(dim=0, keepdim=True).values
            self.opt_min = torch.min(self.opt_min, batch_min)
            self.opt_max = torch.max(self.opt_max, batch_max)


class DummyScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
    def step(self):
        pass

def MAPE(actual, forecast):
    """
    Calculate the Mean Absolute Percentage Error (MAPE)

    Parameters:
    actual (Tensor): The actual values.
    forecast (Tensor): The forecasted or predicted values.

    Returns:
    float: The MAPE value.
    """
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
    for batch, dat in enumerate(dataloader):
        X = dat[0]
        y = dat[1]
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        scheduler.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch*len(X)
            # writer.add_scalar('training loss', loss, epoch*len(dataloader) + batch)
            # print(f"Epoch: {epoch}, loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss = 0
    num_batches = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss_fn_value = loss_fn(pred, y).item()
            test_loss += loss_fn(pred, y).item()
            num_batches += 1
            if num_batches > dataloader.max_batches:
                break

    return test_loss / num_batches


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

        warmup_time = time.time()
        # Do warmup
        pred = model(X)
        pred = model(X)
        torch.cuda.synchronize()
        print("Warmup time: ", time.time() - warmup_time)

        for i in range(10):
            start = time.time()
            pred = model(X)
            torch.cuda.synchronize()
            end = time.time()
            total_time += (end - start)

        average_time = total_time / 10
        # If average time is more than 1 second,
        # we are not going to do the 100 trials
        # 10 is probably enough
        if average_time > 1:
            return average_time * 1000

        total_time = 0
        for i in range(trials):
            start = time.time()
            pred = model(X)
            torch.cuda.synchronize()
            end = time.time()
            total_time += (end - start)

    average_time = total_time / trials
    if writer:
      writer.add_scalar('inference time', average_time*1000, 0)
    return average_time * 1000

def train_test_infer_loop(nn_class, train_dl, test_dl, early_stopper, arch_params, hyper_params):
    learning_rate = hyper_params.get("learning_rate")
    epochs = hyper_params.get("epochs")
    batch_size = hyper_params.get("batch_size")
    weight_decay = hyper_params.get("weight_decay")

    if 'dropout' in hyper_params:
        arch_params['dropout'] = hyper_params['dropout']

    model = nn_class(arch_params).to(DATATYPE).to(device)
    model.calculate_and_save_normalization_parameters(train_dl)

    print(model)
    writer = None
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = weight_decay)
    scheduler = DummyScheduler(optimizer)
    best_test_loss = np.inf
    model_epoch = 0

    for t in range(model_epoch, epochs):
        epoch_start = time.time()
        tl_start = time.time()
        train_loop(writer, train_dl, model, loss_fn, optimizer, scheduler, t+1)
        tl_end = time.time()
        print(f"Training time: {tl_end - tl_start}")
        test_loop_start = time.time()
        test_loss = test_loop(test_dl, model, loss_fn)
        test_loop_end = time.time()
        print(f"Test loop time: {(test_loop_end - test_loop_start)}")
        epoch_time = time.time() - epoch_start
        print(f"Epoch {t+1}\n-------------------------------")
        print(f"Test Error: \n Avg loss: {test_loss:>8f}, Time: {epoch_time:>8f}\n")
        if early_stopper.early_stop(test_loss):
            print(f'Early stopping at epoch {t+1} after max patience reached.')
            break

    infer_start = time.time()
    infer_time = infer_loop(model, test_dl, 100, writer)
    infer_end = time.time()
    print(f"Inference time: {(infer_end - infer_start)}")
    return test_loss, infer_time

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
    def get_specifier(cls, name):
        if name == 'particlefilter':
            return ParticleFilterSpecifier()
        elif name == 'miniweather':
            return MiniWeatherSpecifier()
        elif name == 'binomial_options':
            return BinomialOptionsSpecifier()
        elif name == 'bonds':
            return BondsOptionsSpecifier()


def get_all_infer_data(dataloader):
    data = list(dataloader)
    # concatenate everything
    item = torch.cat([x[0] for x in data])
    return item


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


@click.command()
@click.option('--name', help='Name of the benchmark')
@click.option('--config', default='config.yaml',
              help='Path to the configuration file')
@click.option('--architecture_config', default=None,
              help='Path to the configuration file for the architecture of this run')
@click.option('--output', default='output.yaml',
              help='Path to write the output to')
def main(name, config, architecture_config, output):
    # TODO: This we need to get from the configuration file
    with open(config) as f:
      config = yaml.load(f, Loader=yaml.FullLoader)
    config = config[name]['train_args']
    data_args = config['data_args']
    file_path = data_args['train_test_dataset']
    region_name = data_args['region_name']

    with open(architecture_config) as f:
      arch_params = yaml.load(f, Loader=yaml.FullLoader)
      arch_params, hyper_params = arch_params['arch_parameters'], arch_params['hyper_parameters']
  
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

    ds = HDF5DataSet(file_path, region_name, 'input', 'output',
                     train_end_index=tei, target_type=target_type,
                     load_entire=data_args['load_entire']
                    )
    if 'multiplier' in arch_params:
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

    train_dl = DataLoader(ds, batch_size=batch_size, sampler=train_sampler)
    test_dl = DataLoader(ds, batch_size=batch_size*2, sampler=valid_sampler)
    test_dl.max_batches = max_test_batches

    nn = BenchSpec.get_nn_class()
    early_stop_parms = config['early_stop_args']
    early_stopper = EarlyStopper(early_stop_parms['patience'], 
                                 early_stop_parms['min_delta_percent'])
    test_loss, runtime = train_test_infer_loop(nn, train_dl, test_dl, 
                                               early_stopper, arch_params, 
                                               hyper_params
                                               )
    results = {"average_mse": test_loss, 'inference_time': runtime}
    print(results)
    with open(output, 'w') as f:
        yaml.dump(results, f)

if __name__ == '__main__':
    main()