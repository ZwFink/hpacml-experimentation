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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class HDF5DataSet(Dataset):
    def __init__(self, file_path, group_name, ipt_dataset, opt_dataset, target_type=torch.float64):
        self.open_file = h5py.File(file_path, 'r')
        self.file_path = file_path
        self.target_type = target_type

        self.ipt_dataset, self.opt_dataset = self.get_datasets(group_name, ipt_dataset, opt_dataset)
        assert(len(self.ipt_dataset) == len(self.opt_dataset))
        self.length = len(self.ipt_dataset)

    def get_datasets(self, group_name, ipt_dataset, opt_dataset):
        group = self.open_file[group_name]
        ipt_dataset = group[ipt_dataset]
        opt_dataset = group[opt_dataset]
        return ipt_dataset[0], opt_dataset[0]

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        return torch.tensor(self.ipt_dataset[idx]).to(self.target_type), torch.tensor(self.opt_dataset[idx]).to(self.target_type)


class ParticleFilterNeuralNetwork(nn.Module):
    def __init__(self, network_params):
        super(ParticleFilterNeuralNetwork, self).__init__()

        conv_kernel_size = network_params.get("conv_kernel_size")
        conv_stride = network_params.get("conv_stride")
        maxpool_kernel_size = network_params.get("maxpool_kernel_size")
        maxpool_stride = maxpool_kernel_size

        self.linear_relu_stack = nn.Sequential(
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

        # Linear layers
        self.fc1 = nn.Linear(output_size, 2)

    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.squeeze(-1)
        x = x / 255
        logits = self.linear_relu_stack(x)
        logits = logits.flatten(1)
        logits = F.relu(logits)
        logits = self.fc1(logits)
        logits = logits * 128
        logits = torch.clamp(logits, min=0, max=128)
        return logits

    def calculate_and_save_normalization_parameters(self, train_dl):
        return None

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
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            # correct += (pred.argmax(1)==y).type(torch.double).sum().item()
    
    test_loss /= size
    correct /= size
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")
    return test_loss*size

def infer_loop(model, dataloader, trials, writer=None):

    data_list = []
    for batch in dataloader:
        # Assuming batch[0] is the tensor of data; adjust if your dataloader structure is different
        data_list.append(batch[0])
    X = torch.cat(data_list, dim=0)
    X = X.to(device)

    total_time = 0
    total_to_gpu = 0
    total_to_cpu = 0
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

        X = X.to('cpu')
        for i in range(trials):

            transfer_start = time.time()
            X = X.to(device)
            torch.cuda.synchronize()
            transfer_end = time.time()
            transfer_to_gpu = transfer_end - transfer_start

            start = time.time()
            pred = model(X)
            torch.cuda.synchronize()
            end = time.time()

            transfer_start = time.time()
            pred = pred.to('cpu')
            torch.cuda.synchronize()
            transfer_end = time.time()
            transfer_to_cpu = transfer_end - transfer_start

            if i >= 2:
                total_to_cpu += transfer_to_cpu
                total_to_gpu += transfer_to_gpu
                total_time += (end-start)
        average_forward = total_time / (trials-2)
        average_gpu = total_to_gpu / (trials-2)
        average_cpu = total_to_cpu / (trials-2)
    average_time = total_time / (trials-2)
    if writer:
      writer.add_scalar('inference time', average_time*1000, 0)
    return average_time * 1000

def train_test_infer_loop(nn_class, train_dl, test_dl, arch_params, hyper_params):
    learning_rate = hyper_params.get("learning_rate")
    epochs = hyper_params.get("epochs")
    batch_size = hyper_params.get("batch_size")
    weight_decay = hyper_params.get("weight_decay")

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
        train_loop(writer, train_dl, model, loss_fn, optimizer, scheduler, t+1)
        epoch_time = time.time() - epoch_start
        test_loss = test_loop(test_dl, model, loss_fn)
        test_loss_mape = test_loop(test_dl, model, MAPE)
        infer_time = infer_loop(model, test_dl, 100, writer)
    return test_loss, infer_time

def get_nn_class(name):
    if name == 'particlefilter':
        return ParticleFilterNeuralNetwork
    

@click.command()
@click.option('--name', help='Name of the benchmark')
@click.option('--config', default='config.yaml', help='Path to the configuration file')
@click.option('--architecture_config', default=None, help='Path to the configuration file for the architecture of this run')
@click.option('--output', default='output.yaml', help='Path to write the output to')
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
  
  # TODO: We need to get the region name and the input/output from the configuration file
  ds = HDF5DataSet(file_path, region_name, 'input', 'output', target_type=DATATYPE)
  validation_split = 0.2
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
  test_dl = DataLoader(ds, batch_size=batch_size, sampler=valid_sampler)

  nn = get_nn_class(name)
  test_loss, runtime = train_test_infer_loop(nn, train_dl, test_dl, arch_params, hyper_params)
  results = {"total_mse": test_loss, 'inference_time': runtime}
  print(results)
  with open(output, 'w') as f:
    yaml.dump(results, f)

if __name__ == '__main__':
    main()