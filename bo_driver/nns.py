import torch
import torch.nn.functional as F
from torch import nn
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        else:
            self.activ_fn = nn.LeakyReLU()

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

        conv_kernel_size = int(network_params.get("conv_kernel_size"))
        conv_stride = int(network_params.get("conv_stride"))
        maxpool_kernel_size = int(network_params.get("maxpool_kernel_size"))
        maxpool_stride = int(maxpool_kernel_size)
        fc2_size = int(network_params.get("fc2_size"))


        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=conv_kernel_size, stride=conv_stride, padding=1),
            nn.BatchNorm2d(1),
            nn.PReLU(),
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


class MiniBUDENeuralNetwork(nn.Module):
    def __init__(self, params):
        super(MiniBUDENeuralNetwork, self).__init__()
        print("Params are", params)
        multiplier = params.get("multiplier")
        h1_feature_multiplier = params.get("hidden1_feature_mult")
        h2_feature_multiplier = params.get("hidden2_feature_mult")
        h3_feature_multiplier = params.get("hidden3_feature_mult")
        h4_feature_multiplier = params.get("hidden4_feature_mult")
        dropout = params.get("dropout")

        n_pose_values = 6 * multiplier

        # Magic number: 21552 is the number of constant values
        # in each tensor entry based on the input deck.
        n_input_features = n_pose_values + 21552
        n_output_features = 1 * multiplier

        h1_features = n_input_features * h1_feature_multiplier
        h2_features = h1_features * h2_feature_multiplier
        h3_features = h2_features * h3_feature_multiplier
        h4_features = h3_features * h4_feature_multiplier

        layers = [h1_features, h2_features, h3_features, h4_features]
        layers = list(map(int, layers))

        layers_backwards = layers[::-1]
        while layers_backwards[-1] < n_output_features:
            layers_backwards.pop()

        layers = layers_backwards[::-1]
        layers = list(filter(lambda x: x > 0, layers))
        # input_layer = nn.Linear(n_input_features, layers[0])
        input_layer = nn.Linear(n_input_features, n_input_features)
        nn_layers = [input_layer, nn.PReLU()]
        l0 = nn.Linear(n_input_features, layers[0])
        nn_layers += [l0, nn.PReLU()]
        print('Layers are ', layers)
        for i in range(1, len(layers)):
            nn_layers.append(nn.Linear(layers[i-1], layers[i]))
            nn_layers.append(nn.PReLU())
            nn_layers.append(nn.Dropout(dropout))
        nn_layers.append(nn.Linear(layers[-1], n_output_features))

        self.sequential = nn.Sequential(*nn_layers)

        self.register_buffer('ipt_stdev',
                             torch.full((1, n_input_features), torch.inf)
                             )
        self.register_buffer('ipt_mean',
                             torch.full((1, n_input_features), -torch.inf)
                             )
        self.register_buffer('opt_stdev',
                             torch.full((1, n_output_features), torch.inf)
                             )
        self.register_buffer('opt_mean',
                             torch.full((1, n_output_features), -torch.inf)
                             )

    def forward(self, x):
        x = (x-self.ipt_mean) / (self.ipt_stdev)
        x = self.sequential(x)
        x = (x * self.opt_stdev) + self.opt_mean
        return x

    def calculate_and_save_normalization_parameters(self, train_dl):
        dataset = train_dl.dataset
        ipt_stdev = np.std(dataset.ipt_dataset, axis=0)
        ipt_mean = np.mean(dataset.ipt_dataset, axis=0)

        opt_stdev = np.std(dataset.opt_dataset, axis=0)
        opt_mean = np.mean(dataset.opt_dataset, axis=0)

        self.ipt_stdev = torch.from_numpy(ipt_stdev).to(DATATYPE)
        self.ipt_mean = torch.from_numpy(ipt_mean).to(DATATYPE)
        self.opt_stdev = torch.from_numpy(opt_stdev).to(DATATYPE)
        self.opt_mean = torch.from_numpy(opt_mean).to(DATATYPE)

        self.ipt_stdev = torch.max(self.ipt_stdev,
                                   torch.ones_like(self.ipt_stdev)
                                   )
        self.opt_stdev = torch.max(self.opt_stdev,
                                   torch.ones_like(self.opt_stdev)
                                   )

        print("Input stdev shape: ", ipt_stdev.shape)


class ConfigurableNumHiddenLayersMiniBUDENeuralNetwork(nn.Module):
    def __init__(self, params):
        super(ConfigurableNumHiddenLayersMiniBUDENeuralNetwork, self).__init__()
        print("Params are", params)
        multiplier = params.get("multiplier")
        feature_multiplier = params.get("feature_multiplier")
        num_hidden_layers = params.get("num_hidden_layers")
        h1_features = params.get("hidden_1_features")
        dropout = params.get("dropout")

        n_pose_values = 6 * multiplier
        n_input_features = n_pose_values
        n_output_features = 1 * multiplier

        first_layer = nn.Linear(n_input_features, h1_features)
        nn_layers = [first_layer, nn.PReLU(), nn.Dropout(dropout)]
        prev_features = h1_features
        num_features = int(h1_features * feature_multiplier)

        for i in range(1, num_hidden_layers):
            if i == num_hidden_layers - 1:
                last_layer = nn.Linear(prev_features, n_output_features)
                nn_layers.append(last_layer)
            else:
                num_features = int(prev_features * feature_multiplier)
                if num_features < n_output_features:
                    continue
                next_layer = nn.Linear(prev_features, num_features)
                nn_layers.append(next_layer)
                nn_layers.append(nn.PReLU())
                nn_layers.append(nn.Dropout(dropout))

                prev_features = num_features

        self.sequential = nn.Sequential(*nn_layers)

        self.register_buffer('ipt_stdev',
                             torch.full((1, n_input_features), torch.inf)
                             )
        self.register_buffer('ipt_mean',
                             torch.full((1, n_input_features), -torch.inf)
                             )
        self.register_buffer('opt_stdev',
                             torch.full((1, n_output_features), torch.inf)
                             )
        self.register_buffer('opt_mean',
                             torch.full((1, n_output_features), -torch.inf)
                             )

    def forward(self, x):
        x = (x-self.ipt_mean) / (self.ipt_stdev)
        x = self.sequential(x)
        x = (x * self.opt_stdev) + self.opt_mean
        return x

    def calculate_and_save_normalization_parameters(self, train_dl):
        dataset = train_dl.dataset
        ipt_stdev = np.std(dataset.ipt_dataset, axis=0)
        ipt_mean = np.mean(dataset.ipt_dataset, axis=0)

        opt_stdev = np.std(dataset.opt_dataset, axis=0)
        opt_mean = np.mean(dataset.opt_dataset, axis=0)

        self.ipt_stdev = torch.from_numpy(ipt_stdev).to(DATATYPE)
        self.ipt_mean = torch.from_numpy(ipt_mean).to(DATATYPE)
        self.opt_stdev = torch.from_numpy(opt_stdev).to(DATATYPE)
        self.opt_mean = torch.from_numpy(opt_mean).to(DATATYPE)

        self.ipt_stdev = torch.max(self.ipt_stdev,
                                   torch.ones_like(self.ipt_stdev)
                                   )
        self.opt_stdev = torch.max(self.opt_stdev,
                                   torch.ones_like(self.opt_stdev)
                                   )

        print("Input stdev shape: ", ipt_stdev.shape)
        print(self.ipt_stdev)
