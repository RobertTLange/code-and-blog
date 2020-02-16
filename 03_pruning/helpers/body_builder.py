import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau

import numpy as np
from collections import OrderedDict
from functools import reduce


class BodyBuilder(nn.Module):
    def __init__(self, input_dim, layers_info, output_act, hidden_act,
                 dropout=0.0, batch_norm=False, load_in_path=None):
        """ Network Class for Base Torso of Agents """
        super(BodyBuilder, self).__init__()
        self.input_dims = [tuple(input_dim)]
        self.layers_info = layers_info
        self.num_hidden_layers = len(self.layers_info)

        self.dropout = dropout
        self.batch_norm = batch_norm
        self.recurrence = None

        # Dictionary mapping strings to activations
        self.str_to_act = {"relu": nn.ReLU(), "sigmoid": nn.Sigmoid(),
                           "softplus": nn.Softplus(), "tanh": nn.Tanh(),
                           "softmax": nn.Softmax(dim=1), "identity": nn.Identity()}
        self.hidden_act = hidden_act
        self.output_act = output_act

        # Build the network module
        self.model = self.build_network()

        # Load in pretrained network weights
        self.load_in_path = load_in_path
        if self.load_in_path is not None:
            checkpoint = torch.load(self.load_in_path, map_location='cpu')
            self.model.load_state_dict(checkpoint)

    def build_network(self):
        """ Function builds the network architecture """
        feature_layers = OrderedDict()
        # Define Hidden Layers in Loop
        for layer_id in range(0, self.num_hidden_layers):
            # Add the plain layer
            feature_layers[str(layer_id)] = self.build_layer(self.layers_info[layer_id])
            layer_name = self.layers_info[layer_id][0]
            # Add the elementwise activation if layer is not postprocessing
            if layer_name not in ["max_pool", "avgpool", "adaptivemaxpool",
                                  "adaptiveavgpool", "flatten"]:
                if layer_id != self.num_hidden_layers - 1:
                    feature_layers[str(layer_id) + "-act"] = self.str_to_act[self.hidden_act]
                else:
                    feature_layers[str(layer_id) + "-act"] = self.str_to_act[self.output_act]


            # Add a dropout layer if desired - but not for final layer
            if self.dropout != 0.0 and not layer_id == self.num_hidden_layers - 1:
                feature_layers[str(layer_id) + "-dropout"] = self.postprocess_layer("dropout")

            # Add batchnorm (1d/2d) if desired - but not for final layer
            if self.batch_norm and not layer_id == self.num_hidden_layers - 1:
                if layer_name in ["linear", "rnn", "gru", "lstm"]:
                    feature_layers[str(layer_id) + "-batch-norm-1d"] = self.postprocess_layer("batch-norm-1d")
                elif layer_name == "conv2d":
                    feature_layers[str(layer_id) + "-batch-norm-2d"] = self.postprocess_layer("batch-norm-2d")

        return nn.Sequential(feature_layers)

    def build_layer(self, layer_config):
        """ Function specifies the specific network layer
        Linear L Ex:     ["linear", out_dim, bias_boolean]
        Convol L Ex:     ["conv2d", out_channels,
                          kernel_size, stride, padding]
        Dropout L Ex:    ["dropout", dropout_prob]
        Batch Norm L Ex: ["batch_norm_XXX", num_features]
        Pooling L Ex:    [<Pooling_Name>]
        """
        layer_name = layer_config[0]
        # Base Layers
        if layer_name == "linear":
            layer = nn.Linear(self.input_dims[-1], layer_config[1], bias=layer_config[2])
        elif layer_name == "flatten":
            layer = Flatten()
        elif layer_name == "conv2d":
            layer = nn.Conv2d(in_channels=self.input_dims[-1][0],
                              out_channels=layer_config[1],
                              kernel_size=layer_config[2],
                              stride=layer_config[3],
                              padding=layer_config[4])
        # Pooling Techniques
        elif layer_name == "maxpool":
            layer = nn.MaxPool2d(kernel_size=layer_config[1],
                                 stride=layer_config[2],
                                 padding=layer_config[3])
        elif layer_name == "avgpool":
            layer = nn.AvgPool2d(kernel_size=layer_config[1],
                                 stride=layer_config[2],
                                 padding=layer_config[3])
        elif layer_name == "adaptivemaxpool":
            layer = nn.AdaptiveMaxPool2d(output_size=(layer_config[1], layer_config[2]))
        elif layer_name == "adaptiveavgpool":
            layer = nn.AdaptiveAvgPool2d(output_size=(layer_config[1], layer_config[2]))

        # Update the input dimension for the next layer
        new_dims = self.calc_new_input_dims(self.input_dims[-1], layer_config)
        self.input_dims.append(new_dims)
        return layer

    def postprocess_layer(self, layer_name):
        """ Define post-processing layer (dropout, batch-norm) """
        if layer_name == "dropout":
            layer = nn.Dropout(p=self.dropout)
        elif layer_name == "batch-norm-1d":
            layer = nn.BatchNorm1d(num_features=self.input_dims[-1])
        elif layer_name == "batch-norm-2d":
            layer = nn.BatchNorm2d(num_features=self.input_dims[-1][0])
        return layer

    def calc_new_input_dims(self, input_dim, layer_config):
        """ Calc new dimens of the data after passing through layer """
        layer_name = layer_config[0]
        if layer_name == "conv2d":
            new_channels = layer_config[1]
            kernel, stride, padding = layer_config[2], layer_config[3], layer_config[4]
            new_height = int((input_dim[1] - kernel + 2*padding)/stride) + 1
            new_width = int((input_dim[2] - kernel + 2 * padding) / stride) + 1
            output_dim = (new_channels, new_height, new_width)
        elif layer_name == "flatten":
            output_dim = reduce(lambda x, y: x*y, input_dim)
        elif layer_name in ["maxpool", "avgpool"]:
            new_channels = input_dim[0]
            kernel, stride, padding = layer_config[1], layer_config[2], layer_config[3]
            new_height = int((input_dim[1] - kernel + 2*padding)/stride) + 1
            new_width = int((input_dim[2] - kernel + 2 * padding) / stride) + 1
            output_dim = (new_channels, new_height, new_width)
        elif layer_name in ["adaptivemaxpool", "adaptiveavgpool"]:
            new_channels = input_dim[0]
            output_dim = (new_channels, layer_config[1], layer_config[2])
        elif layer_name == "linear":
            output_dim = layer_config[1]
        elif layer_name in ["rnn", "gru", "lstm"]:
            # Num Hidden Units * Num Directions
            output_dim = layer_config[1] * (1 + layer_config[5])
        return output_dim

    def forward(self, x):
        """ Perform the forward pass through the network """
        return self.model(x)

    def feature_size(self):
        return self.input_dims[-1]

    def get_num_trainable_params(self):
        """ Get the number of trainable parameters in the network """
        model_parameters = filter(lambda p: p.requires_grad,
                                  self.model.parameters())
        trainable_params = sum([np.prod(p.size()) for p in model_parameters])
        return trainable_params


class Flatten(nn.Module):
    """ Flatten Layer that return 1d vector of concatenated entries """
    def forward(self, input):
        return input.view(input.size(0), -1)


def set_optimizer(network, opt_type, l_rate, momentum=0, w_decay=0.):
    """
    Set optimizer for network. If applicable set momentum & regularisation.
    """
    if opt_type == "Adam":
        optimizer = optim.Adam(network.parameters(), lr=l_rate,
                               weight_decay=w_decay)
    elif opt_type == "AdamW":
        optimizer = optim.AdamW(network.parameters(), lr=l_rate,
                                weight_decay=w_decay)
    elif opt_type == "RMSprop":
        optimizer = optim.RMSprop(network.parameters(), lr=l_rate,
                                  momentum=momentum, weight_decay=w_decay)
    elif opt_type == "SGD":
        optimizer = optim.SGD(network.parameters(), lr=l_rate,
                              momentum=momentum, weight_decay=w_decay)
    elif opt_type == "AMSGrad":
        optimizer = optim.Adam(network.parameters(), lr=l_rate,
                               amsgrad=momentum, weight_decay=w_decay)
    elif opt_type == "Adadelta":
        optimizer = optim.Adadelta(network.parameters(), lr=l_rate,
                                   weight_decay=w_decay)
    else:
        raise ValueError
    return optimizer


def set_lrate_schedule(optimizer, schedule_type, schedule_inputs):
    """ Set the learning rate schedule for a specific network """
    # Check https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html for more details
    if schedule_type == "exp-decay":
        scheduler = StepLR(optimizer,
                           step_size=schedule_inputs["step_size"],
                           gamma=schedule_inputs["gamma"])
    elif schedule_type == "multi-step":
        scheduler = MultiStepLR(optimizer,
                                milestones=schedule_inputs["milestones"],
                                gamma=schedule_inputs["gamma"])
    elif schedule_type == "reduce-on-plateu":
        # Note: Need to provide metric to check plateu - optimizer.step(metric)
        scheduler = ReduceLROnPlateau(optimizer,
                                      mode=schedule_inputs["mode"],
                                      factor=schedule_inputs["factor"],
                                      patience=schedule_inputs["patience"])
    else:
        raise ValueError
    return scheduler
