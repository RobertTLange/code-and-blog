from .body_builder import BodyBuilder, Flatten
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class MaskedBodyBuilder(BodyBuilder):
    """ Monkey Patch BodyBuilder Class for Masking/Pruning of Weights """
    def build_layer(self, layer_config):
        """ Function specifies the specific network layer (allow masking)
        Masked Linear L Ex:     ["linear", out_dim, bias_boolean]
        Maske Convol L Ex:     ["conv2d", out_channels,
                                kernel_size, stride, padding]
        """
        layer_name = layer_config[0]
        if layer_name == "linear":
            layer = MaskedLinear(self.input_dims[-1], layer_config[1], bias=layer_config[2])
        elif layer_name == "flatten":
            layer = Flatten()
        elif layer_name == "conv2d":
            layer = MaskedConv2d(in_channels=self.input_dims[-1][0],
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

    def set_masks(self, masks):
        """ Set pruning masks - sequentially match children with masks """
        mask_counter = 0
        for child in self.model.children():
            if isinstance(child, MaskedLinear) or isinstance(child, MaskedConv2d):
                child.set_mask(masks[mask_counter])
                mask_counter += 1
                if mask_counter == len(masks):
                    break

# Credits to https://github.com/wanglouis49/pytorch-weights_pruning

def to_var(x, requires_grad=False, volatile=False):
    """ Set whether parameters will be optimized """
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


class MaskedLinear(nn.Linear):
    """ Masked Version of Linear Layer - Set Mask When Defining the Net"""
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.mask_flag = False

    def set_mask(self, mask):
        self.register_buffer('mask', mask)
        mask_var = self.get_mask()
        self.weight.data = self.weight.data*mask_var.data
        self.mask_flag = True

    def get_mask(self):
        # print(self.mask_flag)
        return to_var(self.mask, requires_grad=False)

    def forward(self, x):
        if self.mask_flag == True:
            mask_var = self.get_mask()
            weight = self.weight * mask_var
            return F.linear(x, weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)


class MaskedConv2d(nn.Conv2d):
    """ Masked Version of Conv2D Layer - Set Mask When Defining the Net"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(MaskedConv2d, self).__init__(in_channels, out_channels,
            kernel_size, stride, padding, dilation, groups, bias)
        self.mask_flag = False

    def set_mask(self, mask):
        self.register_buffer('mask', mask)
        mask_var = self.get_mask()
        self.weight.data = self.weight.data*mask_var.data
        self.mask_flag = True

    def get_mask(self):
        # print(self.mask_flag)
        return to_var(self.mask, requires_grad=False)

    def forward(self, x):
        if self.mask_flag == True:
            mask_var = self.get_mask()
            weight = self.weight * mask_var
            return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
