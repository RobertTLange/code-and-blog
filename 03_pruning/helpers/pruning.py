import numpy as np
from functools import reduce
import torch
from .masked_body_builder import MaskedBodyBuilder
from .train_dojo import TrainDojo

def weight_prune_masks(network, sparsity_level):
    """ Generate set of masks that 0-out smallest magnitude weights """
    all_weights = []
    # Construct list of all weights & determine percentiles (for each layer indiv!)
    # Important sparsity level is in percent 100 = 1
    no_param_layers = len(list(network.parameters()))
    masks = []
    for layer_id, p in enumerate(network.parameters()):
        # Make sure not to prune output layer & flatten layers
        if len(p.data.size()) != 1 and layer_id != no_param_layers-1:
            all_weights = list(p.cpu().data.abs().numpy().flatten())

            # Get threshold weight value - percentile of weight distr of layer
            threshold = np.percentile(np.array(all_weights), sparsity_level)

            # Mask out all weights that are smaller to get desired sparsity
            pruned_inds = p.data.abs() > threshold
            masks.append(pruned_inds.float().to(p.device))
    return masks


def unit_prune_masks(network, sparsity_level):
    """ Generate set of masks that 0-out smallest columns (sum of squared weights) """
    all_weights = []
    # Construct list of all weights & determine percentiles (for each layer indiv!)
    # Important sparsity level is in percent 100 = 1
    no_param_layers = len(list(network.parameters()))
    masks = []
    for layer_id, p in enumerate(network.parameters()):
        # Make sure not to prune output layer & flatten layers
        if len(p.data.size()) != 1 and layer_id != no_param_layers-1:
            squared_weights = p.cpu().data.abs().numpy()**2
            sq_sum_weights = np.sum(squared_weights, axis=0)

            # Get threshold SS-weight cols - percentile! & column indices
            threshold = np.percentile(np.array(sq_sum_weights), sparsity_level)
            keep_col = sq_sum_weights > threshold

            # Mask out all weight columns that are smaller to get sparsity
            pruned_inds = torch.ones(p.size())*torch.Tensor(keep_col)
            masks.append(pruned_inds.float().to(p.device))
    return masks


def filter_prune_masks(network, sparsity_level):
    """ Generate set of masks that 0-out smallest filter in conv layer (sum of squared weights in filter) """
    all_weights = []
    # Construct list of all weights & determine percentiles (for each layer indiv!)
    # Important sparsity level is in percent 100 = 1
    no_param_layers = len(list(network.parameters()))
    masks = []
    for layer_id, p in enumerate(network.parameters()):
        # Make sure not to prune output layer & flatten layers
        if len(p.data.size()) == 4 and layer_id != no_param_layers-1:
            squared_weights = p.cpu().data.abs().numpy()**2
            sq_sum_weights = np.sum(squared_weights, axis=(1,2,3))
            # Get threshold SS-filter cols - percentile! & filter indices
            threshold = np.percentile(sq_sum_weights, sparsity_level)
            keep_filter = sq_sum_weights > threshold
            # Very ugly way how to reshape the column vector of filters to keep
            # into a masking tensor
            pruned_inds = torch.Tensor(keep_filter).expand((p.size(3), p.size(2),
                                                            p.size(1), p.size(0))).T
            # Mask out all weight columns that are smaller to get sparsity
            masks.append(pruned_inds.float().to(p.device))
    return masks


def summarize_pruning(pruned_network, verbose=False):
    """ Summarize the pruning based on the masks (overall + layer-specific) """
    sparsity_layers, total_params, total_pruned = [], 0, 0
    no_param_layers = len(list(pruned_network.parameters()))
    # Loop over different weight layers of the network & track how many 0
    for layer_id, param in enumerate(pruned_network.parameters()):
        if len(param.data.size()) != 1 and layer_id != no_param_layers-1:
            # Calculate non-zero weights
            num_params = reduce(lambda x, y: x*y, param.size())
            zero_params = np.count_nonzero(param.cpu().data.numpy()==0)
            # Track sparsity statistics
            total_params += num_params
            total_pruned += zero_params
            sp_perc = zero_params/num_params
            sparsity_layers.append(sp_perc)
            if verbose:
                layer_type = "Linear" if len(param.size())==2 else "Conv"
                print("Layer {} | Type {} - Sparsity Level: {}".format(layer_id+1, layer_type, sp_perc))
    # Calculate the overall sparsity of the network!
    sparsity_overall = total_pruned/total_params
    if verbose:
        print("Overall mean sparsity: {}".format(sparsity_overall))
    return sparsity_overall, sparsity_layers


def evaluate_pruning(net_config, load_in_path, criterion, device, test_loader,
                     sparsity_levels, prune_type="weight"):
    """ Pruning Test Accuracy Evaluation - Prune Types: weight, unit, filter """
    accuracy_levels = []
    for sparse_level in sparsity_levels:
        # Define and load the full trained network
        prune_net = MaskedBodyBuilder(**net_config, load_in_path=load_in_path)
        prune_net.to(device)
        # Get the weight pruning masks & set them!
        if prune_type == "weight":
            masks_net = weight_prune_masks(prune_net, sparse_level)
        elif prune_type == "unit":
            masks_net = unit_prune_masks(prune_net, sparse_level)
        elif prune_type == "filter":
            masks_net = filter_prune_masks(prune_net, sparse_level)
        else:
            raise ValueError
        prune_net.set_masks(masks_net)
        sparsity_overall, sparsity_layers = summarize_pruning(prune_net,
                                                              verbose=False)
        # Evaluate the performance
        dojo = TrainDojo(prune_net, optimizer=None, criterion=criterion, device=device,
                         problem_type="classification", train_loader=None,
                         test_loader=test_loader)
        loss, accuracy = dojo.get_network_performance(test=True)
        print("Type: {} | Sparsity (Exluding Readout): {:.2f} | Test Accuracy: {:.3f}".format(prune_type,
                                                                                   sparsity_overall,
                                                                                   accuracy))
        accuracy_levels.append(accuracy)
    return accuracy_levels
