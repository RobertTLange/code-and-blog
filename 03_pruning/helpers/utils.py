import os
import torch
import numpy as np
import random
import pickle
import json
import h5py
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

import seaborn as sns
sns.set(context='poster', style='white', palette='Set1',
        font='sans-serif', font_scale=1, color_codes=True, rc=None)


class DotDic(dict):
    """ Helper to load in parameters from json & easily call them """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __deepcopy__(self, memo=None):
        return DotDic(copy.deepcopy(dict(self), memo=memo))


def load_config(config_fname):
    """ Load in a config JSON file and return as a dictionary """
    json_config = json.loads(open(config_fname, 'r').read())
    dict_config = DotDic(json_config)

    # Make inner dictionaries indexable like a class
    for key, value in dict_config.items():
        if isinstance(value, dict):
            dict_config[key] = DotDic(value)
    return dict_config


def set_random_seeds(seed_id, verbose=False):
    """ Set random seed (random, npy, torch, gym) for reproduction """
    os.environ['PYTHONHASHSEED'] = str(seed_id)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed_id)
    random.seed(seed_id)
    np.random.seed(seed_id)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_id)
        torch.cuda.manual_seed(seed_id)
    if verbose:
        print("-- Random seeds (random, numpy, torch) were set to {}".format(seed_id))


def load_log(log_fname):
    """ Load in logging results & mean the results over different runs """
    # Open File & Get array names to load in
    h5f = h5py.File(log_fname, mode="r")
    obj_names = list(h5f[list(h5f.keys())[0]].keys())
    data_to_store = [[] for i in range(len(obj_names))]

    for seed_id in h5f.keys():
        run = h5f[seed_id]
        for i, o_name in enumerate(obj_names):
            data_to_store[i].append(run[o_name][:])
    h5f.close()

    # Post process results - Mean over different runs
    result_dict = {key: {} for key in obj_names}
    for i, o_name in enumerate(obj_names):
        result_dict[o_name]["mean"] = np.mean(data_to_store[i], axis=0)
        result_dict[o_name]["std"] = np.std(data_to_store[i], axis=0)
    # Return as dot-callable dictionary
    return DotDic(result_dict)


def plot_learning(steps, stats_mean_list=None, stats_std_list=None,
                  labels=[], title="Learning Curve", ylabel="Loss",
                  y_lim=[0.0, 1.7], share_x=False, share_y=False,
                  legend_loc="best", filter_ws=0, step_size=10000,
                  xlabel=r"$\times 10^4$ Batch Iterations"):
    """ Plot the learning curve of the different networks trained """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    # Loop over individual lines, smooth the arrays &
    for i in range(len(labels)):
        if filter_ws > 0:
            temp_mean = savgol_filter(stats_mean_list[i], filter_ws, 3)
            if stats_std_list is not None:
                temp_std = savgol_filter(stats_std_list[i], filter_ws, 3)
        else:
            temp_mean = stats_mean_list[i]
            if stats_std_list is not None:
                temp_std = stats_std_list[i]
        ax.plot(steps, temp_mean, label=labels[i])
        if stats_std_list is not None:
            ax.fill_between(steps,
                            temp_mean - 0.5*temp_std,
                            temp_mean + 0.5*temp_std, alpha=0.5)

    its_ticks = np.arange(step_size, np.max(steps), step_size)
    its_labels_temp = [str(int(it/step_size)) for it in its_ticks]
    its_labels = [it_l for it_l in its_labels_temp]
    its_labels[0] = r"$1$"
    ax.set_xticks(its_ticks, minor=False)
    ax.set_xticklabels(its_labels)

    ax.legend(loc=legend_loc, fontsize=12)
    ax.set_title(title)
    if not share_y:
        ax.set_ylabel(ylabel)
    else:
        ax.get_yaxis().set_ticks([])
    if not share_x:
        ax.set_xlabel(xlabel)
    else:
        ax.get_xaxis().set_ticks([])
    ax.set_ylim(bottom=y_lim[0], top=y_lim[1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=25)
    return


def plot_pruning(sparsity_levels, accuracy_1, accuracy_2, labels, title):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(sparsity_levels, accuracy_1, label=labels[0])
    ax.plot(sparsity_levels, accuracy_2, label=labels[1])

    ax.set_xlabel("Sparsity Level")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
