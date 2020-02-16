import torch
import numpy as np
import pandas as pd

import time
import h5py
from tensorboardX import SummaryWriter


class DeepLogger(object):
    def __init__(self, time_to_track, what_to_track, log_fname=None,
                 network_fname=None, seed_id=0, tboard_fname=None,
                 time_to_print=None, what_to_print=[], save_all_ckpth=False,
                 print_every_update=None):
        """
        Logging object for Deep RL experiments - Parameters to specify:
        - Where to log agent (.ckpth) & training stats (.hdf5) to
        - Random seed & folderpath for tensorboard
        - Time index & statistics to print & Verbosity level of logger
        - Whether to save all or only most recent checkpoint of network
        """
        self.current_optim_step = 0
        self.log_save_counter = 0
        self.log_update_counter = 0
        self.seed_id = seed_id
        self.print_every_update = print_every_update if print_every_update is not None else 1
        self.start_time = time.time()

        # Set where to log to (Stats - .hdf5, Network - .ckpth)
        if isinstance(log_fname, str): self.log_save_fname = log_fname
        else: self.log_save_fname = None

        if isinstance(network_fname, str): self.network_save_fname = network_fname
        else: self.network_save_fname = None

        # Boolean and fname list for storing all weights during training
        self.save_all_ckpth = save_all_ckpth
        if self.save_all_ckpth:
            self.ckpth_w_list = []

        # Initialize tensorboard logger/summary writer
        if isinstance(tboard_fname, str):
            self.writer = SummaryWriter(tboard_fname + "_seed_" + str(self.seed_id))
        else:
            self.writer = None

        # Initialize pd dataframes to store logging stats/times
        self.time_to_track = time_to_track + ["time_elapsed"]
        self.what_to_track = what_to_track
        self.clock_to_track = pd.DataFrame(columns=self.time_to_track)
        self.stats_to_track = pd.DataFrame(columns=self.what_to_track)

        # Set up what to print
        self.time_to_print = time_to_print
        self.what_to_print = what_to_print
        self.verbose = len(self.what_to_print)>0

    def update_log(self, clock_tick, stats_tick, network=None, plot_to_tboard=None):
        """ Update with the newest tick of performance stats, net weights """
        # Transform clock_tick, stats_tick lists into pd arrays
        c_tick = pd.DataFrame(columns=self.time_to_track)
        c_tick.loc[0] = clock_tick + [time.time() - self.start_time]
        s_tick = pd.DataFrame(columns=self.what_to_track)
        s_tick.loc[0] = stats_tick

        # Append time tick & results to pandas dataframes
        self.clock_to_track = pd.concat([self.clock_to_track, c_tick], axis=0)
        self.stats_to_track = pd.concat([self.stats_to_track, s_tick], axis=0)

        # Tick up the update counter
        self.log_update_counter += 1

        # Update the tensorboard with the newest event
        if self.writer is not None:
            # Update the tensorboard log
            self.update_tboard(clock_tick, stats_tick, network, plot_to_tboard)

        # Speak to me - bro!!
        if self.verbose and self.log_update_counter % self.print_every_update == 0:
            print(pd.concat([c_tick[self.time_to_print],
                             s_tick[self.what_to_print]], axis=1))

        return

    def update_tboard(self, clock_tick, stats_tick, network, plot_to_tboard=None):
        """ Update the tensorboard with the newest events """
        # Add performance & step counters
        for i, performance_tick in enumerate(self.what_to_track):
            self.writer.add_scalar('performance/' + performance_tick,
                                    stats_tick[i], clock_tick[0])

        # Log the network params & gradients
        if network is not None:
            for name, param in network.named_parameters():
                self.writer.add_histogram('weights/' + name, param.clone().cpu().data.numpy(), stats_tick[i])
                self.writer.add_histogram('gradients/' + name, param.grad.clone().cpu().data.numpy(), stats_tick[i])

        # Add the plot of interest to tboard
        if trace_plot is not None:
            self.writer.add_figure('plot', plot_to_tboard, stats_tick[i])

        # Flush the log event
        self.writer.flush()

    def save_log(self, temp_log_fname=None):
        """ Create compressed .hdf5 file containing group <random-seed-id> """
        if self.log_save_fname is not None: log_name = self.log_save_fname
        else: log_name = None

        # Always use temp_log_fname if it is given!
        if temp_log_fname is not None:
            log_name = temp_log_fname


        if log_name is not None:
            h5f = h5py.File(log_name, 'a')
            g_name = "random-seed-" + str(self.seed_id) + "/"

            # Create the "datasets"/variables to store in the hdf5 file - update counter
            for o_name in self.time_to_track:
                if self.log_save_counter >= 1:
                    del h5f[g_name + o_name]

                h5f.create_dataset(name=g_name + o_name, data=self.clock_to_track[o_name].astype(float),
                                   compression='gzip', compression_opts=4,
                                   dtype='float32')

            # Create the "datasets"/variables to store in the hdf5 file - stats
            for o_name in self.what_to_track:
                if self.log_save_counter >= 1:
                    del h5f[g_name + o_name]
                h5f.create_dataset(name=g_name + o_name, data=self.stats_to_track[o_name].astype(float),
                                   compression='gzip', compression_opts=4,
                                   dtype='float32')

            h5f.flush()
            h5f.close()

            # Tick the log save counter
            self.log_save_counter += 1

    def save_network(self, network, temp_log_fname=None):
        """ Save the current state of the network as a checkpoint """
        # Overwrite the log save hdf5 path
        if temp_log_fname is not None:
            self.network_save_fname = temp_log_fname

        if self.network_save_fname is not None:
            if self.save_all_ckpth:
                # Save the current weights in a designated ckpth file
                base_str = self.network_save_fname.split(".")[0]
                save_fname = base_str + "_" + str(self.log_update_counter) + ".ckpth"
                torch.save(network.model.state_dict(), save_fname)

                # Update the list that stores all ckpth of weights
                self.ckpth_w_list.append(save_fname)
            else:
                # Update the saved weights in a single file!
                torch.save(network.model.state_dict(), self.network_save_fname)
