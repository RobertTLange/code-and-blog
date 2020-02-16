import torch
import torch.nn as nn
import torch.nn.functional as F

class TrainDojo(object):
    """ Learning Loop for basic regression/classification setups """
    def __init__(self, network, optimizer, criterion, device,
                 problem_type,  train_loader, test_loader=None,
                 train_log=None, log_batch_interval=None, scheduler=None):

        self.network = network              # Network to train
        self.criterion = criterion          # Loss criterion to minimize
        self.device = device                # Device for tensors
        self.optimizer = optimizer          # PyTorch optimizer for SGD
        self.scheduler = scheduler          # Learning rate scheduler

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.problem_type = problem_type
        self.logger = train_log
        self.log_batch_interval = log_batch_interval
        self.batch_processed = 0

    def train(self, num_epochs):
        """ Loop over epochs in training & test on all hold-out batches """
        # Get Initial Performance after Network Initialization
        train_performance = self.get_network_performance(test=False)
        if self.test_loader is not None:
            test_performance = self.get_network_performance(test=True)
        else:
            test_performance = [0, 0] if self.problem_type == "classification" else 0

        # Update the logging instance with the initial random performance
        clock_tick = [0, 0, 0]
        stats_tick = train_performance + test_performance
        self.logger.update_log(clock_tick, stats_tick)

        # Save the log & the initial network checkpoint
        self.logger.save_log()
        self.logger.save_network(self.network)


        for epoch_id in range(1, num_epochs+1):
            # Train the network for a single epoch
            self.train_for_epoch(epoch_id)

            # Update the learning rate using the scheduler (if desired)
            if self.scheduler is not None:
                self.scheduler.step()

    def train_for_epoch(self, epoch_id=0):
        """ Perform one epoch of training with train data loader """
        self.network.train()
        # Loop over batches in the training dataset
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Put data on the device
            data, target = data.to(self.device), target.to(self.device)
            # Clear gradients & perform forward as well as backward pass
            self.optimizer.zero_grad()
            output = self.network(data.float())

            # If we are training on a regression problem - change datatype!
            if not self.problem_type == "classification":
                target = target.float()

            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            # Update the batches processed counter
            self.batch_processed += 1

            # Log the performance of the network
            if self.batch_processed % self.log_batch_interval == 0:
                # Get Current Performance after Single Epoch of Training
                train_performance = self.get_network_performance(test=False)
                if self.test_loader is not None:
                    test_performance = self.get_network_performance(test=True)
                else:
                    test_performance = [0, 0] if self.problem_type == "classification" else 0
                # Update the logging instance
                clock_tick = [epoch_id, batch_idx+1, self.batch_processed]
                stats_tick = train_performance + test_performance
                self.logger.update_log(clock_tick, stats_tick)

                # Save the log & the current network checkpoint
                self.logger.save_log()
                self.logger.save_network(self.network)
                self.network.train()
        return

    def get_network_performance(self, test=False):
        """ Get the performance of the network """
        # Log the classifier accuracy if problem is classification
        if self.problem_type == "classification":
            correct_classf = 0

        loader = self.test_loader if test else self.train_loader

        self.network.eval()
        loss = 0
        with torch.no_grad():
            # Loop over batches and get average batch accuracy/loss
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.network(data.float())

                # If we are training on a regression problem - change datatype!
                if not self.problem_type == "classification":
                    target = target.float()

                loss += self.criterion(output, target).sum().item()

                # Get test accuracy
                if self.problem_type == "classification":
                    pred = output.argmax(dim=1, keepdim=True)
                    correct_classf += pred.eq(target.view_as(pred)).sum().item()

        avg_loss = loss/len(loader.dataset)

        if self.problem_type == "classification":
            avg_acc = correct_classf/len(loader.dataset)
            return [avg_loss, avg_acc]
        else:
            return avg_loss


class DistillDojo(TrainDojo):
    """ Monkey Patch TrainDojo Class for Knowledge Distillation Training """
    def __init__(self, teacher_network, alpha, tau, network, optimizer, device,
                 problem_type,  train_loader, test_loader,
                 train_log=None, log_batch_interval=None, scheduler=None):
        TrainDojo.__init__(self, network, optimizer, None, device,
                     problem_type,  train_loader, test_loader,
                     train_log, log_batch_interval, scheduler)
        self.teacher_network = teacher_network
        self.criterion = loss_fn_kd
        self.alpha = alpha
        self.tau = tau

    def train_for_epoch(self, epoch_id=0):
        """ Perform one epoch of training with train data loader """
        self.network.train()
        self.teacher_network.eval()
        # Loop over batches in the training dataset
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # Put data on the device
            data, target = data.to(self.device), target.to(self.device)

            # Clear gradients & perform as well as backward pass
            self.optimizer.zero_grad()
            output = self.network(data)

            # Get Teacher targets & dettach from computation graph
            teacher_output = self.teacher_network(data)
            teacher_output = teacher_output.detach()

            # Compute Knowledge Distillation Loss
            loss = self.criterion(output, target, teacher_output,
                                  self.tau, self.alpha)
            loss.backward()
            self.optimizer.step()

            # Update the batches processed counter
            self.batch_processed += 1

            # Log the performance of the network
            if self.batch_processed % self.log_batch_interval == 0:
                # Get Current Performance after Single Epoch of Training
                train_performance = self.get_network_performance(test=False)
                if self.test_loader is not None:
                    test_performance = self.get_network_performance(test=True)
                else:
                    test_performance = [0, 0] if self.problem_type == "classification" else 0
                # Update the logging instance
                clock_tick = [epoch_id, batch_idx+1, self.batch_processed]
                stats_tick = train_performance + test_performance
                self.logger.update_log(clock_tick, stats_tick)

                # Save the log & the current network checkpoint
                self.logger.save_log()
                self.logger.save_network(self.network)
                self.network.train()
        return

    def get_network_performance(self, test=False):
        """ Get the performance of the network """
        # Log the classifier accuracy if problem is classification
        if self.problem_type == "classification":
            correct_classf = 0

        loader = self.test_loader if test else self.train_loader

        self.network.eval()
        loss = 0
        with torch.no_grad():
            # Loop over batches and get average batch accuracy/loss
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.network(data)

                # Get Teacher targets & dettach from computation graph
                teacher_output = self.teacher_network(data)
                teacher_output = teacher_output.detach()

                # Compute Knowledge Distillation Loss
                loss += self.criterion(output, target.long(), teacher_output,
                                       self.tau, self.alpha).sum().item()

                # Get test accuracy
                if self.problem_type == "classification":
                    pred = output.argmax(dim=1, keepdim=True)
                    correct_classf += pred.eq(target.view_as(pred)).sum().item()

        avg_loss = loss/len(loader.dataset)
        if self.problem_type == "classification":
            avg_acc = correct_classf/len(loader.dataset)
            return [avg_loss, avg_acc]
        else:
            return avg_loss



def loss_fn_kd(output, target, teacher_output, tau, alpha):
    """
    Knowledge-Distillation (KD) loss
    CE(p_student(t=1), y_class) + alpha*CE(soft_p_student(t=tau), soft_p_teacher(t=tau))
    NOTE: Need to input logits to perform temperature trafo
    """
    KD_loss = nn.KLDivLoss()(F.log_softmax(output/tau, dim=1),
                             F.softmax(teacher_output/tau, dim=1)) * (alpha * tau * tau) + \
              F.cross_entropy(output, target) * (1. - alpha)
    return KD_loss
