import os
import time

import torch.nn as nn
import torch


class NN(nn.Module):

    def __init__(self, in_feature, out_feature, nf=64):

        super(NN, self).__init__()

        model = [
            # nn.Linear(in_feature, 2*in_feature//3),
            # nn.BatchNorm1d(2*in_feature//3),
            # nn.ReLU(True),
            # nn.Linear(2*in_feature // 3, out_feature),
            # nn.Sigmoid(),

            nn.Conv1d(in_feature, nf, 3, padding=1),
            nn.BatchNorm1d(nf),
            nn.ReLU(True),

            nn.Conv1d(nf, 2*nf, 3, padding=1),
            nn.BatchNorm1d(2*nf),
            nn.ReLU(True),

            nn.Conv1d(2*nf, 4*nf, 3, padding=1),
            nn.BatchNorm1d(4*nf),
            nn.ReLU(True),

            nn.Conv1d(4*nf, 2*nf, 3, padding=1),
            nn.BatchNorm1d(2*nf),
            nn.ReLU(True),

            nn.Conv1d(2*nf, 1*nf, 3, padding=1),
            nn.BatchNorm1d(1*nf),
            nn.ReLU(True),

            nn.Conv1d(1*nf, out_feature, 3, padding=1),
            nn.Sigmoid(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        """Standard forward."""
        return self.model(x)


class ModelRGB:
    def __init__(self, expr_dir, batch_size=None, epoch_count=1, niter=150, niter_decay=150, beta1=0.5, lr=0.0002, nf=64):

        self.expr_dir = expr_dir
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size

        self.epoch_count = epoch_count
        self.niter = niter
        self.niter_decay = niter_decay
        self.beta1 = beta1
        self.lr = lr
        self.old_lr = self.lr

        self.print_freq = 500

        self.net = NN(in_feature=3, out_feature=1, nf=nf)

        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          lr=self.lr, betas=(self.beta1, 0.999))
        self.loss = nn.MSELoss()
#        self.loss = nn.L1Loss()

        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)

    def train(self, dataset):

        self.batch_size = dataset.batch_size

        out_f = open(f"{self.expr_dir}/results.txt", 'w')

        total_steps = 0
        print_start_time = time.time()

        self.net.to(self.device)
        for epoch in range(self.epoch_count, self.niter + self.niter_decay + 1):
            epoch_start_time = time.time()
            epoch_iter = 0

            for rgb, dose in dataset:
                total_steps += self.batch_size
                epoch_iter += self.batch_size

                dose_predicted = self.net.forward(rgb.transpose(1,2))
                self.optimizer.zero_grad()
                loss = self.loss(dose_predicted, dose)
                loss.backward()
                self.optimizer.step()
                loss_value = loss.data.item()

                if total_steps % self.print_freq == 0:
                    t = (time.time() - print_start_time) / self.batch_size
                    print_log(out_f, format_log(epoch, epoch_iter, loss_value, t))
                    print_start_time = time.time()

            if epoch % 2 == 0:
                print_log(out_f, 'saving the model at the end of epoch %d, iterations %d' %
                          (epoch, total_steps))
                self.save('latest')

            print_log(out_f, 'End of epoch %d / %d \t Time Taken: %d sec' %
                      (epoch, self.niter + self.niter_decay, time.time() - epoch_start_time))

            if epoch > self.niter:
                self.update_learning_rate()

        out_f.close()

    def update_learning_rate(self):
        """Update learning rate."""
        lrd = self.lr / self.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def save(self, checkpoint_name):
        """
        Save the model and optimizer.

        :param checkpoint_name: name of the checkpoint.
        :type checkpoint_name: str
        """
        checkpoint_path = os.path.join(self.expr_dir, checkpoint_name)
        checkpoint = {
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)


class ModelRBGB:
    def __init__(self, expr_dir, batch_size=None, epoch_count=1, niter=150, niter_decay=150, beta1=0.5, lr=0.0002, nf=64):

        self.expr_dir = expr_dir
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size

        self.epoch_count = epoch_count
        self.niter = niter
        self.niter_decay = niter_decay
        self.beta1 = beta1
        self.lr = lr
        self.old_lr = self.lr

        self.print_freq = 500

        self.net = NN(in_feature=2, out_feature=1, nf=nf)

        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          lr=self.lr, betas=(self.beta1, 0.999))
        self.loss = nn.MSELoss()
#        self.loss = nn.L1Loss()

        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)

    def train(self, dataset):

        self.batch_size = dataset.batch_size

        out_f = open(f"{self.expr_dir}/results.txt", 'w')

        total_steps = 0
        print_start_time = time.time()

        self.net.to(self.device)
        for epoch in range(self.epoch_count, self.niter + self.niter_decay + 1):
            epoch_start_time = time.time()
            epoch_iter = 0

            for rgb, dose in dataset:
                total_steps += self.batch_size
                epoch_iter += self.batch_size


                dose_predicted = self.net.forward(rgb.transpose(1,2))
                self.optimizer.zero_grad()
                loss = self.loss(dose_predicted, dose)
                loss.backward()
                self.optimizer.step()
                loss_value = loss.data.item()

                if total_steps % self.print_freq == 0:
                    t = (time.time() - print_start_time) / self.batch_size
                    print_log(out_f, format_log(epoch, epoch_iter, loss_value, t))
                    print_start_time = time.time()

            if epoch % 2 == 0:
                print_log(out_f, 'saving the model at the end of epoch %d, iterations %d' %
                          (epoch, total_steps))
                self.save('latest')

            print_log(out_f, 'End of epoch %d / %d \t Time Taken: %d sec' %
                      (epoch, self.niter + self.niter_decay, time.time() - epoch_start_time))

            if epoch > self.niter:
                self.update_learning_rate()

        out_f.close()

    def update_learning_rate(self):
        """Update learning rate."""
        lrd = self.lr / self.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def save(self, checkpoint_name):
        """
        Save the model and optimizer.

        :param checkpoint_name: name of the checkpoint.
        :type checkpoint_name: str
        """
        checkpoint_path = os.path.join(self.expr_dir, checkpoint_name)
        checkpoint = {
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)


def print_log(out_f, message):
    """
    Writes in log file.

    :param out_f: I/O stream.
    :type out_f:
    :param message: message to display.
    :type message: str
    """
    out_f.write(message + "\n")
    out_f.flush()
    print(message)


def format_log(epoch, iteration, error, t):
    """
    Generic format for print/log.

    :param epoch: epoch.
    :type epoch: int
    :param iteration: iteration.
    :type iteration: int
    :param error: error.
    :type error: float
    :param t: time.
    :type t: float
    :return: message.
    :rtype: str
    """
    message = '(epoch: %d, iteration: %d, time: %.3f) ' % (epoch, iteration, t)
    message += '%s: %.3f ' % ("Loss", error)
    return message
