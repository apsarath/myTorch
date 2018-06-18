# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import quadprog
from myTorch.projects.overfeeding.recurrent_net import Recurrent
import torch.nn.functional as F
from collections import deque

# Auxiliary functions useful for GEM's inner optimization.

def compute_offsets(task, nc_per_task, curriculum):
    """
        Compute offsets for curriculum tasks to determine which
        outputs to select for a given task.
    """
    if curriculum:
        offset1 = task * nc_per_task
        offset2 = (task + 1) * nc_per_task
    else:
        offset1 = 0
        offset2 = nc_per_task
    return offset1, offset2


def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1


def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector

        This part could possibly become a bottleneck.
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose())
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))


class GemModel(Recurrent):
    def __init__(self,
                 device,
                 input_size,
                 output_size,
                 num_layers,
                 layer_size,
                 cell_name,
                 activation,
                 output_activation,
                 n_tasks,
                 args,
                 task):

        super(GemModel, self).__init__(device,
                 input_size,
                 output_size,
                 num_layers,
                 layer_size,
                 cell_name,
                 activation,
                 output_activation, task)
        self.margin = args["memory_strength"]
        self.is_curriculum = args["is_curriculum"]

        self.n_outputs = output_size

        # self.opt = optim.SGD(self.parameters(), args.lr)

        self.num_memories = args["num_memories"]

        # allocate episodic memory
        self.memory_data = dict()

        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())
        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks)
        self.grads = self.grads.to(device)

        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.mem_cnt = 0
        if self.is_curriculum:
            self.nc_per_task = int(self.num_memories / n_tasks)
        else:
            self.nc_per_task = self.num_memories

    def forward(self, x, t=0):
        output = super().forward(x)
        # if self.is_curriculum:
            # make sure we predict classes within the current task
            # offset1 = int(t * self.nc_per_task)
            # offset2 = int((t + 1) * self.nc_per_task)
            # if offset1 > 0:
            #     output[:, :offset1].data.fill_(-10e10)
            # if offset2 < self.n_outputs:
            #     output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

    def train_over_one_data_iterate(self, data, t, seqloss, average_accuracy):
        # update memory
        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t
            self.memory_data[t] = deque(maxlen=self.nc_per_task)

        # This entire thing can be replaced by saving the data itself
        self.memory_data[t].append(data)

        # compute gradient on previous tasks
        if len(self.observed_tasks) > 1:
            for tt in range(len(self.observed_tasks) - 1):
                self.zero_grad()
                # fwd/bwd on the examples in the memory
                past_task = self.observed_tasks[tt]
                ptloss = 0
                for _data in self.memory_data[past_task]:
                    ptloss, _ = super()._compute_loss_and_metrics(_data, seqloss=ptloss)
                ptloss.backward()
                store_grad(self.parameters, self.grads, self.grad_dims,
                           past_task)

        # now compute the grad on the current minibatch
        seqloss, average_accuracy = super()._compute_loss_and_metrics(data, seqloss, average_accuracy)
        seqloss.backward(retain_graph=False)

        # check if gradient violates constraints
        if len(self.observed_tasks) > 1:
            # copy gradient
            store_grad(self.parameters, self.grads, self.grad_dims, t)
            indx =  torch.LongTensor(self.observed_tasks[:-1]).to(self._device)
            dotp = torch.mm(self.grads[:, t].unsqueeze(0),
                            self.grads.index_select(1, indx))
            if (dotp < 0).sum() != 0:
                project2cone2(self.grads[:, t].unsqueeze(1),
                              self.grads.index_select(1, indx), self.margin)
                # copy gradients back
                overwrite_grad(self.parameters, self.grads[:, t],
                               self.grad_dims)

        return seqloss, average_accuracy