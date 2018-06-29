# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import quadprog
from collections import deque

import numpy as np
import torch

from myTorch.projects.overfeeding.recurrent_net import Recurrent


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
                 memory_strength,
                 num_memories,
                 task,
                 use_regularisation,
                 regularisation_constant):

        super(GemModel, self).__init__(device,
                                       input_size,
                                       output_size,
                                       num_layers,
                                       layer_size,
                                       cell_name,
                                       activation,
                                       output_activation, task)
        self.margin = memory_strength
        # self.is_curriculum = args["is_curriculum"]

        self.n_outputs = output_size

        # self.opt = optim.SGD(self.parameters(), args.lr)

        self.num_memories = num_memories

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
        self.nc_per_task = int(self.num_memories / n_tasks)
        # if self.is_curriculum:
        #     self.nc_per_task = int(self.num_memories / n_tasks)
        # else:
        #     self.nc_per_task = self.num_memories
        self.use_regularisation = use_regularisation
        self.regularisation_constant = regularisation_constant

    #
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

    def train_over_one_data_iterate(self, data, task, retain_graph=True):
        # update memory
        if task != self.old_task:
            self.observed_tasks.append(task)
            self.old_task = task
            self.memory_data[task] = deque(maxlen=self.nc_per_task)

        # This entire thing can be replaced by saving the data itself
        self.memory_data[task].append(data)

        # compute gradient on previous tasks
        if len(self.observed_tasks) > 1:
            for tt in range(len(self.observed_tasks) - 1):
                self.zero_grad()
                # fwd/bwd on the examples in the memory
                past_task = self.observed_tasks[tt]
                ptloss = 0.0
                for _data in self.memory_data[past_task]:
                    current_ptloss, _ = super()._compute_loss_and_metrics(data=_data)
                    ptloss += current_ptloss
                ptloss /= len(self.memory_data[past_task])
                ptloss.backward()
                store_grad(self.parameters, self.grads, self.grad_dims,
                           past_task)

        if self.use_regularisation:
            _retain_graph = True
        else:
            _retain_graph = retain_graph
        seqloss, num_correct, num_total = super().train_over_one_data_iterate(data=data,
                                                                              task=task, retain_graph=_retain_graph)
        regularisation_loss = torch.zeros_like(seqloss)

        # check if gradient violates constraints
        if len(self.observed_tasks) > 1:
            # copy gradient
            store_grad(self.parameters, self.grads, self.grad_dims, task)
            indx = torch.LongTensor(self.observed_tasks[:-1]).to(self._device)
            curr_grad = self.grads[:, task].unsqueeze(0)
            prev_grad = self.grads.index_select(1, indx)
            dotp = torch.mm(curr_grad, prev_grad)
            if self.use_regularisation:
                cosine_similarity = dotp / (torch.norm(prev_grad, 2) * torch.norm(curr_grad))
                cosine_loss = torch.sum(torch.min(
                    torch.zeros_like(cosine_similarity), cosine_similarity)[0])
                regularisation_loss = regularisation_loss - \
                                      self.regularisation_constant * cosine_loss
            else:
                if (dotp < 0).sum() != 0:
                    project2cone2(self.grads[:, task].unsqueeze(1),
                                  self.grads.index_select(1, indx), self.margin)
                    # copy gradients back
                    overwrite_grad(self.parameters, self.grads[:, task],
                                   self.grad_dims)
        if (self.use_regularisation):
            self.zero_grad()
            (seqloss + regularisation_loss).backward(retain_graph=retain_graph)
        return seqloss, num_correct, num_total
