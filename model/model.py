"""
Some code are adapted from https://github.com/liyaguang/DCRNN
and https://github.com/xlwang233/pytorch-DCRNN, which are
licensed under the MIT License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from data.data_utils import computeFFT
from model.cell import DCGRUCell
from torch.autograd import Variable
import utils
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import config


def apply_tuple(tup, fn):
    """Apply a function to a Tensor or a tuple of Tensor"""
    if isinstance(tup, tuple):
        return tuple((fn(x) if isinstance(x, torch.Tensor) else x) for x in tup)
    else:
        return fn(tup)


def concat_tuple(tups, dim=0):
    """Concat a list of Tensors or a list of tuples of Tensor"""
    if isinstance(tups[0], tuple):
        return tuple(
            (torch.cat(xs, dim) if isinstance(xs[0], torch.Tensor) else xs[0])
            for xs in zip(*tups)
        )
    else:
        return torch.cat(tups, dim)


class DCRNNEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        max_diffusion_step,
        hid_dim,
        num_nodes,
        num_rnn_layers,
        dcgru_activation=None,
        filter_type="laplacian",
        device=None,
    ):
        super(DCRNNEncoder, self).__init__()
        self.hid_dim = hid_dim
        self.num_rnn_layers = num_rnn_layers
        self._device = device

        encoding_cells = list()
        # the first layer has different input_dim
        encoding_cells.append(
            DCGRUCell(
                input_dim=input_dim,
                num_units=hid_dim,
                max_diffusion_step=max_diffusion_step,
                num_nodes=num_nodes,
                nonlinearity=dcgru_activation,
                filter_type=filter_type,
            )
        )

        # construct multi-layer rnn
        for _ in range(1, num_rnn_layers):
            encoding_cells.append(
                DCGRUCell(
                    input_dim=hid_dim,
                    num_units=hid_dim,
                    max_diffusion_step=max_diffusion_step,
                    num_nodes=num_nodes,
                    nonlinearity=dcgru_activation,
                    filter_type=filter_type,
                )
            )
        self.encoding_cells = nn.ModuleList(encoding_cells)

    def forward(self, inputs, initial_hidden_state, supports):
        seq_length = inputs.shape[0]
        batch_size = inputs.shape[1]
        # (seq_length, batch_size, num_nodes*input_dim)
        inputs = torch.reshape(inputs, (seq_length, batch_size, -1))

        current_inputs = inputs
        # the output hidden states, shape (num_layers, batch, outdim)
        output_hidden = []
        for i_layer in range(self.num_rnn_layers):
            hidden_state = initial_hidden_state[i_layer]
            output_inner = []
            for t in range(seq_length):
                _, hidden_state = self.encoding_cells[i_layer](
                    supports, current_inputs[t, ...], hidden_state
                )
                output_inner.append(hidden_state)
            output_hidden.append(hidden_state)
            current_inputs = torch.stack(output_inner, dim=0).to(
                self._device
            )  # (seq_len, batch_size, num_nodes * rnn_units)
        output_hidden = torch.stack(output_hidden, dim=0).to(
            self._device
        )  # (num_layers, batch_size, num_nodes * rnn_units)
        return output_hidden, current_inputs

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_rnn_layers):
            init_states.append(self.encoding_cells[i].init_hidden(batch_size))
        # (num_layers, batch_size, num_nodes * rnn_units)
        return torch.stack(init_states, dim=0)


class DCRNNModel_DoubleEncoder(nn.Module):
    def __init__(self, device=None):
        super(DCRNNModel_DoubleEncoder, self).__init__()

        num_nodes = config.num_nodes
        num_rnn_layers = config.rnn_layers
        rnn_units = config.rnn_units
        enc_input_dim = config.input_dim
        max_diffusion_step = config.diffusion_step
        # print(config.classes)
        num_classes = config.classes

        self.num_nodes = num_nodes
        self.num_rnn_layers = num_rnn_layers
        self.rnn_units = rnn_units
        self._device = device
        self.num_classes = num_classes

        # First DCRNN Encoder
        self.encoder1 = DCRNNEncoder(
            input_dim=enc_input_dim,
            max_diffusion_step=max_diffusion_step,
            hid_dim=rnn_units,
            num_nodes=num_nodes,
            num_rnn_layers=num_rnn_layers,
            dcgru_activation="tanh",
            filter_type="laplacian",
        )

        # Second DCRNN Encoder with potentially different parameters
        self.encoder2 = DCRNNEncoder(
            input_dim=enc_input_dim,
            max_diffusion_step=max_diffusion_step,
            hid_dim=rnn_units,
            num_nodes=num_nodes,
            num_rnn_layers=num_rnn_layers,
            dcgru_activation="tanh",
            filter_type="laplacian",
        )

        # Final FC layer after concatenating outputs
        self.fc = nn.Linear(rnn_units * 2, num_classes)  # Concatenating outputs, so *2

        self.dropout = nn.Dropout(config.dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, input_seq, seq_lengths, supports1, supports2):
        """
        Args:
            input_seq: input sequence, shape (batch, seq_len, num_nodes, input_dim)
            seq_lengths: actual seq lengths w/o padding, shape (batch,)
            supports: list of supports from laplacian or dual_random_walk filters
        Returns:
            pool_logits: logits from last FC layer (before sigmoid/softmax)
        """
        batch_size, max_seq_len = input_seq.shape[0], input_seq.shape[1]

        # Transpose input_seq to (seq_len, batch, num_nodes, input_dim)
        input_seq = torch.transpose(input_seq, dim0=0, dim1=1)

        # Initialize hidden states for both encoders
        init_hidden_state1 = self.encoder1.init_hidden(batch_size).to(self._device)
        init_hidden_state2 = self.encoder2.init_hidden(batch_size).to(self._device)

        # Pass through first encoder
        _, final_hidden1 = self.encoder1(input_seq, init_hidden_state1, supports1)

        # Pass through second encoder
        _, final_hidden2 = self.encoder2(input_seq, init_hidden_state2, supports2)

        # Concatenate final hidden states from both encoders
        final_hidden_concat = torch.cat((final_hidden1, final_hidden2), dim=-1)

        # Transpose to (batch_size, seq_len, rnn_units * 2)
        final_hidden_concat = torch.transpose(final_hidden_concat, dim0=0, dim1=1)

        # Extract last relevant output
        last_out = utils.last_relevant_pytorch(
            final_hidden_concat, seq_lengths, batch_first=True
        )

        # Reshape last_out to match the input dimensions of self.fc
        last_out = last_out.view(batch_size, self.num_nodes, self.rnn_units * 2)

        # Final FC layer
        logits = self.fc(self.relu(self.dropout(last_out)))

        # Max-pooling over nodes
        pool_logits, _ = torch.max(logits, dim=1)  # (batch_size, num_classes)

        return pool_logits
