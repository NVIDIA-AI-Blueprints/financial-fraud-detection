# Copyright (c) 2025, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv


class GraphSAGE(torch.nn.Module):
    """
    GraphSAGE model for graph-based learning.

    This model learns node embeddings by aggregating information from a node's
    neighborhood using multiple graph convolutional layers.

    Parameters:
    ----------
    in_channels : int
        The number of input features for each node.
    hidden_channels : int
        The number of hidden units in each layer, controlling
        the embedding dimension.
    out_channels : int
        The number of output features (or classes) for the final layer.
    n_hops : int
        The number of GraphSAGE layers (or hops) used to aggregate information
        from neighboring nodes.
    dropout_prob : float, optional (default=0.25)
        The probability of dropping out nodes during training for
        regularization.
    """

    def __init__(
        self, in_channels, hidden_channels, out_channels, n_hops, dropout_prob=0.25
    ):
        super(GraphSAGE, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.n_hops = n_hops

        # list of conv layers
        self.convs = nn.ModuleList()
        # add first conv layer to the list
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        # add the remaining conv layers to the list
        for _ in range(n_hops - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        # output layer
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.dropout_prob = dropout_prob

    def forward(self, x, edge_index, return_hidden: bool = False):

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if return_hidden:
            return x
        else:
            return self.fc(x)
