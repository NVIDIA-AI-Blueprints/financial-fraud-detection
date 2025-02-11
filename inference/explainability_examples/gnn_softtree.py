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
from torch_geometric.nn import GCNConv
from captum.attr import IntegratedGradients

# Define the Soft Decision Tree
class SoftDecisionTree(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SoftDecisionTree, self).__init__()
        self.gate_weights = nn.Parameter(torch.randn(hidden_dim, input_dim))
        self.gate_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.leaf_weights = nn.Parameter(torch.randn(hidden_dim, num_classes))

    def forward(self, x):
        gates = torch.sigmoid(torch.matmul(x, self.gate_weights.T) + self.gate_bias)
        outputs = torch.matmul(gates, self.leaf_weights)
        return outputs

# Define the GNN with Soft Decision Tree
class GNNWithSoftTree(nn.Module):
    def __init__(self, input_dim, gnn_hidden_dim, tree_hidden_dim, num_classes):
        super(GNNWithSoftTree, self).__init__()
        self.conv1 = GCNConv(input_dim, gnn_hidden_dim)
        self.conv2 = GCNConv(gnn_hidden_dim, tree_hidden_dim)
        self.tree = SoftDecisionTree(tree_hidden_dim, tree_hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        outputs = self.tree(x)
        return outputs

# Parameters
input_dim = 10          # Number of input features
gnn_hidden_dim = 32     # Hidden dimension for GNN
tree_hidden_dim = 16    # Hidden dimension for Soft Decision Tree
num_classes = 3         # Number of output classes
num_nodes = 100         # Number of nodes in the graph
num_edges = 500         # Number of edges in the graph

# Generate dummy data
node_features = torch.rand(num_nodes, input_dim)  # (num_nodes, input_dim)
edge_index = torch.randint(0, num_nodes, (2, num_edges))  # (2, num_edges)

# Ensure edge_index has valid indices
assert edge_index.max().item() < num_nodes, "Edge index contains out-of-range node indices."
assert edge_index.min().item() >= 0, "Edge index contains negative indices."

labels = torch.randint(0, num_classes, (num_nodes,))  # Random labels for nodes

# Initialize the model
model = GNNWithSoftTree(input_dim, gnn_hidden_dim, tree_hidden_dim, num_classes)

# Forward pass
logits = model(node_features, edge_index)
print("Logits shape:", logits.shape)  # (num_nodes, num_classes)

# Training example
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

loss = criterion(logits, labels)
loss.backward()
optimizer.step()

print("Training loss:", loss.item())

# Captum Integrated Gradients for interpretability
model.eval()

def forward_with_edge_index(x):
    return model(x, edge_index)

# Initialize Integrated Gradients
ig = IntegratedGradients(forward_with_edge_index)

# Compute attributions for the GNN model
attributions = ig.attribute(inputs=node_features, target=0)  # Target class index
print("Attributions shape:", attributions.shape)  # (num_nodes, input_dim)

