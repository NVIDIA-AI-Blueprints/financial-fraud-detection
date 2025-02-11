import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import xgboost as xgb
from captum.attr import ShapleyValueSampling, IntegratedGradients
import shap
import numpy as np

# Define the GNN model
class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# Instantiate GNN
input_dim = 10
hidden_dim = 32
output_dim = 16
gnn_model = GNN(input_dim, hidden_dim, output_dim)

# Instantiate XGBoost
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=3)

# Dummy graph data
node_features = torch.rand(100, input_dim)  # 100 nodes, 10 features each
edge_index = torch.randint(0, 100, (2, 500))  # 500 edges
edge_index = edge_index.to(torch.long)  # Convert edge_index to long tensor
train_mask = torch.rand(100) > 0.5
labels = torch.randint(0, 3, (100,))  # Random labels

# Step 1: Train GNN and extract embeddings
gnn_model.eval()  # Set GNN to evaluation mode
embeddings = gnn_model(node_features, edge_index).detach()

# Step 2: Train XGBoost on GNN embeddings
xgb_model.fit(embeddings[train_mask].numpy(), labels[train_mask].numpy())
# Define a callable wrapper for the model
def model_predict(data):
    return xgb_model.predict_proba(data)

# Step 3: SHAP for XGBoost
explainer = shap.SamplingExplainer(model_predict, embeddings.numpy())
shap_values = explainer.shap_values(embeddings.numpy())

# Visualize SHAP values for XGBoost
shap.summary_plot(shap_values, embeddings.numpy())
#print(shap_values)

# Step 4: Use Captum for SHAP values on GNN
gnn_model.train()  # Back to trainable mode
captum_shap = ShapleyValueSampling(gnn_model)

# Explain GNN embeddings for a specific node
#attributions = captum_shap.attribute(
#    inputs=node_features, 
#    target=0,  # Target class
#    n_samples = 5
#)

def forward_with_edge_index(x):
    return gnn_model(x, edge_index)

# Visualize Captum results (Integrated Gradients example)
ig = IntegratedGradients(forward_with_edge_index)
#print (ig)
print ("node, edge", node_features.shape, edge_index.shape)
ig_attr = ig.attribute(
    inputs=node_features, 
    target=0  # Target class
)

print("Captum SHAP Attributions:", ig_attr)

