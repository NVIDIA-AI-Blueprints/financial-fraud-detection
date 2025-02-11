import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import xgboost as xgb
from captum.attr import ShapleyValues, IntegratedGradients

class GNNWithXGB(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_classes):
        super(GNNWithXGB, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.num_classes = num_classes

        # Initialize XGBoost model
        self.xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=num_classes)

    def forward(self, x, edge_index, train_mask=None, labels=None):
        # Step 1: GNN forward pass to get embeddings
        x = self.conv1(x, edge_index).relu()
        embeddings = self.conv2(x, edge_index)

        # Convert embeddings to numpy for XGBoost
        embeddings_np = embeddings.detach().cpu().numpy()

        # Step 2: Use XGBoost for classification
        if self.training:
            # Train XGBoost if in training mode
            assert train_mask is not None and labels is not None, "Training requires train_mask and labels."
            xgb_input = embeddings_np[train_mask.cpu().numpy()]
            y_train = labels[train_mask].cpu().numpy()
            self.xgb_model.fit(xgb_input, y_train)
        else:
            # Predict using XGBoost in evaluation mode
            predictions = self.xgb_model.predict(embeddings_np)
            predictions = torch.tensor(predictions, device=x.device)
            return predictions

        return embeddings  # Return embeddings for potential chaining

# Initialize model and dummy data
input_dim = 10
hidden_dim = 32
output_dim = 16
num_classes = 3
model = GNNWithXGB(input_dim, hidden_dim, output_dim, num_classes)

node_features = torch.rand(100, input_dim)  # 100 nodes, 10 features each
edge_index = torch.randint(0, 100, (2, 500))  # 500 edges
train_mask = torch.rand(100) > 0.5
labels = torch.randint(0, num_classes, (100,))  # Random labels

# Train GNN + XGBoost hybrid
model.train()
model(node_features, edge_index, train_mask, labels)

# Step 3: Use Captum for GNN Attribution
model.eval()  # Set model to evaluation mode
gnn_only = lambda x: model.conv2(model.conv1(x, edge_index).relu(), edge_index)

# Captum Integrated Gradients
ig = IntegratedGradients(gnn_only)
attributions = ig.attribute(inputs=node_features, target=0)

print("Integrated Gradients Attributions:", attributions)

