# Financial Fraud Detection

## 1.0.0

### New Features

- Initial release of the NVIDIA AI Blueprint for Financial Fraud Detection.
- End-to-end workflow: data preparation, model training via the `financial-fraud-training` container, and inference with NVIDIA Dynamo-Triton.
- GNN (GraphSAGE / TransformerConv) embeddings combined with an XGBoost classifier, with Shapley-value explainability.
- Two modeling formulations on the IBM TabFormer dataset: link prediction (`financial-fraud-usage-link-prediction.ipynb`) and node prediction (`financial-fraud-usage-np.ipynb`).
