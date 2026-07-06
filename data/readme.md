# Data Folder

This folder holds the dataset and all generated artifacts for the Financial Fraud Detection workflow.

**Place the downloaded `transactions.tgz` under `data/TabFormer/raw/`** — the notebook untars it in place to produce `card_transaction.v1.csv`. The other directories are **created later by the workflow**: the preprocessing step produces the graph inputs and the training container produces the model artifacts. Which graph directory appears depends on which notebook you run: `gnn/` for `financial-fraud-usage-link-prediction.ipynb` (link prediction) and `gnn_np/` for `financial-fraud-usage-np.ipynb` (node prediction).

The workflow uses the **IBM TabFormer** credit card transaction dataset (Apache 2.0, ~24M records / 15 fields, one field being the `is_fraud` label). See the main [README](../README.md) and [notebooks/extra/download.md](../notebooks/extra/download.md) for how to obtain it.

## Layout

```
data/
└── TabFormer/
    ├── raw/          # place transactions.tgz here; notebook untars it to card_transaction.v1.csv
    ├── gnn/          # link-prediction graph (created by financial-fraud-usage-link-prediction.ipynb)
    ├── gnn_np/       # node-prediction graph (created by financial-fraud-usage-np.ipynb)
    └── trained_models*/   # model artifacts (created by the financial-fraud-training container)
```

When each is created:

- `raw/` — place `transactions.tgz` here (the notebook untars it to `card_transaction.v1.csv`).
- `gnn/` — created when you run `financial-fraud-usage-link-prediction.ipynb` (link-prediction preprocessing).
- `gnn_np/` — created when you run `financial-fraud-usage-np.ipynb` (node-prediction preprocessing).
- `trained_models*/` — created by the **training** step (the `financial-fraud-training` container).

The temporal split produced during preprocessing follows the transaction year: training = before 2018, validation = 2018, test = after 2018.
