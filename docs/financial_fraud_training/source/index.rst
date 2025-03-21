Financial Fraud Training Documentation
======================================

~~~~~~~~~~~~
Introduction
~~~~~~~~~~~~

Based on user provided training configuration, the NIM first builds a GNN model that 
produces embeddings for credit card transactions, and then the NIM uses the transaction
embeddings to train an XGBoost model to predict fraud scores of the transactions. 
The NIM encapsulate the complexity of creating the graph in cuGraph and building key-value 
attribute store in WholeGraph.Once the graph is created, the GNN model is 
trained and used to produce the embeddings that are then feed to XGBoost.


~~~~~~~~~~~~~~~~~
Table of Contents
~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 2

   data/index
   configuration/index






Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

This source code and/or documentation ("Licensed Deliverables") are
subject to NVIDIA intellectual property rights under U.S. and
international Copyright laws.