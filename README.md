# Great-Multimodal-LightGCN-Recommender-Model

This project implements a multi-modal graph recommender system using various graph neural network (GNN) architectures. The system provides recommendations based on user-item interactions and incorporates text, image, and other product features from the Amazon Fashion dataset.

## Features

This repository includes implementations of the following models:

* **GraphSAGE:** A general inductive framework that leverages node feature information to generate node embeddings.
* **Graph Attention Network (GAT):** A neural network architecture that operates on graph-structured data, leveraging masked self-attentional layers.
* **LightGCN:** A simplified GNN model for recommendation that only includes the most essential components of GCNs.

## Dataset

This project uses the [Amazon Fashion dataset](https://nijianmo.github.io/amazon/index.html), which contains product reviews and metadata from Amazon.

## Setup

1.  Clone the repository:
    ```bash
    git clone [https://github.com/your-username/your-repository.git](https://github.com/your-username/your-repository.git)
    ```
2.  Install the required dependencies using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Data Processing:**
    Run the `process_data.py` script to preprocess the raw data and create the necessary graph data files (`items_features.parquet`, `user_item_edges.parquet`, `item_images.parquet`).

    ```bash
    python process_data.py
    ```

2.  **Model Training:**
    You can train any of the implemented models using their respective training scripts:

    * **GraphSAGE:**
        ```bash
        python train_recommender.py
        ```
    * **GAT:**
        ```bash
        python train_recommender_gat.py
        ```
    * **LightGCN:**
        ```bash
        python train_recommender_lightgcn.py
        ```

## File Descriptions

* `process_data.py`: Preprocesses the Amazon Fashion dataset and creates the necessary files for training.
* `train_recommender.py`: Trains the GraphSAGE model.
* `train_recommender_gat.py`: Trains the GAT model.
* `train_recommender_lightgcn.py`: Trains the LightGCN model.
* `Multi Modal Graph Recommender System.ipynb`: A Jupyter Notebook containing all the code for data processing, model training, and analysis.

## Citations:
Jianmo Ni, Jiacheng Li, Julian McAuley. Justifying recommendations using distantly-labeled reviews and fined-grained aspects. Empirical Methods in Natural Language Processing (EMNLP), 2019.

@inproceedings{ni2019justifying,
  title={Justifying recommendations using distantly-labeled reviews and fined-grained aspects},
  author={Ni, Jianmo and Li, Jiacheng and McAuley, Julian},
  booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
  year={2019}
}
