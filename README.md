# cmse802_project

* Project Title: Graph Learning with Topological Regularization

* Brief Description: This project explores the integration of topological invariants such as betti numbers and Mayer betti numbers into graph learning models using topological regularization. In traditional machine learning, graphs are processed through Graph Neural Networks (GNNs), but these methods may not adequately capture higher-order structures like loops, voids, or other topological features that can be essential for graph-based problems. The goal of this project is to incorporate persistent Mayer homology as a regularization term in the training of GNNs. This regularization will act to preserve certain topological features of the graph during learning, helping the model to respect the underlying topology of the data. This is especially relevant in applications where the graph’s topology is a key component of the information being represented, such as in social networks, biological networks, or citation graphs.


1. **Objective 1: Develop a Graph Neural Network (GNN) capable of integrating topological regularization using persistent homology.**
   - **Measurable**: Successfully implement the GNN and confirm that it incorporates a persistent homology-based regularization term.
   - **Alignment with CMSE 802**: This objective ties into the course's coverage of graph learning models and advanced neural networks, including the discussion on regularization techniques and graph-based machine learning algorithms.
2. **Objective 2: Evaluate the impact of topological regularization on the performance of the GNN across multiple datasets.**
   - **Measurable**: Compare the performance (accuracy, precision, recall, etc.) of the GNN with and without topological regularization on at least two datasets (e.g., social or citation networks).
   - **Alignment with CMSE 802**: This aligns with the course topics on model evaluation and metrics, particularly in comparing different model architectures and the effectiveness of regularization techniques in improving machine learning outcomes.
3. **Objective 3: Experiment with different topological regularization strengths and analyze their influence on learning outcomes.**
   - **Measurable**: Test three different regularizations such as persistent homology, persistent Mayer homology and Mayer Laplacian strengths and provide quantitative results showing their effect on the model’s performance and topological features.
   - **Alignment with CMSE 802**: This objective aligns with the homework and topics on hyperparameter tuning and optimization, as well as discussions around the control of model complexity through regularization.
4. **Objective 4: Prepare a final report and presentation that communicates the implementation, results, and analysis of topological regularization in GNNs.**
   - **Measurable**: Deliver a written report and a presentation that clearly outlines the project's methods, results, and conclusions, using appropriate visualizations and metrics.
   - **Alignment with CMSE 802**: This objective supports the final project submission and presentation requirements in the course, emphasizing clarity of communication and effective use of visual tools for explaining complex models.



The folder structure of the project is as follows:
```
project_root/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   ├── exploratory/
│   └── final/
│
├── experiments/
│   ├── exp_001/
│   └── exp_002/
|
├── src/
│   ├── data_loading/
|   │   └── data_read.py
│   ├── topology/
|   │   ├── path_complexes.py
|   │   ├── boundary_maps.py
|   │   └── betti_numbers.py
│   ├── model_architecture/
│   ├── evaluation/
│   └── training/
│
├── tests/
│
├── results/
│   ├── figures/
│   └── models/
│
├── config/
│
├── .gitignore
├── README.md
└── requirements.txt
```