# cmse802_project

* Project Title: Homology Fed Neural Network

* Brief Description: This project explores the integration of topological invariants such as betti
numbers into graph learning models using topological regularization. In traditional
machine learning, graphs are processed through Graph Neural Networks (GNNs), but
these methods may not adequately capture higher-order structures like loops, voids,
or other topological features that can be essential for graph-based problems. The
goal of this project is to incorporate homology as a regularization term in the train-
ing of NNs. This regularization will act to preserve certain topological features of
the graph during learning, helping the model to respect the underlying topology of
the data. This is especially relevant in applications where the graph’s topology is
a key component of the information being represented, such as in social networks,
biological networks, or citation graphs.


1. **Objective 1: Develop a Neural Network Framework Incorporating Topological Invariants.**
   - **Updated Goal**:Implement a neural network capable of integrating Betti numbers, persistent homology, and Mayer homology as input features for molecular property prediction.
   - **Measurable**: Successfully process and utilize homology-derived features, ensuring that the network predicts at least 17 molecular properties with measurable accuracy.
   - **Alignment with CMSE 802**: Ties to the course's emphasis on advanced machine learning techniques and their application to domain-specific data.

2. **Objective 2: Address Computational Challenges in Homology Computations**
   - **Updated Goal**:Optimize the computation of Betti numbers, persistence diagrams, and Mayer homology invariants to handle larger datasets efficiently.
   - **Measurable**: Achieve a significant reduction in computation time (e.g., 30–50% improvement compared to the current baseline) without loss of accuracy.
   - **Alignment with CMSE 802**: Relates to discussions on computational efficiency and algorithmic optimization in data science workflows.

3. **Objective 3: Evaluate the Contribution of Different Topological Invariants**
  - **Updated Goal**:Analyze the predictive power of individual and combined homological features (Betti numbers, persistence, Mayer homology) in the neural network's performance across molecular datasets.   
   - **Measurable**: Provide quantitative comparisons (e.g., mean absolute error and mean squared error) across three different models using separate and integrated topological features.
   - **Alignment with CMSE 802**: Aligns with the course topics on model evaluation, metrics, and feature selection in machine learning.
4. **Objective 4: Prepare a final report and presentation that communicates the implementation, results, and analysis of topological regularization in NNs.**
   - **Measurable**: Deliver a written report and a presentation that clearly outlines the project's methods, results, and conclusions, using appropriate visualizations and metrics.
   - **Alignment with CMSE 802**: This objective supports the final project submission and presentation requirements in the course, emphasizing clarity of communication and effective use of visual tools for explaining complex models.



**Future Work**: Extend the Framework to Diverse Datasets
- **Updated Goal**: Test the framework on multiple datasets (e.g., QM9, citation networks) to verify its adaptability to different types of graph-based problems.
- **Measurable**: Demonstrate improved or competitive performance on at least two datasets compared to traditional graph neural networks.
- **Alignment with CMSE 802**: Connects to discussions on generalizability and the robustness of machine learning models.


The folder structure of the project is as follows:
```
project_root/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── experiments/
│   ├── exp_001/
|
├── src/
│   ├── data_loading/
|   │   └── data_read.py
│   ├── topology/
|   │   ├── path_complexes.py
|   │   ├── boundary_maps.py
|   │   └── betti_numbers.py
|
├── results/
│   └── figures/
│
├── config/
│
├── .gitignore
├── README.md
├── Karaguler_Dilan_HomologyFedNN_FinalReport.pdf
├── Karaguler_Dilan_HomologyFedNN.pptx
└── requirements.txt
```