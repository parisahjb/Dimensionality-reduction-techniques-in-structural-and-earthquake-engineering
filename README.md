Abstract


Dimensionality reduction (DR) techniques are increasingly crucial in engineering and scientific applications for feature extraction and data visualization purposes. This repository presents a comprehensive evaluation of various dimensionality reduction techniques for analyzing a cognate set of computational and experimental data in the field of structural and earthquake engineering, considering varying numbers of classes.

In the context of class-imbalanced data sets, where observations in each class are disproportionate, the main objective of this study is to assess the performance of different DR techniques and classifiers. The research utilizes a synthetic data set generated through computer simulations, characterizing risks posed by earthquakes to structures. The data is classified into three, five, and eight classes based on the severity of damage in the simulations. Additionally, three existing multi-class imbalanced data sets, developed independently, are also used for evaluation.

The following dimensionality reduction techniques are investigated:

Principal Component Analysis (PCA)
Linear Discriminant Analysis (LDA)
t-Distributed Stochastic Neighbor Embedding (t-SNE)
To address the class-imbalance problem, the Synthetic Minority Oversampling Technique (SMOTE) is employed in conjunction with various classifiers, including:

K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)
Decision Trees (DT)
Random Forest (RF)
MLPClassifier (MLP)
AdaBoost
The evaluation includes both linear and non-linear as well as supervised and unsupervised DR techniques. The goal is to identify the most effective combination of DR and classification methods for these engineering data sets.

Paper Reference
This research study has been published in the Journal of Your Journal Name (DOI: Your DOI). Please refer to the paper for more detailed insights into the methodology, experiments, and findings.

Repository Structure
The repository is organized as follows:

Copy code
|- notebooks/
|  |- 3class_TowerDamage_C.ipynb
|  |- 5class_TowerDamage_C.ipynb
|  |- AdditionalDataSets_C.ipynb
|- utils.py
|- README.md
notebooks/: This directory contains Jupyter notebooks with the implementation of different options for dimensionality reduction and classification.
3class_TowerDamage_C.ipynb: Notebook for analyzing the 3-class tower damage dataset.
5class_TowerDamage_C.ipynb: Notebook for analyzing the 5-class tower damage dataset. (It also covers an 8-class dataset.)
AdditionalDataSets_C.ipynb: Notebook for analyzing additional imbalance datasets.
utils.py: This Python file contains utility functions used in the notebooks.
README.md: You are currently reading this file, which provides an overview of the project.
Requirements
To run the notebooks and utilize the utility functions, ensure you have the following dependencies installed:

Python (version 3.x)
Jupyter Notebook
NumPy
SciPy
scikit-learn
imbalanced-learn
matplotlib
seaborn
How to Use
Clone the repository to your local machine:
bash
Copy code
git clone [https://github.com/yourusername/your-repo-name.git](https://github.com/parisahjb/Dimensionality-reduction-techniques-in-structural-and-earthquake-engineering.git)
Install the required dependencies using pip:
Copy code
pip install numpy scipy scikit-learn imbalanced-learn matplotlib seaborn
Launch Jupyter Notebook and open the desired notebook (e.g., 3class_TowerDamage_C.ipynb).

Follow the instructions in the notebook to execute the code and reproduce the experiments.


