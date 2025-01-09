# Stylometry-Based-Author-Profiling-on-Bangla-Text-Using-Text-Clustering
Stylometry-based Author Profiling on Bangla Text Using Text Clustering
This repository contains the final-year capstone project developed by Ishmam-ur-Rahman, MD Azman Ahmed, Md Suman Akanda, and Md. Sanim Hossain under the supervision of Dr. Mohammad Rezwanul Huq at East West University, Bangladesh.

Project Overview
This project applies stylometry-based techniques and unsupervised text clustering to analyze and profile authors of Bangla literature. Using advanced machine learning methods, the system identifies writing styles and patterns unique to individual authors.

Key Features
Dimensionality Reduction: Combines PCA, LSA, and t-SNE for efficient feature space reduction.
Clustering Algorithms: Implements K-Means and DBSCAN for robust text grouping.
High Accuracy: Achieved 95.81% homogeneity, 94% ARI and NMI scores, and 93.45% completeness.
Table of Contents
Introduction
Features
Technologies Used
Dataset
Code Structure
Installation
Usage
Results
Future Work
Acknowledgments
Introduction
Author profiling involves analyzing textual data to uncover stylistic patterns that differentiate one author from another. This project focuses on Bangla text, leveraging machine learning to provide insights into linguistic characteristics.

Features
Preprocessing: Custom Bangla tokenizer for text cleaning and preparation.
Feature Extraction: TF-IDF vectorization for unigram and bigram features.
Clustering Techniques: Employs K-Means and DBSCAN with robust evaluation metrics.
Visualization: Uses t-SNE for 3D cluster visualization.
Technologies Used
Programming Language: Python 3.10
Libraries:
scikit-learn for clustering, dimensionality reduction, and evaluation metrics.
Pandas for data preprocessing.
Matplotlib for visualization.
Dataset
The BAAD16 dataset contains writings from 16 Bangla authors, with over 13.4 million words. Each document is preprocessed into a uniform length for consistency.



The project achieved the following:

Without Dimensionality Reduction:
ARI: 66.17%, NMI: 85.54%.
With PCA-then-t-SNE:
ARI: 93.86%, NMI: 94.61%.
Future Work
Expand the dataset to include more authors and genres.
Explore additional clustering algorithms like hierarchical clustering.
Optimize t-SNE hyperparameters for computational efficiency.
Acknowledgments
We extend our gratitude to:

Dr. Mohammad Rezwanul Huq for his invaluable guidance.
East West University for providing the opportunity to complete this project.
Would you like to customize any section further?



