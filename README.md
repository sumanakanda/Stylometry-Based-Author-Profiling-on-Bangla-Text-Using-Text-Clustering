Here’s a detailed and structured README file for your GitHub project based on the analysis:

---

# Stylometry-based Author Profiling on Bangla Text Using Text Clustering

This repository contains the implementation of our final-year capstone project. It leverages advanced machine learning techniques to analyze and profile Bangla text authors based on their unique writing styles.

## Project Overview

The project uses stylometry, a computational approach to author profiling, to cluster Bangla texts from 16 authors. With advanced text preprocessing, feature extraction, dimensionality reduction, and clustering algorithms, this system identifies distinct authorship patterns in Bangla literature.

---

## Features
- **Custom Bangla Tokenization**: Removes punctuation and cleans text for meaningful analysis.
- **TF-IDF Vectorization**: Extracts unigram and bigram features for numerical representation.
- **Dimensionality Reduction**:
  - **PCA and LSA**: Initial reduction to remove noise.
  - **t-SNE**: Final reduction for effective clustering and visualization.
- **Clustering Algorithms**:
  - **K-Means**: Groups texts based on similarity.
  - **DBSCAN**: Identifies clusters of varying densities.
- **Evaluation Metrics**: Assesses clustering performance using ARI, NMI, FMI, and more.

---

## Technologies Used
- **Programming Language**: Python 3.10
- **Libraries**:
  - `scikit-learn` for clustering, dimensionality reduction, and evaluation.
  - `pandas` and `numpy` for data manipulation.
  - `matplotlib` for visualizations.
  - `tqdm` for tracking execution progress.

---

## Dataset
We used the **BAAD16** dataset, a collection of writings from 16 Bangla authors. It contains over 13.4 million words, partitioned into equal segments of 1500 words for analysis.

---

## Code Workflow

### 1. **Preprocessing**
- **Custom Tokenization**:
  A regular expression-based function cleans and tokenizes Bangla text, removing punctuation and extraneous characters.

### 2. **Feature Extraction**
- **TF-IDF Vectorization**:
  Transforms text into numerical features with a maximum of 1000 features, capturing both unigram and bigram patterns.

### 3. **Dimensionality Reduction**
- **Step 1**: PCA or LSA reduces the 1000-dimensional TF-IDF features to 100 components.
- **Step 2**: t-SNE further reduces the dimensionality to 3 components for clustering and visualization.

### 4. **Clustering**
- **K-Means**:
  Groups data into 16 clusters corresponding to the 16 authors in the dataset.
- **DBSCAN**:
  Identifies clusters and outliers based on data density.

### 5. **Evaluation**
- External Metrics:
  - Adjusted Rand Index (ARI): Measures similarity with ground truth labels.
  - Normalized Mutual Information (NMI): Quantifies the overlap between predicted clusters and true labels.
- Internal Metrics:
  - Homogeneity, Completeness, and V-measure assess cluster quality.

---

## Results

### Without Dimensionality Reduction
- **K-Means**: ARI: 66.17%, NMI: 85.54%
- **DBSCAN**: Unable to form clusters due to high-dimensional noise.

### PCA or LSA with 3 Components
- **K-Means**:
  - PCA: ARI: 45.45%, NMI: 62.18%
  - LSA: ARI: 33.87%, NMI: 55.13%
- **DBSCAN**:
  - PCA: ARI: 25.25%, NMI: 44.94%
  - LSA: ARI: 38.56%, NMI: 50.02%

### PCA/LSA + t-SNE
- **PCA + t-SNE**:
  - K-Means: ARI: 65.45%, NMI: 87.16%
  - DBSCAN: ARI: 93.86%, NMI: 94.61%
- **LSA + t-SNE**:
  - K-Means: ARI: 64.32%, NMI: 85.47%
  - DBSCAN: ARI: 85.29%, NMI: 90.76%

---

## Visualizations
Clusters visualized in 2D/3D space show distinct groupings when using t-SNE after PCA or LSA.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name/stylometry-bangla.git
   cd stylometry-bangla
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Add the dataset to the `/dataset` directory.

---

## Usage
1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook CSE400.ipynb
   ```
2. Follow the steps for:
   - Preprocessing
   - Feature extraction
   - Dimensionality reduction
   - Clustering and evaluation

---

## Future Work
- Expand the dataset to include more authors and diverse genres.
- Optimize clustering and dimensionality reduction techniques.
- Integrate additional visualization methods for insights.

---

## Acknowledgments
We thank:
- **Dr. Mohammad Rezwanul Huq** for his guidance.
- **East West University** for providing resources to complete this project.

---
