

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



## Code and Process Description

### 1. **Environment Setup**
   The environment is configured using Python libraries like `scikit-learn`, `pandas`, and `matplotlib`. The project runs on Google Colab to utilize cloud resources for computational tasks.

   **Key Code Snippets**:
   ```python
   from sklearn.decomposition import PCA, TruncatedSVD
   from sklearn.manifold import TSNE
   from sklearn.cluster import KMeans, DBSCAN
   from sklearn.feature_extraction.text import TfidfVectorizer
   from sklearn.metrics import silhouette_score, adjusted_rand_score
   ```

---

### 2. **Data Preprocessing**
   - **Dataset Description**: The dataset consists of Bangla text samples from 16 authors, containing 13.4+ million words. Each document is standardized to ~750 words.
   - **Preprocessing Steps**:
     - **Text Cleaning**: Removal of punctuation and stopwords using regular expressions.
     - **Custom Tokenizer**: A Bangla-specific tokenizer to handle linguistic nuances.
     - **Merging**: Rows are merged to create richer context (~1500 words per row).

   **Key Code Snippets**:
   ```python
   def tokenize_bangla(text):
       r = re.compile(r'([\s\।{}]+)'.format(re.escape(punctuation)))
       tokens = [t.strip() for t in r.split(text) if t.strip()]
       return tokens
   ```

---

### 3. **Feature Extraction**
   - **TF-IDF Vectorization**:
     - Converts text into numerical vectors for machine learning algorithms.
     - Configured to capture **unigram** and **bigram** features for richer representation.

   **TF-IDF Parameters**:
   - `max_features=1000` (limits feature size to top 1000 terms).
   - `ngram_range=(1, 2)` (captures unigrams and bigrams).
   - `sublinear_tf=True` (applies logarithmic scaling to term frequencies).

   **Key Code Snippets**:
   ```python
   vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), sublinear_tf=True)
   tfidf_matrix = vectorizer.fit_transform(text_data)
   ```

---

### 4. **Dimensionality Reduction**
   - **Principal Component Analysis (PCA)**:
     - Reduces high-dimensional data to principal components while preserving variance.
   - **Latent Semantic Analysis (LSA)**:
     - Captures semantic relationships using matrix factorization.
   - **t-SNE**:
     - Visualizes high-dimensional data in 3D space while preserving local structure.

   **Key Code Snippets**:
   ```python
   pca = PCA(n_components=100)
   reduced_data = pca.fit_transform(tfidf_matrix.toarray())
   
   tsne = TSNE(n_components=3, perplexity=30)
   tsne_data = tsne.fit_transform(reduced_data)
   ```

---

### 5. **Clustering**
   - **K-Means**:
     - Clusters data into predefined groups (16 clusters for 16 authors).
   - **DBSCAN**:
     - Detects clusters based on density, allowing for arbitrary shapes and noise detection.

   **Key Code Snippets**:
   ```python
   kmeans = KMeans(n_clusters=16, random_state=42)
   kmeans_labels = kmeans.fit_predict(tsne_data)

   dbscan = DBSCAN(eps=2.5, min_samples=8)
   dbscan_labels = dbscan.fit_predict(tsne_data)
   ```

---

### 6. **Evaluation Metrics**
   - **External Metrics**:
     - Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), Fowlkes-Mallows Index (FMI).
   - **Internal Metrics**:
     - Homogeneity, Completeness, V-measure.

   **Key Code Snippets**:
   ```python
   from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure

   ari = adjusted_rand_score(true_labels, kmeans_labels)
   homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(true_labels, kmeans_labels)
   ```

---

### 7. **Visualization**
   - Clustering results are visualized in 2D and 3D space for better interpretability.
   - Tools like `matplotlib` are used for generating plots.

   **Key Code Snippets**:
   ```python
   plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=kmeans_labels, cmap='viridis')
   plt.title("K-Means Clustering Visualization")
   plt.show()
   ```

---

## Results
- **Without Dimensionality Reduction**: ARI = 66.17%, NMI = 85.54%.
- **PCA + t-SNE**: ARI = 93.86%, NMI = 94.61%.
- **DBSCAN Performance**: Superior in detecting noise and outliers.


---

## Visualizations
Clusters visualized in 2D/3D space show distinct groupings when using t-SNE after PCA or LSA.

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
