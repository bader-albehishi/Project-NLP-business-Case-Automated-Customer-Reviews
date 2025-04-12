# 📦 Product Category Clustering using HDBSCAN & Sentence Transformers

This project performs unsupervised clustering of Amazon product reviews by combining sentence embeddings with HDBSCAN and visualizing results using UMAP. The objective is to automatically group similar product names and categories into meaningful meta-categories and generate descriptive names using GPT.

---

## 📌 Project Goals

- Clean and preprocess product name and category text data.
- Generate sentence embeddings using **Sentence Transformers**.
- Cluster the embeddings using **HDBSCAN** to automatically discover product groups.
- Use **UMAP** to visualize the clusters in 2D.
- Evaluate the clustering with metrics like **Silhouette Score** and **Davies-Bouldin Index**.
- Use **OpenAI GPT** to suggest human-readable names for each cluster.
- Export the results for further analysis or dashboard integration.

---

## 🛠️ Technologies Used

- Python, Pandas, NumPy
- [Sentence-Transformers](https://www.sbert.net/)
- [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/)
- [UMAP](https://umap-learn.readthedocs.io/en/latest/)
- OpenAI GPT (via `openai` package)
- Scikit-learn for evaluation
- WordCloud, Matplotlib, Seaborn for visualization

---

## 📂 Dataset

The dataset used is a CSV file:  
**`1429_1 - Copy.csv`**  
It contains at least the following columns:
- `name`: Product name
- `categories`: Product categories

---

## 🚀 How It Works

1. **Preprocessing**  
   Cleans the `name` and `categories` columns, removes stopwords, punctuation, and applies lemmatization.

2. **Embedding**  
   Converts text into numerical vectors using `all-MiniLM-L6-v2` model from `sentence-transformers`.

3. **Clustering**  
   Clusters the embeddings using HDBSCAN, which automatically detects the number of clusters and noise points.

4. **Visualization**  
   Projects high-dimensional embeddings into 2D using UMAP and plots the clusters with color labels.

5. **Evaluation**  
   Calculates:
   - Silhouette Score
   - Davies-Bouldin Index  
   *(excluding noise for accurate evaluation)*

6. **Meta Category Naming**  
   GPT analyzes the top keywords from each cluster and suggests an appropriate category label.

7. **Export**  
   Results are saved in `clustered_reviews.csv` with `meta_category` column included.





---

## 📁 Output Files

- `clustered_reviews.csv` – Final dataset with cluster labels and meta-categories.
- Pickled models (optional) – To reuse clustering and embedding models.
- WordClouds – For keyword visualization in each cluster (optional).

---

## 🧠 Notes

- HDBSCAN is preferred over K-Means for this task due to automatic cluster count and better noise handling.
- Evaluation metrics are helpful, but visualization and human judgment are key in unsupervised tasks.

---

## ✅ Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
