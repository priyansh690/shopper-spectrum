# 🛒 Shopper Spectrum
### Customer Segmentation & Product Recommendations in E-Commerce

> **Domain:** E-Commerce & Retail Analytics  
> **Problem Type:** Unsupervised ML (Clustering) + Collaborative Filtering  

---

## 📦 Project Structure

```
gallant-heisenberg/
├── data/
│   ├── online_retail.csv        # Transaction dataset (2022–2023)
│   └── rfm_segments.csv         # RFM scores + cluster labels
├── models/
│   ├── kmeans_model.pkl         # Trained KMeans model (K=4)
│   ├── scaler.pkl               # StandardScaler for RFM normalisation
│   ├── cluster_labels_map.pkl   # Cluster ID → Segment label mapping
│   ├── product_similarity.pkl   # Cosine similarity matrix (products)
│   └── product_names.pkl        # Indexed product names list
├── shopper_spectrum.ipynb       # Full ML notebook (EDA → Models)
├── app.py                       # Streamlit web application
├── generate_data.py             # Synthetic dataset generator
├── train_models.py              # Standalone model training script
└── requirements.txt             # Python dependencies
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset
```bash
python generate_data.py
```

### 3. Train Models
Either run the Jupyter notebook (recommended):
```bash
jupyter notebook shopper_spectrum.ipynb
```
Or use the standalone script:
```bash
python train_models.py
```

### 4. Launch Streamlit App
```bash
streamlit run app.py
```

---

## 🧠 Methodology

### RFM Analysis
| Metric | Definition |
|--------|-----------|
| **Recency** | Days since last purchase |
| **Frequency** | Number of distinct invoices |
| **Monetary** | Total spend (£) |

### Customer Segments
| Segment | Description | Strategy |
|---------|-------------|----------|
| 👑 **High-Value** | Recent, frequent, high spenders | VIP rewards, early access |
| ⭐ **Regular** | Moderate frequency & spend | Cross-sell, bundle offers |
| 🔔 **Occasional** | Infrequent buyers | Seasonal promotions |
| ⚠️ **At-Risk** | Long inactive, low value | Win-back campaigns |

### Recommendation System
- **Algorithm:** Item-based Collaborative Filtering
- **Similarity:** Cosine similarity on Customer × Product purchase matrix
- **Output:** Top-5 similar products for any given product

---

## 📊 Notebook Sections

1. **Dataset Collection & Understanding** – Structure, types, missing values
2. **Data Preprocessing** – Clean cancellations, nulls, bad values
3. **Exploratory Data Analysis** – 7+ visualisations (country, products, trends, RFM)
4. **Feature Engineering & Clustering** – RFM → StandardScaler → KMeans → Segments
5. **Recommendation System** – Pivot table → Cosine similarity → Top-N recommendations
6. **Model Evaluation** – Silhouette score, Davies-Bouldin index, segment KPIs

---

## 🛠 Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `Matplotlib` · `Seaborn` · `Plotly` · `Streamlit` · `Joblib`
