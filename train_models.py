"""
train_models.py - Standalone script to train and save all models.
Run this if you don't want to use the Jupyter notebook.
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import joblib

warnings.filterwarnings('ignore')

DATA_PATH  = 'data/online_retail.csv'
MODELS_DIR = 'models'


def load_and_clean(path: str) -> pd.DataFrame:
    print('\n[1/5] Loading dataset ...')
    df = pd.read_csv(path, dtype={'CustomerID': str})
    print(f'   Raw shape: {df.shape}')

    # Preprocessing
    df = df.dropna(subset=['CustomerID'])
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df = df.drop_duplicates()
    df['InvoiceDate']  = pd.to_datetime(df['InvoiceDate'])
    df['CustomerID']   = df['CustomerID'].astype(str).str.strip().str.split('.').str[0]
    df['TotalAmount']  = df['Quantity'] * df['UnitPrice']

    print(f'   Clean shape : {df.shape}')
    return df


def compute_rfm(df: pd.DataFrame) -> pd.DataFrame:
    print('\n[2/5] Computing RFM ...')
    snapshot = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = (
        df.groupby('CustomerID')
        .agg(
            Recency  =('InvoiceDate', lambda x: (snapshot - x.max()).days),
            Frequency=('InvoiceNo',   'nunique'),
            Monetary =('TotalAmount', 'sum'),
        )
        .reset_index()
    )
    print(f'   Customers: {len(rfm):,}')
    return rfm


def train_clustering(rfm: pd.DataFrame):
    print('\n[3/5] Training KMeans ...')
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

    # Find optimal K (2-8)
    best_sil, best_k = -1, 4
    for k in range(2, 9):
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        labels = km.fit_predict(scaled)
        sil = silhouette_score(scaled, labels)
        print(f'   K={k} -> Silhouette={sil:.4f}')
        if sil > best_sil:
            best_sil, best_k = sil, k

    # Final model - use K=4 for business interpretability
    OPTIMAL_K = 4
    print(f'\n   -> Using K={OPTIMAL_K} (business segments)')
    kmeans = KMeans(n_clusters=OPTIMAL_K, init='k-means++', n_init=20, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(scaled)

    # Label clusters
    summary = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
    avg_r = summary['Recency'].mean()
    avg_f = summary['Frequency'].mean()
    avg_m = summary['Monetary'].mean()

    labels_map = {}
    for c, row in summary.iterrows():
        r, f, m = row['Recency'], row['Frequency'], row['Monetary']
        if r <= avg_r and f >= avg_f and m >= avg_m:
            labels_map[c] = 'High-Value'
        elif r <= avg_r * 1.3 and f >= avg_f * 0.7:
            labels_map[c] = 'Regular'
        elif r > avg_r * 1.5:
            labels_map[c] = 'At-Risk'
        else:
            labels_map[c] = 'Occasional'

    # Ensure all 4 labels are assigned
    assigned = set(labels_map.values())
    required = {'High-Value', 'Regular', 'Occasional', 'At-Risk'}
    if not required.issubset(assigned):
        ranked = summary.sort_values(['Monetary', 'Frequency'], ascending=[False, False])
        label_order = ['High-Value', 'Regular', 'Occasional', 'At-Risk']
        labels_map  = {c: l for c, l in zip(ranked.index, label_order)}

    rfm['Segment'] = rfm['Cluster'].map(labels_map)
    print(f'   Segment distribution:\n{rfm["Segment"].value_counts().to_string()}')
    return kmeans, scaler, labels_map, rfm


def build_recommendation(df: pd.DataFrame):
    print('\n[4/5] Building recommendation model ...')
    pc = df.groupby(['CustomerID', 'Description'])['Quantity'].sum().reset_index()

    # Filter products with >= 10 buyers
    popular = (
        pc.groupby('Description')['CustomerID']
        .nunique()
        .reset_index(name='n')
        .query('n >= 10')['Description']
    )
    pc = pc[pc['Description'].isin(popular)]

    pivot = pc.pivot_table(index='Description', columns='CustomerID',
                           values='Quantity', fill_value=0)
    print(f'   Pivot: {pivot.shape[0]} products x {pivot.shape[1]} customers')

    sim = cosine_similarity(pivot)
    sim_df = pd.DataFrame(sim, index=pivot.index, columns=pivot.index)
    product_names = sim_df.index.tolist()
    print(f'   Similarity matrix: {sim_df.shape[0]} x {sim_df.shape[1]}')
    return sim_df, product_names


def save_models(kmeans, scaler, labels_map, sim_df, product_names, rfm):
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(kmeans,      f'{MODELS_DIR}/kmeans_model.pkl')
    joblib.dump(scaler,      f'{MODELS_DIR}/scaler.pkl')
    joblib.dump(labels_map,  f'{MODELS_DIR}/cluster_labels_map.pkl')
    joblib.dump(sim_df,      f'{MODELS_DIR}/product_similarity.pkl')
    joblib.dump(product_names, f'{MODELS_DIR}/product_names.pkl')
    rfm.to_csv('data/rfm_segments.csv', index=False)
    print('\n[5/5] All models saved:')
    for f in os.listdir(MODELS_DIR):
        size = os.path.getsize(f'{MODELS_DIR}/{f}') / 1024
        print(f'   {MODELS_DIR}/{f}  ({size:.1f} KB)')
    print('   data/rfm_segments.csv')


def main():
    print('=' * 55)
    print('SHOPPER SPECTRUM - Model Training Pipeline')
    print('=' * 55)

    # Generate dataset if missing
    if not os.path.exists(DATA_PATH):
        print('[!] Dataset not found. Generating ...')
        import generate_data
        generate_data.generate_dataset(output_path=DATA_PATH)

    df = load_and_clean(DATA_PATH)
    rfm = compute_rfm(df)
    kmeans, scaler, labels_map, rfm = train_clustering(rfm)
    sim_df, product_names = build_recommendation(df)
    save_models(kmeans, scaler, labels_map, sim_df, product_names, rfm)

    print('\nTraining complete! Run: streamlit run app.py')


if __name__ == '__main__':
    main()
