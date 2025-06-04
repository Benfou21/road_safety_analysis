import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score



def prepare_clustering_df(caract, lieux, vehicules, whithout_num = False):
    """
    Merge des datasets pour clustering et préparation :
    - merge sur 'Num_Acc'
    - drop colonnes inutiles (an, id_vehicule, dep, mois, lat, long)
    - one hot encoding des variables catégoriques
    - standardisation des variables numériques
    """
    df = caract.merge(lieux, on=["Num_Acc"]).merge(vehicules, on=["Num_Acc"])

    # Suppression colonnes inutiles ou à faible corrélation
    to_drop = ["Num_Acc", "an", "id_vehicule", "dep", "mois", "larrout", "lat", "long"]
    
    if whithout_num : #Suppression des variables à plus forte corrélation pour faire ressortir d'autres clusters
        to_drop.extend(["nbv", "vma",])
        
    df = df.drop(columns=to_drop)

    # Colonnes catégoriques à one-hot encoder
    col_cate = ["heure_categorie","mois_categorie","jour_categorie","lum","agg","int",
                "atm","col","catr","circ","prof","plan","surf","infra","situ",
                "catv_categorie","obs","obsm","choc","manv","motor","dep_cat"]
    col_num = [col for col in df.columns if col not in col_cate]

    df = pd.get_dummies(df, columns=col_cate, drop_first=True)

    # Standardisation variables numériques
    if len(col_num) > 0:
        scaler = StandardScaler()
        df[col_num] = scaler.fit_transform(df[col_num])

    return df

def kmeans_clustering(df, k, plot=True):
    """
    Effectue un clustering KMeans avec k clusters.
    Affiche la distribution et les variables les plus discriminantes.
    Retourne le DataFrame avec la colonne 'cluster'.
    """
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(df)

    summary = df.groupby('cluster').mean()

    if plot:
        plt.figure(figsize=(7, 4))
        df['cluster'].value_counts().sort_index().plot(kind='bar', color='skyblue')
        plt.xlabel('Cluster')
        plt.ylabel('Nombre d’observations')
        plt.title(f'Distribution des clusters (k={k})')
        plt.grid(True)
        plt.show()

        diffs = summary.max() - summary.min()
        top_vars = diffs.sort_values(ascending=False).head(15).index

        plt.figure(figsize=(12, 6))
        sns.heatmap(summary[top_vars].T, cmap='coolwarm', center=0)
        plt.title(f'Top variables discriminantes par cluster (k = {k})')
        plt.xlabel('Cluster')
        plt.ylabel('Variables')
        plt.tight_layout()
        plt.show()

        # Heatmaps individuelles par cluster
        for c in summary.index:
            values = summary.loc[c]
            others_mean = summary.drop(index=c).mean()
            diffs = (values - others_mean).abs()
            top_vars = diffs.sort_values(ascending=False).head(5).index.tolist()

            plt.figure(figsize=(6, len(top_vars) * 0.4 + 1))
            sns.heatmap(summary.loc[:, top_vars].T, annot=True, cmap="coolwarm", center=0,
                        cbar=False, linewidths=0.5, linecolor='gray')
            plt.title(f"Variables les plus discriminantes – Cluster {c}")
            plt.xlabel("Cluster")
            plt.ylabel("Variable")
            plt.tight_layout()
            plt.show()

    return df

def plot_accidents_per_month(caract):
    """Trace un histogramme du nombre d'accidents par mois."""
    acc_per_month = caract.groupby('mois').size().reset_index(name='accidents')
    import matplotlib.pyplot as plt

    plt.bar(acc_per_month["mois"], acc_per_month["accidents"], color="orange")
    plt.title("Nombre d'accidents au cours des mois")
    plt.xlabel("Mois")
    plt.ylabel("Nb d'accidents")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    print("Ce script contient des fonctions de feature engineering et clustering. À importer.")
