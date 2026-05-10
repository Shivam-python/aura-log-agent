import pandas as pd
from sklearn.cluster import DBSCAN


class LogClusterer:

    def __init__(self, eps=0.2, min_samples=1):
        self.eps = eps
        self.min_samples = min_samples

    def cluster(self, embeddings):

        clustering = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric='cosine'
        )

        return clustering.fit_predict(embeddings)

    @staticmethod
    def print_clusters(df, min_size=10):

        grouped = (
            df.groupby("cluster")["log_message"]
            .apply(list)
            .sort_values(key=lambda x: x.map(len), ascending=False)
        )

        print("\nClustered Patterns:\n")

        for cluster_id, messages in grouped.items():

            if len(messages) < min_size:
                continue

            print(f"\nCluster {cluster_id} ({len(messages)} logs):")

            for msg in messages[:5]:
                print(f"  - {msg}")