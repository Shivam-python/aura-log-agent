import pandas as pd

from config import (
    DATASET_PATH,
    REGEX_CONFIG_PATH,
    MODEL_OUTPUT_PATH,
    EMBEDDING_MODEL,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES
)

from regex_engine import RegexClassifier
from embedding_engine import EmbeddingEngine
from clustering import LogClusterer
from classifier import LogClassifier


def main():

    # ------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------

    df = pd.read_csv(DATASET_PATH)

    print(f"Loaded {len(df)} logs")

    # ------------------------------------------------
    # REGEX CLASSIFICATION
    # ------------------------------------------------

    regex_classifier = RegexClassifier(REGEX_CONFIG_PATH)

    df["regex_label"] = df["log_message"].apply(
        regex_classifier.classify
    )

    # ------------------------------------------------
    # SPLIT DATA
    # ------------------------------------------------

    df_non_regex = df[df["regex_label"].isnull()].copy()

    df_legacy = df_non_regex[
        df_non_regex.source == "LegacyCRM"
    ]

    df_non_legacy = df_non_regex[
        df_non_regex.source != "LegacyCRM"
    ]

    print(f"Regex matched logs: {len(df) - len(df_non_regex)}")
    print(f"ML logs remaining: {len(df_non_legacy)}")

    # ------------------------------------------------
    # EMBEDDINGS
    # ------------------------------------------------

    embedding_engine = EmbeddingEngine(EMBEDDING_MODEL)

    embeddings = embedding_engine.encode(
        df_non_legacy["log_message"].tolist()
    )

    # ------------------------------------------------
    # CLUSTERING
    # ------------------------------------------------

    clusterer = LogClusterer(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES
    )

    clusters = clusterer.cluster(embeddings)

    df_non_legacy["cluster"] = clusters

    clusterer.print_clusters(df_non_legacy)

    # ------------------------------------------------
    # TRAIN CLASSIFIER
    # ------------------------------------------------

    X = embeddings
    y = df_non_legacy["target_label"].values

    classifier = LogClassifier()

    classifier.train(X, y)

    classifier.save(MODEL_OUTPUT_PATH)

    print(f"\nSaved model to: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()