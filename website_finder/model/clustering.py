from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering
import numpy as np


class ClusterClassifier:
    def __init__(self, data, embedder_model, text_col, target_col) -> None:
        self.data = data
        self.embedder = SentenceTransformer(embedder_model)
        self.text_col = text_col
        self.target_col = target_col
        self.clustered_sentences = None
        pass

    def prepare_text_cluster(self, predicted_category):
        """
        This is a simple application for sentence embeddings: clustering

        Sentences are mapped to sentence embeddings and then agglomerative clustering with a threshold is applied.
        """
        text_corpus = list(
            self.data[self.data[self.target_col] == predicted_category][self.text_col]
        )
        corpus_embeddings = self.embedder.encode(text_corpus)
        corpus_embeddings = corpus_embeddings / np.linalg.norm(
            corpus_embeddings, axis=1, keepdims=True
        )

        clustering_model = AgglomerativeClustering(
            n_clusters=None, distance_threshold=1.5
        )  # , affinity='cosine', linkage='average', distance_threshold=0.4)
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_

        clustered_sentences = {}
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            # print(sentence_id, cluster_id)
            if cluster_id not in clustered_sentences:
                clustered_sentences[cluster_id] = []

            clustered_sentences[cluster_id].append(text_corpus[sentence_id])

        self.clustered_sentences = clustered_sentences
        return self.clustered_sentences
        # for i, cluster in clustered_sentences.items():
        #     print("Cluster ", i + 1)
        #     print(cluster)
        #     print("")

    def predict_top_n(self, text, n):
        if self.clustered_sentences is None:
            return []

        text_embed = self.embedder.encode(text)
        topn_sites = []
        for i, cluster in self.clustered_sentences.items():
            similarity_score = util.cos_sim(
                [text_embed.mean()], [self.embedder.encode(cluster).mean()]
            ).item()
            topn_sites.append(similarity_score)

        return sorted(topn_sites, reverse=True)[:n]
