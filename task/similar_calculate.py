import time
from collections import defaultdict

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils.label_utils import TagManager
from model.encoder import BaseEmbedder, batch_encode


class SimilarCalculate:
    def __init__(self, embedder: BaseEmbedder, label_file):
        self.embedder = embedder
        self.manager = TagManager.load_from_json(label_file)
        self.label_embed = {}

    def get_labels_embed(self, mode="separate"):
        """
        Generate embeddings for labels based on their descriptions

        Args:
            mode: "separate" - embed each description separately
                  "combined" - combine all descriptions for each label
        """
        labels = list(self.manager.tag_index.keys())

        if mode == "separate":
            label_map = []
            label_des_list = []
            for label in labels:
                des_list = self.manager.get_descriptions(label)
                label_map.extend([label] * len(des_list))
                label_des_list.extend(des_list)

        elif mode == "combined":
            label_map = labels
            label_des_list = [
                ', '.join(self.manager.get_descriptions(label)) for label in labels
            ]
        else:
            raise NotImplementedError(f"Mode '{mode}' not implemented")

        self.label_embed = {
            "label_map": label_map,
            "embedding": batch_encode(self.embedder, label_des_list, show_progress=True)
        }

    def calculate_top_k_similar_label(self, to_match_phrase: str, mode="separate", q=6, top_k=4):
        """
        Calculate top-k most similar labels for a given phrase

        Args:
            to_match_phrase: Input phrase to match
            mode: "separate" or "combined" (must match get_labels_embed mode)
            q: Number of top matches to consider per query (separate mode only)
            top_k: Number of top labels to return

        Returns:
            List of (label, similarity_score) tuples
        """
        if mode == "separate":
            query_list = to_match_phrase.split(', ')
        else:
            query_list = [to_match_phrase]

        query_embedding = self.embedder.embed(query_list)
        if query_embedding is None or len(query_embedding) == 0:
            return []

        similarities = cosine_similarity(query_embedding, self.label_embed['embedding'])  # (n_query, n_labels)

        if mode == "separate":
            # Aggregate similarity scores per label
            label_score_counter = defaultdict(float)

            for sim_row in similarities:
                # Get top q most similar labels
                top_q_idx = np.argsort(sim_row)[-q:][::-1]
                for idx in top_q_idx:
                    label = self.label_embed['label_map'][idx]
                    label_score_counter[label] += sim_row[idx]  # Accumulate similarity scores

            # Sort and return top_k labels with scores
            sorted_labels = sorted(label_score_counter.items(), key=lambda x: x[1], reverse=True)
            return sorted_labels[:top_k]

        elif mode == "combined":
            sim_row = similarities[0]  # Only one query
            top_k_idx = np.argsort(sim_row)[-top_k:][::-1]
            return [(self.label_embed['label_map'][idx], sim_row[idx]) for idx in top_k_idx]

        else:
            raise NotImplementedError(f"Mode '{mode}' not implemented")

    def delete_model(self):
        """Clean up model and free GPU memory"""
        if isinstance(self.embedder, BaseEmbedder):
            del self.embedder.model
            del self.embedder.tokenizer
        torch.cuda.empty_cache()
        time.sleep(10)  # Allow time for memory cleanup