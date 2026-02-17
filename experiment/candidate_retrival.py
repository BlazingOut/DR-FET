from task.similar_calculate import SimilarCalculate

import json
from tqdm import tqdm
from utils.utils import complete_hierarchy_tags, read_json_data, write_json_data, SimpleTextWriter
from utils.metrics import calculate_recall


def find_best_accept_chain(candidate_tags, max_chains=2):
    """
    Find top max_chains valid label chains from candidate tags

    Args:
        candidate_tags: List of candidate tags
        max_chains: Maximum number of chains to return

    Returns:
        list: List of valid accepted tag chains
    """
    all_chains = []

    def backtrack(start_idx, current_chain, current_level):
        """Backtrack to search for all possible acceptance chains"""
        if len(all_chains) >= max_chains:
            return

        # Try selecting tags starting from start_idx
        for i in range(start_idx, len(candidate_tags)):
            tag = candidate_tags[i]
            tag_level = tag.count('/')

            # Condition 1: Tag level must be greater than current level
            if tag_level <= current_level:
                continue

            # Condition 2: Check if all accepted tags are prefixes of current tag
            is_all_prefix = True
            for accepted_tag in current_chain:
                if not tag.startswith(accepted_tag):
                    is_all_prefix = False
                    break

            if not is_all_prefix:
                continue

            # Accept this tag
            new_chain = current_chain.copy()

            # Add all missing parent tags
            if tag_level > 1:
                parts = tag.split('/')[1:]
                for j in range(1, tag_level):
                    parent_tag = '/' + '/'.join(parts[:j])
                    if parent_tag not in new_chain:
                        new_chain.append(parent_tag)

            new_chain.append(tag)

            # Save current chain
            all_chains.append(new_chain.copy())

            # Continue search (with depth limit)
            if len(all_chains) < max_chains:
                backtrack(i + 1, new_chain, tag_level)

    # Search starting from each position
    for start in range(len(candidate_tags)):
        if len(all_chains) >= max_chains:
            break
        backtrack(start, [], 0)

    return all_chains


def filter_by_candidate_tags_stage1(similar_calculate: SimilarCalculate, data_file, save_file):
    """Filter data based on candidate tags (stage 1)"""

    filtered_data = []
    accept_cnt = 0
    data = read_json_data(data_file)
    writer = SimpleTextWriter(save_file, 200)

    for idx, item in tqdm(enumerate(data)):
        description = item["description"]
        true_labels = item["label"]

        # Get candidate tags
        candidates = similar_calculate.calculate_top_k_similar_label(description, 'separate', q=5, top_k=5)
        candidate_tags = [c[0] for c in candidates]

        # Find top 2 valid acceptance chains
        accept_chains = find_best_accept_chain(candidate_tags, max_chains=2)

        # Check if any chain contains all true labels
        is_valid = False
        best_chain = []

        for chain in accept_chains:
            # Complete hierarchy
            full_chain = complete_hierarchy_tags(chain)
            if set(true_labels).issubset(set(full_chain)):
                is_valid = True
                best_chain = chain
                break

        if is_valid:
            new_item = item.copy()
            new_item["candidates"] = candidate_tags
            new_item["accepted_chain"] = best_chain
            writer.write(json.dumps(new_item))
            accept_cnt += 1

        # Periodic progress logging
        if idx % 1000 == 0:
            print(f"Processed: {idx}, Accepted: {accept_cnt}")

    return accept_cnt

def filter_by_candidate_tags_stage2(similar_calculate: SimilarCalculate, data_file, save_file):
    """Filter data based on candidate tags (stage 2)"""

    filtered_data = []
    accept_cnt = 0
    data = read_json_data(data_file)
    writer = SimpleTextWriter(save_file, 200)

    for idx, item in tqdm(enumerate(data)):
        description = item["description"]
        true_labels = item["label"]

        # Get candidate tags
        candidates = similar_calculate.calculate_top_k_similar_label(description, 'separate', q=5, top_k=5)
        candidate_tags = [c[0] for c in candidates]

        # Process candidate tags sequentially, building acceptance list
        accepted_tags = []
        cur_level = 0

        for tag in candidate_tags:
            # Calculate tag level (number of '/')
            tag_level = tag.count('/')

            # Condition 1: Tag level must be greater than current level
            if tag_level <= cur_level:
                continue

            # Condition 2: Check if all accepted tags are prefixes of current tag
            is_all_prefix = True
            for accepted_tag in accepted_tags:
                if not tag.startswith(accepted_tag):
                    is_all_prefix = False
                    break

            if not is_all_prefix:
                continue

            # Accept this tag
            # Add missing parent tags if needed
            if tag_level > 1:
                parts = tag.split('/')[1:]  # Remove initial empty string
                for i in range(1, tag_level):  # From 1 to tag_level-1
                    parent_tag = '/' + '/'.join(parts[:i])
                    if parent_tag not in accepted_tags:
                        accepted_tags.append(parent_tag)

            cur_level = tag_level
            accepted_tags.append(tag)

        # Check if true labels are subset of accepted tags
        accepted_tags = complete_hierarchy_tags(accepted_tags)
        is_equal = set(true_labels) == set(accepted_tags)

        if is_equal:
            new_item = item.copy()
            new_item["candidates"] = candidate_tags
            writer.write(json.dumps(new_item))
            accept_cnt += 1

        # Periodic progress logging
        if idx % 1000 == 0:
            print(f"Processed: {idx}, Accepted: {accept_cnt}")

    writer.close()
    return accept_cnt

def generate_and_test_candidates(similar_calculate: SimilarCalculate, data, save_file, q=5, top_k=5):
    gp_pairs = []
    process_data = []
    for idx, item in tqdm(enumerate(data)):
        description = item["description"]
        g_labels = item["label"]
        candidates = similar_calculate.calculate_top_k_similar_label(description, q=q, top_k=top_k)
        candidate_tags = [c[0] for c in candidates]
        hierarchy_tags = complete_hierarchy_tags(candidate_tags)
        gp_pairs.append((g_labels, hierarchy_tags))
        item["candidates"] = hierarchy_tags
        process_data.append(item)
        if idx % 1000 == 0:
            recall = calculate_recall(gp_pairs)
            print(recall)

    recall = calculate_recall(gp_pairs)
    write_json_data(save_file, process_data)
    return recall
