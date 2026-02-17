def calculate_recall(gp_pairs):
    """
    Calculate average recall

    For each sample pair (g, p):
    - True labels list g and predicted labels list p can have different lengths
    - Recall = number of correctly predicted labels / total number of true labels
    - Average recall = mean of all sample recalls

    Args:
        gp_pairs: List of tuples [([true_label1, ...], [pred_label1, ...]), ...]

    Returns:
        float: Average recall
    """
    recalls = []

    for true_labels, pred_labels in gp_pairs:
        # Convert to sets
        true_set = set(true_labels)
        pred_set = set(pred_labels)

        # Calculate intersection (correctly predicted labels)
        correct_count = len(true_set & pred_set)
        truth_count = len(true_set)

        # Calculate recall for this sample
        if truth_count > 0:
            sample_recall = correct_count / truth_count
            recalls.append(sample_recall)

    # Calculate average recall
    if recalls:
        average_recall = sum(recalls) / len(recalls)
        return average_recall
    else:
        return 0.0


def count_match(label_true, label_pred):
    return sum(1 if t in label_pred else 0 for t in label_true)


def calc_f1(p, r):
    return 2 * p * r / float(p + r + 1e-5)


def micro_f1(gold_pred_label_pairs):
    l_true_cnt, l_pred_cnt, hit_cnt = 0, 0, 0
    for labels_true, labels_pred in gold_pred_label_pairs:
        hit_cnt += count_match(labels_true, labels_pred)
        l_true_cnt += len(labels_true)
        l_pred_cnt += len(labels_pred)
    p = hit_cnt / (l_pred_cnt + 1e-5)
    r = hit_cnt / (l_true_cnt + 1e-5)
    return p, r, calc_f1(p, r)


def macro_f1_gptups(true_and_prediction):
    # num_examples = len(true_and_prediction)
    p, r = 0., 0.
    pred_example_count, gold_example_count = 0., 0.
    pred_label_count, gold_label_count = 0., 0.
    for true_labels, predicted_labels in true_and_prediction:
        # print(predicted_labels)
        if len(predicted_labels) > 0:
            pred_example_count += 1
            pred_label_count += len(predicted_labels)
            per_p = len(set(predicted_labels).intersection(set(true_labels))) / float(len(predicted_labels))
            p += per_p
        if len(true_labels) > 0:
            gold_example_count += 1
            per_r = len(set(predicted_labels).intersection(set(true_labels))) / float(len(true_labels))
            r += per_r
    precision, recall = 0, 0
    if pred_example_count > 0:
        precision = p / pred_example_count
    if gold_example_count > 0:
        recall = r / gold_example_count
    # avg_elem_per_pred = pred_label_count / pred_example_count
    return precision, recall, calc_f1(precision, recall)


def strict_match(labels_true, labels_pred):
    if len(labels_pred) != len(labels_true):
        return False
    for lt in labels_true:
        if lt not in labels_pred:
            return False
    return True


def strict_acc_gp_pairs(gp_pairs):
    hit_cnt = sum(1 if strict_match(labels_true, labels_pred) else 0 for labels_true, labels_pred in gp_pairs)
    return hit_cnt / (len(gp_pairs) + 0.0001)

def compute_and_print_all_metrics(gp_pairs):
    acc = strict_acc_gp_pairs(gp_pairs)
    p, r, mif1 = micro_f1(gp_pairs)
    _, _, macf1 = macro_f1_gptups(gp_pairs)
    metircs = {
        'accuracy': acc,
        'micro_f1': mif1,
        'micro_precision': p,
        'micro_recall': r,
        'macro_f1': macf1
    }
    return metircs