import os

from torch.utils.data import DataLoader
from tqdm import tqdm

from model.llm import LlmVLLM, finetune_model
from task.llm_icl_data import ICLDataset, icl_collate_fn

from utils.utils import *
from utils.metrics import compute_and_print_all_metrics


def llm_icl_ner(llm, set_name, model_name, general_mode):
    if general_mode == "candidate":
        data_file = f"data/{set_name}/candidate_{model_name}/candidate_test.jsonl" 
    elif general_mode == 'direct':
        data_file = f"data/{set_name}/test.jsonl"
    with open(data_file, "r", encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    demo_file = f"demo/{set_name}_llm_icl_{general_mode}_demo.json"
    dataset = ICLDataset(data, demo_file, llm.model.get_tokenizer(), general_mode)
    # dataloader = DataLoader(dataset, batch_size=8, collate_fn=icl_collate_fn)

    # pred_labels = []
    gp_pairs = []
    all_prompts = [dataset[i] for i in range(len(dataset))]
    # 创建简单的缓冲写入器
    save_path = f"data/{set_name}/llm_icl_{general_mode}_test.json"
    writer = SimpleTextWriter(save_path, buffer_size=500)
    responses = llm.get_responses(all_prompts, temperature=0.5)
    gp_pairs = []  # (ground_truth, predictions) pairs
    for idx, res_text in enumerate(responses):
        item = data[idx]
        true_labels = item["label"]
        try:
            # Extract JSON data from response
            res_json = extract_json_data(res_text)
            if not isinstance(res_json, list):
                # Invalid format, use empty list
                pred_labels = []
            else:
                pred_labels = res_json
        except Exception as e:
            # Parsing error, use empty list
            pred_labels = []
            print(f"Error processing response: {e}")
        # Collect ground truth and predictions
        gp_pairs.append((true_labels, pred_labels))
        writer.write(json.dumps(pred_labels))

    # Finalize
    writer.close()
    final_metrics = compute_and_print_all_metrics(gp_pairs)
    print(f"Final metrics: {final_metrics}")


def llm_ft_ner_train(save_dir, set_name, model_name, model_path, general_mode, seed=42):
    train_data_file = f"{save_dir}/candidate_train.jsonl"
    dev_data_file = f"{save_dir}/candidate_dev.jsonl"
    train_data = read_json_data(train_data_file)
    dev_data = read_json_data(dev_data_file)
    demo_file = f"demo/{set_name}_llm_icl_{general_mode}_demo.json"

    with open(demo_file, "r", encoding='utf-8') as f:
        system_prompt = json.load(f)["system"]

    save_path = f"checkpoints/{set_name}/{general_mode}_{model_name}_s{seed}/"
    os.makedirs(save_path, exist_ok=True)
    finetune_model(train_data, system_prompt, model_path, save_path, general_mode)



def llm_ft_ner_test(model, set_name, data_file, save_file, general_mode="candidate"):
    """Test fine-tuned LLM for NER evaluation"""

    demo_file = f"demo/{set_name}_llm_icl_{general_mode}_demo.json"
    data = read_json_data(data_file)
    dataset = ICLDataset(data, demo_file, model.model.get_tokenizer(), general_mode, is_sytem=True, is_demo=False)
    all_prompts = [dataset[i] for i in range(len(dataset))]

    writer = SimpleTextWriter(save_file, buffer_size=500)
    responses = model.get_responses(all_prompts, temperature=0.5)
    gp_pairs = []  # (ground_truth, predictions) pairs
    for idx, res_text in enumerate(responses):
        item = data[idx]
        true_labels = item["label"]
        try:
            # Extract JSON data from response
            res_json = extract_json_data(res_text)
            if not isinstance(res_json, list):
                # Invalid format, use empty list
                pred_labels = []
            else:
                pred_labels = res_json
        except Exception as e:
            # Parsing error, use empty list
            pred_labels = []
            print(f"Error processing response: {e}")
        # Collect ground truth and predictions
        gp_pairs.append((true_labels, pred_labels))
        writer.write(json.dumps(pred_labels))
        # if (idx + 1) % 500 == 0:
        #     final_metrics = compute_and_print_all_metrics(gp_pairs)
        #     print(f"Final metrics: {final_metrics}")

    # Finalize
    writer.close()
    final_metrics = compute_and_print_all_metrics(gp_pairs)
    print(f"Final metrics: {final_metrics}")


if __name__ == "__main__":
    model_path = r"C:\0store\LLM\Qwen2.5-3B-Instruct"
    # llm = Llm(model_path)

    # llm_icl_ner(llm, set_name="ontonotes", general_mode="candidate")
    # llm_ft_ner_train(model_path, set_name="ontonotes", general_mode="direct")
    llm_ft_ner_test(model_path, "ontonotes", "direct")