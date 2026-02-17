import os

from torch.utils.data import DataLoader
from tqdm import tqdm

from model.llm import Llm, finetune_model
from task.llm_icl_data import ICLDataset, icl_collate_fn

from utils.utils import *
from utils.metrics import compute_and_print_all_metrics


def llm_icl_ner(llm, set_name, general_mode):
    data_file = f"data/{set_name}/test.jsonl"
    with open(data_file, "r", encoding='utf-8') as f:
        data = [json.loads(line) for line in f]

    demo_file = f"demo/{set_name}_llm_icl_{general_mode}_demo.json"
    dataset = ICLDataset(data, demo_file, llm.tokenizer, general_mode)
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=icl_collate_fn)

    # 创建简单的缓冲写入器
    save_path = f"data/{set_name}/llm_icl_{general_mode}_test.json"
    writer = SimpleTextWriter(save_path, buffer_size=100)

    # pred_labels = []
    gp_pairs = []
    for idx, batch in tqdm(enumerate(dataloader)):
        batch_prompt = batch["prompt"]

        responses = llm.get_response(batch_prompt, max_new_tokens=64)
        data_item = batch["data"]

        for res, item in zip(responses, data_item):
            g_labels = item["entities"][0]["label"]
            try:
                # 尝试提取JSON数据
                res_json = extract_json_data(res)
                if not isinstance(res_json, list):
                    # 过滤该数据
                    # 直接返回空列表
                    item_pred = []
                else:
                    item_pred = res_json
            except Exception as e:  # 捕获extract_json_data可能抛出的任何异常
                item_pred = []
                print(f"处理响应时出错: {e}")
            gp_pairs.append((g_labels, item_pred))
            writer.write(json.dumps(item_pred))
        if idx % 100 == 0:
            metrics = compute_and_print_all_metrics(gp_pairs)
            print(metrics)
    # 处理完成后关闭写入器
    writer.close()
    metrics = compute_and_print_all_metrics(gp_pairs)
    print(metrics)


def llm_ft_ner_train(save_dir, set_name, model_path, general_mode):
    train_data_file = f"{save_dir}/candidate_train.jsonl"
    dev_data_file = f"{save_dir}/candidate_dev.jsonl"
    train_data = read_json_data(train_data_file)
    dev_data = read_json_data(dev_data_file)
    demo_file = f"demo/{set_name}_llm_icl_{general_mode}_demo.json"

    with open(demo_file, "r", encoding='utf-8') as f:
        system_prompt = json.load(f)["system"]

    save_path = f"checkpoints/{set_name}/{general_mode}/"
    os.makedirs(save_path, exist_ok=True)
    finetune_model(train_data, system_prompt, model_path, save_path, general_mode)


def llm_ft_ner_test(set_name, data_file, save_file, model_path, lora_ckpt_path, general_mode="candidate"):
    """Test fine-tuned LLM for NER evaluation"""

    model = Llm(model_path, lora_ckpt_path)
    demo_file = f"demo/{set_name}_llm_icl_{general_mode}_demo.json"
    data = read_json_data(data_file)
    dataset = ICLDataset(data, demo_file, model.tokenizer, general_mode, is_sytem=True, is_demo=False)

    writer = SimpleTextWriter(save_file, buffer_size=100)
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=icl_collate_fn)

    gp_pairs = []  # (ground_truth, predictions) pairs

    for idx, batch in tqdm(enumerate(dataloader)):
        batch_prompt = batch["prompt"]
        responses = model.get_response(batch_prompt, max_new_tokens=64)
        data_items = batch["data"]

        for res, item in zip(responses, data_items):
            true_labels = item["label"]

            try:
                # Extract JSON data from response
                res_json = extract_json_data(res)
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

        # Periodically compute and print metrics
        if idx % 100 == 0:
            metrics = compute_and_print_all_metrics(gp_pairs)
            print(f"Batch {idx} metrics: {metrics}")

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