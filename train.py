import os
import random
import re

from model.llm import Llm, LlmVLLM
from model.encoder import get_encoder
from experiment.pre_filter import filter_basic_train_data_simple
from experiment.candidate_retrival import filter_by_candidate_tags_stage1, filter_by_candidate_tags_stage2, generate_and_test_candidates
from experiment.des_generate import des_generate
from experiment.llm_ner import llm_ft_ner_train, llm_ft_ner_test, llm_icl_ner

from task.similar_calculate import SimilarCalculate

from utils.label_utils import build_tag_tree
from utils.utils import read_json_data, write_json_data

def find_latest_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        return None
    else:
        steps = []
        pattern = re.compile(r'checkpoint-(\d+)')
        for filename in os.listdir(checkpoint_dir):
            match = pattern.fullmatch(filename)
            if match:
                steps.append(int(match.group(1)))
        if len(steps) == 0:
            return None
        else:
            max_step = max(steps)
            return f'checkpoint-{max_step}'

def candidate_filter(config, llm, save_dir, step=1):
    set_name = config["data"]["set_name"]
    # 1.2.1
    if step == 1:
        data_file = f"{save_dir}/pre_filter_train.jsonl"
    else:
        data_file = f"{save_dir}/filter_train_1.jsonl"
    save_file = f"{save_dir}/pre_filter_train_{step}_description.jsonl"
    des_generate(llm, data_file, save_file, 200, gen_mode="test")

    # 1.2.2 build tag_tree
    description_data_file = f"{save_dir}/pre_filter_train_{step}_description.jsonl"
    save_file_path = f"{save_dir}/{set_name}-tag-hierarchy-pre-{step}.json"
    build_tag_tree(set_name, description_data_file, save_file_path)

    # 1.2.3
    pretrained_model_path = config["model"]["encoder_path"]
    encoder_name = config["model"]["encoder_name"]
    embedder = get_encoder(encoder_name, pretrained_model_path)
    label_file = f"{save_dir}/{set_name}-tag-hierarchy-pre-{step}.json"
    similar_calculate = SimilarCalculate(embedder, label_file)
    similar_calculate.get_labels_embed()

    data_file = description_data_file
    save_file = f"{save_dir}/filter_train_{step}.jsonl"
    if step == 1:
        filter_by_candidate_tags_stage1(similar_calculate, data_file, save_file)
    else:
        filter_by_candidate_tags_stage2(similar_calculate, data_file, save_file)

def process_pre_filter(config, step_1=True, step_2=True, step_3=True):
    model_name = config["model"]["model_name"]
    model_path = config["model"]['model_path']
    set_name = config["data"]["set_name"]
    llm = LlmVLLM(model_path)
    save_dir = f"data/{set_name}/pre_filter_{model_name}"
    os.makedirs(save_dir, exist_ok=True)
    # 1.1 sy

    if step_1:
        print("Start pre-filtering Synatics")
        data_file_path = f"data/{set_name}/train.jsonl"
        save_file_path = f"{save_dir}/pre_filter_train.jsonl"
        filter_basic_train_data_simple(llm, data_file_path, save_file_path, 0)
        print("pre-filter done!!")
        print(f"File saved to {save_dir}/pre_filter_train.jsonl")

    if step_2:
        candidate_filter(config, llm, save_dir, step=1)
    if step_3:
        candidate_filter(config, llm, save_dir, step=2)



def process_train(config, step_1=True, step_2=True, step_3=True, step_4=True):
    """Main training pipeline"""

    set_name = config["data"]["set_name"]
    model_path = config["model"]["model_path"]
    model_name = config["model"]["model_name"]
    if step_1 or step_2:
        llm = LlmVLLM(model_path)

    # Setup directories
    pref_save_dir = f"data/{set_name}/pre_filter_{model_name}"
    des_save_dir = f"data/{set_name}/description_{model_name}"
    cand_save_dir = f"data/{set_name}/candidate_{model_name}"
    os.makedirs(des_save_dir, exist_ok=True)
    os.makedirs(cand_save_dir, exist_ok=True)
    tag_save_file = f"{des_save_dir}/{set_name}-tag-hierarchy-des.json"
    
    if step_1:
        # # Step 1: Generate descriptions for filtered training data
        train_data_file = f"{pref_save_dir}/filter_train_2.jsonl"
        train_save_file = f"{des_save_dir}/des_train.jsonl"
        des_generate(llm, train_data_file, train_save_file, 100, "train")
        # Rebuild tag tree with new descriptions
        build_tag_tree(set_name, train_save_file, tag_save_file)
    if step_2:
        # Step 2: Create or load dev set
        dev_data_file = f"data/{set_name}/dev.jsonl"
        dev_save_file = f"data/{set_name}/description_{model_name}/des_dev.jsonl"

        if not os.path.exists(dev_data_file):
            data = read_json_data(train_save_file)
            train_data = random.sample(data, int(len(data) * 0.8))
            dev_data = [x for x in data if x not in train_data]
            write_json_data(dev_data_file, dev_data)
            write_json_data(train_save_file, train_data)

        des_generate(llm, dev_data_file, dev_save_file, 100, "test")
    if step_3:
        # Step 3: Get candidate tags for train and test sets
        pretrained_model_path = config["model"]["encoder_path"]
        encoder_name = config["model"]["encoder_name"]
        embedder = get_encoder(encoder_name, pretrained_model_path)
        similar_calculate = SimilarCalculate(embedder, tag_save_file)
        similar_calculate.get_labels_embed()

        for split in ["train", "dev"]:
            data_file = f"{des_save_dir}/des_{split}.jsonl"
            save_file = f"{cand_save_dir}/candidate_{split}.jsonl"
            data = read_json_data(data_file)
            generate_and_test_candidates(similar_calculate, data, save_file, q=5, top_k=10)
    if step_4:
        # Step 4: Fine-tune final decision model
        model_path = config["model"]["model_path"]
        
        llm_ft_ner_train(cand_save_dir, set_name, model_name, model_path, "candidate", seed=config["seed"])

def process_test(config,step1=True, step2=True, step3=True):
    
    model_path = config["model"]["model_path"]
    model_name = config["model"]["model_name"]
    set_name = config["data"]["set_name"]
    
    data_file = f"data/{set_name}/test.jsonl"
    save_file = f"data/{set_name}/description_{model_name}/des_test.jsonl"
    if step1:
        llm = LlmVLLM(model_path)
        des_generate(llm, data_file, save_file, 100, "test")
    if step2:
        pretrained_model_path = config["model"]["encoder_path"]
        encoder_name = config["model"]["encoder_name"]
        embedder = get_encoder(encoder_name, pretrained_model_path)
        tag_file = f"data/{set_name}/description_{model_name}/{set_name}-tag-hierarchy-des.json"
        similar_calculate = SimilarCalculate(embedder, tag_file)
        similar_calculate.get_labels_embed()
        data = read_json_data(save_file)
        print(data[:5])
        save_file = f"data/{set_name}/candidate_{model_name}/candidate_test.jsonl"
        generate_and_test_candidates(similar_calculate, data, save_file, q=5, top_k=20)
    if step3:
        seed = config["seed"]
        lora_ckpt_dir = f"checkpoints/{set_name}/candidate_{model_name}_s{seed}/"
        lora_ckpt_path = lora_ckpt_dir + find_latest_checkpoint(lora_ckpt_dir)
        data_file = f"data/{set_name}/candidate_{model_name}/candidate_test.jsonl"
        save_path = f"data/{set_name}/llm_ft_candidate_test_s{seed}.jsonl"
        llm = LlmVLLM(model_path, lora_ckpt_path)
        llm_ft_ner_test(llm, set_name, data_file, save_path, general_mode="candidate")


def process_test_icl(config):
    model_path = config["model"]["model_path"]
    model_name = config["model"]["model_name"]
    set_name = config["data"]["set_name"]

    llm = LlmVLLM(model_path)
    llm_icl_ner(llm, set_name, model_name, "candidate")
    