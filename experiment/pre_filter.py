import json
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from utils.utils import extract_json_data, SimpleTextWriter, read_json_data
from task.llm_icl_data import get_prompt

class LlmPreFilterDataset(Dataset):
    def __init__(self, data, demo_file_path, tokenizer):
        self.tokenizer = tokenizer
        self.data = data
        self.demo_prompt = get_prompt(demo_file_path)
        self.demo_prompt_text = self.tokenizer.apply_chat_template(self.demo_prompt, tokenize=False, add_generation_prompt=False)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        entity = item["mention"]

        new_item = [{
            "role": "user",
            "content": f"Text: {text}\nPhrase: {entity}"
        }]
        demo_chat_list = self.demo_prompt.copy()
        chat_list = demo_chat_list + new_item
        prompt = self.tokenizer.apply_chat_template(chat_list, tokenize=False, add_generation_prompt=True)

        return prompt

from vllm import SamplingParams

def filter_basic_train_data_simple(llm, data_file, save_path, data_start=None):
    data = read_json_data(data_file)
    if data_start:
        data = data[data_start:]
    not_val_list = ["preposition", "pronoun", "conjunction", "verb",
                    "adjective", "adverb", "incomplete sentence", "definite article"]

    dataset = LlmPreFilterDataset(data, "demo/pre_filter_demo.json", llm.model.get_tokenizer())
    
    # 3. 提取所有 Prompt (这一步很快，只是内存操作)
    print("Preparing prompts...")
    all_prompts = [dataset[i] for i in range(len(dataset))]
    print(all_prompts[:5])
    responses = llm.get_responses(all_prompts, temperature=0.5)

    # 6. 后处理与写入
    # with open("data/bbn/pre_filter_train_1.jsonl", 'w', encoding='utf-8') as f:
    #     for idx, res_text in enumerate(responses):
    #         item = {
    #             "id": idx, 
    #             "text": data[idx]['text'],
    #             'gen_text': res_text
    #             }
    #         f.write(json.dumps(item)+'\n')
    with open(save_path, 'w', encoding='utf-8') as writer:
        for i, res_text in enumerate(responses):
            val = extract_json_data(res_text)

            # 你的原有逻辑
            if not isinstance(val, dict) or "component" not in val or "valid" not in val:
                continue
                
            valid_flag = not (val["component"] in not_val_list or val["valid"] == "false")
            
            if valid_flag:
                # 写入原始数据
                writer.write(json.dumps(data[i], ensure_ascii=False) + "\n")
            
    # print(f"Done! Processed {processed_count} items.")


