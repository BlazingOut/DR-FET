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


def filter_basic_train_data_simple(llm, data_file, save_path, buffer_size=100, data_start=None):
    data = read_json_data(data_file)
    if data_start:
        data = data[data_start:]
    not_val_list = ["preposition", "pronoun", "conjunction", "verb",
                    "adjective", "adverb", "incomplete sentence", "definite article"]

    dataset = LlmPreFilterDataset(data, "demo/pre_filter_demo.json", llm.tokenizer)
    dataloader = DataLoader(dataset, batch_size=8)

    writer = SimpleTextWriter(save_path, buffer_size, write_mode="a")

    processed_count = 0
    for batch in tqdm(dataloader):
        responses = llm.get_response(batch)

        for res in responses:
                val = extract_json_data(res)
                if not isinstance(val, dict) or not ("component" in val) or not ("valid" in val):
                    continue
                valid_flag = not (val["component"] in not_val_list or val["valid"] == "false")
                if valid_flag:
                    writer.write(json.dumps(data[processed_count]))
                processed_count += 1

    writer.close()
    print(f"Process Down！Total {processed_count} samples")

