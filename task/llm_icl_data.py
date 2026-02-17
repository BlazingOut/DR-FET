import json

from torch.utils.data import Dataset

def get_prompt(demo_file_path, is_system=True, is_demo=True):
    with open(demo_file_path, 'r', encoding="utf-8") as f:
        demo = json.load(f)
    is_example = ("example" in demo)
    demo_chat_list = [{
        "role": "system",
        "content": demo["system"]
    }]
    if "example" in demo and is_demo:
        for example in demo["example"]:
            demo_chat_list.append({
                "role": "user",
                "content": example["query"]
            })
            demo_chat_list.append({
                "role": "assistant",
                "content": example["answer"]
            })
    elif is_demo:
        raise ValueError("Demo doesn't include examples!")
    if not is_system and is_example:
        demo_chat_list[1]["content"] = demo["system"] + "\n" + demo_chat_list[1]["content"]
        return demo_chat_list[1:]
    elif not is_system:
        demo_chat_list[0]["role"] = "user"
        return demo_chat_list
    else:
        return demo_chat_list

class ICLDataset(Dataset):
    def __init__(self, data, demo_file_path, tokenizer, general_mode="pre", is_sytem=True, is_demo=True):
        self.tokenizer = tokenizer
        self.data = data
        self.demo_prompt = get_prompt(demo_file_path, is_sytem, is_demo)
        self.demo_prompt_text = self.tokenizer.apply_chat_template(self.demo_prompt, tokenize=False, add_generation_prompt=False)
        self.general_mode = general_mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        entity = item["mention"]

        if self.general_mode == "pre":
            new_item = [{
                "role": "user",
                "content": f"Text: {text}\nPhrase: {entity}",
            }]
        elif self.general_mode == "candidate":
            candidates = sorted(item['candidates'])
            new_item = [{
                "role": "user",
                "content": f"Text: {text}\nEntity: {entity}\nCandidates: {candidates}",
            }]
        elif self.general_mode == "direct":
            new_item = [{
                "role": "user",
                "content": f"Text: {text}\nEntity: {entity}",
            }]
        elif self.general_mode == "des_train":
            label = sorted(item["label"])
            new_item = [{
                "role": "user",
                "content": f"Text: {text}\nEntity: {entity}\nLabel: {json.dumps(label)}",
            }]
        else:
            raise ValueError("Unknown general mode")

        demo_chat_list = self.demo_prompt.copy()
        chat_list = demo_chat_list + new_item
        prompt = self.tokenizer.apply_chat_template(chat_list, tokenize=False, add_generation_prompt=True)
        return {
            "prompt": prompt,
            "data": item
        }

def icl_collate_fn(batch):
    prompt = [b["prompt"] for b in batch]
    data = [b["data"] for b in batch]
    return {"prompt": prompt, "data": data}
