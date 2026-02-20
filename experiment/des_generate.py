import json
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.utils import extract_json_data, SimpleTextWriter, read_json_data
from task.llm_icl_data import icl_collate_fn, ICLDataset


def des_generate(llm, data_file, save_path, buffer_size=100, gen_mode="test"):
    """Generate descriptions using LLM with in-context learning"""

    data = read_json_data(data_file)

    # Load appropriate demonstration examples based on mode
    if gen_mode == "test":
        demo_file = "demo/descriptor_gen_demo_test.json"
        dataset = ICLDataset(data, demo_file, llm.model.get_tokenizer(), general_mode="direct")
    elif gen_mode == "train":
        demo_file = "demo/descriptor_gen_demo_train.json"
        dataset = ICLDataset(data, demo_file, llm.model.get_tokenizer(), general_mode="des_train")
    else:
        raise ValueError(f"Unknown generation mode: {gen_mode}")
    all_prompts = [dataset[i] for i in range(len(dataset))]
    # dataloader = DataLoader(dataset, batch_size=8, collate_fn=icl_collate_fn)

    # Create buffered writer for results
    writer = SimpleTextWriter(save_path, buffer_size)

    responses = llm.get_responses(all_prompts, temperature=0.5)
    for idx, res_text in enumerate(responses):
        # Extract JSON data from response
        res_json = extract_json_data(res_text)
        # Validate extracted JSON
        try:
            if not isinstance(res_json, dict) or "description" not in res_json:
                print("Invalid response format, skipping")
                continue
            # Create new item with generated description
            new_item = data[idx].copy()
            new_item["description"] = res_json["description"]
            writer.write(json.dumps(new_item))
        except Exception as e:
            # Handle any parsing errors
            print(f"Error processing response: {e}")
            continue
        
    