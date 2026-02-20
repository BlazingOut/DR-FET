import json

import torch

from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerFast
from peft import PeftModel, LoraConfig, TaskType, get_peft_model


from utils.utils import complete_hierarchy_tags
class Llm:
    def __init__(self, model_path: str, lora_ckpt_path=None):
        # 加载模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
        # 加载lora
        self.lora_path = lora_ckpt_path
        if lora_ckpt_path is not None:
            lora_config = LoraConfig.from_pretrained(lora_ckpt_path)
            self.model = PeftModel.from_pretrained(self.model, lora_ckpt_path, config=lora_config)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.model.device)
        print("pipeline built successfully!!")

        # 1. 正确初始化Tokenizer，确保使用左填充
        # 注意：此设置应在类初始化时完成，这里为演示目的放在函数内
        # 理想做法：在您的类 __init__ 方法中添加：self.tokenizer.padding_side = 'left'
        self.tokenizer.padding_side = 'left'

        # 如果tokenizer没有定义pad_token，将其设置为eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_response(self, batch, max_new_tokens=128):

        # 2. 对输入进行编码和填充
        inputs = self.tokenizer(
            batch,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt"
        )
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        # 3. 准备生成参数
        # 明确指定EOS Token ID。对于Qwen模型，通常是151645 (<|im_end|>) 和 151643 (<|endoftext|>)
        eos_token_id = [
            151645,  # <|im_end|>
            151643  # <|endoftext|>
        ]

        # 4. 调用generate函数（关键修改）
        generated_sequence = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            num_return_sequences=1,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,  # 设置EOS标记，使生成过程遇到它们时停止
            pad_token_id=self.tokenizer.pad_token_id,  # 明确设置PAD Token ID
            early_stopping=True  # 当所有序列都遇到EOS时提前停止
        )

        # 5. 解码生成的序列，跳过特殊标记
        gen_texts_raw = self.tokenizer.batch_decode(
            generated_sequence[:, input_ids.shape[1]:],
            skip_special_tokens=True  # 这将过滤掉所有<|im_end|>等特殊标记
        )
        return gen_texts_raw


import json
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


class LlmVLLM:
    def __init__(self, model_path: str, lora_ckpt_path=None, gpu_memory_utilization=0.8):
        """
        gpu_memory_utilization: 控制 vLLM 占用显存的比例，默认 0.9，
                                如果还要跑其他程序，可以调低一点。
        """
        # 1. 初始化 vLLM 引擎
        self.enable_lora = lora_ckpt_path is not None
        self.model = LLM(
            model=model_path,
            enable_lora=self.enable_lora,
            max_lora_rank=64,  # 根据你 LoRA 的实际 rank 调整，通常 64 足够
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            tensor_parallel_size=1  # 如果有多张 GPU，可以增加这个值实现模型并行
        )

        # 保存 LoRA 配置（如果存在）
        self.lora_request = None
        if self.enable_lora:
            self.lora_request = LoRARequest("my_lora_adapter", 1, lora_ckpt_path)

        print("vLLM engine built successfully!!")

    def get_responses(self, prompts, max_new_tokens=128, temperature=0):
        """
        直接传入整个 List[str]
        """
        # 2. 设置采样参数 (对应之前 do_sample=False)
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            # vLLM 会自动识别模型的 stop words，通常不需要手动传 eos_token_id
            # 如果需要强制停止，可以使用 stop=["<|im_end|>", "<|endoftext|>"]
        )

        # 3. 批量推理
        outputs = self.model.generate(
            prompts,
            sampling_params,
            lora_request=self.lora_request
        )

        # 4. 提取文本结果
        responses = [output.outputs[0].text for output in outputs]
        return responses


def general_finetune_pipeline(model, tokenizer, save_path, tokenized_dataset):
    # 定义LoraConfig
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()

    model.enable_input_require_grads()
    # tokenizer.pad_token = tokenizer.eos_token
    args = TrainingArguments(
        output_dir=save_path,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        logging_steps=50,
        num_train_epochs=1,
        save_steps=500,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
    )
    for i in range(5):
        print(tokenizer.decode(tokenized_dataset[i]['input_ids']))
    trainer = Trainer(
        model = model,
        args = args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    )
    trainer.train()
    del model
    torch.cuda.empty_cache()
    del tokenizer
MAX_LENTH = 1024
def general_process_data(item, system_prompt, tokenizer: PreTrainedTokenizerFast, is_system, general_method):

    if general_method == 'candidate':
        prompt, answer = process_data(item)
    else:
        prompt, answer = process_data_direct_infer(item)
    if not is_system:
        chat_list = [
            {'role': 'user', 'content': system_prompt + '\n' + prompt},
            {'role': 'assistant', 'content': answer}
        ]
    else:
        chat_list = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': answer}
        ]
    prompt_text = tokenizer.apply_chat_template(chat_list[:-1], tokenize=False, add_generation_prompt=True)
    answer_text = tokenizer.apply_chat_template(chat_list, tokenize=False, add_generation_prompt=False)[len(prompt_text):]
    answer_text = answer_text.strip()
    instruction = tokenizer(prompt_text, add_special_tokens=False)
    response = tokenizer(answer_text, add_special_tokens=False)
    input_ids = instruction['input_ids'] + response['input_ids']
    attention_mask = instruction['attention_mask'] + response['attention_mask']
    labels = [-100] * len(instruction['input_ids']) + response['input_ids']
    # 截断
    if len(input_ids) > MAX_LENTH:
        input_ids = input_ids[:MAX_LENTH]
        attention_mask = attention_mask[:MAX_LENTH]
        labels = labels[:MAX_LENTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def process_data(item):

    """
    text: 文本内容
    mention_span: 实体的文本内容
    labels: 实体的标签
    description: 实体的描述
    candidates: 实体的候选标签
    """
    # item['candidates'] = complete_hierarchy_tags(item['candidates'])
    truth_label_in_candidates = [label for label in item["label"] if label in item['candidates']]
    truth_label_in_candidates = sorted(truth_label_in_candidates)
    answer = f"```json\n{json.dumps(truth_label_in_candidates)}\n```"
    prompt = f"Text: {item['text']}\nEntity: {item['mention']}\nCandidates: {item['candidates']}"
    return prompt, answer

def process_data_direct_infer(item):
    prompt = f"Text: {item['text']}\nEntity: {item['mention']}"
    truth_label = sorted(item["label"])
    answer = f"```json\n{json.dumps(truth_label)}\n```"
    return prompt, answer

from datasets import Dataset
def finetune_model(data, system_prompt, model_path, save_path, general_method):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

    raw_dataset = Dataset.from_list(data)
    tokenized_dataset = raw_dataset.map(
        general_process_data,
        fn_kwargs={"system_prompt": system_prompt, "tokenizer": tokenizer, "is_system": True,
                   "general_method": general_method},
        remove_columns=raw_dataset.column_names,
        batched=False
    )
    for i in range(5):
        print(tokenizer.decode(tokenized_dataset[i]['input_ids']))
    general_finetune_pipeline(model, tokenizer, save_path, tokenized_dataset)
