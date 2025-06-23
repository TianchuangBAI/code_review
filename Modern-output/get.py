import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel

# 1. 加载本地模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained("./Qwen/Qwen3-14B", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./Qwen/Qwen3-14B", device_map="auto", torch_dtype=torch.bfloat16)

# 加载LoRA模型
model = PeftModel.from_pretrained(model, model_id="./output/final_model")

# 2. 定义模型推理函数（添加医生提示词）
def generate_response(text, model, tokenizer):
    # 拼接提示词和用户输入
    prompt = (
        "You are a doctor, please give some advice to solve the question from patient.\n\n"
        f"Patient's question: {text}\n\n"
        "Doctor's advice:"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

# 3. 处理输入文件并保存输出
input_file = "descriptions.txt"
output_file = "lora.txt"

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    
    for line in tqdm(infile, desc="Processing descriptions"):
        text = line.strip()
        if text:  # 跳过空行
            try:
                response = generate_response(text, model, tokenizer)
                outfile.write(response.strip() + '\n')
            except Exception as e:
                print(f"处理出错: {text[:50]}... 错误: {str(e)}")
                outfile.write("[ERROR]\n")

print(f"处理完成！结果已保存到 {output_file}")