import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 加载模型和tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "./Qwen/Qwen3-14B",
    use_fast=False,
    trust_remote_code=True
)
base_model = AutoModelForCausalLM.from_pretrained(
    "./Qwen/Qwen3-14B",
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, model_id="./output/Qwen3-14B/checkpoint-5066")
model.eval()
print("Model loaded successfully.")

def generate_answer(instruction):
    # 构造提示
    prompt = f"请回答以下选择题：\n{instruction}\n答案是："
    
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=10,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码输出
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取答案部分（假设模型输出格式为"答案是：A"）
    answer = answer.replace(prompt, "").strip()
    
    # 只保留第一个字符（A/B/C/D）
    if len(answer) > 0:
        return answer[0].upper()
    return ""

def process_json_file(input_file, output_file):
    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 处理每条数据
    for item in data:
        if "instruction" in item and "input" in item:
            instruction = item["instruction"]
            answer = generate_answer(instruction)
            item["input"] = answer
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"处理完成，结果已保存到 {output_file}")

# 使用示例
input_json = "hwtcm.json"
output_json = "hwtcm_with_answers.json"
process_json_file(input_json, output_json)