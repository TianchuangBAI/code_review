import json
from tqdm import tqdm
import dashscope
from dashscope import Generation
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 设置你的阿里云API密钥
dashscope.api_key = ""  # 替换为你的实际API密钥
MAX_WORKERS = 3  # 并发数
lock = threading.Lock()  # 用于线程安全的文件写入

def predict_with_api(messages):
    """使用阿里云API调用Qwen3-14B模型"""
    response = Generation.call(
        model='qwen3-14b',
        messages=messages,
        result_format='message',
        enable_thinking=False,
        seed=1234,
        temperature=0.01
    )
    
    if response.status_code == 200:
        return response.output.choices[0].message.content
    else:
        raise Exception(f"API调用失败: {response.code} - {response.message}")

def process_question(data, outfile):
    """处理单个问题并写入结果"""
    question = data['question']
    options = data['options']
    
    # 构建提示信息
    prompt = f"""请根据以下问题选择一个最合适的答案：
问题: {question}
选项:
"""
    for opt in options:
        prompt += f"- {opt}\n"
    prompt += "\n请直接给出选项字母（如A、B、C等）作为答案，不要包含其他内容。"
    
    messages = [
        {"role": "system", "content": "你是一个专业的问答助手，能够准确理解问题并从给定选项中选择最合适的答案。"},
        {"role": "user", "content": prompt}
    ]
    
    try:
        answer = predict_with_api(messages)
        result = {
            "question": question,
            "options": options,
            "model_answer": answer
        }
        
        # 线程安全的写入
        with lock:
            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
        return True
    except Exception as e:
        print(f"处理问题时出错: {question}, 错误: {str(e)}")
        return False

def process_jsonl(input_file, output_file):
    """使用线程池并发处理问题"""
    # 读取所有问题
    with open(input_file, 'r', encoding='utf-8') as infile:
        questions = [json.loads(line) for line in infile]
    
    # 使用进度条
    progress = tqdm(total=len(questions), desc="Processing questions")
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交所有任务
            futures = {
                executor.submit(process_question, data, outfile): data
                for data in questions
            }
            
            # 等待完成并更新进度条
            for future in as_completed(futures):
                future.result()  # 获取结果或捕获异常
                progress.update(1)
    
    progress.close()

if __name__ == "__main__":
    input_jsonl = "qawest.jsonl"  # 输入文件路径
    output_jsonl = "outputs.jsonl"  # 输出文件路径
    
    process_jsonl(input_jsonl, output_jsonl)
    print(f"处理完成，结果已保存到 {output_jsonl}")