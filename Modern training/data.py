from datasets import load_dataset
import os

# 下载数据集
print("正在下载数据集...")
dataset = load_dataset("petkopetkov/MedDialog")

# 确保输出目录存在
os.makedirs("output", exist_ok=True)

# 提取所有description并保存到文件
output_file = "output/descriptions.txt"
total_count = 0

with open(output_file, 'w', encoding='utf-8') as f:
    # 处理每个split（train/validation/test）
    for split in dataset.keys():
        print(f"正在处理 {split} 数据...")
        split_data = dataset[split]
        
        # 写入每个description
        for item in split_data:
            description = item['description'].strip()
            if description:  # 确保description不为空
                f.write(description + '\n')
                total_count += 1

print(f"处理完成！共保存 {total_count} 条description到 {output_file}")