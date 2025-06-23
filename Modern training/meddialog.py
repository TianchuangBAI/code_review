from datasets import load_dataset
import os
import shutil
import logging

# 配置日志以查看下载进度
logging.basicConfig(level=logging.INFO)

def download_meddialog_to_local(output_dir="meddialog_en"):
    """
    下载 MedDialog 数据集到当前目录下的指定文件夹（默认：meddialog_en）
    :param output_dir: 数据集存储目录名称（英文）
    :return: 数据集对象（如果成功）
    """
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据集（自动下载到 Hugging Face 默认缓存目录）
        logging.info("正在下载数据集...")
        ds = load_dataset("petkopetkov/MedDialog", trust_remote_code=True)
        
        # 获取 Hugging Face 缓存路径
        from datasets.config import HF_DATASETS_CACHE
        cache_path = os.path.join(HF_DATASETS_CACHE, "petkopetkov___MedDialog")
        
        # 检查缓存路径是否存在
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"数据集缓存路径 {cache_path} 不存在，可能下载失败！")
        
        # 将缓存数据复制到当前目录下的目标文件夹
        logging.info(f"正在复制数据集到本地目录: ./{output_dir}")
        shutil.copytree(cache_path, output_dir, dirs_exist_ok=True)
        
        # 打印数据集信息
        print("\n" + "=" * 50)
        print(f"数据集已成功下载到: ./{output_dir}")
        print(f"数据集结构: {ds}")
        print(f"训练集样本数: {len(ds['train'])}")
        
        if 'validation' in ds:
            print(f"验证集样本数: {len(ds['validation'])}")
        if 'test' in ds:
            print(f"测试集样本数: {len(ds['test'])}")
        
        print("\n第一条训练数据示例:")
        print(ds["train"][0])
        print("=" * 50 + "\n")
        
        return ds
    
    except Exception as e:
        logging.error(f"下载失败: {str(e)}")
        print("\n可能的原因：")
        print("1. 网络问题（可能需要科学上网）")
        print("2. Hugging Face 访问权限问题")
        print("3. 磁盘空间不足")
        return None

# 使用示例
if __name__ == "__main__":
    # 下载到当前目录下的 "meddialog_en" 文件夹
    dataset = download_meddialog_to_local()
    
    if dataset is not None:
        # 在这里继续你的数据处理代码
        # 例如：访问训练数据
        # train_data = dataset["train"]
        pass