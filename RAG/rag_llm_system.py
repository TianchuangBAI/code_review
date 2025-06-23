import json
import os
import numpy as np
from typing import List, Dict, Any
import faiss
import pickle
from pathlib import Path
import re

# PDF处理
import PyPDF2
from io import BytesIO

# 模型相关
from openai import OpenAI
from modelscope import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import CrossEncoder

# 文本处理
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RAGSystem:
    def __init__(self, 
                 model_name="Qwen/Qwen3-14B",
                 embedding_model="text-embedding-v4",
                 rerank_model="BAAI/bge-reranker-base",
                 chunk_size=500,
                 chunk_overlap=50,
                 top_k=10,
                 rerank_top_k=3):
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        
        # 初始化本地LLM
        print("正在加载本地LLM模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        print("本地LLM模型加载完成")
        
        # 初始化embedding客户端
        self.embedding_client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.embedding_model = embedding_model
        
        # 初始化rerank模型
        print("正在加载重排序模型...")
        try:
            self.reranker = CrossEncoder(rerank_model)
            print("重排序模型加载完成")
        except:
            print("重排序模型加载失败，将使用余弦相似度作为备选")
            self.reranker = None
        
        # 向量库相关
        self.index = None
        self.chunks = []
        self.chunk_embeddings = []
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """从PDF文件中提取文本"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"读取PDF文件 {pdf_path} 时出错: {e}")
        return text
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """将文本分割成chunks"""
        # 简单的文本分割策略
        sentences = re.split(r'[。！？\n]', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + "。"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + "。"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """获取文本的embeddings"""
        embeddings = []
        batch_size = 10  # 批量处理以避免API限制
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            try:
                response = self.embedding_client.embeddings.create(
                    model=self.embedding_model,
                    input=batch_texts,
                    dimensions=1024,
                    encoding_format="float"
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
                print(f"已处理 {min(i+batch_size, len(texts))}/{len(texts)} 条文本的embedding")
                
            except Exception as e:
                print(f"获取embedding时出错: {e}")
                # 如果出错，添加零向量作为占位符
                for _ in batch_texts:
                    embeddings.append([0.0] * 1024)
        
        return np.array(embeddings)
    
    def build_knowledge_base(self, pdf_folder: str, save_path: str = "knowledge_base.pkl"):
        """构建知识库"""
        print("正在构建知识库...")
        
        # 检查是否已存在知识库文件
        if os.path.exists(save_path):
            print("发现已存在的知识库文件，正在加载...")
            with open(save_path, 'rb') as f:
                data = pickle.load(f)
                self.chunks = data['chunks']
                self.chunk_embeddings = data['embeddings']
                self.index = data['index']
            print("知识库加载完成")
            return
        
        # 读取所有PDF文件
        all_text = ""
        pdf_files = list(Path(pdf_folder).glob("*.pdf"))
        
        for pdf_file in pdf_files:
            print(f"正在处理: {pdf_file}")
            text = self.extract_text_from_pdf(str(pdf_file))
            all_text += f"\n\n=== {pdf_file.name} ===\n\n" + text
        
        # 分割文本
        print("正在分割文本...")
        self.chunks = self.split_text_into_chunks(all_text)
        print(f"共生成 {len(self.chunks)} 个文本块")
        
        # 生成embeddings
        print("正在生成embeddings...")
        embeddings = self.get_embeddings(self.chunks)
        self.chunk_embeddings = embeddings
        
        # 构建FAISS索引
        print("正在构建FAISS索引...")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # 使用内积相似度
        
        # 归一化embeddings以使用内积计算余弦相似度
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.index.add(normalized_embeddings.astype(np.float32))
        
        # 保存知识库
        print("正在保存知识库...")
        with open(save_path, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'embeddings': self.chunk_embeddings,
                'index': self.index
            }, f)
        
        print("知识库构建完成")
    
    def retrieve_relevant_docs(self, query: str) -> List[Dict[str, Any]]:
        """检索相关文档"""
        # 获取查询的embedding
        query_embedding = self.get_embeddings([query])[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # 检索
        scores, indices = self.index.search(
            query_embedding.reshape(1, -1).astype(np.float32), 
            self.top_k
        )
        
        # 整理检索结果
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):  # 确保索引有效
                results.append({
                    'text': self.chunks[idx],
                    'score': float(score),
                    'index': int(idx)
                })
        
        return results
    
    def rerank_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """重排序文档"""
        if not documents:
            return documents
        
        if self.reranker is not None:
            try:
                # 使用CrossEncoder进行重排序
                pairs = [(query, doc['text']) for doc in documents]
                scores = self.reranker.predict(pairs)
                
                # 更新分数并排序
                for i, score in enumerate(scores):
                    documents[i]['rerank_score'] = float(score)
                
                documents = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
            except Exception as e:
                print(f"重排序失败，使用原始分数: {e}")
        
        # 返回top_k个结果
        return documents[:self.rerank_top_k]
    
    def generate_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成答案"""
        # 构建上下文
        context = "\n\n".join([f"参考文档{i+1}:\n{doc['text']}" for i, doc in enumerate(context_docs)])
        
        # 构建prompt
        prompt = f"""基于以下参考文档回答问题，如果参考文档中没有相关信息，请说明无法从提供的文档中找到答案。

参考文档:
{context}

问题: {query}

请提供准确、详细的答案:"""

        try:
            # 准备模型输入
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            # 生成答案
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            
            # 解析thinking内容
            try:
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0
            
            thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            
            return {
                'answer': content,
                'thinking': thinking_content,
                'context_docs': context_docs,
                'success': True
            }
            
        except Exception as e:
            print(f"生成答案时出错: {e}")
            return {
                'answer': f"生成答案时出现错误: {e}",
                'thinking': "",
                'context_docs': context_docs,
                'success': False
            }
    
    def process_single_query(self, query: str) -> Dict[str, Any]:
        """处理单个查询"""
        print(f"正在处理查询: {query}")
        
        # 检索相关文档
        retrieved_docs = self.retrieve_relevant_docs(query)
        
        # 重排序
        reranked_docs = self.rerank_documents(query, retrieved_docs)
        
        # 生成答案
        result = self.generate_answer(query, reranked_docs)
        
        return {
            'query': query,
            'retrieved_docs': retrieved_docs,
            'reranked_docs': reranked_docs,
            'answer': result['answer'],
            'thinking': result['thinking'],
            'success': result['success']
        }
    
    def batch_inference(self, input_json_path: str, output_json_path: str):
        """批量推理"""
        print(f"开始批量推理，输入文件: {input_json_path}")
        
        # 读取输入数据
        with open(input_json_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        results = []
        
        for i, item in enumerate(input_data):
            query = item.get('input', '')
            expected_output = item.get('output', '')
            
            print(f"\n处理第 {i+1}/{len(input_data)} 个查询...")
            
            # 处理查询
            result = self.process_single_query(query)
            
            # 添加原始数据和期望输出
            result.update({
                'original_instruction': item.get('instruction', ''),
                'original_input': item.get('input', ''),
                'expected_output': expected_output,
                'index': i
            })
            
            results.append(result)
            
            # 定期保存中间结果
            if (i + 1) % 10 == 0:
                temp_output_path = output_json_path.replace('.json', f'_temp_{i+1}.json')
                with open(temp_output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"已保存中间结果到: {temp_output_path}")
        
        # 保存最终结果
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n批量推理完成，结果已保存到: {output_json_path}")
        print(f"总共处理了 {len(results)} 个查询")


def main():
    # 配置参数
    PDF_FOLDER = "knowledge_base"  # PDF文件夹路径
    INPUT_JSON = "input_data.json"  # 输入JSON文件路径
    OUTPUT_JSON = "rag_results.json"  # 输出JSON文件路径
    
    # 检查环境变量
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("请设置环境变量 DASHSCOPE_API_KEY")
        return
    
    # 初始化RAG系统
    rag_system = RAGSystem(
        chunk_size=500,
        chunk_overlap=50,
        top_k=10,
        rerank_top_k=3
    )
    
    # 构建知识库
    rag_system.build_knowledge_base(PDF_FOLDER)
    
    # 批量推理
    rag_system.batch_inference(INPUT_JSON, OUTPUT_JSON)


if __name__ == "__main__":
    main()
