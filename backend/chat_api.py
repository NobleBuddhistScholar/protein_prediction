import json
import logging
from typing import Optional
from openai import OpenAI
import os
from datetime import datetime
from creat_knowledgeDB import search_knowledge

api_key = "sk-755fa616aac649b5be5d47c6af5ed44a" # 请替换为您的实际API密钥

def generate_summary_stream(genome_id: str, genome_data: dict, save_path: str, rag = "virus_knowledge", if_rag = True,):
    #检索知识库
    if if_rag == True:
        arg_data = search_knowledge(genome_id = genome_id, genome_data=genome_data, collection_name=rag, top_k=3, return_results=True)
    else:
        arg_data = '无'
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    messages = [
        {
            "role": "system",
            "content": "你是一个专业的生物信息学助手，擅长分析病毒基因组序列和蛋白质功能预测。请用专业但易懂的语言进行解释，并给出有依据的推测。"
            "请从以下方面进行专业分析：\n"
            "1. 整体基因组结构特征\n"
            "2. 各预测蛋白的功能分析\n"
            "3. 可能的病毒分类及生物学意义\n"
            "4. 特别关注关键蛋白的特征\n"
        },
        {
            "role": "user",
            "content": f"""请分析以下基因组预测结果：
            
            **基因组基本信息**
            - 编号: {genome_id}
            - 长度: {genome_data['metadata']['length']} bp
            
            **预测的开放阅读框(ORFs)**:
            {json.dumps(genome_data['features'], indent=2, ensure_ascii=False)}

            **知识库提供的信息**:
            {arg_data}
            
            请从以下方面进行专业分析：
            1. 整体基因组结构特征
            2. 各预测蛋白的功能分析
            3. 可能的病毒分类及生物学意义
            4. 特别关注关键蛋白的特征"""
        }
    ]
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
            top_p=0.9,
            stream=True
        )
        summary = ""
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="", flush=True)
                summary += chunk.choices[0].delta.content
                yield chunk.choices[0].delta.content
        if summary:
            os.makedirs(save_path, exist_ok=True)
            current_date = datetime.now().strftime("%Y-%m-%d")
            summary_path = os.path.join(save_path, f"{genome_id}_基因组分析报告_{current_date}.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary)
    except Exception as e:
        yield f"[ERROR]{str(e)}"