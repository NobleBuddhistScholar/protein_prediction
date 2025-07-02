import json
import logging
from typing import Optional
from openai import OpenAI
import os
from datetime import datetime  # Import datetime for date formatting
# 初始化客户端 - 从环境变量获取API密钥
client = OpenAI(
    api_key='sk-2197cbc777064fc6b7fff4ac5c81959d',  # 请设置环境变量DEEPSEEK_API_KEY
    base_url="https://api.deepseek.com"  # 使用基础URL
)

def generate_summary(genome_id: str, genome_data: dict, save_path: str) -> Optional[str]:
    try:
        # 1. 准备对话消息
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的生物信息学助手，擅长分析病毒基因组序列和蛋白质功能预测。"
                           "请用专业但易懂的语言进行解释，并给出有依据的推测。"
            },
            {
                "role": "user",
                "content": f"""请分析以下基因组预测结果：
                
                **基因组基本信息**
                - 编号: {genome_id}
                - 长度: {genome_data['metadata']['length']} bp
                
                **预测的开放阅读框(ORFs)**:
                {json.dumps(genome_data['features'], indent=2, ensure_ascii=False)}
                
                请从以下方面进行专业分析：
                1. 整体基因组结构特征
                2. 各预测蛋白的功能分析
                3. 可能的病毒分类及生物学意义
                4. 特别关注关键蛋白的特征"""
            }
        ]

        # 2. 调用DeepSeek API
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
            top_p=0.9,
            stream=False
        )

        # 3. 提取生成的总结
        summary = response.choices[0].message.content.strip()
        
        if not summary:
            logging.error("API返回空响应")
            return None

        # 4. 保存结果
        os.makedirs(save_path, exist_ok=True)
        
        # Get the current date in the desired format
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Create the filename with the new format
        summary_path = os.path.join(save_path, f"{genome_id}_基因组分析报告_{current_date}.txt")
        
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        
        logging.info(f"总结已保存到: {summary_path}")
        return summary

    except Exception as e:
        logging.error(f"生成总结时出错: {str(e)}", exc_info=True)
        return None