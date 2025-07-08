import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
import sys
import json
import torch

# 1. 全局变量设置
PERSIST_DIRECTORY = "./chroma_db"  # 向量库存储目录
COLLECTION_NAME = "virus_knowledge"  # 集合名称

# 2. 全局变量用于延迟加载模型
embedder = None

def get_embedder():
    """延迟加载模型，仅在实际需要时加载"""
    global embedder
    if embedder is None:
        print("正在加载文本向量化模型...")
        try:
            embedder = SentenceTransformer('embedding_model/shibing624_text2vec-base-chinese',
                                          device="cuda" if torch.cuda.is_available() else "cpu")
            print("文本向量化模型加载完成")
        except Exception as e:
            print(f"加载文本向量化模型出错: {str(e)}")
            raise
    return embedder

# 3. 只按固定长度切分文本（支持多段落合并）
def load_text_chunks(file_path, chunk_size=500, overlap=50):
    """
    按固定长度切分文本，支持段落合并和重叠窗口
    
    参数:
    - chunk_size: 每个文本块的目标长度（字符数）
    - overlap: 相邻文本块的重叠长度（增强上下文连续性）
    """
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在: {file_path}")
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"读取文件时出错 {file_path}: {e}")
        raise
    
    # 移除多余空白字符，但保留段落分隔符
    text = text.replace('\n\n', '|PARAGRAPH|')  # 标记段落分隔符
    text = " ".join(text.split())  # 合并所有空白字符为单个空格
    text = text.replace('|PARAGRAPH|', '\n\n')  # 恢复段落分隔符
    
    # 按固定长度切分，保留重叠
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i+chunk_size])
    
    return chunks

# 构建向量库
def build_knowledge_base(path, chunk_size=500, overlap=50):
    """
    构建向量知识库，支持单个文件或整个文件夹
    
    参数:
    - path: 文件路径或文件夹路径
    - chunk_size: 文本分块大小
    - overlap: 文本块重叠长度
    """
    # 初始化 Chroma 本地数据库
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
    )
    
    # 处理路径是文件还是文件夹
    all_chunks = []
    file_paths = []
    
    if os.path.isdir(path):
        print(f"检测到文件夹: {path}")
        # 处理文件夹中的所有txt文件
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    print(f"处理文件: {file_path}")
                    try:
                        chunks = load_text_chunks(file_path, chunk_size, overlap)
                        all_chunks.extend(chunks)
                        # 记录每个块对应的源文件
                        file_paths.extend([file] * len(chunks))
                    except Exception as e:
                        print(f"处理文件 {file_path} 时出错: {e}")
    else:
        # 处理单个文件
        print(f"处理文件: {path}")
        all_chunks = load_text_chunks(path, chunk_size, overlap)
        file_name = os.path.basename(path)
        file_paths = [file_name] * len(all_chunks)
    
    print(f"文本已切分为 {len(all_chunks)} 个片段")
    
    # 如果没有提取到任何文本，直接返回
    if len(all_chunks) == 0:
        print("没有发现有效的文本内容，请检查文件路径")
        return chroma_client, collection
    
    # 计算每段的 embedding（使用批处理加速）
    print("正在计算文本向量表示...")
    embeddings = get_embedder().encode(all_chunks, batch_size=64).tolist()
    
    # 生成唯一ID
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    
    # 添加文件来源元数据
    metadatas = [{"source": file_path} for file_path in file_paths]
    
    # 批量添加到向量库
    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )
    
    # 数据已自动持久化到磁盘
    print("知识库已向量化并入库！")
    print("数据已自动持久化到磁盘（ChromaDB自动持久化）")
    return chroma_client, collection
def list_collections():
    """
    列出当前知识库中的所有集合
    """
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    collections = chroma_client.list_collections()
    
    print(f"当前知识库中共有 {len(collections)} 个集合:")
    for i, collection in enumerate(collections):
        print(f"{i+1}. {collection.name} (ID: {collection.id})")
    
    return collections

def add_to_knowledge_base(path, collection_name=COLLECTION_NAME, chunk_size=500, overlap=50):
    """
    向现有知识库集合中添加新的文本内容，支持单个文件或整个文件夹
    
    参数:
    - path: 新知识文本的路径(文件或文件夹)
    - collection_name: 要更新的集合名称
    - chunk_size: 文本分块大小
    - overlap: 文本块重叠长度
    """
    # 连接到 Chroma 数据库
    print(f"正在连接到知识库: {PERSIST_DIRECTORY}")
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    
    # 获取指定集合
    print(f"获取集合: {collection_name}")
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception as e:
        print(f"错误: 无法获取集合 '{collection_name}', {e}")
        return None
    
    # 获取当前集合中的条目数量，用于生成新的唯一ID
    collection_count = collection.count()
    print(f"当前集合中已有 {collection_count} 个条目")
    
    # 处理路径是文件还是文件夹
    all_chunks = []
    file_paths = []
    
    if os.path.isdir(path):
        print(f"检测到文件夹: {path}")
        # 处理文件夹中的所有txt文件
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    print(f"处理文件: {file_path}")
                    try:
                        chunks = load_text_chunks(file_path, chunk_size, overlap)
                        all_chunks.extend(chunks)
                        # 记录每个块对应的源文件
                        file_paths.extend([file] * len(chunks))
                    except Exception as e:
                        print(f"处理文件 {file_path} 时出错: {e}")
    else:
        # 处理单个文件
        print(f"处理文件: {path}")
        all_chunks = load_text_chunks(path, chunk_size, overlap)
        file_name = os.path.basename(path)
        file_paths = [file_name] * len(all_chunks)
    
    print(f"新文本已切分为 {len(all_chunks)} 个片段")
    
    # 如果没有提取到任何文本，直接返回
    if len(all_chunks) == 0:
        print("没有发现有效的文本内容，请检查文件路径")
        return collection
    
    # 计算每段的 embedding
    print("正在计算文本向量表示...")
    embeddings = get_embedder().encode(all_chunks, batch_size=64).tolist()
    
    # 生成唯一ID，从当前数量开始递增
    ids = [f"chunk_{collection_count + i}" for i in range(len(all_chunks))]
    
    # 添加文件来源元数据
    metadatas = [{"source": file_path} for file_path in file_paths]
    
    # 批量添加到向量库
    print("正在添加新内容到知识库...")
    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )
    
    print(f"成功添加 {len(all_chunks)} 个新文本块到集合 '{collection_name}'")
    print(f"集合现在总共有 {collection.count()} 个条目")
    
    return collection

def get_collection_info(collection_name=COLLECTION_NAME):
    """
    获取指定集合的详细信息
    """
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    
    try:
        collection = chroma_client.get_collection(name=collection_name)
        
        # 获取集合的基本信息
        count = collection.count()
        
        print(f"\n集合信息: {collection_name}")
        print(f"总条目数: {count}")
        
        # 获取示例数据
        if count > 0:
            sample = collection.peek(limit=1)
            print("\n示例数据:")
            print(f"ID: {sample['ids'][0]}")
            print(f"内容: {sample['documents'][0][:200]}...")
            
        return {"name": collection_name, "count": count}
    
    except Exception as e:
        print(f"错误: 无法获取集合 '{collection_name}' 的信息, {e}")
        return None
# 检索
def search_knowledge(genome_id: str, genome_data: dict, collection_name=COLLECTION_NAME, top_k=3, return_results=False):
    """
    根据查询在知识库中检索相关内容
    
    参数:
    - genome_id: 基因组ID
    - genome_data: 基因组数据
    - collection_name: 集合名称
    - top_k: 返回的最相关结果数量
    - return_results: 是否返回结果（用于RAG集成）
    
    返回:
    - 如果 return_results=True，返回检索到的文档和其相似度分数
    - 否则直接打印结果
    """
    # 初始化集合
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    try:
        collection = chroma_client.get_collection(collection_name)
    except Exception as e:
        print(f"错误: 无法获取集合 '{collection_name}', {e}")
        print("尝试创建新集合...")
        collection = chroma_client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    # 获取查询关键词
    print("正在获取查询关键词...")
    query = get_qurey(genome_id, genome_data)
    # 对查询进行向量化
    query_embedding = get_embedder().encode([query]).tolist()[0]
    
    # 执行向量检索
    print("正在执行向量检索...")
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances", "metadatas"]
    )
    
    if return_results:
        # 返回用于RAG的结果
        retrieved_docs = []
        for i, (doc, dist) in enumerate(zip(results['documents'][0], results['distances'][0])):
            retrieved_docs.append({
                "content": doc,
                "similarity": 1-dist
            })
        return retrieved_docs
    else:
        # 打印结果（用于调试）
        print(f"\n查询: {query}")
        for i, (doc, dist) in enumerate(zip(results['documents'][0], results['distances'][0])):
            print(f"匹配 {i+1} (相似度: {1-dist:.4f}):")
            print(f"{doc[:200]}...\n")  # 显示前200个字符

# 获取qurey
def get_qurey(genome_id: str, genome_data: dict):
    """
    获得查询关键词
    """
    from openai import OpenAI
    client = OpenAI(api_key="sk-755fa616aac649b5be5d47c6af5ed44a", base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个专业的生物信息学助手,你需要从用户提供的病毒极影上的蛋白质的注释信息提取关键信息，这些关键信息将用于后续知识库的检索,关键信息需要简洁精准"},
            {"role": "user",
            "content": f"""请分析以下基因组预测结果：
            
            **基因组基本信息**
            - 编号: {genome_id}
            - 长度: {genome_data['metadata']['length']} bp
            
            **预测的开放阅读框(ORFs)**:
            {json.dumps(genome_data['features'], indent=2, ensure_ascii=False)}"""},
        ],
        stream=False
    )
    
    print(response.choices[0].message.content)
    return response.choices[0].message.content
# 主程序
if __name__ == "__main__":
    print(f"ChromaDB 版本: {chromadb.__version__}")
    
    # 显示菜单
    print("\n==== 病毒知识库管理系统 ====")
    print("1. 列出当前知识库中的所有集合")
    print("2. 查看集合详情")
    print("3. 构建新的知识库（文件）")
    print("4. 构建新的知识库（文件夹）")
    print("5. 向知识库添加内容（文件）")
    print("6. 向知识库添加内容（文件夹）")
    print("7. 执行知识检索")
    print("0. 退出")
    
    choice = input("\n请选择操作 [0-7]: ")
    
    if choice == "1":
        # 列出所有集合
        list_collections()
        
    elif choice == "2":
        # 查看集合详情
        collection_name = input(f"请输入集合名称（默认: {COLLECTION_NAME}）: ") or COLLECTION_NAME
        get_collection_info(collection_name)
        
    elif choice == "3":
        # 构建新知识库（文件）
        file_path = input("请输入知识文本文件路径: ")
        if os.path.isfile(file_path):
            build_knowledge_base(file_path)
        else:
            print(f"错误: 文件 '{file_path}' 不存在")
            
    elif choice == "4":
        # 构建新知识库（文件夹）
        folder_path = input("请输入知识文本文件夹路径: ")
        if os.path.isdir(folder_path):
            build_knowledge_base(folder_path)
        else:
            print(f"错误: 文件夹 '{folder_path}' 不存在")
            
    elif choice == "5":
        # 添加内容到知识库（文件）
        file_path = input("请输入知识文本文件路径: ")
        collection_name = input(f"请输入目标集合名称 (默认: {COLLECTION_NAME}): ") or COLLECTION_NAME
        if os.path.isfile(file_path):
            add_to_knowledge_base(file_path, collection_name)
        else:
            print(f"错误: 文件 '{file_path}' 不存在")
            
    elif choice == "6":
        # 添加内容到知识库（文件夹）
        folder_path = input("请输入知识文本文件夹路径: ")
        collection_name = input(f"请输入目标集合名称 (默认: {COLLECTION_NAME}): ") or COLLECTION_NAME
        if os.path.isdir(folder_path):
            add_to_knowledge_base(folder_path, collection_name)
        else:
            print(f"错误: 文件夹 '{folder_path}' 不存在")
            
    elif choice == "7":
        # 执行测试查询
        search_knowledge(genome_id="BS016439.1", genome_data={
          "metadata": {
            "genome_id": "MN025317.1",
            "length": 1014,
            "detection_parameters": {
              "min_confidence": 0.5,
              "min_length": 100,
              "model_version": "HybridModel"
            }
          },
          "features": [
            {
              "type": "nsp1",
              "location": "706..1014",
              "strand": "+",
              "qualifiers": {
                "confidence": 0.879,
                "protein_length": 103
              }
            }
          ]
        }, collection_name=COLLECTION_NAME, top_k=3, return_results=False)
        
    elif choice == "0":
        print("程序已退出")
        
    else:
        print("无效的选择，请重新运行程序")
