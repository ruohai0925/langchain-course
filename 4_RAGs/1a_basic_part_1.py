# RAG: 检索增强生成（Retrieval-Augmented Generation）
# 检索增强生成是一种结合了检索和生成的技术，它使用检索到的相关文档来增强生成模型的回答

import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# 定义包含文本文件的目录和持久化目录
# 使用 os.path.dirname(os.path.abspath(__file__)) 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建文本文件的完整路径，假设文件在 documents 子目录下
file_path = os.path.join(current_dir, "documents", "lord_of_the_rings.txt")
# 构建向量数据库的持久化存储目录
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# 检查 Chroma 向量存储是否已经存在
# 如果不存在，则初始化向量存储
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # 确保文本文件存在
    # 如果文件不存在，抛出 FileNotFoundError 异常
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )

    # 从文件中读取文本内容
    # TextLoader 用于加载纯文本文件
    loader = TextLoader(file_path)
    documents = loader.load()

    # 将文档分割成块
    # CharacterTextSplitter 按字符数分割文档
    # chunk_size=1000：每个块最多1000个字符
    # chunk_overlap=50：相邻块之间重叠50个字符，保持上下文连贯性
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50) 
    docs = text_splitter.split_documents(documents)

    # 显示分割后文档的信息
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    # 创建嵌入向量
    print("\n--- Creating embeddings ---")
    # OpenAIEmbeddings 用于将文本转换为向量表示
    # model="text-embedding-3-small"：使用 OpenAI 的 text-embedding-3-small 模型
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )  # 如果需要，可以更新为其他有效的嵌入模型
    print("\n--- Finished creating embeddings ---")

    # 创建向量存储并自动持久化
    print("\n--- Creating vector store ---")
    # Chroma.from_documents 将文档、嵌入向量和持久化目录作为参数
    # 自动将文档转换为向量并存储到指定的持久化目录
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating vector store ---")

else:
    # 如果向量存储已经存在，则不需要重新初始化
    print("Vector store already exists. No need to initialize.")

# 要问的问题
# Who is the Ring-bearer?
# Where does Gandalf meet Frodo?