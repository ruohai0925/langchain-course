# 与 1a_basic_part_1.py 的主要区别对比分析
# 
# | 方面 | 1a_basic_part_1.py | 2a_rag_basics_metadata.py |
# |------|-------------------|---------------------------|
# | **文件处理** | 处理单个文件 | 处理目录中的多个文件 |
# | **元数据管理** | 无元数据 | 为每个文档添加来源文件元数据 |
# | **文档分割** | chunk_overlap=50 | chunk_overlap=0 |
# | **目录结构** | 简单的文件路径 | 更复杂的目录结构管理 |
# | **错误处理** | 检查单个文件 | 检查整个目录 |
# | **可扩展性** | 基础功能 | 支持多文档管理 |
# | **文件发现** | 手动指定文件 | 自动发现目录中的.txt文件 |
# | **数据库名称** | chroma_db | chroma_db_with_metadata |
# | **向量库导入** | langchain_chroma | langchain_community.vectorstores |
#
# 关键改进点：
# 1. 多文档支持：可以处理整个目录中的多个文档文件
# 2. 元数据管理：为每个文档添加来源信息，便于后续追踪和过滤
# 3. 更好的目录结构：分离文档目录和数据库目录，便于管理
# 4. 自动文件发现：自动发现目录中的文本文件，无需手动指定

import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 定义包含文本文件的目录和持久化目录
# 与之前的代码相比，这里支持处理多个文件，并添加了元数据管理
current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "documents")  # 文档目录，包含多个文本文件
db_dir = os.path.join(current_dir, "db")  # 数据库目录
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")  # 带元数据的向量数据库路径

# 打印目录信息，便于调试和确认路径
print(f"Books directory: {books_dir}")
print(f"Persistent directory: {persistent_directory}")

# 检查 Chroma 向量存储是否已经存在
# 如果不存在，则初始化向量存储
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # 确保文档目录存在
    # 如果目录不存在，抛出 FileNotFoundError 异常
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            f"The directory {books_dir} does not exist. Please check the path."
        )

    # 列出目录中的所有文本文件
    # 使用列表推导式筛选出所有 .txt 结尾的文件
    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]

    # 从每个文件中读取文本内容，并存储带有元数据的文档
    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = TextLoader(file_path)  # 为每个文件创建加载器
        book_docs = loader.load()  # 加载文档
        for doc in book_docs:
            # 为每个文档添加元数据，标明其来源文件
            # 这是与之前代码的主要区别：添加了元数据管理
            doc.metadata = {"source": book_file}
            documents.append(doc)

    # 将文档分割成块
    # 注意：这里 chunk_overlap=0，与之前代码不同
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # 显示分割后文档的信息
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")

    # 创建嵌入向量
    print("\n--- Creating embeddings ---")
    # 使用与之前相同的嵌入模型
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )  # 如果需要，可以更新为其他有效的嵌入模型
    print("\n--- Finished creating embeddings ---")

    # 创建向量存储并持久化
    print("\n--- Creating and persisting vector store ---")
    # 将文档、嵌入向量和持久化目录作为参数创建向量数据库
    # 元数据会自动保存到数据库中
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating and persisting vector store ---")

else:
    # 如果向量存储已经存在，则不需要重新初始化
    print("Vector store already exists. No need to initialize.")
