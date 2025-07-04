# 与 1a_basic_part_1.py 的对比分析
# 
# | 方面 | 1a_basic_part_1.py (构建阶段) | 1b_basic_part_2.py (检索阶段) |
# |------|------------------------------|------------------------------|
# | **主要目的** | 构建向量数据库 | 从向量数据库中检索相关文档 |
# | **文件处理** | 加载、分割、向量化文档 | 不处理文件，只加载已存在的数据库 |
# | **嵌入模型** | 创建嵌入模型实例 | 使用相同的嵌入模型实例 |
# | **Chroma 使用** | `Chroma.from_documents()` 创建新数据库 | `Chroma()` 加载已存在的数据库 |
# | **输出结果** | 向量数据库文件 | 检索到的相关文档列表 |
# | **性能考虑** | 计算密集型（向量化） | 查询密集型（相似度搜索） |
# | **执行频率** | 一次性构建，离线处理 | 实时查询，在线处理 |
# | **资源消耗** | 高（需要处理整个文档） | 低（只处理查询） |
# | **依赖关系** | 无依赖，独立构建 | 依赖 part_1 构建的数据库 |
#
# 关键区别总结：
# - part_1 是离线构建阶段，将文档转换为向量并存储到数据库中，这个过程比较耗时，但只需要执行一次
# - part_2 是在线检索阶段，根据用户查询快速检索相关文档，这个过程很快，可以实时执行
# - 这种分离设计使得系统既高效又灵活，构建好的向量数据库可以被多次查询使用

import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Define the persistent directory
# 与 part_1 相同，使用相同的路径来加载已创建的向量数据库
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Define the embedding model
# 注意：必须使用与 part_1 中相同的嵌入模型，否则向量不兼容
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the existing vector store with the embedding function
# 这里不需要重新创建数据库，而是加载 part_1 中已经构建好的数据库
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

# Define the user's question
# 这是我们要在文档中搜索的查询
query = "Where does Gandalf meet Frodo?"

# Retrieve relevant documents based on the query
# 创建检索器，用于从向量数据库中查找最相关的文档
retriever = db.as_retriever(
    search_type="similarity_score_threshold",  # 使用相似度分数阈值搜索
    search_kwargs={"k": 3, "score_threshold": 0.5},  # 检索参数
)
# 执行检索，获取相关文档
relevant_docs = retriever.invoke(query)

# Display the relevant results with metadata
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")  # 显示文档内容
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")  # 显示文档来源
