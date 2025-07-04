# 与 2a_rag_basics_metadata.py 的对比分析
# 
# | 方面 | 2a_rag_basics_metadata.py (构建阶段) | 2b_rag_basics_metadata.py (检索+生成阶段) |
# |------|-----------------------------------|------------------------------------------|
# | **主要目的** | 构建带元数据的向量数据库 | 从向量数据库中检索并生成回答 |
# | **文件处理** | 加载、分割、向量化多个文档 | 不处理文件，只加载已存在的数据库 |
# | **元数据使用** | 创建元数据 | 读取和显示元数据 |
# | **输出结果** | 向量数据库文件 | 检索结果 + 生成的回答 |
# | **模型使用** | OpenAIEmbeddings (向量化) | OpenAIEmbeddings + ChatOpenAI (检索+生成) |
# | **检索参数** | 无 | k=3, score_threshold=0.2 |
# | **生成功能** | 无 | 有（但被注释掉） |
# | **执行频率** | 一次性构建，离线处理 | 实时查询，在线处理 |
# | **依赖关系** | 无依赖，独立构建 | 依赖 2a 构建的数据库 |
# | **查询示例** | 无 | "Where is Dracula's castle located?" |
#
# 关键区别总结：
# - 2a 是离线构建阶段，将多个文档转换为向量并存储到数据库中，添加元数据管理
# - 2b 是在线检索阶段，根据用户查询检索相关文档，并可以生成回答（当前被注释）
# - 2b 展示了完整的 RAG 流程：检索(Retrieval) + 生成(Generation)

import os

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# 定义持久化目录
# 与 2a 使用相同的路径，确保能加载到正确的向量数据库
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_with_metadata")

# 定义嵌入模型
# 必须使用与 2a 中相同的嵌入模型，确保向量兼容性
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 加载已存在的向量存储，并指定嵌入函数
# 这里加载的是 2a 中构建的带元数据的向量数据库
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

# 定义用户的问题
# 这是一个具体的查询示例，用于测试检索功能
query = "Where is Dracula's castle located?"

# 基于查询检索相关文档
# 创建检索器，配置搜索参数
retriever = db.as_retriever(
    search_type="similarity_score_threshold",  # 使用相似度分数阈值搜索
    search_kwargs={"k": 3, "score_threshold": 0.2},  # 检索参数：最多3个文档，相似度阈值0.2
)
# 执行检索，获取相关文档
relevant_docs = retriever.invoke(query)

# 显示相关结果及其元数据
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")  # 显示文档内容
    print(f"Source: {doc.metadata['source']}\n")  # 显示文档来源（元数据）

# 以下是生成回答的代码，当前被注释掉
# 这部分展示了完整的 RAG 流程：检索 + 生成
# combined_input = (
#     "Here are some documents that might help answer the question: "
#     + query
#     + "\n\nRelevant Documents:\n"
#     + "\n\n".join([doc.page_content for doc in relevant_docs])
#     + "\n\nPlease provide a rough answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
# )

# # 创建 ChatOpenAI 模型用于生成回答
# model = ChatOpenAI(model="gpt-4o")

# # 定义发送给模型的消息
# messages = [
#     SystemMessage(content="You are a helpful assistant."),  # 系统角色设定
#     HumanMessage(content=combined_input),  # 用户消息，包含查询和相关文档
# ]

# # print(messages, "messages")

# # 调用模型生成回答
# result = model.invoke(messages)

# # 显示完整结果和仅内容
# print("\n--- Generated Response ---")
# print("Full result:")
# # print(result)
# print("Content only:")
# print(result.content)
