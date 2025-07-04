# 与 2b_rag_basics_metadata.py 的对比分析
# 
# | 方面 | 2b_rag_basics_metadata.py | 3_rag_one_off_question.py |
# |------|---------------------------|---------------------------|
# | **主要目的** | 检索文档并显示元数据 | 完整的 RAG 问答系统 |
# | **生成功能** | 被注释掉 | 已启用，完整实现 |
# | **检索类型** | similarity_score_threshold | similarity |
# | **检索参数** | k=3, score_threshold=0.2 | k=3 |
# | **元数据显示** | 显示文档来源 | 不显示元数据 |
# | **查询示例** | Dracula's castle location | Dracula's fears |
# | **输出格式** | 仅检索结果 | 检索结果 + 生成回答 |
# | **环境变量** | 无 load_dotenv() | 有 load_dotenv() |
# | **库导入** | langchain.schema | langchain_core.messages |
# | **功能完整性** | 部分功能 | 完整 RAG 流程 |
# | **使用场景** | 调试和测试 | 实际问答应用 |
#
# 关键区别总结：
# - 2b 主要是检索功能，生成部分被注释，适合调试和查看检索结果
# - 3_rag_one_off_question 是完整的 RAG 系统，包含检索和生成两个完整步骤
# - 3_rag_one_off_question 更适合实际应用，可以直接回答用户问题
# - 检索策略不同：2b 使用阈值过滤，3_rag_one_off_question 使用纯相似度排序

import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 加载 .env 文件中的环境变量
# 确保能正确读取 OpenAI API 密钥等配置
load_dotenv()

# 定义持久化目录
# 使用与之前相同的路径，加载已构建的向量数据库
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(
    current_dir, "db", "chroma_db_with_metadata")

# 定义嵌入模型
# 必须使用与构建阶段相同的嵌入模型，确保向量兼容性
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 加载已存在的向量存储，并指定嵌入函数
# 这里加载的是之前构建的带元数据的向量数据库
db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

# 定义用户的问题
# 这是一个具体的查询示例，用于测试完整的 RAG 系统
query = "What does dracula fear the most?"

# 基于查询检索相关文档
# 创建检索器，使用纯相似度搜索（无阈值过滤）
retriever = db.as_retriever(
    search_type="similarity",  # 使用纯相似度搜索，返回最相似的文档
    search_kwargs={"k": 3},    # 检索参数：返回最相似的3个文档
)
# 执行检索，获取相关文档
relevant_docs = retriever.invoke(query)

# 显示相关结果（不显示元数据）
print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")  # 只显示文档内容，不显示来源

# 将查询和相关文档内容组合成输入
# 这是 RAG 系统的核心：将检索到的文档与用户查询结合
combined_input = (
    "Here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant Documents:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])  # 将所有相关文档内容拼接
    + "\n\nPlease provide a rough answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
)

# 创建 ChatOpenAI 模型用于生成回答
# 使用 GPT-3.5-turbo 模型
model = ChatOpenAI(model="gpt-3.5-turbo")

# 定义发送给模型的消息
# 使用系统消息和人类消息的标准格式
messages = [
    SystemMessage(content="You are a helpful assistant."),  # 系统角色设定
    HumanMessage(content=combined_input),  # 用户消息，包含查询和相关文档
]

# 调用模型生成回答
# 这是 RAG 系统的生成阶段
result = model.invoke(messages)

# 显示生成的结果
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(result.content)  # 只显示生成的回答内容
