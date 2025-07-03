# 导入必要的模块
from dotenv import load_dotenv  # 用于加载环境变量
from langchain_openai import ChatOpenAI  # OpenAI聊天模型接口
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # 消息类型定义

# 从.env文件加载环境变量（包含API密钥等配置）
load_dotenv()

# 创建ChatOpenAI模型实例
# 使用gpt-4o模型，这是OpenAI最新的多模态模型，支持文本和图像输入
model = ChatOpenAI(model="gpt-3.5-turbo")

# 初始化聊天历史记录列表
# 用于存储整个对话过程中的所有消息
chat_history = []

# 设置初始系统消息（可选）
# 系统消息用于定义AI助手的行为和角色
system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message)  # 将系统消息添加到聊天历史中

# 开始交互式聊天循环
while True:
    # 获取用户输入
    query = input("You: ")
    
    # 检查退出条件：如果用户输入"exit"（不区分大小写），则退出程序
    if query.lower() == "exit":
        break
    
    # 将用户消息添加到聊天历史中
    # HumanMessage表示用户输入的消息
    chat_history.append(HumanMessage(content=query))

    # 使用完整的聊天历史调用AI模型
    # 这样AI可以记住之前的对话内容，实现上下文连贯的对话
    result = model.invoke(chat_history)
    response = result.content
    
    # 将AI的回复添加到聊天历史中
    # AIMessage表示AI助手的回复消息
    chat_history.append(AIMessage(content=response))

    # 打印AI的回复
    print(f"AI: {response}")

# 程序结束后，打印完整的消息历史记录
# 这有助于调试和了解对话的完整流程
print("---- Message History ----")
print(chat_history)

# 实际生产中，chat_history存在云端，用户回来后，可以随时查看对话历史
