# 导入必要的模块
# SystemMessage: 用于设置AI助手的系统角色和行为
# HumanMessage: 表示用户输入的消息
# AIMessage: 表示AI助手的回复消息
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 导入OpenAI的聊天模型接口
from langchain_openai import ChatOpenAI

# 导入环境变量加载工具，用于从.env文件加载API密钥等配置
from dotenv import load_dotenv

# 加载.env文件中的环境变量（如OPENAI_API_KEY）
load_dotenv()

# 创建ChatOpenAI实例，使用gpt-3.5-turbo模型
# 这个模型是OpenAI提供的对话优化模型，适合聊天和问答任务
llm = ChatOpenAI(model="gpt-3.5-turbo")

# 定义对话消息列表
# 消息列表包含对话的上下文，按时间顺序排列
messages = [
    # 系统消息：设置AI助手的角色和行为
    # 这里将AI助手设置为社交媒体内容策略专家
    SystemMessage("You are an expert in social media content strategy"), 
    
    # 人类消息：用户的输入
    # 询问如何创建吸引人的Instagram帖子
    HumanMessage("Give a short tip to create engaging posts on Instagram"), 
]

# 调用语言模型，传入消息列表
# invoke方法会处理整个对话上下文并生成回复
result = llm.invoke(messages)

# 打印AI助手的回复内容
# result.content包含AI生成的文本回复
print(result.content)