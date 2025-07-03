# 导入不同AI提供商的聊天模型接口
# ChatGoogleGenerativeAI: Google的Gemini模型接口
# ChatAnthropic: Anthropic的Claude模型接口  
# ChatOpenAI: OpenAI的GPT模型接口
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

# 导入环境变量加载工具和消息类型
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

# LangChain聊天模型文档链接
# https://python.langchain.com/docs/integrations/chat/

# 加载环境变量（包含各AI提供商的API密钥）
load_dotenv()

# 定义统一的对话消息
# 包含系统指令和用户问题
messages = [
    SystemMessage(content="Solve the following math problems"),  # 系统消息：设置AI为数学问题解答者
    HumanMessage(content="What is the square root of 49?"),      # 用户消息：询问49的平方根
]

# ===== OpenAI GPT模型示例 =====

# 创建OpenAI聊天模型实例
# gpt-3.5-turbo是OpenAI的对话优化模型，性价比高
model = ChatOpenAI(model="gpt-3.5-turbo")

# 调用模型处理消息
result = model.invoke(messages)
print(f"Answer from OpenAI: {result.content}")


# ===== Anthropic Claude模型示例 =====

# 创建Anthropic聊天模型实例
# claude-3-opus-20240229是Claude 3系列中最强大的模型
# Anthropic模型文档: https://docs.anthropic.com/en/docs/models-overview
model = ChatAnthropic(model="claude-3-opus-20240229")

# 调用模型处理相同的消息
result = model.invoke(messages)
print(f"Answer from Anthropic: {result.content}")


# ===== Google Gemini模型示例 =====

# 创建Google聊天模型实例
# gemini-1.5-flash是Google的快速响应模型，适合实时应用
# Google AI控制台: https://console.cloud.google.com/gen-app-builder/engines
# Gemini API文档: https://ai.google.dev/gemini-api/docs/models/gemini
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# 调用模型处理相同的消息
result = model.invoke(messages)
print(f"Answer from Google: {result.content}")
