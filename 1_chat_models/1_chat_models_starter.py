# 导入OpenAI的聊天模型类
# ChatOpenAI是LangChain提供的与OpenAI GPT模型交互的接口
from langchain_openai import ChatOpenAI

# 导入环境变量加载工具
# dotenv用于从.env文件中加载环境变量，比如API密钥
from dotenv import load_dotenv

# 加载.env文件中的环境变量
# 这会将.env文件中的变量（如OPENAI_API_KEY）加载到系统环境中
load_dotenv()

# 创建ChatOpenAI实例
# model参数指定使用的模型名称，这里使用GPT-4
# 如果没有设置API密钥，会从环境变量OPENAI_API_KEY中获取
llm = ChatOpenAI(model="gpt-3.5-turbo")

# 调用语言模型
# invoke()方法用于向模型发送消息并获取响应
# 这里询问印度的当前时间
result = llm.invoke("Who is the current president of the United States?")

# 打印模型的响应结果
# result是一个AIMessage对象，包含模型的回复内容
print(result)
# print(result.content)

