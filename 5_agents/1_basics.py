# LangChain Agent 基础示例
# 这个文件展示了如何创建一个简单的 Agent，它可以使用工具来执行任务
# Agent 是一个能够使用工具来解决问题的 AI 系统

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
import datetime
from langchain.agents import tool

# 加载环境变量，确保能正确读取 OpenAI API 密钥
load_dotenv()

# 使用 @tool 装饰器定义一个工具函数
# 这个工具可以让 Agent 获取当前系统时间
@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"): # adjust the format to your needs
    """ 
    返回指定格式的当前日期和时间
    Args:
        format: 时间格式字符串，默认为 "%Y-%m-%d %H:%M:%S"
    Returns:
        格式化后的当前时间字符串
    """
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

# 创建语言模型实例
# 使用 GPT-4 模型，确保 Agent 有足够的推理能力
llm = ChatOpenAI(model="gpt-4")

# 定义用户的查询
# 这是一个需要 Agent 使用工具来回答的问题
query = "What is the current time in New York? (You are in China). Just show the current time and not the date"

# 从 LangChain Hub 拉取 ReAct 提示模板
# ReAct 是一个流行的 Agent 框架，结合了推理(Reasoning)和行动(Acting)
prompt_template = hub.pull("hwchase17/react")

# 定义 Agent 可以使用的工具列表
# 这里只包含一个工具：获取系统时间
# tools = []
tools = [get_system_time]

# 创建 ReAct Agent
# 将语言模型、工具和提示模板组合成一个 Agent
agent = create_react_agent(llm, tools, prompt_template)

# 创建 Agent 执行器
# AgentExecutor 负责管理 Agent 的执行过程，包括工具调用和错误处理
# verbose=True 会显示详细的执行过程，便于调试
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 执行 Agent，处理用户查询
# Agent 会自动分析问题，决定是否需要使用工具，然后生成回答
agent_executor.invoke({"input": query})


