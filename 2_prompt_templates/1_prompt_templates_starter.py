# 导入必要的模块
from langchain_openai import ChatOpenAI  # OpenAI聊天模型接口
from dotenv import load_dotenv  # 环境变量管理
from langchain_core.prompts import ChatPromptTemplate  # 聊天提示模板

# 加载环境变量（包含API密钥等配置）
load_dotenv()

# 创建ChatOpenAI模型实例
llm = ChatOpenAI(model="gpt-3.5-turbo")

# ===== 示例1: 使用模板的提示 =====
# 定义一个包含变量的模板字符串
# 使用花括号{}来标记需要动态填充的变量
template = "Write a {tone} email to {company} expressing interest in the {position} position, mentioning {skill} as a key strength. Keep it to 4 lines max"

# 从模板字符串创建ChatPromptTemplate对象
# ChatPromptTemplate是LangChain提供的提示模板类，用于管理动态提示
prompt_template = ChatPromptTemplate.from_template(template)

# 使用invoke方法填充模板变量
# 传入一个字典，键名对应模板中的变量名
prompt = prompt_template.invoke({
    "tone": "energetic",      # 语气：充满活力的
    "company": "samsung",     # 公司：三星
    "position": "AI Engineer", # 职位：AI工程师
    "skill": "AI"             # 技能：AI
})


# 调用语言模型生成回复
result = llm.invoke(prompt)

# 打印结果
print(result)

# ===== 示例2: 使用系统消息和人类消息的提示（使用元组） =====
# 定义消息列表，每个元素是一个元组 (角色, 内容)
# 角色可以是 "system"（系统消息）或 "human"（人类消息）
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),  # 系统消息：设置AI为喜剧演员
    ("human", "Tell me {joke_count} jokes."),                         # 人类消息：要求讲笑话
]

# 从消息列表创建ChatPromptTemplate对象
# 这种方式更适合复杂的对话场景，可以设置AI的角色和行为
prompt_template = ChatPromptTemplate.from_messages(messages)

# 填充模板变量
prompt = prompt_template.invoke({
    "topic": "lawyers",    # 话题：律师
    "joke_count": 1        # 笑话数量：1个
})

# 调用语言模型生成回复
result = llm.invoke(prompt)

# 打印结果
print(result)

