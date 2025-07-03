# 导入必要的模块
from dotenv import load_dotenv  # 环境变量管理
from langchain.prompts import ChatPromptTemplate  # 聊天提示模板
from langchain.schema.output_parser import StrOutputParser  # 字符串输出解析器
from langchain_openai import ChatOpenAI  # OpenAI聊天模型

# 从.env文件加载环境变量（包含API密钥等配置）
load_dotenv()

# 创建ChatOpenAI模型实例
# 使用gpt-3.5-turbo模型，这是OpenAI最新的多模态模型
model = ChatOpenAI(model="gpt-3.5-turbo")

# 定义提示模板（不需要单独的Runnable链）
# 使用from_messages方法创建包含系统消息和人类消息的模板
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a facts expert who knows facts about {animal}."),  # 系统消息：设置AI为特定动物的知识专家
        ("human", "Tell me {fact_count} facts."),                              # 人类消息：要求提供指定数量的事实
    ]
)

# 使用LangChain表达式语言（LCEL）创建组合链
# 管道操作符 | 用于连接不同的组件
# 链的执行顺序：prompt_template -> model -> StrOutputParser
chain = prompt_template | model | StrOutputParser()
# chain = prompt_template | model
# 如果不使用StrOutputParser，输出将是AIMessage对象而不是纯字符串

# 运行链
# invoke方法传入包含模板变量的字典
result = chain.invoke({"animal": "elephant", "fact_count": 2})

# 输出结果
print(result)
