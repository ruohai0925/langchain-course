# 这个代码展示了LangChain链的内部工作原理，通过手动构建可运行组件来理解LCEL（LangChain Expression Language）的底层实现机制。

# 导入必要的模块
from dotenv import load_dotenv  # 环境变量管理
from langchain.prompts import ChatPromptTemplate  # 聊天提示模板
from langchain.schema.runnable import RunnableLambda, RunnableSequence  # 可运行组件和序列
from langchain_openai import ChatOpenAI  # OpenAI聊天模型

# 从.env文件加载环境变量（包含API密钥等配置）
load_dotenv()

# 创建ChatOpenAI模型实例
model = ChatOpenAI(model="gpt-3.5-turbo")

# 定义提示模板
# 使用from_messages方法创建包含系统消息和人类消息的模板
prompt_template = ChatPromptTemplate.from_messages(
     [
        ("system", "You love facts and you tell facts about {animal}"),  # 系统消息：设置AI为热爱事实的专家
        ("human", "Tell me {count} facts."),                             # 人类消息：要求提供指定数量的事实
    ]
)

# 创建单独的可运行组件（链中的步骤）
# 这些组件展示了链的内部工作原理

# 步骤1: 格式化提示
# RunnableLambda将普通函数包装为可运行组件
# format_prompt方法将字典参数转换为格式化的提示对象
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))

# 步骤2: 调用模型
# 将格式化的提示转换为消息列表，然后调用模型
# to_messages()方法将提示对象转换为消息列表
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))

# 步骤3: 解析输出
# 从AIMessage对象中提取content字段，获取纯文本内容
parse_output = RunnableLambda(lambda x: x.content)

# 创建RunnableSequence（等同于LCEL链）
# RunnableSequence是链的底层实现，明确指定了执行顺序
# first: 第一个组件（格式化提示）
# middle: 中间组件列表（调用模型）
# last: 最后一个组件（解析输出）
chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

# 运行链
# invoke方法传入包含模板变量的字典
response = chain.invoke({"animal": "cat", "count": 2})

# 输出结果
print(response)


# 字典 → PromptValue → 消息列表 → AIMessage → 字符串