# 导入必要的模块
from dotenv import load_dotenv  # 环境变量管理
from langchain.prompts import ChatPromptTemplate  # 聊天提示模板
from langchain.schema.output_parser import StrOutputParser  # 字符串输出解析器
from langchain.schema.runnable import RunnableLambda  # 可运行Lambda函数包装器
from langchain_openai import ChatOpenAI  # OpenAI聊天模型

# 从.env文件加载环境变量（包含API密钥等配置）
load_dotenv()

# 创建ChatOpenAI模型实例
# 使用gpt-4o模型，这是OpenAI最新的多模态模型
model = ChatOpenAI(model="gpt-4o")

# 定义第一个提示模板：动物事实生成
# 这个模板用于生成关于特定动物的事实
animal_facts_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You like telling facts and you tell facts about {animal}."),  # 系统消息：设置AI为动物事实专家
        ("human", "Tell me {count} facts."),                                      # 人类消息：要求提供指定数量的事实
    ]
)

# 定义第二个提示模板：翻译功能
# 这个模板用于将文本翻译成指定语言
translation_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a translator and convert the provided text into {language}."),  # 系统消息：设置AI为翻译专家
        ("human", "Translate the following text to {language}: {text}"),                   # 人类消息：要求翻译指定文本
    ]
)

# 定义额外的处理步骤，使用RunnableLambda包装自定义函数

# 步骤1: 计算单词数量
# 这个函数接收文本输入，计算单词数量并在文本前添加计数信息
# 输入: 字符串文本
# 输出: 包含单词计数的格式化字符串
# 函数作用: 对文本进行预处理，添加元数据信息
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")

# 步骤2: 准备翻译参数
# 这个函数将前一步的输出转换为翻译模板所需的参数格式
# 输入: 字符串文本（来自前一步的输出）
# 输出: 字典，包含text和language两个键
# 函数作用: 将字符串输出重新格式化为翻译模板需要的参数结构
prepare_for_translation = RunnableLambda(lambda output: {"text": output, "language": "Chinese"})

# 使用LangChain表达式语言（LCEL）创建组合链
# 管道操作符 | 连接多个组件，形成复杂的工作流
# 执行顺序：animal_facts_template → model → StrOutputParser → prepare_for_translation → translation_template → model → StrOutputParser
chain = animal_facts_template | model | StrOutputParser() | prepare_for_translation | translation_template | model | StrOutputParser() 

# 运行链
# invoke方法传入初始参数，整个链会自动处理数据流
result = chain.invoke({"animal": "cat", "count": 2})

# 输出最终结果
print(result)
