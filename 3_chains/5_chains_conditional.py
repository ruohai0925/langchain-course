from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from langchain_openai import ChatOpenAI

# 加载 .env 文件中的环境变量（如 OpenAI API 密钥等）
# 注意：.env 文件要放在项目根目录，并且要包含 OPENAI_API_KEY，否则后续模型调用会失败
load_dotenv()

# 创建一个 OpenAI 聊天模型实例，指定使用 gpt-3.5-turbo
# 注意：模型名称要和你的 API 权限相符，否则会报错
model = ChatOpenAI(model="gpt-3.5-turbo")

# 定义不同类型反馈的提示模板
positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),  # 系统角色设定为助手
        ("human", "Generate a thank you note for this positive feedback: {feedback}."),  # 针对正面反馈生成感谢信
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Generate a response addressing this negative feedback: {feedback}."),  # 针对负面反馈生成回应
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Generate a request for more details for this neutral feedback: {feedback}."),  # 针对中性反馈请求更多信息
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Generate a message to escalate this feedback to a human agent: {feedback}."),  # 需要升级为人工处理的反馈
    ]
)

# 定义反馈情感分类的提示模板
classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        # ("human", "Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}."),  # 分类反馈情感
        ("human", "Classify the sentiment of this feedback as exactly one of: positive, negative, neutral, escalate. Only output the label."),  # 分类反馈情感
    ]
)

# 定义处理反馈的条件分支
branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        RunnableLambda(lambda x: print("[DEBUG] 走了 positive 分支") or x) | positive_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "negative" in x,
        RunnableLambda(lambda x: print("[DEBUG] 走了 negative 分支") or x) | negative_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "neutral" in x,
        RunnableLambda(lambda x: print("[DEBUG] 走了 neutral 分支") or x) | neutral_feedback_template | model | StrOutputParser()
    ),
    RunnableLambda(lambda x: print("[DEBUG] 走了 escalate 分支") or x) | escalate_feedback_template | model | StrOutputParser()
)

# 创建情感分类链
classification_chain = classification_template | model | StrOutputParser()

# 组合分类和响应生成为一个完整的链
# 先分类，再根据分类结果走不同分支
chain = classification_chain | branches

# 用一个示例评论运行链
# 示例：
# 好评 - "The product is excellent. I really enjoyed using it and found it very helpful."
# 差评 - "The product is terrible. It broke after just one use and the quality is very poor."
# 中评 - "The product is okay. It works as expected but nothing exceptional."
# 默认 - "I'm not sure about the product yet. Can you tell me more about its features and benefits?"

# review = "The product is terrible. It broke after just one use and the quality is very poor."
# review = "The product is excellent. I really enjoyed using it and found it very helpful."
# review = "The product is neutral. It works as expected but nothing exceptional."
review = "I'm not sure about the product yet. Can you tell me more about its features and benefits?"

result = chain.invoke({"feedback": review})

# 输出最终结果
print(result)
