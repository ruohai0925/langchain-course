# 代码整体流程说明
# 1. 加载环境变量，初始化 OpenAI 聊天模型。
# 2. 定义了三个提示模板：电影摘要、剧情分析、角色分析。
# 3. 分别用函数封装剧情和角色分析的提示模板。
# 4. 定义 combine_verdicts 函数用于合并剧情和角色分析的结果。
# 5. 用 LCEL 构建了一个多分支并行链：先生成摘要，然后并行分析剧情和角色，最后合并输出。
# 6. 运行链，输入电影名，输出完整的影评分析。

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI

# 加载 .env 文件中的环境变量（如 OpenAI API 密钥等）
# 注意：.env 文件要放在项目根目录，并且要包含 OPENAI_API_KEY，否则后续模型调用会失败
load_dotenv()

# 创建一个 OpenAI 聊天模型实例，指定使用 gpt-3.5-turbo
# 注意：模型名称要和你的 API 权限相符，否则会报错
model = ChatOpenAI(model="gpt-3.5-turbo")

# 定义电影摘要的提示模板
# ChatPromptTemplate.from_messages 用于构建多轮对话的提示
summary_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a movie critic."),  # 系统角色设定为影评人
        ("human", "Provide a brief summary of the movie {movie_name}."),  # 用户输入，要求简要总结电影
    ]
)

# 定义剧情分析步骤
def analyze_plot(plot):
    # 这里每次调用都会新建一个剧情分析的提示模板
    plot_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a movie critic."),  # 系统角色设定
            ("human", "Analyze the plot: {plot}. What are its strengths and weaknesses?"),  # 用户输入，分析剧情优缺点
        ]
    )
    # format_prompt 返回的是格式化后的 PromptValue 对象
    return plot_template.format_prompt(plot=plot)  # 返回格式化后的提示

# 定义角色分析步骤
def analyze_characters(characters):
    # 这里每次调用都会新建一个角色分析的提示模板
    character_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a movie critic."),  # 系统角色设定
            ("human", "Analyze the characters: {characters}. What are their strengths and weaknesses?"),  # 用户输入，分析角色优缺点
        ]
    )
    return character_template.format_prompt(characters=characters)  # 返回格式化后的提示

# 合并剧情和角色分析，输出最终评价
def combine_verdicts(plot_analysis, character_analysis):
    # 注意：这里直接拼接字符串，实际生产环境可考虑更复杂的格式化
    return f"Plot Analysis:\n{plot_analysis}\n\nCharacter Analysis:\n{character_analysis}"

# 用 LCEL（LangChain Expression Language）简化分支链
# 这里用 RunnableLambda 包装分析函数，保证链式调用兼容性
plot_branch_chain = (
    RunnableLambda(lambda x: analyze_plot(x))  # 首先用 lambda 包装剧情分析函数
    | model                                   # 传递给模型，生成分析内容
    | StrOutputParser()                       # 输出解析为字符串
)

character_branch_chain = (
    RunnableLambda(lambda x: analyze_characters(x))  # 首先用 lambda 包装角色分析函数
    | model                                          # 传递给模型
    | StrOutputParser()                              # 输出解析为字符串
)

# 使用 LCEL 创建组合链
chain = (
    summary_template
    | model
    | StrOutputParser()
    # RunnableParallel 用于并行执行多个分支
    # branches 参数是一个字典，key 是分支名，value 是对应的链
    | RunnableParallel(branches={"plot": plot_branch_chain, "characters": character_branch_chain})
    # 上一步输出会被并行传递到两个分支：剧情分析和角色分析
    # 注意：这里 x["branches"]["plot"] 取的是并行分支的结果
    | RunnableLambda(lambda x: combine_verdicts(x["branches"]["plot"], x["branches"]["characters"]))
    # 合并两个分支的结果，输出最终评价
)

# 运行链，输入电影名为 Inception（盗梦空间）
# 注意：invoke 方法的参数要和 summary_template 的变量名一致
result = chain.invoke({"movie_name": "Inception"})

print(result)  # 打印最终结果