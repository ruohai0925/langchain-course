# 示例来源: https://python.langchain.com/v0.2/docs/integrations/memory/google_firestore/
# 这个示例展示了如何使用Firebase Firestore来持久化存储聊天消息历史

# 导入必要的模块
from dotenv import load_dotenv  # 环境变量管理
from google.cloud import firestore  # Google Firestore数据库客户端
from langchain_google_firestore import FirestoreChatMessageHistory  # LangChain的Firestore消息历史集成
from langchain_openai import ChatOpenAI  # OpenAI聊天模型
import time  # 用于添加延迟和调试

"""
设置步骤说明（复制此示例的步骤）:
1. 创建Firebase账户，网址是https://firebase.google.com/ (注意不要使用学校邮箱，我用的是zdsjtu@gmail.com)
2. 创建新的Firebase项目和FireStore数据库
3. 获取项目ID
4. 在计算机上安装Google Cloud CLI
    - 安装链接: https://cloud.google.com/sdk/docs/install
    - 使用Google账户认证Google Cloud CLI
        - 认证指南: https://cloud.google.com/docs/authentication/provide-credentials-adc#local-dev
    - 将默认项目设置为新创建的Firebase项目
5. 安装依赖: pip install langchain-google-firestore
6. 在Google Cloud Console中启用Firestore API:
    - 启用链接: https://console.cloud.google.com/apis/enableflow?apiid=firestore.googleapis.com&project=crewai-automation
"""

# 加载环境变量（包含API密钥等配置）
load_dotenv()

# ===== Firebase Firestore配置 =====
PROJECT_ID = "langchain-e6855"  # Firebase项目ID，需要替换为你自己的项目ID
SESSION_ID = "user_session_new"  # 会话ID，可以是用户名或唯一标识符
COLLECTION_NAME = "chat_history"  # Firestore集合名称，用于存储聊天历史

try:
    # 初始化Firestore客户端
    print("正在初始化Firestore客户端...")
    print(f"项目ID: {PROJECT_ID}")
    client = firestore.Client(project=PROJECT_ID)
    print("Firestore客户端初始化成功！")
    
    # 测试连接
    print("正在测试Firestore连接...")
    test_doc = client.collection('test').document('connection_test')
    test_doc.set({'timestamp': time.time()})
    print("Firestore连接测试成功！")
    
    # 清理测试文档
    test_doc.delete()
    print("测试文档已清理")

    # 初始化Firestore聊天消息历史管理器
    print("正在初始化Firestore聊天消息历史...")
    chat_history = FirestoreChatMessageHistory(
        session_id=SESSION_ID,      # 会话标识符
        collection=COLLECTION_NAME, # 集合名称
        client=client,              # Firestore客户端实例
    )
    print("聊天历史初始化完成。")
    print("当前聊天历史:", chat_history.messages)

    # 初始化聊天模型
    print("正在初始化聊天模型...")
    model = ChatOpenAI(model="gpt-3.5-turbo")
    print("聊天模型初始化完成！")

    # 开始交互式聊天循环
    print("开始与AI聊天。输入'exit'退出。")

    while True:
        # 获取用户输入
        human_input = input("用户: ")
        
        # 检查退出条件
        if human_input.lower() == "exit":
            break

        try:
            # 将用户消息添加到Firestore历史记录中
            # 这会自动将消息保存到云端数据库
            print("正在保存用户消息到Firestore...")
            chat_history.add_user_message(human_input)
            print("用户消息保存成功！")

            # 使用完整的聊天历史调用AI模型
            # 模型可以访问之前保存的所有对话内容
            print("正在调用AI模型...")
            ai_response = model.invoke(chat_history.messages)
            print("AI模型调用成功！")
            
            # 将AI回复添加到Firestore历史记录中
            # 同样会自动保存到云端
            print("正在保存AI回复到Firestore...")
            chat_history.add_ai_message(ai_response.content)
            print("AI回复保存成功！")

            # 显示AI回复
            print(f"AI: {ai_response.content}")
            
        except Exception as e:
            print(f"对话过程中出现错误: {e}")
            print("尝试继续对话...")

except Exception as e:
    print(f"初始化过程中出现错误: {e}")
    print("\n可能的解决方案:")
    print("1. 检查Google Cloud CLI是否正确安装和认证")
    print("2. 确认项目ID是否正确")
    print("3. 确认Firestore API是否已启用")
    print("4. 检查网络连接")
    print("5. 尝试运行: gcloud auth application-default login")
