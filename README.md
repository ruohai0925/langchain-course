# LangChain 课程项目

> 基于 LangChain 框架的 AI 应用开发学习项目

## 📖 项目简介

这是一个全面的 LangChain 学习项目，涵盖了从基础的聊天模型到高级的 RAG 系统和 Agent 开发的完整学习路径。项目采用渐进式学习方式，每个模块都包含详细的注释和实际应用示例。

## 🏗️ 项目结构

```
langchain-course/
├── 1_chat_models/           # 聊天模型基础
├── 2_prompt_templates/      # 提示模板
├── 3_chains/               # 链式处理
├── 4_RAGs/                 # 检索增强生成
├── 5_agents/               # Agent 系统
├── documents/              # 示例文档
└── db/                     # 向量数据库
```

## 📚 学习模块

### 1. 聊天模型 (1_chat_models/)

**学习目标**: 掌握 LangChain 中聊天模型的基本使用

**文件列表**:
- `1_chat_models_starter.py` - 基础聊天模型使用
- `2_chat_models_conversation.py` - 多轮对话实现
- `3_chat_models-alternative_models.py` - 替代模型使用
- `4_chat_model_conversation_with_user.py` - 用户交互对话
- `5_chat_model_save_message_history_firebase.py` - 消息历史保存
- `token_calculation.py` - Token 计算

**核心概念**:
- ChatOpenAI 模型配置
- 多轮对话管理
- 消息历史存储
- Token 使用优化

### 2. 提示模板 (2_prompt_templates/)

**学习目标**: 学习如何创建和管理提示模板

**文件列表**:
- `1_prompt_templates_starter.py` - 提示模板基础

**核心概念**:
- ChatPromptTemplate 使用
- 模板变量管理
- 系统消息和用户消息

### 3. 链式处理 (3_chains/)

**学习目标**: 掌握 LangChain 的链式处理能力

**文件列表**:
- `1_chains_basics.py` - 基础链式处理
- `2_chains_inner_workings.py` - 链的内部工作原理
- `3_chains_sequential.py` - 顺序链
- `4_chains_parallel.py` - 并行链
- `5_chains_conditional.py` - 条件链

**核心概念**:
- LCEL (LangChain Expression Language)
- 顺序链和并行链
- 条件分支处理
- 链的组合和优化

### 4. 检索增强生成 (4_RAGs/)

**学习目标**: 构建完整的 RAG 系统

**文件列表**:
- `1a_basic_part_1.py` - 基础 RAG 构建
- `1b_basic_part_2.py` - 基础 RAG 检索
- `2a_rag_basics_metadata.py` - 带元数据的 RAG 构建
- `2b_rag_basics_metadata.py` - 带元数据的 RAG 检索
- `3_rag_one_off_question.py` - 完整 RAG 问答系统

**示例文档**:
- `lord_of_the_rings.txt` - 指环王
- `Dracula.txt` - 德古拉
- `Frankenstein.txt` - 弗兰肯斯坦
- `Alice's Adventures in Wonderland.txt` - 爱丽丝梦游仙境
- `about_me.txt` - 个人信息

**核心概念**:
- 文档加载和分割
- 向量化和存储
- 相似度检索
- 检索结果生成

### 5. Agent 系统 (5_agents/)

**学习目标**: 开发智能 Agent 系统

**文件列表**:
- `1_basics.py` - Agent 基础实现

**核心概念**:
- 工具定义和使用
- ReAct 框架
- Agent 执行器
- 工具调用管理

## 🚀 快速开始

### 环境要求

- Python 3.8+
- OpenAI API 密钥

### 安装依赖

```bash
pip install langchain langchain-openai langchain-community langchain-chroma chromadb python-dotenv
```

### 环境配置

1. 创建 `.env` 文件
2. 添加你的 OpenAI API 密钥：

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 运行示例

```bash
# 运行基础聊天模型
python 1_chat_models/1_chat_models_starter.py

# 运行链式处理示例
python 3_chains/1_chains_basics.py

# 运行 RAG 系统
python 4_RAGs/1a_basic_part_1.py
python 4_RAGs/1b_basic_part_2.py

# 运行 Agent 示例
python 5_agents/1_basics.py
```

## 📝 学习路径建议

### 初学者路径
1. **聊天模型** → 理解基础 AI 交互
2. **提示模板** → 学习提示工程
3. **链式处理** → 掌握组件组合
4. **RAG 系统** → 构建知识问答系统
5. **Agent 系统** → 开发智能应用

### 进阶路径
- 深入研究每个模块的高级特性
- 尝试组合不同模块构建复杂应用
- 优化性能和成本
- 添加错误处理和监控

## 🔧 技术栈

- **LangChain**: AI 应用开发框架
- **OpenAI**: GPT 模型 API
- **Chroma**: 向量数据库
- **Python-dotenv**: 环境变量管理

## 📖 学习要点

### 关键概念
- **LCEL**: LangChain 表达式语言，用于组合组件
- **RAG**: 检索增强生成，结合检索和生成能力
- **Agent**: 能够使用工具的智能系统
- **Vector Store**: 向量数据库，用于存储文档嵌入

### 最佳实践
- 合理设置 chunk_size 和 chunk_overlap
- 使用适当的相似度阈值
- 优化 Token 使用
- 添加错误处理机制

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进这个项目！

## 📄 许可证

本项目基于 MIT 许可证开源。

## 🙏 致谢

- **原始作者**: Harish Neel
- **教程视频**: ZDSJTU
- **GitHub**: https://github.com/ruohai0925/langchain-course

---

**注意**: 请确保在使用前正确配置 OpenAI API 密钥，并注意 API 使用成本。