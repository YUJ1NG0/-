以下是一个完整的、带详细注释的 **基于 LangChain + DeepSeek + RAG 技术构建的论坛疑难解答 Agent** 脚本，文件名建议为 `forum_rag_agent.py`。

------

## ✅ `forum_rag_agent.py`

```python
# forum_rag_agent.py

# ✅ 依赖导入
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain.llms import HuggingFacePipeline
import os

# ✅ 第一步：加载论坛规则与指引文档（作为知识源）
def load_documents():
    loader1 = TextLoader("forum_rules.txt", encoding="utf-8")
    loader2 = TextLoader("forum_guide.txt", encoding="utf-8")

    docs1 = loader1.load()
    docs2 = loader2.load()
    documents = docs1 + docs2

    print(f"✅ 共加载文档 {len(documents)} 篇")
    return documents

# ✅ 第二步：将文档分割成小块用于向量化处理
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"✅ 文档分割为 {len(chunks)} 个片段用于向量存储")
    return chunks

# ✅ 第三步：构建向量知识库（RAG 的 Retrieval 部分）
def build_vector_store(chunks):
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vectordb = Chroma.from_documents(chunks, embedding=embedding, persist_directory="./forum_knowledge")
    vectordb.persist()
    print("✅ 向量知识库构建完成并已持久化")
    return vectordb

# ✅ 第四步：加载 DeepSeek LLM（RAG 的 Generation 部分）
def load_llm():
    print("✅ 正在加载 DeepSeek 模型...")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-chat")
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/deepseek-llm-7b-chat",
        torch_dtype="auto",
        device_map="auto"
    )

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, do_sample=True)
    llm = HuggingFacePipeline(pipeline=pipe)
    print("✅ 模型加载完成")
    return llm

# ✅ 第五步：构建 RAG 问答链（检索 + 生成）
def build_qa_chain(llm, vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})  # Top-3 相关段落
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    print("✅ RAG 问答链构建完成")
    return qa

# ✅ 主函数：运行 Agent，进行问答交互
def run_agent():
    # 检查是否已有向量数据库
    if os.path.exists("./forum_knowledge/index"):
        print("✅ 加载已存在的向量数据库")
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        vectordb = Chroma(persist_directory="./forum_knowledge", embedding_function=embedding)
    else:
        docs = load_documents()
        chunks = split_documents(docs)
        vectordb = build_vector_store(chunks)

    llm = load_llm()
    qa_chain = build_qa_chain(llm, vectordb)

    print("\n🎯 论坛疑难解答 Agent 启动完成，输入你的问题（输入 exit 退出）\n")
    while True:
        question = input("🧾 你的提问：")
        if question.strip().lower() == "exit":
            print("👋 再见！")
            break

        result = qa_chain({"query": question})
        print("\n💡 Agent 回答：")
        print(result["result"])
        print("-" * 60)

# ✅ 脚本入口
if __name__ == "__main__":
    run_agent()
```

------

## 📝 使用说明

1. **准备文档**：
   - 在脚本同级目录下准备 `forum_rules.txt` 和 `forum_guide.txt` 两个文件；
   - 内容建议包括论坛发帖规则、评论规范、敏感词处理说明、常见问题等。
2. **运行脚本**：

```bash
python forum_rag_agent.py
```

首次运行会构建知识库，之后再运行会直接加载缓存的向量。

------

## ✅ 项目描述模板（可放在简历或报告中）

> 构建了一个基于 RAG（检索增强生成）的智能问答 Agent，用于回答用户在论坛使用过程中遇到的问题。项目采用 LangChain 框架结合 DeepSeek LLM，使用论坛规则与使用指引文档构建本地知识库，通过向量检索获取相关内容并结合大模型生成答案，确保输出准确、可控。该系统具备快速响应、离线运行、可扩展等优点。

------

是否需要我继续帮你：

- 封装为 REST API（Flask/FastAPI）供前端调用？
- 或者改造成你现有微服务架构中的一个服务模块？

欢迎继续提问。