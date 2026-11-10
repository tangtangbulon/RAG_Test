import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from FlagEmbedding import FlagReranker
import pickle


load_dotenv()

print("⏳ 正在加载向量库和 BM25 索引...")
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)

with open("splits.pkl", "rb") as f:
    splits = pickle.load(f)

# 构建向量检索器 (召回 Top 10)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# 构建 BM25 关键词检索器 (召回 Top 10)
bm25_retriever = BM25Retriever.from_documents(splits)
bm25_retriever.k = 10

# 构建混合检索器：将向量检索和 BM25 的结果融合 (各自权重 0.5)
hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)
print("⏳ 正在加载重排模型...")
reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True, normalize=True)


llm = ChatOpenAI(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=os.getenv("BASE_URL"))
prompt_template ="""你是一个严谨的企业政策问答助手。请**仅根据**下面提供的上下文来回答问题。
规则：
1. 如果上下文里有答案，请准确回答，并在回答末尾用括号标出出处页码，如：doc.metadata.get('source')。
2. 如果上下文里没有答案，请务必直接回答：“根据提供的文档，我无法找到相关信息。”
3. 绝对不要利用你自己的常识去编造答案。

上下文：
{context}

用户问题：{question}
你的回答："""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def ask_system(query):
    # 【第一步：混合检索】召回 10+10，去重后大概剩下 10-15 个候选块
    print("🔍 正在执行混合检索...")
    candidate_docs = hybrid_retriever.invoke(query)

    if not candidate_docs:
        return "根据提供的文档，我无法找到相关信息。"
        
    # 【第二步：重排】将问题与候选块逐对输入重排模型，计算精准相关性得分
    print(f"⚖️ 正在对 {len(candidate_docs)} 个候选片段进行重排...")

    pairs = [(query, doc.page_content) for doc in candidate_docs]

    # 输出结果为浮点数列表，分数越高越相关
    scores = reranker.compute_score(pairs)

     # 将得分与文档绑定，并按得分从高到低排序
    docs_scores_pairs = list(zip(candidate_docs, scores))
    docs_scores_pairs.sort(key=lambda x: x[1], reverse=True)

    # 提取重排后的 Top 3 文档
    top_k=3
    final_docs = [pair[0] for pair in docs_scores_pairs[:top_k]]
    top_score = docs_scores_pairs[0][1]
    
    print(f"📊 最高相关性得分: {top_score:.4f}")

    # 【第三步：阈值过滤】
    # normalize=True 后得分经 sigmoid 压到 0–1，不相关的问题通常接近 0
    REJECT_THRESHOLD = 0.5
    if top_score < REJECT_THRESHOLD:
        return "根据提供的文档，我无法找到相关信息。"

    # 【第四步：格式化答案】将 Top 3 文档按页码拼接成上下文
    context = ""
    for doc in final_docs:
        page_num = doc.metadata.get('page', 0) + 1
        source = doc.metadata.get('source', '未知文档')
        context += f"\n[出处: {source} 第{page_num}页]: {doc.page_content}\n"


    formatted_prompt = prompt.format(context=context, question=query)
    response = llm.invoke(formatted_prompt)
    return response.content

if __name__ == "__main__":
    test_questions = [
        "不支持“七天无理由退货有哪些？", # 替换成你准备的真实问题
        "文档里关于火星登陆的条款是什么？", # 替换成你准备的瞎编问题
    ]
    for q in test_questions:
        print(f"\n🔍 用户提问：{q}")
        print(f"🤖 回答：{ask_system(q)}")