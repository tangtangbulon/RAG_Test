from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)

# 测试一个“有答案”和一个“没答案”的问题
query_yes = "不支持“七天无理由退货有哪些？"
query_no = "文档里关于火星登陆的条款是什么？"

for query in [query_yes, query_no]:
    print(f"\n🔍 用户提问：{query}")
    # similarity_search_with_relevance_scores 会返回相似度分数（0-1之间）
    results = vectorstore.similarity_search_with_relevance_scores(query, k=3)
    
    for doc, score in results:
        print(f"  [相似度: {score:.3f}] 页码: {doc.metadata.get('page', '未知')} | 内容: {doc.page_content[:50]}...")