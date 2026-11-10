from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
import pickle


print("正在加载文本")
loader = DirectoryLoader(
    "data", 
    glob="*.md", 
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
    )
docs = loader.load()

print("正在切分文本...")

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)
splits=text_splitter.split_documents(docs)


with open("splits.pkl", "wb") as f:
    pickle.dump(splits, f)
print("已保存切分后的文本块 (splits.pkl)")

print("正在向量化并存入数据库（这步需要下载嵌入模型，需等待）...")

embedding_model=HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

vectorstore=Chroma.from_documents(
    documents=splits,
    embedding=embedding_model,
    persist_directory="./chroma_db",
)

print("数据库构建完成")

