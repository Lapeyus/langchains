from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from chroma import save_to_chroma

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=0
)

all_splits = text_splitter.split_documents(data)

oembed = OllamaEmbeddings(
    base_url="http://localhost:11434", 
    model="llama2"
)
save_to_chroma(all_splits,oembed)
