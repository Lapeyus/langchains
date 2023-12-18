from langchain.document_loaders import GoogleDriveLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from chroma import save_to_chroma

import os
from dotenv import load_dotenv

load_dotenv()

FOLDER_ID = os.getenv("FOLDER_ID")

loader = GoogleDriveLoader(
    folder_id=FOLDER_ID,
    recursive=False
)

docs = loader.load()
metadata = {
    "source": "gdrive", 
    "rag": "yes"
}
updated_docs = []
for doc in docs:
    combined_metadata = {**doc.metadata, **metadata}
    updated_doc = Document(
        page_content=doc.page_content, 
        metadata=combined_metadata
    )
    updated_docs.append(updated_doc)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000, 
    chunk_overlap=0, 
    separators=[" ", ",", "\n"]
)

all_splits = text_splitter.split_documents(docs)
print(all_splits)

oembed = OllamaEmbeddings(
    base_url="http://localhost:11434", 
    model="llama2"
)

save_to_chroma(all_splits,oembed)

