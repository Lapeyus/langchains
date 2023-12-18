from langchain.vectorstores import Chroma

def save_to_chroma(docs,embedder):
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedder,
        ids=None,
        collection_name="langchain",
        persist_directory="./chroma_db"
    )

    vectorstore.persist()
