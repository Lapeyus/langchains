from langchain.chains import RetrievalQA
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import Ollama

embeddings = OllamaEmbeddings(
    base_url="http://localhost:11434", 
    model="llama2"
)
db = Chroma(
    persist_directory="./chroma_db", 
    embedding_function=embeddings
)
retriever = db.as_retriever()

llm = Ollama(
    model="llama2", 
    callback_manager=CallbackManager(
        [StreamingStdOutCallbackHandler()]
    )
)
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever,
)
qa.combine_documents_chain.verbose = True
qa.combine_documents_chain.llm_chain.verbose = True
qa.combine_documents_chain.llm_chain.llm.verbose = True

while True:
    query = input("> ")
    answer = qa.run(query)
    print(answer)
