from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

DATA_PATH = "courses/"
VECTOR_STORE_PATH = "vector_store"
def create_vector_store():
    """
    Creates a FAISS vector store from documents in the DATA_PATH.
    This function needs to be run only once, or when documents are updated.
    """
    print("Loading documents...")
    loader = DirectoryLoader(DATA_PATH, glob="*.txt")
    documents = loader.load()

    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    print("Creating embeddings and FAISS vector store...")
    # Using a popular open-source embedding model
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    
    # Create the vector store and save it locally
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"Vector store created and saved at {VECTOR_STORE_PATH}")


if __name__=="__main__":
    create_vector_store()

