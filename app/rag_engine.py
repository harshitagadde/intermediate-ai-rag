import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DeterministicFakeEmbedding
from langchain_community.vectorstores import FAISS
from transformers import pipeline

DB_FAISS_PATH = 'vectorstore/db_faiss'

class RAGEngine:
    def __init__(self):
        self.embeddings = DeterministicFakeEmbedding(size=384)
        self.vector_store = None
        # Load a lightweight local text generation pipeline
        self.generator = pipeline(
            "text-generation", 
            model="google/flan-t5-small", 
            max_new_tokens=150
        )

    def ingest_pdf(self, pdf_path: str):
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        text_chunks = text_splitter.split_documents(documents)

        self.vector_store = FAISS.from_documents(text_chunks, self.embeddings)
        self.vector_store.save_local(DB_FAISS_PATH)
        return len(text_chunks)

    def load_vector_store(self):
        if os.path.exists(DB_FAISS_PATH):
            self.vector_store = FAISS.load_local(
                DB_FAISS_PATH, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )

    def query(self, user_query: str, k: int = 3):
        if not self.vector_store:
            self.load_vector_store()
            if not self.vector_store:
                raise ValueError("Vector DB not initialized. Ingest a document first.")

        docs = self.vector_store.similarity_search(user_query, k=k)
        retrieved_context = "\n".join([doc.page_content for doc in docs])
        
        # Ground question on retrieved context
        prompt = f"Answer the question based only on the context below.\nContext: {retrieved_context}\nQuestion: {user_query}\nAnswer:"
        generated_text = self.generator(prompt)[0]['generated_text']
        
        return {
            "answer": generated_text,
            "retrieved_contexts": [doc.page_content for doc in docs]
        }

rag_service = RAGEngine()
