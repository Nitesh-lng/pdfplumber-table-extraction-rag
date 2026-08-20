from langchain_community.vectorstores import FAISS
from src.config import INDEX_PATH
from langchain_huggingface import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL
from langchain_community.vectorstores.utils import DistanceStrategy

class VectorStore:

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={
                "normalize_embeddings": True
            },
        )

    def build(self,chunks):
        return FAISS.from_documents(
            chunks,
            self.embeddings,
            distance_strategy=DistanceStrategy.COSINE
        )

    def save(self,vector_store):
        vector_store.save_local(INDEX_PATH)

    def load(self):
        return FAISS.load_local(
            INDEX_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
