from langchain_experimental.text_splitter import SemanticChunker
from src.config import BREAKPOINT_THRESHOLD_TYPE
from src.config import EMBEDDING_MODEL
from langchain_huggingface import HuggingFaceEmbeddings

class TextSemanticChunker:

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.threshold = BREAKPOINT_THRESHOLD_TYPE
        self.splitter = SemanticChunker(
            embeddings = self.embeddings,
            breakpoint_threshold_type = self.threshold
        )

    def chunk(self,documents):
        return self.splitter.split_documents(documents)
