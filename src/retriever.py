from collections import defaultdict

from langchain_community.retrievers import BM25Retriever

from src.config import TOP_K


class HybridRetriever:

    def __init__(self, vector_store, chunks, rrf_k=60):

        self.vector_store = vector_store
        self.chunks = chunks
        self.rrf_k = rrf_k

        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks)
        self.bm25_retriever.k = TOP_K

    def retrieve(self, query):
        vector_results = self.vector_store.similarity_search(
            query,
            k=TOP_K)
        bm25_results = self.bm25_retriever.invoke(query)
        return self._rrf_fusion(
            vector_results,
            bm25_results)

    def _rrf_fusion(self, vector_results, bm25_results):
        scores = defaultdict(float)
        documents = {}
        for rank, document in enumerate(vector_results, start=1):
            document_id = self._get_document_id(document)
            scores[document_id] += (
                1 / (self.rrf_k + rank)
            )
            documents[document_id] = document
        for rank, document in enumerate(bm25_results, start=1):
            document_id = self._get_document_id(document)

            scores[document_id] += (
                1 / (self.rrf_k + rank)
            )

            documents[document_id] = document
        ranked_documents = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True
        )
        results = []
        for document_id, score in ranked_documents[:TOP_K]:
            document = documents[document_id]
            document.metadata["rrf_score"] = score
            results.append(document)
        return results

    def _get_document_id(self, document):
        return (
            document.page_content,
            tuple(sorted(document.metadata.items()))
        )