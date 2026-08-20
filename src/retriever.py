from collections import defaultdict

from langchain_community.retrievers import BM25Retriever

from src.config import TOP_K


class HybridRetriever:

    def __init__(self, vector_store, chunks, rrf_k=60, debug=False):

        self.vector_store = vector_store
        self.chunks = chunks
        self.rrf_k = rrf_k
        self.debug = debug

        self.bm25_retriever = BM25Retriever.from_documents(
            self.chunks)
        self.bm25_retriever.k = TOP_K

    def retrieve(self, query):
        vector_results = self.vector_store.similarity_search(
            query,
            k=TOP_K)
        bm25_results = self.bm25_retriever.invoke(query)
        if self.debug:
            self._log_ranked_documents(
                "VECTOR SEARCH RESULTS", vector_results
            )
            self._log_ranked_documents(
                "BM25 RESULTS", bm25_results
            )

        results = self._rrf_fusion(
            vector_results,
            bm25_results)
        if self.debug:
            self._log_ranked_documents("RRF RESULTS", results)
        return results

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

    def _log_ranked_documents(self, title, documents):
        print(f"\n{title}")
        for rank, document in enumerate(documents, start=1):
            print(f"Rank: {rank}")
            if "rrf_score" in document.metadata:
                print(f"RRF score: {document.metadata['rrf_score']}")
            print(f"Page content:\n{document.page_content}")
            print(f"Metadata: {document.metadata}")

    def _get_document_id(self, document):
        return (
            document.page_content,
            tuple(sorted(document.metadata.items()))
        )
