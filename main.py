from src.document_builder import DocumentBuilder
from src.chunker import TextSemanticChunker
from src.vector_store import VectorStore
from src.retriever import HybridRetriever
from src.augmenter import Augmenter
from src.generator import Generator


def main():
    document_builder = DocumentBuilder()
    all_documents = document_builder.build_documents()

    print(f"Total documents: {len(all_documents)}")

    chunker = TextSemanticChunker()
    chunks = chunker.chunk(all_documents)

    print(f"Total chunks: {len(chunks)}")

    vector_store_manager = VectorStore()
    vector_store = vector_store_manager.build(chunks)
    vector_store_manager.save(vector_store)

    print("FAISS index created and saved.")

    retriever = HybridRetriever(
        vector_store=vector_store,
        chunks=chunks,
        debug=False
    )

    augmenter = Augmenter()

    generator = Generator()

    query = input("\nAsk your question: ")

    retrieved_documents = retriever.retrieve(query)

    print(
        f"\nRetrieved documents: "
        f"{len(retrieved_documents)}"
    )

    prompt = augmenter.augment(
        query=query,
        documents=retrieved_documents
    )

    answer = generator.generate(prompt)

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(answer)

if __name__ == "__main__":
    main()
