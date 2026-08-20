class Augmenter:

    def __init__(self):
        self.system_instruction = (
            "Answer the user's question using only the provided context. "
            "If the answer is not present in the context, say that the "
            "information is not available in the provided context."
        )

    def augment(self, query, documents):

        context_parts = []

        for index, document in enumerate(documents, start=1):

            context_parts.append(
                f"[Context {index}]\n"
                f"{document.page_content}"
            )

        context = "\n\n".join(context_parts)

        prompt = (
            f"{self.system_instruction}\n\n"
            f"Context:\n"
            f"{context}\n\n"
            f"Question:\n"
            f"{query}\n\n"
            f"Answer:"
        )

        return prompt