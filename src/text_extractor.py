from src.loader import DataLoader


class TextExtractor:

    def __init__(self):
        loader = DataLoader()
        self.documents = loader.load()

    def inspect_documents(self):
        for page_number, document in enumerate(self.documents, start=1):

            print(f"\n{'=' * 50}")
            print(f"PAGE {page_number}")
            print(f"{'=' * 50}")

            print(document.page_content)