from langchain_community.document_loaders import PDFPlumberLoader
from src.config import FINANCIAL_REPORT_PATH

class DataLoader:

    def __init__(self):
        self.file_path = FINANCIAL_REPORT_PATH

    def load(self):
        if not self.file_path.exists():
            raise FileNotFoundError('File not found!')

        if self.file_path.suffix.lower() != '.pdf':
            raise ValueError('File extension have different')

        loader = PDFPlumberLoader(self.file_path)
        documents = loader.load()

        if not documents:
            print('File is empty')

        return documents
        