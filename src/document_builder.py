from langchain_core.documents import Document

from src.text_extractor import TextExtractor
from src.table_extractor import TableExtractor


class DocumentBuilder:

    def __init__(self):
        self.text_extractor = TextExtractor()
        self.table_extractor = TableExtractor()

    def build_documents(self):

        documents = []

        text_documents = self.text_extractor.documents

        for document in text_documents:

            documents.append(
                Document(
                    page_content=document.page_content,
                    metadata={
                        **document.metadata,
                        "type": "text"
                    }
                )
            )

        tables = self.table_extractor.extract_tables()

        for table in tables:

            table_documents = self._create_table_documents(table)

            documents.extend(table_documents)

        return documents

    def _create_table_documents(self, table):

        documents = []

        page = table["page"]
        table_number = table["table_number"]

        for row in table["rows"]:

            metric = row["metric"]
            values = row["values"]

            lines = [
                "Financial Table",
                f"Metric: {metric}"
            ]

            for period, value in values.items():

                lines.append(f"{period}: {value}")

            table_text = "\n".join(lines)

            documents.append(
                Document(
                    page_content=table_text,
                    metadata={
                        "type": "table",
                        "page": page,
                        "table_number": table_number,
                        "metric": metric
                    }
                )
            )

        return documents