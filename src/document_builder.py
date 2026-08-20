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
            documents.extend(
                self._create_table_comparison_documents(table)
            )

        return documents

    def _create_table_comparison_documents(self, table):
        """Keep adjacent table metrics together for comparison queries."""
        documents = []
        rows = table["rows"]

        for start in range(0, len(rows), 3):
            group = rows[start:start + 3]
            lines = [
                "Financial comparison table."
            ]

            for row in group:
                metric = row["metric"]

                # Queries commonly use the singular form, while the report
                # labels this metric as "Revenues" or "Total revenues".
                if metric.lower() in {"revenues", "total revenues"}:
                    metric = "Revenue"

                values = "; ".join(
                    f"{period}: {value}"
                    for period, value in row["values"].items()
                )
                lines.append(f"{metric}: {values}")

            documents.append(
                Document(
                    page_content="\n".join(lines),
                    metadata={
                        "type": "table",
                        "page": table["page"],
                        "table_number": table["table_number"],
                        "table_scope": "comparison_group",
                        "row_start": start
                    }
                )
            )

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
