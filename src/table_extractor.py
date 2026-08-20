import re
import pdfplumber

from src.config import FINANCIAL_REPORT_PATH


class TableExtractor:

    PERIOD_PATTERN = re.compile(
        r"^(?:Q[1-4][’']\d{2}|20\d{2})$"
    )

    def __init__(self):
        self.file_path = FINANCIAL_REPORT_PATH
    def clean_row(self, row):

        cleaned_row = []

        for cell in row:

            if cell is None:
                continue

            cell = str(cell).strip()

            if cell:
                cleaned_row.append(cell)

        return cleaned_row

    def normalize_row(self, row):

        normalized = []

        i = 0

        while i < len(row):

            value = row[i].strip()

            # Case:
            # value + ')' or ')%'
            if i + 1 < len(row):

                next_value = row[i + 1].strip()

                if next_value == ")":
                    value = value + ")"
                    i += 2

                    if i < len(row) and row[i].strip() == "%":
                        value = value + "%"
                        i += 1

                    normalized.append(value)
                    continue

                if next_value == ")%":
                    value = value + ")%"
                    i += 2

                    normalized.append(value)
                    continue

            # Case:
            # value + '%'
            if i + 1 < len(row):

                next_value = row[i + 1].strip()

                if next_value == "%":
                    normalized.append(value + "%")
                    i += 2
                    continue

            normalized.append(value)

            i += 1

        return normalized

    def is_period_row(self, row):

        if len(row) < 2:
            return False

        return all(
            self.PERIOD_PATTERN.match(value)
            for value in row
        )

    def is_section_row(self, text):

        text = text.lower().strip()

        return (
            text.startswith("adjustment to ")
            or text.startswith("adjustments to ")
        )

    def structure_table(self, table, page_number, table_number):

        rows = []

        for row in table:

            cleaned_row = self.clean_row(row)

            if not cleaned_row:
                continue

            normalized_row = self.normalize_row(cleaned_row)

            if normalized_row:
                rows.append(normalized_row)

        if not rows:
            return None
        header_index = None

        for index, row in enumerate(rows):

            if self.is_period_row(row):
                header_index = index
                break
        if header_index is None:
            return None

        periods = rows[header_index]

        structured_rows = []

        pending_labels = []

        for row in rows[header_index + 1:]:

            if len(row) == 1:

                text = row[0]

                if self.is_section_row(text):

                    # Section heading should not become
                    # part of the next metric.
                    pending_labels = []

                else:

                    pending_labels.append(text)

                continue
            metric = row[0]

            if pending_labels:

                metric = " ".join(
                    pending_labels + [metric]
                )

                pending_labels = []

            values = row[1:]

            if len(values) < len(periods):

                values = values + (
                    [""] * (len(periods) - len(values))
                )

            elif len(values) > len(periods):

                values = values[:len(periods)]

            period_values = dict(
                zip(periods, values)
            )

            structured_rows.append(
                {
                    "metric": metric,
                    "values": period_values
                }
            )
        # Final structured table

        return {
            "page": page_number,
            "table_number": table_number,
            "periods": periods,
            "rows": structured_rows
        }

    def extract_tables(self):

        structured_tables = []

        with pdfplumber.open(self.file_path) as pdf:

            for page_number, page in enumerate(
                pdf.pages,
                start=1
            ):

                tables = page.extract_tables()

                if not tables:
                    continue

                for table_number, table in enumerate(
                    tables,
                    start=1
                ):

                    structured_table = self.structure_table(
                        table=table,
                        page_number=page_number,
                        table_number=table_number
                    )

                    if structured_table:

                        structured_tables.append(
                            structured_table
                        )

        return structured_tables

    def display_tables(self):

        tables = self.extract_tables()

        for table in tables:

            print("\n" + "=" * 70)

            print(
                f"PAGE: {table['page']} | "
                f"TABLE: {table['table_number']}"
            )

            print("=" * 70)

            print(
                "PERIODS:",
                table["periods"]
            )

            print()

            for row in table["rows"]:

                print(
                    f"Metric: {row['metric']}"
                )

                for period, value in row["values"].items():

                    print(
                        f"  {period}: {value}"
                    )

                print()