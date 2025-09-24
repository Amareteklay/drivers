#   py -3.12 src/eppo_llm_extraction.py
# %% == LOAD PACKAGES ===

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
import json


# %% === EXTRACTOR CLASSS DEFINITION ===
class EPPOReportingExtractor:
    def __init__(self, csv_path, model="gpt-5-nano"):
        self.csv_path = csv_path
        self.model = model
        self.client = OpenAI()
        self.reports_df = self.load_reports()
        self.results = []

    def load_reports(self):
        try:
            return pd.read_csv(self.csv_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load reporting CSV file: {e}")

    def classify_and_extract(self, row):
        """Classify record type and extract disease occurrences if relevant."""
        eppocode = row["EPPO_code"]
        name = row.get("Name", "")
        title = row["Title"]
        content = row["Content"]
        number = row.get("Number", "")
        date = row.get("Date", "")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": f"""
                Classify the type of record described in the following plant health report into exactly one of these categories:

                **Record Type Definitions:**
                - **First record**: Initial detection or confirmation of a plant disease/pest in a country or region where it has never been officially recorded before
                - **Update of outbreak**: New information about an ongoing disease outbreak, including spread to new areas, changes in severity, or additional affected locations within a country
                - **Eradication of disease**: Official confirmation that a plant disease/pest has been successfully eliminated from a country or region
                - **Absence of disease**: Official confirmation that a suspected disease/pest is not present in a country or region, or that previous records were incorrect
                - **Revised cause of disease**: Correction of previous reports where the causative agent was misidentified or reclassified
                - **New type of hosts**: Discovery that a known disease/pest can affect plant species not previously known to be susceptible
                - **New type of symptoms**: Discovery of new symptoms or damage patterns caused by a known disease/pest
                - **Others**: Reports that don't fit the above categories, such as general surveys, methodology papers, or unclear cases. Often cases are "New EU Regulations", "EPPO Distribution List for...", etc.

                Then, ONLY if the type_of_record is one of:
                - First record
                - Update of outbreak
                - Eradication of disease
                - Absence of disease

                extract **all occurrences** where plant diseases are mentioned. For each occurrence return:
                - country
                - year (exact or approximate)
                - continent (if present or inferable)

                For other record types, leave the occurrences array empty.

                Return a JSON object with fields:
                {{
                "type_of_record": "<classified type>",
                "occurrences": [
                    {{
                    "country": "...",
                    "year": "...",
                    "continent": "..."
                    }},
                    ...
                ]
                }}
                #NOTE Take the title and content to classify the record and extract occurrences from.
                TITLE: {title}
                TEXT: {content}
                """,
                    }
                ],
                max_completion_tokens=5000,
            )

            raw_json = response.choices[0].message.content
            if raw_json is None:
                return []

            # Strip markdown fences if present
            if raw_json.startswith("```json"):
                raw_json = raw_json.removeprefix("```json").strip()
            if raw_json.endswith("```"):
                raw_json = raw_json.removesuffix("```").strip()

            parsed = json.loads(raw_json)
            type_of_record = parsed.get("type_of_record", "Others")
            occurrences = parsed.get("occurrences", [])

            # Check if this record type should have occurrence data extracted
            relevant_types = [
                "First record",
                "Update of outbreak",
                "Eradication of disease",
                "Absence of disease",
            ]

            if type_of_record in relevant_types and occurrences:
                # Return all occurrences for relevant record types
                return [
                    {
                        "eppocode": eppocode,
                        "name": name,
                        "title": title,
                        "number": number,
                        "date": date,
                        "type_of_record": type_of_record,
                        "year": entry.get("year"),
                        "country": entry.get("country"),
                        "continent": entry.get("continent"),
                    }
                    for entry in occurrences
                ]
            else:
                # For irrelevant types or relevant types with no occurrences,
                # still include the record but with empty occurrence fields
                return [
                    {
                        "eppocode": eppocode,
                        "name": name,
                        "title": title,
                        "number": number,
                        "date": date,
                        "type_of_record": type_of_record,
                        "year": None,
                        "country": None,
                        "continent": None,
                    }
                ]

        except Exception as e:
            print(f"❌ Error processing {eppocode}: {e}")
            return []

    def process_all_reports(self):
        for _, row in self.reports_df.iterrows():
            print(f"📦 Processing report: {row['Title']}")
            extracted = self.classify_and_extract(row)
            self.results.extend(extracted)

    def save_results(self, output_path):
        try:
            # Create DataFrame from results
            results_df = pd.DataFrame(self.results)

            # Join with original data to add URL and Content columns
            # Use a subset of original columns to avoid duplication
            original_columns = [
                "EPPO_code",
                "Name",
                "Title",
                "Number",
                "Date",
                "URL",
                "Content",
            ]
            join_df = self.reports_df[original_columns].copy()

            # Perform left join on the key columns
            final_df = results_df.merge(
                join_df,
                left_on=["eppocode", "name", "title", "number", "date"],
                right_on=["EPPO_code", "Name", "Title", "Number", "Date"],
                how="left",
            )

            # Drop duplicate columns from the join
            final_df = final_df.drop(
                columns=["EPPO_code", "Name", "Title", "Number", "Date"]
            )

            # Reorder columns for better readability
            column_order = [
                "eppocode",
                "name",
                "title",
                "number",
                "date",
                "type_of_record",
                "year",
                "country",
                "continent",
                "URL",
                "Content",
            ]
            final_df = final_df.reindex(columns=column_order)

            final_df.to_csv(output_path, index=False)
            print(f"✅ Results saved to {output_path}")
            print(f"📊 Total records processed: {len(final_df)}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")


# %% == EXECUTION ===


def main():
    load_dotenv(dotenv_path="src/.env")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise RuntimeError("OpenAI API key not found. Please set it in the .env file.")

    # === CONFIGURATION ===
    REPORTING_CSV_PATH = "data/eppo_downloads/eppo_reporting_retrieved copy.csv"
    OUTPUT_PATH = "data/eppo_downloads/eppo_reporting_occurrence_LLM.csv"

    extractor = EPPOReportingExtractor(REPORTING_CSV_PATH)
    extractor.process_all_reports()
    extractor.save_results(OUTPUT_PATH)


if __name__ == "__main__":
    main()
