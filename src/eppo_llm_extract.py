"""
EPPO LLM Extractor

This script `extracts all occurrences` of EPPO disease code
from the `"GEOGRAPHICAL DISTRIBUTION"` section of EPPO datasheets using an LLM (OpenAI GPT model).

For each mention, it extracts:
- country (required)
- year (exact or approximate)
- continent (if stated or can be inferred)

Usage:
- Configure  OpenAI API key in src/.env
- Place  EPPO datasheet sections CSV at data/eppo_downloads/eppo_datasheet_sections.csv
- Adjust eppocode and section_titles in the main() function as needed

Results are saved to data/eppo_downloads/structured_occurrences.csv
"""

import pandas as pd
import asyncio
from openai import OpenAI
from dotenv import load_dotenv
import os
import json


class EPPOExtractor:
    def __init__(self, csv_path, model="gpt-4o"):
        self.csv_path = csv_path
        self.model = model
        self.client = OpenAI()  # Initialize the OpenAI client
        self.sections_df = self.load_datasheet_sections()
        self.results = []

    def load_datasheet_sections(self):
        """Load the datasheet sections CSV into a DataFrame."""
        try:
            return pd.read_csv(self.csv_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load CSV file: {e}")

    def extract_information(self, eppocode, section_title):
        """Extract information for a specific eppocode and section using the LLM."""
        # Filter the DataFrame for the specific eppocode and section
        section_data = self.sections_df[
            (self.sections_df["DocumentID"] == eppocode)
            & (self.sections_df["Section Title"] == section_title)
        ]

        if section_data.empty:
            return (
                f"No data found for eppocode: {eppocode} and section: {section_title}"
            )

        section_content = section_data["Section Content"].values[0]

        # Call the LLM to extract information
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": f"""
                        From the text below, extract **all occurrences** where the citrus disease (huanglongbing / greening / citrus dieback / yellow branch disease) is mentioned.

                        For each occurrence, return:
                        - **country** (required)
                        - **year** (exact year if possible, or approximate like "18th century", "1920s", etc.)
                        - **continent** (if stated or can be inferred)

                        Include mentions even if the disease is:
                        - Described with different names
                        - Mentioned as **absent**
                        - Mentioned with vague or approximate years
                        - Mentioned as part of a group of countries

                        Return results in a JSON array format, like:

                        [
                        {{
                            "country": "India",
                            "year": "18th century",
                            "continent": "Asia"
                        }},
                        {{
                            "country": "China",
                            "year": "Late 19th century",
                            "continent": "Asia"
                        }},
                        ...
                        ]

                        Return **only** the JSON array, with no explanation, no markdown, and no extra text.

                        TEXT:
                        {section_content}
                        """,
                    }
                ],
                temperature=0.3,
                max_tokens=2000,
            )
            content = response.choices[0].message.content
            if content is None:
                print("🚨 No content returned from LLM.")
                return []
            raw_json = content
            if raw_json.startswith("```json"):
                raw_json = raw_json.removeprefix("```json").strip()
            if raw_json.endswith("```"):
                raw_json = raw_json.removesuffix("```").strip()
            # print(
            #     "🧪 Raw Response:", raw_json[:500]
            # )  # Print just the first 500 chars for safety
            try:
                extracted_data = json.loads(raw_json)
            except json.JSONDecodeError as je:
                print("🚨 JSON Decode Error:", je)
                print("🔎 Raw response that failed to parse:", raw_json)
                return []
            return [
                {
                    "eppocode": eppocode,
                    "section_title": section_title,
                    "year": entry.get("year"),
                    "country": entry.get("country"),
                    "continent": entry.get("continent"),
                }
                for entry in extracted_data
            ]
        except Exception as e:
            print(f"❌ Error processing {eppocode}: {e}")
            return []

    def process_all_sections(self, eppocode, section_titles):
        """Process multiple sections for a given eppocode."""
        for section_title in section_titles:
            print(f"📦 Processing eppocode: {eppocode}, section: {section_title}")
            extracted = self.extract_information(eppocode, section_title)
            self.results.extend(extracted)  # Add all entries

    def save_results(self, output_path):
        try:
            df = pd.DataFrame(self.results)
            df.to_csv(output_path, index=False)
            print(f"✅ Results saved to {output_path}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")


async def run_batch(extractor, eppocodes, section_titles):
    """Process a batch of eppocodes asynchronously."""
    for eppocode in eppocodes:
        extractor.process_all_sections(eppocode, section_titles)


def main():
    # Load environment variables
    load_dotenv(dotenv_path="src/.env")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise RuntimeError("OpenAI API key not found. Please set it in the .env file.")

    # === CONFIGURATION ===
    CODES_CSV_PATH = "data/eppo_downloads/eppo_code.csv"  # The one with codeid
    SECTIONS_CSV_PATH = "data/eppo_downloads/eppo_datasheet_sections.csv"
    OUTPUT_PATH = "data/eppo_downloads/structured_occurrences.csv"
    SECTION_TITLES = ["GEOGRAPHICAL DISTRIBUTION"]

    try:
        # === LOAD EPPO CODE CSV WITH codeid as index ===
        code_df = pd.read_csv(CODES_CSV_PATH, index_col="codeid")

        # === Get user input for codeid range ===
        start_id = int(input("Enter the START_ID (inclusive): ").strip())
        end_id = int(input("Enter the END_ID (exclusive): ").strip())

        if start_id >= end_id:
            print("❌ Error: START_ID must be less than END_ID.")
            return

        # Filter by index using .loc (codeid)
        selected_df = code_df.loc[start_id : end_id - 1]
        selected_codes = selected_df["eppocode"].dropna().unique()
        selected_codes = [code.strip() for code in selected_codes]

        print(f"🧪 Selected {len(selected_codes)} eppocode(s):", selected_codes)

        # === Initialize the extractor with the full datasheet ===
        extractor = EPPOExtractor(SECTIONS_CSV_PATH)

        # === Run the batch processing ===
        asyncio.run(run_batch(extractor, selected_codes, SECTION_TITLES))

        # === Save the extracted result ===
        extractor.save_results(OUTPUT_PATH)

    except ValueError:
        print("❌ Error: Please enter valid numeric values for START_ID and END_ID.")
    except KeyError as ke:
        print(f"❌ Invalid codeid: {ke}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
