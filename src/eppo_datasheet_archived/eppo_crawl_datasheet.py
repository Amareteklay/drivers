# the script need to be ran in Terminal: py -3.12 src/eppo_crawl_datasheet.py

import asyncio
from crawl4ai import *
from bs4 import BeautifulSoup
import pandas as pd
import re

# SECTION HEADINGS TO PARSE
main_headings = [
    "IDENTITY",
    "HOSTS",
    "GEOGRAPHICAL DISTRIBUTION",
    "BIOLOGY",
    "DETECTION AND IDENTIFICATION",
    "PATHWAYS FOR MOVEMENT",
    "PEST SIGNIFICANCE",
    "PHYTOSANITARY MEASURES",
    # "REFERENCES",
    # "ACKNOWLEDGEMENTS",
]

# === CONFIGURATION ===
CSV_PATH = "data/eppo_downloads/eppo_code.csv"
START_ID = 62927  # Inclusive
END_ID = 62930  # Exclusive
OUTPUT_CSV = "data/eppo_downloads/eppo_datasheet_sections.csv"

# === LOAD CSV WITH codeid AS INDEX ===
df = pd.read_csv(CSV_PATH, index_col="codeid")
selected_df = df.loc[START_ID : END_ID - 1]
selected_codes = selected_df["eppocode"].dropna().unique()
selected_codes = [code.strip() for code in selected_codes]


# === FUNCTION TO EXTRACT AND SPLIT A DATASHEET ===
async def extract_sections(eppo_code):
    url = f"https://gd.eppo.int/taxon/{eppo_code}/datasheet"
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        soup = BeautifulSoup(result.html, "html.parser")

        # Get raw text
        full_text = soup.get_text(separator="\n", strip=True)

        # Extract full name
        full_name_match = re.search(r"EPPO Datasheet:\s*([^\n]+)", full_text)
        full_name = full_name_match.group(1).strip() if full_name_match else ""

        # Split into sections
        pattern = "|".join([re.escape(h) for h in main_headings])
        split_sections = re.split(f"^({pattern})", full_text, flags=re.MULTILINE)

        rows = []
        for i in range(1, len(split_sections) - 1, 2):
            heading = split_sections[i].strip()
            content = split_sections[i + 1].strip()
            content = re.sub(r"^\d{4}-\d{2}-\d{2}\s*", "", content)

            rows.append(
                {
                    "DocumentID": eppo_code,
                    "Full Name": full_name,
                    "Section Title": heading,
                    "Section Content": content,
                }
            )

        return rows


# === RUNNER FUNCTION FOR ALL CODES ===
async def run_batch(codes):
    all_rows = []
    for code in codes:
        print(f"📦 Processing {code}")
        try:
            rows = await extract_sections(code)
            all_rows.extend(rows)
        except Exception as e:
            print(f"❌ Failed for {code}: {e}")
    return all_rows


# === MAIN ===
if __name__ == "__main__":
    results = asyncio.run(run_batch(selected_codes))
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Done. Extracted {len(results)} section rows into {OUTPUT_CSV}")
