#  python .\src\eppo_crawl_reporting.py
# %% === CONFIGURATION ===
import asyncio
from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup
import pandas as pd


CSV_PATH = "data/eppo_downloads/eppo_code.csv"
# START_ID = 12780  # LAPHFR fall armyworm
# END_ID = 12781  # LAPHFR fall armyworm

# START_ID = 3243  # Xylella fastidiosa(XYLEFA)
# END_ID = 3244  # Xylella fastidiosa(XYLEFA)

START_ID = 105001  # 'Candidatus Liberibacter africanus'(LIBEAF)
END_ID = 126094  # 'Candidatus Liberibacter africanus'(LIBEAF)

OUTPUT_CSV = "data/eppo_downloads/eppo_reporting_126k.csv"
BASE_URL = "https://gd.eppo.int"


# %% === LOAD EPPO codes ===
df = pd.read_csv(CSV_PATH, index_col="codeid")
selected_df = df.loc[START_ID : END_ID - 1]
selected_codes = selected_df["eppocode"].dropna().unique()
selected_codes = [code.strip() for code in selected_codes]


# %% === Extract list of reports for a given EPPO code ===
async def extract_reports(eppo_code):
    url = f"{BASE_URL}/taxon/{eppo_code}/reporting"
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        soup = BeautifulSoup(result.html, "html.parser")  # type: ignore

        # NOTE Extract full name from the <h2> header
        h2 = soup.select_one("div.hero h2")
        if h2:
            ital = h2.find("i")
            name = ital.get_text(strip=True) if ital else ""
        else:
            name = ""

        rows = []
        for tr in soup.select("table.table tbody tr"):
            cols = tr.find_all("td")
            if len(cols) < 3:
                continue

            number = cols[0].get_text(strip=True)
            title = cols[1].get_text(strip=True)
            link = BASE_URL + cols[1].find("a")["href"]
            date = cols[2].get_text(strip=True)

            # Fetch article content
            try:
                art_result = await crawler.arun(url=link)
                art_soup = BeautifulSoup(art_result.html, "html.parser")  # type: ignore

                content_div = art_soup.find("div", class_="content")
                content = (
                    content_div.get_text(separator="\n", strip=True)
                    if content_div
                    else art_soup.get_text()[:2000]
                )
            except Exception as e:
                print(f"❌ Failed to fetch {link}: {e}")
                content = ""

            rows.append(
                {
                    "EPPO_code": eppo_code,
                    "Name": name,
                    "Number": number,
                    "Title": title,
                    "Date": date,
                    "URL": link,
                    "Content": content,
                }
            )

        return rows


# === Runner for batch of codes ===
async def run_batch(codes):
    all_rows = []
    for code in codes:
        print(f"📦 Processing reporting page for {code}")
        try:
            rows = await extract_reports(code)
            all_rows.extend(rows)
        except Exception as e:
            print(f"❌ Failed for {code}: {e}")
    return all_rows


# === MAIN ===
if __name__ == "__main__":
    results = asyncio.run(run_batch(selected_codes))
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"✅ Done. Extracted {len(results)} reports into {OUTPUT_CSV}")
