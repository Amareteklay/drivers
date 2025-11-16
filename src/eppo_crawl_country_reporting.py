# python .\src\eppo_crawl_country_reporting.py

import asyncio
import logging
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd
from bs4 import BeautifulSoup

# === CONFIGURATION ===
BASE_URL = "https://gd.eppo.int"
OUTPUT_CSV = "data/eppo_downloads/eppo_reporting_countries.csv"
MAX_RETRIES = 3
TIMEOUT = 30.0

# Complete country list with ISO 2-letter codes
COUNTRIES = {
    "Afghanistan": "AF",
    "Albania": "AL",
    "Algeria": "DZ",
    "American Samoa": "AS",
    "Andorra": "AD",
    "Angola": "AO",
    "Anguilla": "AI",
    "Antarctica": "AQ",
    "Antigua and Barbuda": "AG",
    "Argentina": "AR",
    "Armenia": "AM",
    "Aruba": "AW",
    "Australia": "AU",
    "Austria": "AT",
    "Azerbaijan": "AZ",
    "Bahamas": "BS",
    "Bahrain": "BH",
    "Bangladesh": "BD",
    "Barbados": "BB",
    "Belarus": "BY",
    "Belgium": "BE",
    "Belize": "BZ",
    "Benin": "BJ",
    "Bermuda": "BM",
    "Bhutan": "BT",
    "Bolivia": "BO",
    "Bosnia and Herzegovina": "BA",
    "Botswana": "BW",
    "Bouvet Island": "BV",
    "Brazil": "BR",
    "British Indian Ocean Territory": "IO",
    "Brunei Darussalam": "BN",
    "Bulgaria": "BG",
    "Burkina Faso": "BF",
    "Burundi": "BI",
    "Cabo Verde": "CV",
    "Cambodia": "KH",
    "Cameroon": "CM",
    "Canada": "CA",
    "Canton and Enderbury Islands": "CT",  # historical / non-standard
    "Cayman Islands": "KY",
    "Central African Republic": "CF",
    "Chad": "TD",
    "Chile": "CL",
    "China": "CN",
    "Christmas Island": "CX",
    "Cocos Islands": "CC",
    "Colombia": "CO",
    "Comoros": "KM",
    "Congo": "CG",
    "Congo, The Democratic Republic of the": "CD",
    "Cook Islands": "CK",
    "Costa Rica": "CR",
    "Cote d'Ivoire": "CI",
    "Croatia": "HR",
    "Cuba": "CU",
    "Cyprus": "CY",
    "Czechia": "CZ",
    "Czechoslovakia (former)": "CS",
    "Denmark": "DK",
    "Djibouti": "DJ",
    "Dominica": "DM",
    "Dominican Republic": "DO",
    "East Timor": "TP",
    "Ecuador": "EC",
    "Egypt": "EG",
    "El Salvador": "SV",
    "Equatorial Guinea": "GQ",
    "Eritrea": "ER",
    "Estonia": "EE",
    "Eswatini": "SZ",
    "Ethiopia": "ET",
    "Falkland Islands": "FK",
    "Faroe Islands": "FO",
    "Fiji": "FJ",
    "Finland": "FI",
    "France": "FR",
    "French Guiana": "GF",
    "French Polynesia": "PF",
    "French Southern Territories": "TF",
    "Gabon": "GA",
    "Gambia": "GM",
    "Georgia": "GE",
    "Germany": "DE",
    "Ghana": "GH",
    "Gibraltar": "GI",
    "Greece": "GR",
    "Greenland": "GL",
    "Grenada": "GD",
    "Guadeloupe": "GP",
    "Guam": "GU",
    "Guatemala": "GT",
    "Guernsey": "GG",
    "Guinea": "GN",
    "Guinea-Bissau": "GW",
    "Guyana": "GY",
    "Haiti": "HT",
    "Heard and McDonald Islands": "HM",
    "Holy See (Vatican City State)": "VA",
    "Honduras": "HN",
    "Hungary": "HU",
    "Iceland": "IS",
    "India": "IN",
    "Indonesia": "ID",
    "Iran, Islamic Republic of": "IR",
    "Iraq": "IQ",
    "Ireland": "IE",
    "Israel": "IL",
    "Italy": "IT",
    "Jamaica": "JM",
    "Japan": "JP",
    "Jersey": "JS",
    "Jordan": "JO",
    "Kazakhstan": "KZ",
    "Kenya": "KE",
    "Kiribati": "KI",
    "Korea, Democratic People's Republic of": "KP",
    "Korea, Republic of": "KR",
    "Kuwait": "KW",
    "Kyrgyzstan": "KG",
    "Lao People's Democratic Republic": "LA",
    "Latvia": "LV",
    "Lebanon": "LB",
    "Lesotho": "LS",
    "Liberia": "LR",
    "Libya": "LY",
    "Liechtenstein": "LI",
    "Lithuania": "LT",
    "Luxembourg": "LU",
    "Madagascar": "MG",
    "Malawi": "MW",
    "Malaysia": "MY",
    "Maldives": "MV",
    "Mali": "ML",
    "Malta": "MT",
    "Marshall Islands": "MH",
    "Martinique": "MQ",
    "Mauritania": "MR",
    "Mauritius": "MU",
    "Mayotte": "YT",
    "Mexico": "MX",
    "Micronesia, Federated States of": "FM",
    "Moldova, Republic of": "MD",
    "Monaco": "MC",
    "Mongolia": "MN",
    "Montenegro": "ME",
    "Montserrat": "MS",
    "Morocco": "MA",
    "Mozambique": "MZ",
    "Myanmar": "MM",
    "Namibia": "NA",
    "Nauru": "NR",
    "Nepal": "NP",
    "Netherlands": "NL",
    "Netherlands Antilles": "AN",
    "New Caledonia": "NC",
    "New Zealand": "NZ",
    "Nicaragua": "NI",
    "Niger": "NE",
    "Nigeria": "NG",
    "Niue": "NU",
    "Norfolk Island": "NF",
    "North Macedonia": "MK",
    "Northern Mariana Islands": "MP",
    "Norway": "NO",
    "Oman": "OM",
    "Pakistan": "PK",
    "Palau": "PW",
    "Panama": "PA",
    "Papua New Guinea": "PG",
    "Paraguay": "PY",
    "Peru": "PE",
    "Philippines": "PH",
    "Pitcairn": "PN",
    "Poland": "PL",
    "Portugal": "PT",
    "Puerto Rico": "PR",
    "Qatar": "QA",
    "Reunion": "RE",
    "Romania": "RO",
    "Russian Federation (the)": "RU",
    "Rwanda": "RW",
    "Saint Helena": "SH",
    "Saint Kitts and Nevis": "KN",
    "Saint Lucia": "LC",
    "Saint Pierre and Miquelon": "PM",
    "Saint Vincent and the Grenadines": "VC",
    "Samoa": "WS",
    "San Marino": "SM",
    "Sao Tome and Principe": "ST",
    "Saudi Arabia": "SA",
    "Senegal": "SN",
    "Serbia": "RS",
    "Serbia and Montenegro": "YU",  # historical
    "Seychelles": "SC",
    "Sierra Leone": "SL",
    "Singapore": "SG",
    "Slovakia": "SK",
    "Slovenia": "SI",
    "Solomon Islands": "SB",
    "Somalia": "SO",
    "South Africa": "ZA",
    "South Georgia and South Sandwich Islands": "GS",
    "South Sudan": "SS",
    "Spain": "ES",
    "Sri Lanka": "LK",
    "Sudan": "SD",
    "Suriname": "SR",
    "Svalbard and Jan Mayen Islands": "SJ",
    "Sweden": "SE",
    "Switzerland": "CH",
    "Syrian Arab Republic": "SY",
    "Taiwan": "TW",
    "Tajikistan": "TJ",
    "Tanzania, United Republic of": "TZ",
    "Thailand": "TH",
    "Togo": "TG",
    "Tokelau": "TK",
    "Tonga": "TO",
    "Trinidad and Tobago": "TT",
    "Tunisia": "TN",
    "Turkmenistan": "TM",
    "Turks and Caicos Islands": "TC",
    "Tuvalu": "TV",
    "Türkiye": "TR",
    "Uganda": "UG",
    "Ukraine": "UA",
    "United Arab Emirates": "AE",
    "United Kingdom": "GB",
    "United States Minor Outlying Islands (the)": "UM",
    "United States of America": "US",
    "Unknown": "ZZ",
    "Uruguay": "UY",
    "Uzbekistan": "UZ",
    "Vanuatu": "VU",
    "Venezuela": "VE",
    "Vietnam": "VN",
    "Virgin Islands (British)": "VG",
    "Virgin Islands (US)": "VI",
    "Wallis and Futuna Islands": "WF",
    "Western Sahara": "EH",
    "Yemen": "YE",
    "Zaire": "ZR",  # historical, equivalent to DR Congo (CD)
    "Zambia": "ZM",
    "Zimbabwe": "ZW",
}

# Optional: filter subset of countries:
TARGET_COUNTRIES = None  # e.g. ["US", "ID", "CN"]


# ================================
# LOGGING
# ================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ================================
# ASYNC HTTP HELPERS
# ================================
async def fetch_page(
    client: httpx.AsyncClient, url: str, retries: int = MAX_RETRIES
) -> Optional[str]:
    """
    Fetch page with retry + exponential backoff.
    """
    for attempt in range(retries):
        try:
            response = await client.get(url, timeout=TIMEOUT, follow_redirects=True)
            if response.status_code in (200, 301, 302):
                return response.text

            logger.warning(f"❌ HTTP {response.status_code} for {url}")

        except httpx.TimeoutException:
            logger.warning(f"⏱️ Timeout at {url} (attempt {attempt + 1}/{retries})")

        except Exception as e:
            logger.error(f"❌ Error fetching {url}: {e}")

        # Backoff
        if attempt < retries - 1:
            await asyncio.sleep(2**attempt)

    return None


# ================================
# PARSE A COUNTRY REPORTING PAGE
# ================================
def parse_reporting_table(
    html: str, country_code: str, country_name: str
) -> list[dict]:
    """
    Parse reporting table, extracting ONLY:
    - country
    - number
    - title
    - date
    - url (to report page)
    """
    soup = BeautifulSoup(html, "html.parser")
    rows = []

    table = soup.select_one("table.table tbody")
    if not table:
        logger.warning(f"⚠️ No reporting table found for {country_name}")
        return rows

    for tr in table.find_all("tr"):
        cols = tr.find_all("td")
        if len(cols) < 3:
            continue

        number = cols[0].get_text(strip=True)
        title_cell = cols[1]
        title = title_cell.get_text(strip=True)

        link_tag = title_cell.find("a")
        url = BASE_URL + link_tag["href"] if link_tag and link_tag.get("href") else ""

        date = cols[2].get_text(strip=True)

        rows.append(
            {
                "country": country_name,
                "number": number,
                "title": title,
                "date": date,
                "url": url,
            }
        )

    return rows


# ================================
# CRAWL A SINGLE COUNTRY
# ================================
async def crawl_country(
    client: httpx.AsyncClient, country_name: str, country_code: str
) -> list[dict]:
    logger.info(f"📦 Crawling {country_name} ({country_code})")

    url = f"{BASE_URL}/country/{country_code}/reporting"
    html = await fetch_page(client, url)

    if not html:
        logger.error(f"❌ Cannot load reporting page for {country_name}")
        return []

    rows = parse_reporting_table(html, country_code, country_name)
    logger.info(f"📄 Found {len(rows)} reporting entries for {country_name}")

    return rows


# ================================
# MAIN EXECUTION PIPELINE
# ================================
async def main():
    # Apply optional filter
    if TARGET_COUNTRIES:
        countries = {k: v for k, v in COUNTRIES.items() if v in TARGET_COUNTRIES}
    else:
        countries = COUNTRIES

    logger.info(f"🌍 Starting crawl for {len(countries)} countries")

    all_rows = []

    async with httpx.AsyncClient() as client:
        for country_name, country_code in countries.items():
            try:
                rows = await crawl_country(client, country_name, country_code)
                all_rows.extend(rows)
            except Exception as e:
                logger.error(f"❌ Error processing {country_name}: {e}")

    # Save CSV
    if all_rows:
        output_path = Path(OUTPUT_CSV)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(all_rows)
        df.to_csv(output_path, index=False, encoding="utf-8")

        logger.info(f"✅ Done. Extracted {len(all_rows)} total rows → {OUTPUT_CSV}")
    else:
        logger.warning("⚠️ No data collected.")


if __name__ == "__main__":
    asyncio.run(main())

"""
Countries with NO TABLE (page exists but no reporting data)
| Country                                   |
| ----------------------------------------- |
| Anguilla                                  |
| Antarctica                                |
| Bouvet Island                             |
| British Indian Ocean Territory            |
| Christmas Island                          |
| Cocos Islands                             |
| Djibouti                                  |
| Eritrea                                   |
| Falkland Islands                          |
| Faroe Islands                             |
| Greenland                                 |
| Heard and McDonald Islands                |
| Holy See (Vatican City State)             |
| Lesotho                                   |
| Maldives                                  |
| Pitcairn                                  |
| Svalbard and Jan Mayen Islands            |
| Tokelau                                   |
| Turks and Caicos Islands                  |
| Tuvalu                                    |
| United States Minor Outlying Islands (UM) |
| Unknown (ZZ)                              |
| Wallis and Futuna Islands                 |
| Zaire (ZR)                                |
"""
