# =========================================================
# DESCRIPTION
# Scrapes all catalog pages to get the list of all the assessments. from here 
# https://www.shl.com/products/product-catalog/?start=0&type=1&type=1
# =========================================================




import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
import csv
import time

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/products/product-catalog/?start={start}&type=1&type=1"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# SHL test type mapping
TEST_TYPE_MAP = {
    "A": "Ability",
    "B": "Behavioral",
    "C": "Competency",
    "D": "Development",
    "E": "Experience",
    "K": "Knowledge",
    "P": "Personality",
    "S": "Simulation"
}

results = []
seen_urls = set()

start = 0
STEP = 12

while True:
    url = CATALOG_URL.format(start=start)

    print(f"Scraping: {url}")

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)

        if response.status_code != 200:
            print(f"Failed page: {url}")
            break

        soup = BeautifulSoup(response.text, "html.parser")

        # Find all table wrappers
        wrappers = soup.select("div.custom__table-wrapper")

        target_table = None

        for wrapper in wrappers:
            heading = wrapper.select_one("th.custom__table-heading__title")

            if heading and heading.get_text(strip=True) == "Individual Test Solutions":
                target_table = wrapper
                break

        # Stop if no matching table found
        if not target_table:
            print("No Individual Test Solutions table found.")
            break

        rows = target_table.select("tr[data-entity-id]")

        # Stop if no rows found
        if not rows:
            print("No more data found.")
            break

        page_count = 0

        for row in rows:
            title_link = row.select_one(
                "td.custom__table-heading__title a"
            )

            if not title_link:
                continue

            title = title_link.get_text(strip=True)

            relative_url = title_link.get("href", "").strip()

            full_url = urljoin(BASE_URL, relative_url)

            # Skip duplicates
            if full_url in seen_urls:
                continue

            seen_urls.add(full_url)


            general_columns = row.select(
                "td.custom__table-heading__general"
            )

            # Remote Testing
            remote_testing = False
            if len(general_columns) >= 1:
                remote_testing = (
                    general_columns[0].select_one(
                        ".catalogue__circle.-yes"
                    ) is not None
                )

            # Adaptive/IRT
            adaptive_irt = False
            if len(general_columns) >= 2:
                adaptive_irt = (
                    general_columns[1].select_one(
                        ".catalogue__circle.-yes"
                    ) is not None
                )

            # Test Type Codes
            test_type_codes = []

            if len(general_columns) >= 3:
                keys = general_columns[2].select(
                    ".product-catalogue__key"
                )

                for key in keys:
                    code = key.get_text(strip=True)

                    if code:
                        test_type_codes.append(code)

            # Human-readable test types
            test_types = [
                TEST_TYPE_MAP.get(code, code)
                for code in test_type_codes
            ]

            item = {
                "title": title,
                "url": full_url,
                "remote_testing": remote_testing,
                "adaptive_irt": adaptive_irt,
                "test_type_codes": ", ".join(test_type_codes),
                "test_types": ", ".join(test_types)
            }

            results.append(item)
            page_count += 1

        print(f"Collected {page_count} items from offset {start}")

        # Stop if no new rows collected
        if page_count == 0:
            break

        start += STEP

        # Prevent hammering server
        time.sleep(1)

    except Exception as e:
        print(f"Error on page {start}: {e}")
        break

# ------------------------
# SAVE JSON
# ------------------------

with open("shl_catalog.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Saved JSON: shl_catalog.json")

# ------------------------
# SAVE CSV
# ------------------------

csv_columns = [
    "title",
    "url",
    "remote_testing",
    "adaptive_irt",
    "test_type_codes",
    "test_types"
]

with open(
    "shl_catalog.csv",
    "w",
    newline="",
    encoding="utf-8"
) as csvfile:

    writer = csv.DictWriter(
        csvfile,
        fieldnames=csv_columns
    )

    writer.writeheader()

    for row in results:
        writer.writerow(row)

print("Saved CSV: shl_catalog.csv")

print(f"\nFinished scraping.")
print(f"Total records collected: {len(results)}")