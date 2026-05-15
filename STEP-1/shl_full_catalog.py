# =========================================================
# DESCRIPTION
# Scrapes all catalog pages also the details of the assessments like description, duration, language, etc.
# 
# =========================================================


import requests
from bs4 import BeautifulSoup
from urllib.parse import (
    urljoin,
    quote,
    urlsplit,
    urlunsplit
)
import json
import csv
import time
import re

# =========================================================
# CONFIG
# =========================================================

BASE_URL = "https://www.shl.com"

CATALOG_URL = (
    "https://www.shl.com/products/product-catalog/"
    "?start={start}&type=1&type=1"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

STEP = 12
REQUEST_DELAY = 1

# =========================================================
# SHL TEST TYPE MAPPING
# =========================================================

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

# =========================================================
# HELPERS
# =========================================================

def clean_text(text):

    if not text:
        return ""

    return " ".join(text.strip().split())


def extract_minutes(text):

    """
    Example:
    Approximate Completion Time in minutes = 9
    """

    match = re.search(r"(\d+)", text)

    if match:
        return int(match.group(1))

    return None


def encode_url(url):

    """
    Encodes spaces and special characters in URLs.
    """

    parts = urlsplit(url)

    encoded_path = quote(parts.path)

    return urlunsplit((
        parts.scheme,
        parts.netloc,
        encoded_path,
        parts.query,
        parts.fragment
    ))


# =========================================================
# FUNCTION 1
# SCRAPE CATALOG
# =========================================================

def scrape_catalog():

    results = []
    seen_urls = set()

    start = 0

    while True:

        url = CATALOG_URL.format(start=start)

        print(f"\nScraping catalog page: {url}")

        try:

            response = requests.get(
                url,
                headers=HEADERS,
                timeout=30
            )

            if response.status_code != 200:
                print("Failed to load page.")
                break

            soup = BeautifulSoup(
                response.text,
                "html.parser"
            )

            wrappers = soup.select(
                "div.custom__table-wrapper"
            )

            target_table = None

            # ONLY scrape:
            # "Individual Test Solutions"

            for wrapper in wrappers:

                heading = wrapper.select_one(
                    "th.custom__table-heading__title"
                )

                if (
                    heading and
                    clean_text(heading.text)
                    == "Individual Test Solutions"
                ):
                    target_table = wrapper
                    break

            if not target_table:
                print(
                    "No Individual Test Solutions table found."
                )
                break

            rows = target_table.select(
                "tr[data-entity-id]"
            )

            if not rows:
                print("No rows found.")
                break

            page_count = 0

            for row in rows:

                link = row.select_one(
                    "td.custom__table-heading__title a"
                )

                if not link:
                    continue

                title = clean_text(link.text)

                relative_url = link.get(
                    "href",
                    ""
                ).strip()

                full_url = urljoin(
                    BASE_URL,
                    relative_url
                )

                # Avoid duplicates
                if full_url in seen_urls:
                    continue

                seen_urls.add(full_url)

                columns = row.select(
                    "td.custom__table-heading__general"
                )

                # =====================================
                # Remote Testing
                # =====================================

                remote_testing = False

                if len(columns) >= 1:

                    remote_testing = (
                        columns[0].select_one(
                            ".catalogue__circle.-yes"
                        ) is not None
                    )

                # =====================================
                # Adaptive / IRT
                # =====================================

                adaptive_irt = False

                if len(columns) >= 2:

                    adaptive_irt = (
                        columns[1].select_one(
                            ".catalogue__circle.-yes"
                        ) is not None
                    )

                # =====================================
                # Test Type Codes
                # =====================================

                test_type_codes = []

                if len(columns) >= 3:

                    keys = columns[2].select(
                        ".product-catalogue__key"
                    )

                    for key in keys:

                        code = clean_text(
                            key.text
                        )

                        if code:
                            test_type_codes.append(
                                code
                            )

                test_types = [
                    TEST_TYPE_MAP.get(code, code)
                    for code in test_type_codes
                ]

                item = {
                    "title": title,
                    "url": full_url,
                    "remote_testing": remote_testing,
                    "adaptive_irt": adaptive_irt,
                    "test_type_codes": test_type_codes,
                    "test_types": test_types
                }

                results.append(item)

                page_count += 1

            print(
                f"Collected {page_count} assessments"
            )

            if page_count == 0:
                break

            start += STEP

            time.sleep(REQUEST_DELAY)

        except Exception as e:

            print(f"Error: {e}")
            break

    return results


# =========================================================
# FUNCTION 2
# SCRAPE DETAIL PAGE
# =========================================================

def scrape_detail_page(item):

    url = item["url"]

    print(f"Scraping detail: {url}")

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=30
        )

        if response.status_code != 200:

            print(f"Failed: {url}")
            return item

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        rows = soup.select(
            ".product-catalogue-training-calendar__row"
        )

        description = ""

        # Default values

        item["job_levels"] = []
        item["languages"] = []
        item["assessment_length_minutes"] = None
        item["downloads"] = []

        for row in rows:

            heading = row.find("h4")

            if not heading:
                continue

            heading_text = clean_text(
                heading.text
            )

            # =====================================
            # DESCRIPTION
            # =====================================

            if heading_text == "Description":

                p = row.find("p")

                if p:

                    description = clean_text(
                        p.text
                    )

            # =====================================
            # JOB LEVELS
            # =====================================

            elif heading_text == "Job levels":

                p = row.find("p")

                if p:

                    item["job_levels"] = [

                        clean_text(x)

                        for x in p.text.split(",")

                        if clean_text(x)
                    ]

            # =====================================
            # LANGUAGES
            # =====================================

            elif heading_text == "Languages":

                p = row.find("p")

                if p:

                    item["languages"] = [

                        clean_text(x)

                        for x in p.text.split(",")

                        if clean_text(x)
                    ]

            # =====================================
            # ASSESSMENT LENGTH
            # =====================================

            elif heading_text == "Assessment length":

                p = row.find("p")

                if p:

                    item[
                        "assessment_length_minutes"
                    ] = extract_minutes(
                        p.text
                    )

        item["description"] = description

        # =================================================
        # DOWNLOADS
        # =================================================

        downloads = []

        download_rows = soup.select(
            ".product-catalogue__download"
        )

        for d in download_rows:

            title_el = d.select_one(
                ".product-catalogue__download-title a"
            )

            language_el = d.select_one(
                ".product-catalogue__download-language"
            )

            if not title_el:
                continue

            download_title = clean_text(
                title_el.text
            )

            # =====================================
            # FIX DOWNLOAD URL ENCODING
            # =====================================

            download_url_raw = title_el.get(
                "href",
                ""
            ).strip()

            download_url = encode_url(
                download_url_raw
            )

            download_language = ""

            if language_el:

                download_language = clean_text(
                    language_el.text
                )

            downloads.append({
                "title": download_title,
                "url": download_url,
                "language": download_language
            })

        item["downloads"] = downloads

        return item

    except Exception as e:

        print(f"Detail scrape error: {e}")

        return item


# =========================================================
# SAVE JSON
# =========================================================

def save_json(data, filename):

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Saved JSON: {filename}")


# =========================================================
# SAVE CSV
# =========================================================

def save_csv(data, filename):

    rows = []

    for item in data:

        rows.append({

            "title":
                item.get("title"),

            "url":
                item.get("url"),

            "description":
                item.get("description"),

            "job_levels":
                ", ".join(
                    item.get("job_levels", [])
                ),

            "languages":
                ", ".join(
                    item.get("languages", [])
                ),

            "assessment_length_minutes":
                item.get(
                    "assessment_length_minutes"
                ),

            "remote_testing":
                item.get("remote_testing"),

            "adaptive_irt":
                item.get("adaptive_irt"),

            "test_type_codes":
                ", ".join(
                    item.get(
                        "test_type_codes",
                        []
                    )
                ),

            "test_types":
                ", ".join(
                    item.get(
                        "test_types",
                        []
                    )
                ),

            "downloads":
                json.dumps(
                    item.get("downloads", []),
                    ensure_ascii=False
                )
        })

    fieldnames = [
        "title",
        "url",
        "description",
        "job_levels",
        "languages",
        "assessment_length_minutes",
        "remote_testing",
        "adaptive_irt",
        "test_type_codes",
        "test_types",
        "downloads"
    ]

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(rows)

    print(f"Saved CSV: {filename}")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    print("\n==============================")
    print("STEP 1: SCRAPING CATALOG")
    print("==============================")

    catalog_data = scrape_catalog()

    print(
        f"\nTotal catalog items: "
        f"{len(catalog_data)}"
    )

    print("\n==============================")
    print("STEP 2: SCRAPING DETAIL PAGES")
    print("==============================")

    full_results = []

    for index, item in enumerate(
        catalog_data,
        start=1
    ):

        print(
            f"\n[{index}/{len(catalog_data)}]"
        )

        enriched_item = scrape_detail_page(
            item
        )

        full_results.append(
            enriched_item
        )

        time.sleep(REQUEST_DELAY)

    print("\n==============================")
    print("SAVING FILES")
    print("==============================")

    save_json(
        full_results,
        "shl_full_catalog.json"
    )

    save_csv(
        full_results,
        "shl_full_catalog.csv"
    )

    print("\nDONE")
    print(
        f"Total assessments scraped: "
        f"{len(full_results)}"
    )