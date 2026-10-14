"""Fetch a versioned, explicit-relation external assay table; no pChEMBL labels."""

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import truststore

BASE = "https://www.ebi.ac.uk"
QUERY = {
    "target_chembl_id": "CHEMBL240",
    "standard_type": "IC50",
    "standard_units": "nM",
    "limit": 1000,
}


def request(url):
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read()
        except (TimeoutError, urllib.error.URLError):
            if attempt == 3:
                raise
            time.sleep(2**attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/chembl-herg"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    truststore.inject_into_ssl()
    status = request(BASE + "/chembl/api/data/status.json")
    (args.output / "status.json").write_bytes(status)
    target = request(BASE + "/chembl/api/data/target/CHEMBL240.json")
    (args.output / "target.json").write_bytes(target)
    url = BASE + "/chembl/api/data/activity.json?" + urllib.parse.urlencode(QUERY)
    rows = []
    pages = []
    while url:
        blob = request(url)
        content = json.loads(blob)
        index = len(pages)
        path = args.output / f"page-{index:03d}.json"
        path.write_bytes(blob)
        pages.append(
            {
                "url": url,
                "sha256": hashlib.sha256(blob).hexdigest(),
                "file": path.name,
                "rows": len(content["activities"]),
            }
        )
        rows.extend(content["activities"])
        print(
            "downloaded",
            len(rows),
            "/",
            content["page_meta"]["total_count"],
            flush=True,
        )
        next_page = content["page_meta"]["next"]
        url = BASE + next_page if next_page else None
        if next_page:
            time.sleep(0.2)
    with (args.output / "activities.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    manifest = {
        "query": QUERY,
        "status": json.loads(status),
        "target": json.loads(target)["pref_name"],
        "pages": pages,
        "rows": len(rows),
        "license": "ChEMBL data: CC BY-SA 3.0; derived tables retain attribution/share-alike.",
        "note": "Independent IC50 assay domain, not peptide MIC replication.",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
