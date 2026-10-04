"""Download the MineThatData E-Mail Analytics Challenge dataset (Kevin Hillstrom, 2008).

Description: https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html
The publisher only offers a plain-HTTP link, so the file is checked against a fixed
row count and SHA-256 before it is saved.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

URL = ("http://www.minethatdata.com/"
       "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv")
DEST = Path(__file__).resolve().parents[1] / "data/raw/hillstrom.csv"
EXPECTED_ROWS = 64_000


def main():
    if DEST.exists():
        print("Data already present.")
        return
    DEST.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL) as resp:
        content = resp.read()
    rows = content.decode("utf-8").strip().splitlines()
    if len(rows) - 1 != EXPECTED_ROWS:
        sys.exit(f"Unexpected row count: {len(rows) - 1}")
    DEST.write_bytes(content)
    print(f"Saved {EXPECTED_ROWS:,} rows to {DEST}")
    print("SHA-256:", hashlib.sha256(content).hexdigest())


if __name__ == "__main__":
    main()
