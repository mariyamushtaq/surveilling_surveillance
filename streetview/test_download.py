"""
Download a small test batch of Street View images from philly_meta.csv.

Usage (unsigned, API key only):
    python -m streetview.test_download --key YOUR_API_KEY

Usage (signed, with secret):
    python -m streetview.test_download --key YOUR_API_KEY --sec YOUR_SIGNING_SECRET

Optional:
    --n          Number of images to download (default: 20)
    --meta_path  Path to metadata CSV (default: data/input_metadata/philly_meta.csv)
    --save_dir   Where to save images (default: ./data/rawdata/image_test)
"""

import os
import fire
import pandas as pd
import requests
import hashlib
import hmac
import base64
import urllib.parse as urlparse
from tqdm import tqdm

from util import constants as C


def _build_url(panoid, heading, key, sec=None):
    """Build a GSV Static API URL, optionally signed."""
    url_str = (f"https://maps.googleapis.com/maps/api/streetview?"
               f"size={C.SV_SIZE}&pano={panoid}&fov={C.SV_FOV}&"
               f"heading={heading}&pitch={C.SV_PITCH}&key={key}")

    if sec:
        parsed = urlparse.urlparse(url_str)
        url_to_sign = parsed.path + "?" + parsed.query
        decoded_key = base64.urlsafe_b64decode(sec)
        signature = hmac.new(decoded_key,
                             url_to_sign.encode(),
                             hashlib.sha1)
        encoded_sig = base64.urlsafe_b64encode(signature.digest()).decode()
        url_str += "&signature=" + encoded_sig

    return url_str


def test_download(key,
                  sec=None,
                  n=20,
                  meta_path="data/input_metadata/philly_meta.csv",
                  save_dir="./data/rawdata/image_test"):
    """Download *n* test images and report results."""
    df = pd.read_csv(meta_path)
    sample = df.head(n)

    os.makedirs(save_dir, exist_ok=True)
    print(f"Downloading {n} test images to {save_dir}/ …")

    ok = 0
    errors = []
    for _, row in tqdm(sample.iterrows(), total=n):
        panoid = row["panoid"]
        heading = row["heading"]
        url = _build_url(panoid, heading, key, sec)
        filename = f"{panoid}_{heading}.jpg"
        filepath = os.path.join(save_dir, filename)

        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200 and len(resp.content) > 1000:
                with open(filepath, "wb") as f:
                    f.write(resp.content)
                ok += 1
            else:
                errors.append((panoid, resp.status_code, len(resp.content)))
        except Exception as e:
            errors.append((panoid, "ERROR", str(e)))

    print(f"\n✅ Downloaded: {ok}/{n}")
    if errors:
        print(f"❌ Failed: {len(errors)}")
        for panoid, status, detail in errors:
            print(f"   {panoid}: status={status}, detail={detail}")
    print(f"\nImages saved to: {os.path.abspath(save_dir)}/")


if __name__ == "__main__":
    fire.Fire(test_download)
