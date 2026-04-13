# ALPR Data Pipeline

Retrieves Automated License Plate Reader (ALPR) camera locations from crowdsourced databases, downloads Google Street View imagery at those locations, and runs the surveillance camera detection model on the images.

## Pipeline Overview

### Step 1: Fetch ALPR Locations (`fetch_alpr_data.py`)

Pulls ALPR camera locations from two sources and merges them:

| Source | Method | Output |
|--------|--------|--------|
| **OpenStreetMap** | Overpass API query for ALPR/surveillance tags | `data/philly_alpr_osm.csv` |
| **DeFlock / ALPRWatch** | KMZ download from alprwatch.org, filtered to Philly bbox | `data/philly_alpr_deflock.csv` |
| **Combined** | Deduplicated by rounding lat/lon to 4 decimals (~11 m) | `data/philly_alpr_combined.csv` |

```bash
# Fetch from both sources
python -m alpr_data.fetch_alpr_data

# Skip one source if needed
python -m alpr_data.fetch_alpr_data --skip_osm
python -m alpr_data.fetch_alpr_data --skip_deflock
```

### Step 2: Download & Detect (`run_alpr_pipeline.py`)

Takes the combined ALPR locations, fetches Street View imagery, and runs detection:

1. **Load locations** from `data/philly_alpr_combined.csv`
2. **Query GSV Metadata API** to find panorama IDs near each ALPR location
3. **Download images** — 4 headings per panorama (0, 90, 180, 270) for full coverage
4. **Run FasterRCNN detection** on all downloaded images
5. **Save results** — annotated images, per-image JSON, summary CSV, and a per-location detection summary

```bash
# Full pipeline
python -m alpr_data.run_alpr_pipeline --key YOUR_API_KEY

# With options
python -m alpr_data.run_alpr_pipeline \
    --key YOUR_API_KEY \
    --ckpt_path detection/model/best.ckpt \
    --conf_threshold 0.3 \
    --device cpu \
    --n_images 40

# Re-run detection only (skip download)
python -m alpr_data.run_alpr_pipeline --key YOUR_API_KEY --skip_download
```

## Output Files

```
alpr_data/
├── data/
│   ├── philly_alpr_osm.csv          # OSM ALPR locations
│   ├── philly_alpr_deflock.csv      # DeFlock ALPR locations
│   ├── philly_alpr_combined.csv     # Merged & deduplicated locations
│   ├── alpr_meta.csv                # GSV metadata (panoid, heading, coords)
│   ├── detection_summary.csv        # All detections across all images
│   ├── alpr_location_detections.csv # Per-ALPR-location detection counts
│   ├── images/                      # Downloaded Street View images
│   └── detections/                  # Annotated images + per-image JSON
├── fetch_alpr_data.py
├── run_alpr_pipeline.py
└── README.md
```

## Data Format

**philly_alpr_combined.csv** columns:
- `lat`, `lon` — camera coordinates
- `source` — `osm` or `deflock`
- `name` — placemark name (DeFlock only)

**alpr_location_detections.csv** columns:
- `panoid` — GSV panorama ID
- `lat`, `lon` — panorama coordinates
- `lat_alpr`, `lon_alpr` — original ALPR camera coordinates
- `source` — data source
- `n_detections` — number of cameras detected in imagery
- `max_score` — highest detection confidence
- `classes` — detected camera types (Directed Camera, Dome Camera)
