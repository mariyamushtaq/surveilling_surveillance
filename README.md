# Surveilling Surveillance: Estimating the Prevalence of Surveillance Cameras with Street View Data

### [Project page](https://policylab.stanford.edu/surveillance/) | [Paper](https://arxiv.org/abs/2105.01764)

![detections](.github/image/detections.png)
*Locations of verified cameras in 10 large U.S. cities for the period 2016–2020.*

This repository extends [Surveilling Surveillance](https://arxiv.org/abs/2105.01764) (Sheng et al., 2021) with Philadelphia-specific analysis, ALPR camera detection, demographic regression, and anonymity penalty routing.

```bibtex
@article{sheng2021surveilling,
  title={Surveilling Surveillance: Estimating the Prevalence of Surveillance Cameras with Street View Data},
  author={Sheng, Hao and Yao, Keniel and Goel, Sharad},
  journal={Artificial Intelligence, Ethics, and Society},
  year={2021}
}
```

---

## Quick Start

### Requirements

- Python ≥ 3.6 (macOS or Linux)
- [PyTorch](https://pytorch.org/) ≥ 1.6 + [torchvision](https://github.com/pytorch/vision/)
- [Detectron2](https://github.com/facebookresearch/detectron2)
- R + tidyverse, sf, pscl (for analysis scripts)

### Installation

```bash
pip install -r requirements.txt
```

### Download Pre-trained Model

Download the [FasterRCNN model](https://storage.googleapis.com/scpl-surveillance/model.zip) (472 MB) and extract to `detection/model/`.

---

## Repository Structure

```
├── detection/           # FasterRCNN camera detection model
├── streetview/          # Street View image download & sampling
├── scripts/             # Pipeline runners
├── analysis/            # R analysis scripts
│   ├── results_philly_combined_revision.Rmd   # Philadelphia regression
│   ├── pull_philly_demographics.R             # Census data pipeline
│   └── output/                                # Generated figures
├── alpr_data/           # ALPR camera detection pipeline
├── anonymity_penalty/   # Surveillance-aware routing
├── outputs/             # Result figures
├── data/                # Metadata & results (images not included)
└── plot/                # Visualization modules
```

**Note:** Street View images and large data files are not included. Run the pipelines with your own Google API key to download imagery.

---

## Camera Detection

### Download Street View Images

```bash
python main.py download_streetview_image --key YOUR_API_KEY --sec YOUR_SIGNING_SECRET
```

### Train Model

```bash
cd detection && python main.py train --exp_name EXPERIMENT_NAME
```

### Run Inference

```bash
cd detection && python main.py test --deploy --deploy_meta_path PATH_TO_META.csv
```

---

## Philadelphia Analysis

### 1. ALPR Data Pipeline

Fetch ALPR camera locations from crowdsourced databases and run detection:

```bash
python -m alpr_data.fetch_alpr_data
python -m alpr_data.run_alpr_pipeline --key YOUR_API_KEY
```

### 2. Demographic Regression

Model camera prevalence against census block-group demographics using zero-inflated Poisson regression:

```bash
# Set Census API key
export CENSUS_API_KEY='your_key'

# Pull demographics
Rscript analysis/pull_philly_demographics.R

# Run analysis
Rscript -e "rmarkdown::render('analysis/results_philly_combined_revision.Rmd')"
```

Outputs: maps, bivariate plots, ZIP regression tables in `analysis/output/`.

### 3. Anonymity Penalty

Compute the cost of avoiding surveillance when routing:

```bash
python anonymity_penalty/anonymity_penalty.py --radius 50 --lam 500 --n_pairs 100
```

Outputs: `anonymity_penalty/routing_output/`

---

## Original Paper Analysis

Reproduce figures from the original paper:

```bash
Rscript -e "rmarkdown::render('analysis/results.Rmd')"
```

Download [camera-data.zip](https://storage.googleapis.com/scpl-surveillance/camera-data.zip) (97 MB) into `data/`.

---

## Data & Artifacts

| Resource | Size | Link |
|----------|------|------|
| Camera annotations | — | [meta.csv](https://storage.googleapis.com/scpl-surveillance/meta.csv) |
| Pre-trained model | 472 MB | [model.zip](https://storage.googleapis.com/scpl-surveillance/model.zip) |
| Detection + road data | 97 MB | [camera-data.zip](https://storage.googleapis.com/scpl-surveillance/camera-data.zip) |

---

## License

See [LICENSE](LICENSE).
