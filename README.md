# Surveilling Surveillance

**Estimating the Prevalence of Surveillance Cameras with Street View Data**

[Project page](https://policylab.stanford.edu/surveillance/) | [Paper](https://arxiv.org/abs/2105.01764)

This repository extends the original [Surveilling Surveillance](https://arxiv.org/abs/2105.01764) paper (Sheng et al., 2021) with:

- **Philadelphia case study** — city-wide camera detection from Street View imagery
- **ALPR integration** — detection at known license plate reader locations  
- **Demographic analysis** — zero-inflated Poisson regression against census block-group data
- **Anonymity penalty** — routing analysis measuring the cost of avoiding surveillance

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
- R + tidyverse, sf, pscl (for analysis)

### Installation

```bash
pip install -r requirements.txt
```

### Download Model

Download the [FasterRCNN model](https://storage.googleapis.com/scpl-surveillance/model.zip) (472 MB) and extract to `detection/model/`.

---

## Repository Structure

```
├── detection/           # FasterRCNN detection model
├── streetview/          # Street View download & sampling
├── scripts/             # Pipeline runners
├── analysis/            # R scripts & outputs
│   ├── results_philly_combined_revision.Rmd
│   ├── pull_philly_demographics.R
│   └── output/
├── alpr_data/           # ALPR camera pipeline
├── anonymity_penalty/   # Surveillance-aware routing
├── outputs/             # Result figures
├── data/                # Metadata (images not included)
├── plot/                # Visualization modules
└── util/                # Shared constants
```

> **Note:** Street View images are not included. Run pipelines with your own Google API key.

---

## Camera Detection

### Download Images

```bash
python main.py download_streetview_image --key YOUR_API_KEY --sec YOUR_SIGNING_SECRET
```

### Train

```bash
cd detection && python main.py train --exp_name EXPERIMENT_NAME
```

### Inference

```bash
cd detection && python main.py test --deploy --deploy_meta_path PATH_TO_META.csv
```

---

## Philadelphia Analysis

### 1. ALPR Pipeline

Fetch ALPR locations and run detection:

```bash
python -m alpr_data.fetch_alpr_data
python -m alpr_data.run_alpr_pipeline --key YOUR_API_KEY
```

See [`alpr_data/README.md`](alpr_data/README.md) for details.

### 2. Demographic Regression

Model camera prevalence against census demographics (ZIP regression):

```bash
export CENSUS_API_KEY='your_key'
Rscript analysis/pull_philly_demographics.R
Rscript -e "rmarkdown::render('analysis/results_philly_combined_revision.Rmd')"
```

### 3. Anonymity Penalty

Compute routing cost of avoiding cameras:

```bash
python anonymity_penalty/anonymity_penalty.py --radius 50 --lam 500 --n_pairs 100
```

---

## Original Paper

Reproduce figures from Sheng et al. (2021):

```bash
Rscript -e "rmarkdown::render('analysis/results.Rmd')"
```

Download [camera-data.zip](https://storage.googleapis.com/scpl-surveillance/camera-data.zip) (97 MB) to `data/`.

---

## Downloads

| Resource | Size | Link |
|----------|------|------|
| Annotations | — | [meta.csv](https://storage.googleapis.com/scpl-surveillance/meta.csv) |
| Pre-trained model | 472 MB | [model.zip](https://storage.googleapis.com/scpl-surveillance/model.zip) |
| Detection data | 97 MB | [camera-data.zip](https://storage.googleapis.com/scpl-surveillance/camera-data.zip) |

---

## License

[MIT](LICENSE)
