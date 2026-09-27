# Modeling the Network of Urban Surveillance Infrastructure in Philadelphia

This project maps where surveillance cameras are in Philadelphia, checks whether they cluster in certain neighborhoods, and measures how far out of your way you'd have to walk to avoid them.

## Motivation

This work started from *Surveilling Surveillance: Estimating the Prevalence of Surveillance Cameras with Street View Data* (Sheng, Yao & Goel, 2021), which used Google Street View images and a detection model to count cameras across several major cities. I wanted to see what that method finds when applied to a single city in depth, and what happens when you push past counting into questions about who is being watched and whether it's possible to move through the city without being seen.

[Original project page](https://policylab.stanford.edu/surveillance/) | [Original paper](https://arxiv.org/abs/2105.01764)

```bibtex
@article{sheng2021surveilling,
  title={Surveilling Surveillance: Estimating the Prevalence of Surveillance Cameras with Street View Data},
  author={Sheng, Hao and Yao, Keniel and Goel, Sharad},
  journal={Artificial Intelligence, Ethics, and Society},
  year={2021}
}
```

## What I Added

**Philadelphia camera detection.** I sampled 9,810 Street View panoramas along the city's road network and ran them through a Faster R-CNN detector (Detectron2). At a confidence cutoff of 0.4, the model flagged 629 cameras: about 37% directed cameras and 63% dome cameras. I picked 0.4 on the cautious side, so these counts are best read as an upper bound.

**License plate reader (ALPR) check.** I pulled 103 known ALPR locations from OpenStreetMap and DeFlock and ran the same detector on Street View images at those sites. It found cameras at 9 of them (10 detections total). The low hit rate shows how hard these devices are to spot from street-level imagery, and it's a useful reality check on the detector overall.

**Demographic analysis.** I matched camera counts to 1,240 census block groups using American Community Survey data (race, income, poverty, and similar measures) and fit a zero-inflated Poisson regression, a model built for count data with lots of zeros. Most block groups (851) had no detected cameras. None of the demographic variables were statistically significant predictors of camera count.

**Anonymity penalty.** Using the city's street centerline data, I built a walking network and compared the shortest route between two points to a route that tries to avoid cameras. Each street segment is penalized based on the detection confidence of cameras within 50 m of it, so a street with several high-confidence cameras costs more than one with a single uncertain detection. Routes are found with Dijkstra's algorithm. Across 99 origin-destination pairs, the median detour was about 164 m, roughly 1% longer than the shortest path.

---

## Quick Start

### Requirements

- Python ≥ 3.6 (macOS or Linux)
- [PyTorch](https://pytorch.org/) ≥ 1.6 + [torchvision](https://github.com/pytorch/vision/)
- [Detectron2](https://github.com/facebookresearch/detectron2)
- NetworkX (for routing)
- R + tidyverse, sf, pscl, tidycensus (for analysis)

### Installation

```bash
pip install -r requirements.txt
```

### Download Model

Download the [Faster R-CNN model](https://storage.googleapis.com/scpl-surveillance/model.zip) (472 MB) and extract to `detection/model/`.

---

## Repository Structure

```
├── detection/           # Faster R-CNN detection model
├── streetview/          # Street View download & sampling
├── scripts/             # Pipeline runners
├── analysis/            # R scripts & outputs
│   ├── results_philly_combined_revision.Rmd
│   ├── pull_philly_demographics.R
│   └── output/
├── alpr_data/           # ALPR camera pipeline
├── anonymity_penalty/   # Camera-avoiding routing
├── outputs/             # Result figures
├── data/                # Metadata (images not included)
├── plot/                # Visualization modules
└── util/                # Shared constants
```

> **Note:** Street View images are not included. Run the pipelines with your own Google API key.

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

Fetch ALPR locations and run detection on them:

```bash
python -m alpr_data.fetch_alpr_data
python -m alpr_data.run_alpr_pipeline --key YOUR_API_KEY
```

See [`alpr_data/README.md`](alpr_data/README.md) for details.

### 2. Demographic Regression

Pull census data and fit the zero-inflated Poisson model:

```bash
export CENSUS_API_KEY='your_key'
Rscript analysis/pull_philly_demographics.R
Rscript -e "rmarkdown::render('analysis/results_philly_combined_revision.Rmd')"
```

### 3. Anonymity Penalty

Compare shortest routes to camera-avoiding routes:

```bash
python anonymity_penalty/anonymity_penalty.py --radius 50 --lam 500 --n_pairs 100
```

`--radius` is how close (in meters) a camera has to be to count against a street, `--lam` controls how strongly cameras are penalized relative to distance, and `--n_pairs` is how many random origin-destination pairs to test.

---

## Reproducing the Original Paper

To regenerate figures from Sheng et al. (2021), download [camera-data.zip](https://storage.googleapis.com/scpl-surveillance/camera-data.zip) (97 MB) to `data/`, then run:

```bash
Rscript -e "rmarkdown::render('analysis/results.Rmd')"
```

---

## Downloads

| Resource | Size | Link |
|----------|------|------|
| Annotations | n/a | [meta.csv](https://storage.googleapis.com/scpl-surveillance/meta.csv) |
| Pre-trained model | 472 MB | [model.zip](https://storage.googleapis.com/scpl-surveillance/model.zip) |
| Detection data | 97 MB | [camera-data.zip](https://storage.googleapis.com/scpl-surveillance/camera-data.zip) |

---

## License

[MIT](LICENSE)