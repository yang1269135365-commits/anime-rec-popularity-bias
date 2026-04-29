# Aesthetic Deviation Residuals for De-biasing Anime Recommendation

**Stratified Factorization Machines on MyAnimeList 2023**

DSCI 4093 Senior Project · Spring 2026
Yaocheng Yang · Advisor: Prof. Jaideep Srivastava
DMR / MCAL Lab, University of Minnesota Twin Cities

---

## TL;DR

This study addresses popularity bias in collaborative-filtering recommendation systems on the MyAnimeList 2023 dataset (13.6M ratings · 221K users · 15.5K titles).

Two independent interventions are tested in a fully-crossed design:
- **User stratification** — partition users into Casual / Intermediate / Veteran by rating-count quantile
- **Target reformulation** — predict aesthetic deviation residual Δr = r − r_official instead of r directly

### Key findings

1. **Popularity bias scales with user seniority** — Naive model ARP bias on Veterans is +0.115, ~35× the Casual rate (+0.003). Stable across data scales and architectures.
2. **Residual prediction is the most effective lever** — global ARP −0.063, Coverage +0.22 → +0.30. Far exceeds the largest stratification gain (+0.026).
3. **FM ≈ MF on Top-K** — MF wins RMSE by 0.030–0.078 but Top-K metrics differ by < 0.01. FM's value is architectural (admits side features and Δr targets).
4. **DeepFM doesn't help at this scale** — bottleneck is data, not capacity.

See [`paper/main.pdf`](paper/main.pdf) for the full writeup.

## Repository Structure

- `paper/` — Final paper (PDF, ~40 pages)
- `reports/` — Stage-by-stage progress reports
  - `Semantics, Taste, and Propagation  .docx.pdf` — Initial proposal (Dec 2025)
  - `stage1 report_ (4).pdf` — Phase 1: EDA & dataset selection
  - `stage_2_report (3).pdf` — Phase 2: Theoretical framework, FM architecture, residual formalization
  - `Stage3 report (2).docx` — Phase 3: Three ablation experiments on the 5% sample
  - `stage 4 report (1).docx` — Phase 4: DeepFM and 50% scale verification
- `notebook/` — Experiment notebooks
  - `01_eda.ipynb` — Exploratory data analysis on Datasets A (2023) and B (2020)
  - `02_feature_baseline.ipynb` — Feature engineering & FM/MF baseline comparison
  - `03_deepfm_initial.ipynb` — DeepFM on the 5% sample
  - `04_deepfm_full_train.ipynb` — Full Phase-4 training at 50% scale
- `prototype/` — Early interactive Web UI prototype (`prototype.html`)

## Dataset

MyAnimeList 2023 from Kaggle:
https://www.kaggle.com/datasets/dsfelix/animes-dataset-2023

The dataset is not redistributed in this repository. To reproduce, download from the link above and place the CSVs locally; the notebooks expect three files: `anime-transformed-dataset-2023.csv`, `users-details-transformed-2023.csv`, and `users-scores-transformed-2023.csv`.

## Reproducing

Notebooks were run on Google Colab (T4 GPU for 5% experiments, A100 for the 50% full-scale run).

Stack: Python 3.10 · PyTorch 2.x · DeepCTR-Torch 0.2.x · Surprise 1.1.x

Two random seeds: `random_state=42` (user-level sampling) and `random_state=2026` (train/test split). Test size 0.2 throughout.

## License

MIT — see [LICENSE](LICENSE).

## Contact

Yaocheng Yang · yang8252@umn.edu