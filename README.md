# kgDrift: Modeling the Evolution of User Interests exploiting Summaries

> **kgDrift** is a deterministic, modular framework for query-log-driven knowledge graph summarization and temporal drift analysis. It transforms time-partitioned SPARQL query logs into interpretable trajectories of user attention over a KG, capturing distributional shifts, semantic drift, and structural topic events (emergence, disappearance, splitting, merging).

---

## Table of Contents

- [Overview](#overview)
- [Pipeline](#pipeline)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Input Format](#input-format)
- [Usage](#usage)
  - [Wikidata Benchmark](#wikidata-benchmark-seebenchmark_entities_embeddings2py)
  - [DBpedia Benchmark](#dbpedia-benchmark-benchmark_entities_with_density_fullpy)
- [CLI Reference](#cli-reference)
- [Output Files](#output-files)
- [Metrics](#metrics)
- [Reproducibility](#reproducibility)
- [Datasets](#datasets)
- [Citation](#citation)

---

## Overview

Understanding how user interests evolve over time in knowledge graph (KG) query workloads is key for search personalization, trend analysis, and system optimization. **kgDrift** addresses this by:

1. **Extracting** entities referenced in monthly SPARQL query-log windows.
2. **Embedding** them with a pluggable encoder (default: Sentence-BERT `all-MiniLM-L6-v2`).
3. **Clustering** the embeddings into semantically coherent themes (default: Ward agglomerative, k=11).
4. **Inducing** a quotient graph summary for each time window.
5. **Aligning** consecutive summaries via centroid cosine similarity and Hungarian matching.
6. **Quantifying** longitudinal drift through Jensen–Shannon divergence (distributional drift) and mean cosine distance between aligned centroids (semantic drift).

The best-performing configuration — **Ward agglomerative + Sentence-BERT (gtxt)** at k=11 — dominates structural and hybrid encoders on cluster quality and outperforms the SummaryGPT baseline on all 14 Wikidata month-by-metric comparisons (silhouette 0.088 vs. 0.011; coherence 0.343 vs. 0.271).

---

## Pipeline

```
Query Log Qt  →  Entity Extraction (Vt)  →  SBERT Embedding (Xt)
     →  Clustering (At / Ct)  →  Quotient Graph Summary (St)
     →  Centroid Similarity Matrix S(t)  →  Hungarian Alignment (π*)
     →  Distributional Drift (JSD)  +  Semantic Drift (cosine distance)
```

---

## Repository Structure

```
kgDRIFT/
│
├── SEEbenchmark_entities_embeddings2.py          # Wikidata benchmark script
├── benchmark_entities_with_density_full.py       # DBpedia benchmark script (+ density algorithms)
│
├── data/
│   ├── wikidata/
│   │   ├── M1.txt   # Monthly entity files (entity [tab/comma] label)
│   │   ├── M2.txt
│   │   └── ...      # M3 – M7
│   └── dbpedia/
│       ├── M1.txt
│       ├── M2.txt
│       ├── M3.txt
│       └── M4.txt
│
└── results/
    ├── wikidata_benchmark/
    └── dbpedia_benchmark/
```

---

## Requirements

- Python ≥ 3.9
- numpy
- pandas
- scikit-learn
- scipy
- sentence-transformers
- openpyxl
- hdbscan *(required only for `benchmark_entities_with_density_full.py` when using `--algorithms hdbscan`)*

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/kgDRIFT.git
cd kgDRIFT
pip install numpy pandas scikit-learn scipy sentence-transformers openpyxl hdbscan
```

---

## Input Format

Each monthly input file should be a plain text file with one entity per line. Two formats are accepted:

**Two-column (entity + label), any of: tab, comma, pipe, or 2+ spaces as delimiter:**
```
Paris    city
Berlin   city
Einstein scientist
```

**Single-column (entity id only — label is auto-duplicated from entity):**
```
Paris
Berlin
Einstein
```

Entity IDs with underscores (e.g. `steve_jobs`) are automatically cleaned to human-readable text (`Steve Jobs`) before SBERT encoding. Raw IDs are preserved in all outputs.

---

## Usage

### Wikidata Benchmark — `SEEbenchmark_entities_embeddings2.py`

Designed for the **7-month Wikidata** SPARQL query log (≈195 unique entities/month). Supports 5 clustering algorithms: `agglomerative`, `spectral`, `gmm`, `kmeans`, `dbscan`.

**Basic run (all algorithms, default k values {5, 8, 11, 15, 20, 25}):**

```bash
python SEEbenchmark_entities_embeddings2.py \
  --input-files data/wikidata/M1.txt data/wikidata/M2.txt data/wikidata/M3.txt \
                data/wikidata/M4.txt data/wikidata/M5.txt data/wikidata/M6.txt \
                data/wikidata/M7.txt \
  --outdir results/wikidata_benchmark
```

**Run with the recommended configuration (Ward, k=11):**

```bash
python SEEbenchmark_entities_embeddings2.py \
  --input-files data/wikidata/M1.txt data/wikidata/M2.txt data/wikidata/M3.txt \
                data/wikidata/M4.txt data/wikidata/M5.txt data/wikidata/M6.txt \
                data/wikidata/M7.txt \
  --outdir results/wikidata_ward_k11 \
  --algorithms agglomerative \
  --k-values 11 \
  --agglomerative-linkage ward \
  --seed 42
```

**Run a subset of algorithms with plots:**

```bash
python SEEbenchmark_entities_embeddings2.py \
  --input-files data/wikidata/M1.txt ... data/wikidata/M7.txt \
  --outdir results/wikidata_subset \
  --algorithms agglomerative kmeans \
  --k-values 5 11 20 \
  --plot
```

---

### DBpedia Benchmark — `benchmark_entities_with_density_full.py`

Designed for the **4-month DBpedia** SPARQL query log (≈210 unique entities/month). Extends the Wikidata script with two additional density-based algorithms: `pure_dbscan` (fixed eps, no target k) and `hdbscan` (automatic density clustering).

**Basic run (all algorithms including density methods):**

```bash
python benchmark_entities_with_density_full.py \
  --input-files data/dbpedia/M1.txt data/dbpedia/M2.txt \
                data/dbpedia/M3.txt data/dbpedia/M4.txt \
  --outdir results/dbpedia_benchmark \
  --algorithms agglomerative spectral gmm kmeans dbscan pure_dbscan hdbscan
```

**Run with recommended configuration (Ward, k=11):**

```bash
python benchmark_entities_with_density_full.py \
  --input-files data/dbpedia/M1.txt data/dbpedia/M2.txt \
                data/dbpedia/M3.txt data/dbpedia/M4.txt \
  --outdir results/dbpedia_ward_k11 \
  --algorithms agglomerative \
  --k-values 11 \
  --agglomerative-linkage ward \
  --seed 42
```

**Run density-only algorithms:**

```bash
python benchmark_entities_with_density_full.py \
  --input-files data/dbpedia/M1.txt data/dbpedia/M2.txt \
                data/dbpedia/M3.txt data/dbpedia/M4.txt \
  --outdir results/dbpedia_density \
  --algorithms pure_dbscan hdbscan \
  --pure-dbscan-eps 0.25 \
  --hdbscan-min-cluster-size 5
```

---

## CLI Reference

Both scripts share most arguments. Arguments marked with † are available only in `benchmark_entities_with_density_full.py`.

### Core Arguments

| Argument | Default | Description |
|---|---|---|
| `--input-files` | *(required)* | One or more monthly entity files in order (M1, M2, …) |
| `--outdir` | `benchmark_results` | Root output directory |
| `--model` | `all-MiniLM-L6-v2` | Sentence-BERT model name (HuggingFace) |
| `--algorithms` | all | Algorithms to run (see lists below) |
| `--k-values` | `5 8 11 15 20 25` | Cluster counts to sweep (ignored for density methods) |
| `--seed` | `42` | Global random seed |
| `--plot` | off | Generate per-run plots |

### Available Algorithms

`SEEbenchmark_entities_embeddings2.py`: `agglomerative`, `spectral`, `gmm`, `kmeans`, `dbscan`

`benchmark_entities_with_density_full.py`: all of the above + `pure_dbscan`†, `hdbscan`†

### Agglomerative Options

| Argument | Default | Description |
|---|---|---|
| `--agglomerative-linkage` | `ward` | Linkage: `ward`, `average`, `complete`, `single` |
| `--agglomerative-metric` | `euclidean` | Distance metric (ward requires euclidean) |

### Spectral Options

| Argument | Default | Description |
|---|---|---|
| `--spectral-affinity` | `nearest_neighbors` | Affinity type |
| `--spectral-n-neighbors` | `10` | Number of nearest neighbors |
| `--spectral-assign-labels` | `kmeans` | Label assignment: `kmeans`, `discretize`, `cluster_qr` |

### GMM Options

| Argument | Default | Description |
|---|---|---|
| `--gmm-covariance-type` | `full` | Covariance type: `full`, `tied`, `diag`, `spherical` |
| `--gmm-n-init` | `5` | Number of initializations |
| `--gmm-max-iter` | `300` | Maximum EM iterations |
| `--gmm-reg-covar` | `1e-6` | Regularization added to covariance diagonal |

### K-Means Options

| Argument | Default | Description |
|---|---|---|
| `--kmeans-n-init` | `auto` | Number of initializations (integer or `auto`) |
| `--kmeans-max-iter` | `300` | Maximum iterations |

### Target-k DBSCAN Options

| Argument | Default | Description |
|---|---|---|
| `--dbscan-min-samples` | `5` | Minimum samples for core point |
| `--dbscan-metric` | `cosine` | Distance metric |
| `--dbscan-eps-min` | `0.01` | Lower bound of eps search range |
| `--dbscan-eps-max` | `1.00` | Upper bound of eps search range |
| `--dbscan-eps-steps` | `100` | Number of eps values to try |
| `--dbscan-include-noise-as-cluster` | off | Treat noise points as a single extra cluster |

### Pure DBSCAN Options †

| Argument | Default | Description |
|---|---|---|
| `--pure-dbscan-eps` | `0.25` | Fixed eps (no target-k search) |
| `--pure-dbscan-min-samples` | `5` | Minimum samples for core point |
| `--pure-dbscan-metric` | `cosine` | Distance metric |
| `--pure-dbscan-include-noise-as-cluster` | off | Treat noise as an extra cluster |

### HDBSCAN Options †

| Argument | Default | Description |
|---|---|---|
| `--hdbscan-min-cluster-size` | `5` | Minimum cluster size |
| `--hdbscan-min-samples` | `None` | Minimum samples (defaults to `min_cluster_size`) |
| `--hdbscan-metric` | `euclidean` | Distance metric (use euclidean on ℓ2-normalized embeddings for cosine-equivalent behavior) |
| `--hdbscan-include-noise-as-cluster` | off | Treat noise as an extra cluster |

---

## Output Files

```
<outdir>/
├── benchmark_summary.csv               # Mean metrics per (algorithm, k) across all months
├── benchmark_summary.xlsx              # Same as above + all-month sheet
├── benchmark_all_month_metrics.csv     # Per-month metrics for every (algorithm, k) run
│
├── agglomerative_k11/
│   ├── M1_cluster_map.csv              # Entity → cluster_id mapping for month M1
│   ├── M1_cluster_summary.csv          # Per-cluster stats for M1
│   ├── month_quality_metrics.csv       # Silhouette, coherence, etc. per month
│   ├── cluster_drift_pair_summary.csv  # JSD + semantic drift per consecutive month pair
│   └── cluster_drift_alignment_long.csv # Per-cluster alignment details (Hungarian matching)
│
├── kmeans_k11/
│   └── ...
└── ...
```

`pure_dbscan/` and `hdbscan/` (DBpedia script only) follow the same internal structure without a `_k<k>` suffix.

---

## Metrics

### Geometric Quality (per window)

| Metric | Range | Description |
|---|---|---|
| Silhouette | [−1, 1] ↑ | Cosine silhouette computed in SBERT embedding space |
| Coherence | [0, 1] ↑ | Size-weighted mean intra-cluster pairwise cosine similarity |

### Distributional Structure (per window, DBpedia script)

| Metric | Description |
|---|---|
| Effective k (entropy) | exp(Shannon entropy of cluster mass distribution) |
| Inverse Simpson effective k | 1 / Σ p² — a diversity-based cluster count |
| Gini coefficient | Size inequality across clusters (0 = uniform, 1 = monopoly) |
| Small-cluster ratio | Fraction of clusters with mass < threshold |
| Largest-cluster fraction | Mass share of the dominant cluster |

### Longitudinal Drift (per consecutive window pair)

| Metric | Range | Description |
|---|---|---|
| JSD (distributional drift) | [0, log 2] ↓ | Jensen–Shannon divergence between aligned cluster mass vectors |
| Semantic drift | [0, 2] ↓ | Mean cosine distance between aligned centroids after Hungarian matching |

Reporting **both** metrics jointly is essential: JSD-stable transitions can still hide substantial content reorganization detectable only through semantic drift (observed on DBpedia: JSD=0.039 with drift=0.625).

---

## Reproducibility

Both scripts are fully deterministic when `--seed 42` is set (default). Embeddings are computed once and cached in memory across all algorithm/k combinations to ensure consistent results and minimize runtime.

The SentenceTransformer model (`all-MiniLM-L6-v2`) is downloaded automatically from HuggingFace on first run and cached locally.

All code and data are available at: [https://anonymous.4open.science/r/kgDRIFT-B9CD](https://anonymous.4open.science/r/kgDRIFT-B9CD)

---

## Datasets

| Dataset | Windows | Entities/month | Source |
|---|---|---|---|
| Wikidata | 7 (M1–M7) | ≈195 | [Malyshev et al., ISWC 2018](https://doi.org/10.1007/978-3-030-00668-6_23) |
| DBpedia | 4 (M1–M4) | ≈210 | [Saleem et al., ISWC 2015 (LSQ)](https://doi.org/10.1007/978-3-319-25010-6_15) |

---

## Citation

If you use kgDrift in your research, please cite:

```bibtex
@inproceedings{kgdrift2026,
  title     = {kgDRIFT: Modeling the Evolution of User Interests exploiting Summaries},
  booktitle = {Proceedings of the 25th International Semantic Web Conference (ISWC 2026)},
  year      = {2026}
}
```
