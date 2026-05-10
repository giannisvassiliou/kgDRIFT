#!/usr/bin/env python3
"""
Single-pass clustering benchmark across algorithms and k values
==============================================================

This optimized script loads SentenceTransformer ONCE, encodes each monthly file ONCE,
then runs all requested clustering algorithms and k values on the cached embeddings.

Entity cleaning for SBERT
-------------------------
Raw entity ids are preserved in outputs, but SBERT encodes cleaned text:
    steve_jobs -> Steve Jobs

Algorithms
----------
- agglomerative
- spectral
- gmm
- kmeans
- dbscan        : target-k DBSCAN, searches eps to approximate requested k
- pure_dbscan   : pure DBSCAN, no requested k
- hdbscan       : automatic density clustering, no requested k

Default k values
----------------
- 5, 8, 11, 15, 20, 25

Outputs
-------
<outdir>/
- benchmark_summary.csv
- benchmark_summary.xlsx
- benchmark_all_month_metrics.csv

Per algorithm/k folder:
- <algorithm>_k<k>/ for k-based methods
- pure_dbscan/ for pure DBSCAN
- hdbscan/ for HDBSCAN

Each run folder contains:
- <month>_cluster_map.csv
- <month>_cluster_summary.csv
- month_quality_metrics.csv
- cluster_drift_pair_summary.csv
- cluster_drift_alignment_long.csv
- optional plots

Example
-------
python benchmark_entities_with_density_full.py \
  --input-files TESToutAI1.txt TESToutAI2.txt TESToutAI3.txt \
  --outdir benchmark_results \
  --algorithms agglomerative spectral gmm kmeans dbscan pure_dbscan hdbscan
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd


ALGORITHMS = [
    "agglomerative",
    "spectral",
    "gmm",
    "kmeans",
    "dbscan",
    "pure_dbscan",
    "hdbscan",
]

K_VALUES = [5, 8, 11, 15, 20, 25]


# -----------------------------
# Robust 2-column file reading
# -----------------------------
def detect_delimiter(sample: str) -> str:
    if "\t" in sample:
        return "\t"
    if "," in sample:
        return ","
    if "|" in sample:
        return "|"
    return ""


def read_two_columns(path: str, sep: Optional[str] = None) -> List[Tuple[str, str]]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    if not lines:
        raise ValueError(f"Input file is empty: {path}")

    rows: List[Tuple[str, str]] = []

    for ln in lines:
        parts = re.split(r"\t|,|\||\s{2,}", ln.strip())

        if len(parts) >= 2:
            entity = parts[0].strip()
            label = parts[1].strip()
        else:
            entity = ln.strip()
            label = entity

        rows.append((entity, label))

    return rows


def clean_entity_for_sbert(entity: str) -> str:
    """Convert entity ids into human-readable text before SBERT encoding.

    Examples
    --------
    steve_jobs -> Steve Jobs
    new_york_city -> New York City
    """
    return entity.replace("_", " ").strip().title()


def month_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    return stem


def ensure_dir(p: Path | str) -> None:
    os.makedirs(p, exist_ok=True)


# -----------------------------
# Embedding helpers
# -----------------------------
def l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms


def cosine_similarity_matrix(A_unit: np.ndarray, B_unit: np.ndarray) -> np.ndarray:
    return A_unit @ B_unit.T


def cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    un = np.linalg.norm(u) + 1e-12
    vn = np.linalg.norm(v) + 1e-12
    return float(1.0 - (u @ v) / (un * vn))


# -----------------------------
# Quality / coherence metrics
# -----------------------------
def mean_pairwise_cosine_similarity(emb_unit: np.ndarray) -> float:
    n = emb_unit.shape[0]
    if n < 2:
        return float("nan")
    S = emb_unit @ emb_unit.T
    tri = np.triu_indices(n, k=1)
    return float(S[tri].mean())


def compute_cluster_coherences(
    emb_unit: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_ids: np.ndarray,
) -> np.ndarray:
    out = np.full((len(cluster_ids),), np.nan, dtype=np.float64)
    for i, cid in enumerate(cluster_ids):
        idx = np.where(cluster_labels == cid)[0]
        if len(idx) < 2:
            continue
        out[i] = mean_pairwise_cosine_similarity(emb_unit[idx])
    return out


def compute_cluster_size_metrics(counts: np.ndarray, small_cluster_threshold: int) -> dict:
    """Compute metrics that describe whether clusters have meaningful sizes.

    These metrics complement silhouette/coherence by measuring the cluster-size
    distribution rather than semantic tightness.
    """
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())

    if total <= 0 or len(counts) == 0:
        return {
            "cluster_size_entropy": float("nan"),
            "cluster_size_entropy_normalized": float("nan"),
            "effective_k_entropy": float("nan"),
            "simpson_diversity": float("nan"),
            "inverse_simpson_effective_k": float("nan"),
            "cluster_size_gini": float("nan"),
            "small_cluster_count": 0,
            "small_cluster_ratio": float("nan"),
            "mean_cluster_size": float("nan"),
            "median_cluster_size": float("nan"),
            "min_cluster_size": float("nan"),
            "max_cluster_size": float("nan"),
            "largest_cluster_fraction": float("nan"),
        }

    props = counts / total
    entropy = float(-np.sum(props * np.log(props + 1e-12)))
    entropy_norm = float(entropy / np.log(len(counts))) if len(counts) > 1 else 0.0
    effective_k_entropy = float(np.exp(entropy))

    # Simpson diversity = probability that two random entities fall in different clusters.
    # Inverse Simpson is another effective-number-of-clusters measure.
    simpson_diversity = float(1.0 - np.sum(props ** 2))
    inverse_simpson_effective_k = float(1.0 / (np.sum(props ** 2) + 1e-12))

    # Gini coefficient over raw cluster counts: 0 = equal sizes, 1 = highly unequal.
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    gini = float((2.0 * np.sum((np.arange(1, n + 1)) * sorted_counts) / (n * total)) - ((n + 1.0) / n))

    small_cluster_count = int(np.sum(counts < small_cluster_threshold))
    small_cluster_ratio = float(small_cluster_count / len(counts))

    return {
        "cluster_size_entropy": entropy,
        "cluster_size_entropy_normalized": entropy_norm,
        "effective_k_entropy": effective_k_entropy,
        "simpson_diversity": simpson_diversity,
        "inverse_simpson_effective_k": inverse_simpson_effective_k,
        "cluster_size_gini": gini,
        "small_cluster_count": small_cluster_count,
        "small_cluster_ratio": small_cluster_ratio,
        "mean_cluster_size": float(np.mean(counts)),
        "median_cluster_size": float(np.median(counts)),
        "min_cluster_size": int(np.min(counts)),
        "max_cluster_size": int(np.max(counts)),
        "largest_cluster_fraction": float(np.max(props)),
    }


# -----------------------------
# Drift metrics
# -----------------------------
def jsd(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)

    try:
        from scipy.spatial.distance import jensenshannon

        d = float(jensenshannon(p, q, base=2.0))
        return d * d
    except Exception:
        m = 0.5 * (p + q)

        def kl(a, b):
            a = np.clip(a, 1e-12, 1.0)
            b = np.clip(b, 1e-12, 1.0)
            return float(np.sum(a * np.log2(a / b)))

        return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def hungarian_match(sim: np.ndarray) -> List[Tuple[int, int, float]]:
    try:
        from scipy.optimize import linear_sum_assignment

        cost = -sim
        r, c = linear_sum_assignment(cost)
        return [(int(i), int(j), float(sim[i, j])) for i, j in zip(r, c)]
    except ImportError:
        nA, nB = sim.shape
        used_a, used_b = set(), set()
        pairs = []
        flat_idx = np.argsort(-sim.ravel())
        for idx in flat_idx:
            ai = idx // nB
            bj = idx % nB
            if ai not in used_a and bj not in used_b:
                pairs.append((int(ai), int(bj), float(sim[ai, bj])))
                used_a.add(ai)
                used_b.add(bj)
                if len(pairs) == min(nA, nB):
                    break
        return pairs


def sample_members(entities: List[str], labels: np.ndarray, cluster_id: int, max_n: int = 10) -> List[str]:
    idx = np.where(labels == cluster_id)[0]
    if len(idx) == 0:
        return []
    chosen = np.random.choice(idx, size=min(max_n, len(idx)), replace=False)
    return [entities[i] for i in chosen]


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class MonthInput:
    name: str
    path: str
    entities: List[str]
    labels_from_file: List[str]
    sbert_texts: List[str]
    emb_unit: np.ndarray


@dataclass
class MonthClustering:
    name: str
    path: str
    entities: List[str]
    emb_unit: np.ndarray
    cluster_labels: np.ndarray
    cluster_ids: np.ndarray
    n_clusters_effective: int
    counts: np.ndarray
    props: np.ndarray
    centroids_unit: np.ndarray
    coherences: np.ndarray
    coherence_weighted: float
    silhouette: float
    out_map_csv: str
    out_summary_csv: str


# -----------------------------
# Clustering functions
# -----------------------------
def dbscan_with_target_k(
    X: np.ndarray,
    target_k: int,
    min_samples: int,
    metric: str,
    eps_min: float,
    eps_max: float,
    eps_steps: int,
):
    from sklearn.cluster import DBSCAN

    best_labels = None
    best_eps = None
    best_score_tuple = None

    eps_values = np.linspace(eps_min, eps_max, eps_steps)

    for eps in eps_values:
        labels = DBSCAN(eps=float(eps), min_samples=min_samples, metric=metric).fit_predict(X)
        non_noise = labels[labels != -1]
        n_clusters = len(set(non_noise))
        n_noise = int(np.sum(labels == -1))

        score_tuple = (abs(n_clusters - target_k), n_noise, -float(eps))

        if best_score_tuple is None or score_tuple < best_score_tuple:
            best_score_tuple = score_tuple
            best_labels = labels
            best_eps = float(eps)

    if best_labels is None:
        raise RuntimeError("DBSCAN search failed to produce labels.")

    return best_labels, best_eps


def remap_labels_to_consecutive(labels: np.ndarray, include_noise_as_cluster: bool) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)

    if include_noise_as_cluster:
        raw_ids = sorted(set(labels.tolist()))
    else:
        raw_ids = sorted([x for x in set(labels.tolist()) if x != -1])

    mapping = {old: new for new, old in enumerate(raw_ids)}
    remapped = np.full_like(labels, fill_value=-1, dtype=int)

    for old, new in mapping.items():
        remapped[labels == old] = new

    cluster_ids = np.array(sorted([x for x in set(remapped.tolist()) if x != -1]), dtype=int)
    return remapped, cluster_ids


def run_clustering(emb_unit: np.ndarray, algorithm: str, k: int, args):
    algorithm = algorithm.lower()

    if algorithm == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering

        clusterer = AgglomerativeClustering(
            n_clusters=k,
            linkage=args.agglomerative_linkage,
            metric=args.agglomerative_metric,
        )
        labels = clusterer.fit_predict(emb_unit)
        cluster_ids = np.arange(k)

    elif algorithm == "spectral":
        from sklearn.cluster import SpectralClustering

        kwargs = dict(
            n_clusters=k,
            affinity=args.spectral_affinity,
            random_state=args.seed,
            assign_labels=args.spectral_assign_labels,
        )
        if args.spectral_affinity == "nearest_neighbors":
            kwargs["n_neighbors"] = args.spectral_n_neighbors
        clusterer = SpectralClustering(**kwargs)
        labels = clusterer.fit_predict(emb_unit)
        cluster_ids = np.arange(k)

    elif algorithm == "gmm":
        from sklearn.mixture import GaussianMixture

        clusterer = GaussianMixture(
            n_components=k,
            covariance_type=args.gmm_covariance_type,
            random_state=args.seed,
            n_init=args.gmm_n_init,
            max_iter=args.gmm_max_iter,
            reg_covar=args.gmm_reg_covar,
        )
        labels = clusterer.fit_predict(emb_unit)
        cluster_ids = np.arange(k)

    elif algorithm == "kmeans":
        from sklearn.cluster import KMeans

        clusterer = KMeans(
            n_clusters=k,
            random_state=args.seed,
            n_init=args.kmeans_n_init,
            max_iter=args.kmeans_max_iter,
        )
        labels = clusterer.fit_predict(emb_unit)
        cluster_ids = np.arange(k)

    elif algorithm == "dbscan":
        labels, best_eps = dbscan_with_target_k(
            X=emb_unit,
            target_k=k,
            min_samples=args.dbscan_min_samples,
            metric=args.dbscan_metric,
            eps_min=args.dbscan_eps_min,
            eps_max=args.dbscan_eps_max,
            eps_steps=args.dbscan_eps_steps,
        )
        labels, cluster_ids = remap_labels_to_consecutive(
            labels,
            include_noise_as_cluster=args.dbscan_include_noise_as_cluster,
        )
        print(f"      DBSCAN selected eps={best_eps:.6f}, effective_clusters={len(cluster_ids)}")

    elif algorithm == "pure_dbscan":
        from sklearn.cluster import DBSCAN

        labels = DBSCAN(
            eps=args.pure_dbscan_eps,
            min_samples=args.pure_dbscan_min_samples,
            metric=args.pure_dbscan_metric,
        ).fit_predict(emb_unit)
        labels, cluster_ids = remap_labels_to_consecutive(
            labels,
            include_noise_as_cluster=args.pure_dbscan_include_noise_as_cluster,
        )
        print(f"      Pure DBSCAN effective_clusters={len(cluster_ids)}")

    elif algorithm == "hdbscan":
        try:
            import hdbscan
        except ImportError:
            raise ImportError("hdbscan is not installed. Run: pip install hdbscan")

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=args.hdbscan_min_cluster_size,
            min_samples=args.hdbscan_min_samples,
            metric=args.hdbscan_metric,
        )
        labels = clusterer.fit_predict(emb_unit)
        labels, cluster_ids = remap_labels_to_consecutive(
            labels,
            include_noise_as_cluster=args.hdbscan_include_noise_as_cluster,
        )
        print(f"      HDBSCAN effective_clusters={len(cluster_ids)}")

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return np.asarray(labels, dtype=int), np.asarray(cluster_ids, dtype=int)


# -----------------------------
# Single algorithm/k run
# -----------------------------
def process_algorithm_k(
    month_inputs: List[MonthInput],
    algorithm: str,
    k: int,
    run_outdir: Path,
    args,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.metrics import silhouette_score

    ensure_dir(run_outdir)

    months: List[MonthClustering] = []
    month_quality_rows = []

    k_label = "auto" if algorithm in {"pure_dbscan", "hdbscan"} else str(k)
    print(f"\n=== {algorithm.upper()} | k={k_label} ===")

    for month in month_inputs:
        print(f"  [{month.name}] clustering {len(month.entities)} entities...")

        cluster_labels, cluster_ids = run_clustering(month.emb_unit, algorithm, k, args)
        n_effective = len(cluster_ids)

        if n_effective == 0:
            print(f"  WARNING: {month.name} produced zero non-noise clusters. Skipping.")
            continue

        counts = np.array([np.sum(cluster_labels == c) for c in cluster_ids], dtype=np.int64)
        props = counts / (counts.sum() + 1e-12)
        size_metrics = compute_cluster_size_metrics(
            counts=counts,
            small_cluster_threshold=args.small_cluster_threshold,
        )

        centroids = []
        for c in cluster_ids:
            idx = np.where(cluster_labels == c)[0]
            if len(idx) > 0:
                cent = month.emb_unit[idx].mean(axis=0)
                cent = cent / (np.linalg.norm(cent) + 1e-12)
            else:
                cent = np.zeros(month.emb_unit.shape[1])
            centroids.append(cent)
        centroids_unit = np.array(centroids, dtype=np.float64)

        coherences = compute_cluster_coherences(month.emb_unit, cluster_labels, cluster_ids)
        valid = ~np.isnan(coherences)
        coherence_weighted = float(np.average(coherences[valid], weights=counts[valid])) if valid.any() else float("nan")

        sil = float("nan")
        if n_effective >= 2:
            try:
                mask = np.isin(cluster_labels, cluster_ids)
                masked_labels = cluster_labels[mask]
                if len(set(masked_labels.tolist())) >= 2 and len(masked_labels) > len(set(masked_labels.tolist())):
                    sil = float(silhouette_score(month.emb_unit[mask], masked_labels, metric="cosine"))
            except Exception:
                sil = float("nan")

        out_map_csv = run_outdir / f"{month.name}_cluster_map.csv"
        pd.DataFrame(
            {
                "entity": month.entities,
                "sbert_text": month.sbert_texts,
                "cluster_id": cluster_labels,
            }
        ).to_csv(out_map_csv, index=False)

        out_summary_csv = run_outdir / f"{month.name}_cluster_summary.csv"
        rows_sum = []
        for i, c in enumerate(cluster_ids):
            rows_sum.append(
                {
                    "cluster_id": int(c),
                    "count": int(counts[i]),
                    "proportion": float(props[i]),
                    "mean_pairwise_cosine_similarity_within_cluster": float(coherences[i])
                    if not np.isnan(coherences[i])
                    else np.nan,
                    "sample_members": " | ".join(sample_members(month.entities, cluster_labels, int(c), max_n=12)),
                }
            )
        pd.DataFrame(rows_sum).sort_values("count", ascending=False).to_csv(out_summary_csv, index=False)

        n_noise = int(np.sum(cluster_labels == -1))
        noise_fraction = float(n_noise / max(len(cluster_labels), 1))

        print(
            f"      silhouette={sil:.4f}  pair_cosine={coherence_weighted:.4f}  "
            f"effective_clusters={n_effective}  noise={n_noise}  "
            f"effective_k={size_metrics['effective_k_entropy']:.2f}  "
            f"small_clusters={size_metrics['small_cluster_ratio']:.2f}"
        )

        month_quality_rows.append(
            {
                "algorithm": algorithm,
                "month": month.name,
                "n_entities": int(len(month.entities)),
                "k_requested": int(k),
                "n_clusters_effective": int(n_effective),
                "n_noise": n_noise,
                "noise_fraction": noise_fraction,
                "silhouette_cosine_on_embedding_space": sil,
                "weighted_mean_intra_cluster_pairwise_cosine_similarity": coherence_weighted,
                **size_metrics,
            }
        )

        months.append(
            MonthClustering(
                name=month.name,
                path=month.path,
                entities=month.entities,
                emb_unit=month.emb_unit,
                cluster_labels=cluster_labels,
                cluster_ids=cluster_ids,
                n_clusters_effective=n_effective,
                counts=counts,
                props=props.astype(np.float64),
                centroids_unit=centroids_unit,
                coherences=coherences,
                coherence_weighted=coherence_weighted,
                silhouette=sil,
                out_map_csv=str(out_map_csv),
                out_summary_csv=str(out_summary_csv),
            )
        )

    month_quality_df = pd.DataFrame(month_quality_rows)
    month_quality_df.to_csv(run_outdir / "month_quality_metrics.csv", index=False)

    pair_rows = []
    align_rows = []

    for t in range(1, len(months)):
        A = months[t - 1]
        B = months[t]

        sim = cosine_similarity_matrix(A.centroids_unit, B.centroids_unit)
        pairs = hungarian_match(sim)

        aligned_B_props = np.zeros((len(A.props),), dtype=np.float64)
        centroid_dists = []

        for ai, bj, sij in pairs:
            aligned_B_props[ai] = float(B.props[bj])
            cd = cosine_distance(A.centroids_unit[ai], B.centroids_unit[bj])
            centroid_dists.append(cd)

            align_rows.append(
                {
                    "algorithm": algorithm,
                    "k_requested": k,
                    "from_month": A.name,
                    "to_month": B.name,
                    "from_cluster": int(A.cluster_ids[ai]),
                    "to_cluster": int(B.cluster_ids[bj]),
                    "centroid_cosine_similarity": float(sij),
                    "centroid_cosine_distance": float(cd),
                    "from_count": int(A.counts[ai]),
                    "to_count": int(B.counts[bj]),
                    "from_prop": float(A.props[ai]),
                    "to_prop": float(B.props[bj]),
                    "delta_prop": float(B.props[bj] - A.props[ai]),
                }
            )

        drift_jsd = jsd(A.props, aligned_B_props)
        centroid_dists = np.array(centroid_dists, dtype=np.float64) if centroid_dists else np.array([np.nan])

        pair_rows.append(
            {
                "algorithm": algorithm,
                "k_requested": k,
                "from_month": A.name,
                "to_month": B.name,
                "jsd_cluster_distribution_aligned": float(drift_jsd),
                "centroid_drift_mean": float(np.nanmean(centroid_dists)),
                "centroid_drift_median": float(np.nanmedian(centroid_dists)),
                "centroid_drift_max": float(np.nanmax(centroid_dists)),
                "n_entities_from": int(A.counts.sum()),
                "n_entities_to": int(B.counts.sum()),
                "n_clusters_effective_from": int(A.n_clusters_effective),
                "n_clusters_effective_to": int(B.n_clusters_effective),
                "from_silhouette": A.silhouette,
                "to_silhouette": B.silhouette,
                "from_weighted_intra_cluster_cosine": A.coherence_weighted,
                "to_weighted_intra_cluster_cosine": B.coherence_weighted,
            }
        )

    pair_df = pd.DataFrame(pair_rows)
    align_df = pd.DataFrame(align_rows)

    pair_df.to_csv(run_outdir / "cluster_drift_pair_summary.csv", index=False)
    align_df.to_csv(run_outdir / "cluster_drift_alignment_long.csv", index=False)

    if args.plot and not pair_df.empty:
        import matplotlib.pyplot as plt

        pair_labels = [f"{r['from_month']}→{r['to_month']}" for _, r in pair_df.iterrows()]

        plt.figure()
        plt.plot(range(1, len(pair_df) + 1), pair_df["jsd_cluster_distribution_aligned"].values, marker="o")
        plt.xticks(range(1, len(pair_df) + 1), pair_labels, rotation=45, ha="right")
        plt.ylabel("Jensen–Shannon Divergence")
        plt.xlabel("Consecutive month pair")
        plt.title(f"{algorithm.upper()} Interest Drift, k={k_label}")
        plt.tight_layout()
        plt.savefig(run_outdir / "cluster_drift_plot_jsd.png", dpi=200)
        plt.close()

        plt.figure()
        plt.plot(range(1, len(pair_df) + 1), pair_df["centroid_drift_mean"].values, marker="o")
        plt.xticks(range(1, len(pair_df) + 1), pair_labels, rotation=45, ha="right")
        plt.ylabel("Mean centroid cosine distance")
        plt.xlabel("Consecutive month pair")
        plt.title(f"{algorithm.upper()} Semantic Drift, k={k_label}")
        plt.tight_layout()
        plt.savefig(run_outdir / "cluster_drift_plot_centroid_mean.png", dpi=200)
        plt.close()

    return month_quality_df, pair_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Single-pass benchmark for clustering algorithms over multiple k values."
    )
    parser.add_argument(
        "--input-files",
        nargs="+",
        required=True,
        help="Monthly input files. Each should be a 2-column file: entity, label. Single-column files are also accepted.",
    )
    parser.add_argument("--outdir", default="benchmark_results", help="Output directory.")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Sentence-BERT model name.")
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=ALGORITHMS,
        choices=ALGORITHMS,
        help="Algorithms to run.",
    )
    parser.add_argument("--k-values", nargs="+", type=int, default=K_VALUES, help="Cluster counts to test.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--plot", action="store_true", help="Generate plots for each algorithm/k run.")
    parser.add_argument(
        "--small-cluster-threshold",
        type=int,
        default=5,
        help="Clusters smaller than this size are counted as small clusters in size-quality metrics.",
    )

    # Agglomerative
    parser.add_argument(
        "--agglomerative-linkage",
        default="ward",
        choices=["ward", "average", "complete", "single"],
        help="Agglomerative linkage.",
    )
    parser.add_argument(
        "--agglomerative-metric",
        default="euclidean",
        help="Agglomerative metric. Ward requires euclidean.",
    )

    # Spectral
    parser.add_argument(
        "--spectral-affinity",
        default="nearest_neighbors",
        choices=["nearest_neighbors", "rbf", "precomputed", "precomputed_nearest_neighbors"],
        help="SpectralClustering affinity.",
    )
    parser.add_argument("--spectral-n-neighbors", type=int, default=10, help="SpectralClustering n_neighbors.")
    parser.add_argument(
        "--spectral-assign-labels",
        default="kmeans",
        choices=["kmeans", "discretize", "cluster_qr"],
        help="SpectralClustering label assignment.",
    )

    # GMM
    parser.add_argument(
        "--gmm-covariance-type",
        default="full",
        choices=["full", "tied", "diag", "spherical"],
        help="GMM covariance type.",
    )
    parser.add_argument("--gmm-n-init", type=int, default=5, help="GMM n_init.")
    parser.add_argument("--gmm-max-iter", type=int, default=300, help="GMM max_iter.")
    parser.add_argument("--gmm-reg-covar", type=float, default=1e-6, help="GMM reg_covar.")

    # KMeans
    parser.add_argument("--kmeans-n-init", default="auto", help="KMeans n_init. Use integer or 'auto'.")
    parser.add_argument("--kmeans-max-iter", type=int, default=300, help="KMeans max_iter.")

    # Target-k DBSCAN
    parser.add_argument("--dbscan-min-samples", type=int, default=5, help="Target-k DBSCAN min_samples.")
    parser.add_argument("--dbscan-metric", default="cosine", help="Target-k DBSCAN metric.")
    parser.add_argument("--dbscan-eps-min", type=float, default=0.01, help="Target-k DBSCAN eps minimum.")
    parser.add_argument("--dbscan-eps-max", type=float, default=1.00, help="Target-k DBSCAN eps maximum.")
    parser.add_argument("--dbscan-eps-steps", type=int, default=100, help="Target-k DBSCAN eps search steps.")
    parser.add_argument(
        "--dbscan-include-noise-as-cluster",
        action="store_true",
        help="Treat target-k DBSCAN noise as its own cluster.",
    )

    # Pure DBSCAN
    parser.add_argument(
        "--pure-dbscan-eps",
        type=float,
        default=0.25,
        help="Pure DBSCAN eps. Unlike --dbscan, this does not search for target k.",
    )
    parser.add_argument("--pure-dbscan-min-samples", type=int, default=5, help="Pure DBSCAN min_samples.")
    parser.add_argument("--pure-dbscan-metric", default="cosine", help="Pure DBSCAN metric.")
    parser.add_argument(
        "--pure-dbscan-include-noise-as-cluster",
        action="store_true",
        help="Treat pure DBSCAN noise as its own cluster.",
    )

    # HDBSCAN
    parser.add_argument("--hdbscan-min-cluster-size", type=int, default=5, help="HDBSCAN min_cluster_size.")
    parser.add_argument(
        "--hdbscan-min-samples",
        type=int,
        default=None,
        help="HDBSCAN min_samples. If omitted, HDBSCAN chooses based on min_cluster_size.",
    )
    parser.add_argument(
        "--hdbscan-metric",
        default="euclidean",
        help="HDBSCAN metric. Use euclidean on normalized embeddings for cosine-like behavior.",
    )
    parser.add_argument(
        "--hdbscan-include-noise-as-cluster",
        action="store_true",
        help="Treat HDBSCAN noise as its own cluster.",
    )

    args = parser.parse_args()

    if args.agglomerative_linkage == "ward" and args.agglomerative_metric != "euclidean":
        raise ValueError("Ward linkage requires --agglomerative-metric euclidean.")

    if args.kmeans_n_init != "auto":
        try:
            args.kmeans_n_init = int(args.kmeans_n_init)
        except ValueError:
            raise ValueError("--kmeans-n-init must be integer or 'auto'.")

    np.random.seed(args.seed)

    outdir = Path(args.outdir)
    ensure_dir(outdir)

    print(f"Loading sentence-transformer model ONCE: {args.model}")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers not installed. Run: pip install -U sentence-transformers", file=sys.stderr)
        sys.exit(1)

    model = SentenceTransformer(args.model)

    month_inputs: List[MonthInput] = []

    print("\nEncoding input files ONCE...")
    for path in args.input_files:
        if not os.path.isfile(path):
            print(f"WARNING: File not found: {path}", file=sys.stderr)
            continue

        mname = month_name_from_path(path)
        rows = read_two_columns(path, sep=None)

        entities = [ent for ent, _ in rows]
        labels_from_file = [lab for _, lab in rows]
        sbert_texts = [clean_entity_for_sbert(ent) for ent in entities]

        print(f"  [{mname}] Encoding {len(entities)} cleaned entities for SBERT...")
        emb = model.encode(sbert_texts, show_progress_bar=False, convert_to_numpy=True)
        emb_unit = l2_normalize(emb)

        month_inputs.append(
            MonthInput(
                name=mname,
                path=path,
                entities=entities,
                labels_from_file=labels_from_file,
                sbert_texts=sbert_texts,
                emb_unit=emb_unit,
            )
        )

    if not month_inputs:
        raise RuntimeError("No valid input files were loaded.")

    summary_rows = []
    all_month_metrics = []

    for algorithm in args.algorithms:
        if algorithm in {"pure_dbscan", "hdbscan"}:
            k_values_for_algorithm = [0]
        else:
            k_values_for_algorithm = args.k_values

        for k in k_values_for_algorithm:
            if algorithm in {"pure_dbscan", "hdbscan"}:
                run_outdir = outdir / algorithm
            else:
                run_outdir = outdir / f"{algorithm}_k{k}"

            month_quality_df, _ = process_algorithm_k(
                month_inputs=month_inputs,
                algorithm=algorithm,
                k=k,
                run_outdir=run_outdir,
                args=args,
            )

            if month_quality_df.empty:
                continue

            mean_silhouette = month_quality_df["silhouette_cosine_on_embedding_space"].mean()
            mean_pair_cosine = month_quality_df["weighted_mean_intra_cluster_pairwise_cosine_similarity"].mean()
            mean_noise_fraction = month_quality_df["noise_fraction"].mean() if "noise_fraction" in month_quality_df else np.nan
            mean_effective_k_entropy = month_quality_df["effective_k_entropy"].mean()
            mean_inverse_simpson_effective_k = month_quality_df["inverse_simpson_effective_k"].mean()
            mean_cluster_size_gini = month_quality_df["cluster_size_gini"].mean()
            mean_small_cluster_ratio = month_quality_df["small_cluster_ratio"].mean()
            mean_largest_cluster_fraction = month_quality_df["largest_cluster_fraction"].mean()

            summary_rows.append(
                {
                    "algorithm": algorithm,
                    "k_requested": k,
                    "mean_silhouette": mean_silhouette,
                    "mean_pairwise_cosine": mean_pair_cosine,
                    "mean_noise_fraction": mean_noise_fraction,
                    "mean_effective_k_entropy": mean_effective_k_entropy,
                    "mean_inverse_simpson_effective_k": mean_inverse_simpson_effective_k,
                    "mean_cluster_size_gini": mean_cluster_size_gini,
                    "mean_small_cluster_ratio": mean_small_cluster_ratio,
                    "mean_largest_cluster_fraction": mean_largest_cluster_fraction,
                    "n_months": len(month_quality_df),
                    "mean_effective_clusters": month_quality_df["n_clusters_effective"].mean(),
                    "min_effective_clusters": month_quality_df["n_clusters_effective"].min(),
                    "max_effective_clusters": month_quality_df["n_clusters_effective"].max(),
                    "run_outdir": str(run_outdir),
                }
            )

            month_quality_df = month_quality_df.copy()
            month_quality_df["run_outdir"] = str(run_outdir)
            all_month_metrics.append(month_quality_df)

    summary_df = pd.DataFrame(summary_rows)

    if summary_df.empty:
        raise RuntimeError("No benchmark results were produced.")

    summary_df = summary_df.sort_values(["algorithm", "k_requested"])

    all_month_df = pd.concat(all_month_metrics, ignore_index=True) if all_month_metrics else pd.DataFrame()

    summary_csv = outdir / "benchmark_summary.csv"
    all_month_csv = outdir / "benchmark_all_month_metrics.csv"
    summary_xlsx = outdir / "benchmark_summary.xlsx"

    summary_df.to_csv(summary_csv, index=False)
    all_month_df.to_csv(all_month_csv, index=False)

    with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        all_month_df.to_excel(writer, sheet_name="All_Month_Metrics", index=False)

    print("\nDone.")
    print(f"Wrote summary CSV: {summary_csv}")
    print(f"Wrote all month metrics CSV: {all_month_csv}")
    print(f"Wrote Excel summary: {summary_xlsx}")

    print("\nBenchmark summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
