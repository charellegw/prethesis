"""
Analisis jarak k-Nearest Neighbor per sample dari output KDN HNSW.
Tujuan: cek apakah neighbor yang diambil wajar jaraknya (tidak terlalu jauh).
"""

import numpy as np
import pandas as pd

# ─── Load semua file .npy ───────────────────────────────────────────────────
OUTPUT_DIR = "outputs_hnsw_final"

distances  = np.load(f"{OUTPUT_DIR}/distances_hnsw.npy")      # (n_samples, k)
indices    = np.load(f"{OUTPUT_DIR}/indices_hnsw.npy")        # (n_samples, k)
labels_enc = np.load(f"{OUTPUT_DIR}/labels_y_encoded.npy")   # (n_samples,)
labels_raw = np.load(f"{OUTPUT_DIR}/labels_y.npy",
                     allow_pickle=True)                        # (n_samples,)
kdn_scores = np.load(f"{OUTPUT_DIR}/kdn_hnsw_scores.npy")    # (n_samples,)

# distances dari FAISS adalah L2-squared → ubah ke Euclidean
distances_euclidean = np.sqrt(distances)

n_samples, k = distances_euclidean.shape
print(f"Dataset   : {n_samples:,} samples, k={k} neighbors")
print(f"Shape distances : {distances_euclidean.shape}")
print(f"Shape indices   : {indices.shape}")

# ─── Ringkasan statistik jarak global ───────────────────────────────────────
print("\n" + "="*60)
print("STATISTIK JARAK EUCLIDEAN (seluruh sample & neighbor)")
print("="*60)
flat_dist = distances_euclidean.flatten()
print(f"  Min    : {flat_dist.min():.4f}")
print(f"  Max    : {flat_dist.max():.4f}")
print(f"  Mean   : {flat_dist.mean():.4f}")
print(f"  Median : {np.median(flat_dist):.4f}")
print(f"  Std    : {flat_dist.std():.4f}")
print(f"  P90    : {np.percentile(flat_dist, 90):.4f}")
print(f"  P95    : {np.percentile(flat_dist, 95):.4f}")
print(f"  P99    : {np.percentile(flat_dist, 99):.4f}")

# ─── Statistik per neighbor ke-1 s/d ke-k ──────────────────────────────────
print("\n" + "="*60)
print("RATA-RATA JARAK PER POSISI NEIGHBOR")
print("="*60)
for i in range(k):
    col = distances_euclidean[:, i]
    print(f"  Neighbor ke-{i+1}  mean={col.mean():.4f}  "
          f"median={np.median(col):.4f}  max={col.max():.4f}")

# ─── Statistik per kelas ─────────────────────────────────────────────────────
CLASS_MAP = {
    0: "BENIGN", 1: "Bot", 2: "DDoS", 3: "DoS GoldenEye",
    4: "DoS Hulk", 5: "DoS Slowhttptest", 6: "DoS slowloris",
    7: "FTP-Patator", 8: "Heartbleed", 9: "Infiltration",
    10: "Port Scanning", 11: "SSH-Patator",
    12: "Web Attacks - Brute Force", 13: "Web Attacks - SQL Injection",
    14: "Web Attacks - XSS"
}

mean_dist_per_sample = distances_euclidean.mean(axis=1)   # rata-rata jarak ke 5 NN
max_dist_per_sample  = distances_euclidean.max(axis=1)    # jarak ke NN terjauh

print("\n" + "="*60)
print("JARAK RATA-RATA PER KELAS (mean jarak ke 5 neighbor)")
print("="*60)
class_ids = np.unique(labels_enc)
rows = []
for cid in class_ids:
    mask = labels_enc == cid
    d    = mean_dist_per_sample[mask]
    rows.append({
        "Class"      : CLASS_MAP.get(int(cid), str(cid)),
        "N_samples"  : mask.sum(),
        "Mean_dist"  : round(d.mean(), 4),
        "Median_dist": round(np.median(d), 4),
        "Max_dist"   : round(d.max(), 4),
        "P95_dist"   : round(np.percentile(d, 95), 4),
        "Mean_kDN"   : round(kdn_scores[mask].mean(), 4),
    })

df_class = pd.DataFrame(rows).sort_values("Mean_dist", ascending=False)
print(df_class.to_string(index=False))

# ─── Sample-level DataFrame ──────────────────────────────────────────────────
df_sample = pd.DataFrame({
    "sample_id"   : np.arange(n_samples),
    "class_label" : [CLASS_MAP.get(int(c), str(c)) for c in labels_enc],
    "kdn_score"   : kdn_scores,
    "mean_dist_5nn": mean_dist_per_sample,
    "max_dist_5nn" : max_dist_per_sample,
    "dist_nn1"    : distances_euclidean[:, 0],
    "dist_nn2"    : distances_euclidean[:, 1],
    "dist_nn3"    : distances_euclidean[:, 2],
    "dist_nn4"    : distances_euclidean[:, 3],
    "dist_nn5"    : distances_euclidean[:, 4],
    "idx_nn1"     : indices[:, 0],
    "idx_nn2"     : indices[:, 1],
    "idx_nn3"     : indices[:, 2],
    "idx_nn4"     : indices[:, 3],
    "idx_nn5"     : indices[:, 4],
})

# ─── Deteksi sample dengan neighbor sangat jauh ──────────────────────────────
# Pakai P95 global sebagai threshold "jauh tidak wajar"
THRESHOLD_FAR = np.percentile(mean_dist_per_sample, 95)
print(f"\n[!] Threshold 'jarak jauh' (P95 mean dist): {THRESHOLD_FAR:.4f}")

df_far = df_sample[df_sample["mean_dist_5nn"] > THRESHOLD_FAR].copy()
print(f"    Jumlah sample dengan jarak rata-rata NN > threshold: {len(df_far):,}")
print("\nDistribusi per kelas (sample berjarak jauh):")
print(df_far["class_label"].value_counts().to_string())

# ─── Sample KDN tinggi TAPI jarak NN juga jauh (ini yang perlu diwaspadai) ──
HIGH_KDN   = 0.5   # kDN >= 0.5 → overlap/severe
df_risky = df_sample[
    (df_sample["kdn_score"] >= HIGH_KDN) &
    (df_sample["mean_dist_5nn"] > THRESHOLD_FAR)
].copy()
print(f"\n[!] Sample kDN >= {HIGH_KDN} DAN jarak NN > threshold: {len(df_risky):,}")
if len(df_risky) > 0:
    print("    Kelas distribusi:")
    print(df_risky["class_label"].value_counts().to_string())
    print("\n    Contoh 10 sample teratas:")
    print(df_risky.sort_values("kdn_score", ascending=False)
          [["sample_id","class_label","kdn_score","mean_dist_5nn","max_dist_5nn"]]
          .head(10).to_string(index=False))

# ─── Export ──────────────────────────────────────────────────────────────────
out_path = f"{OUTPUT_DIR}/knn_distance_analysis.csv"
df_sample.to_csv(out_path, index=False)
print(f"\n[OK] File saved: {out_path}")
print(f"     Kolom: {list(df_sample.columns)}")
