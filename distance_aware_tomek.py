"""
Distance-Aware Tomek Links — Threshold Berbasis Within-Class Distance
=====================================================================
Threshold yang principled untuk thesis:

  Hanya hapus Tomek pair jika:
    dist(pair) <= alpha * median_within_class_1NN_distance

  Justifikasi: kalau jarak antar kelas (Tomek pair) masih sebanding dengan
  "seberapa rapat kelas itu sendiri", maka itu genuine boundary/overlap.
  Kalau jauh lebih besar → bukan overlap, itu isolasi karena sparsity.

  Alpha:
    1.0 = strict, hanya yang benar-benar di boundary
    1.5 = recommended (balance antara cleaning dan konservatif)
    2.0 = longgar, agresif
"""

import numpy as np
import pandas as pd
from collections import Counter

OUTPUT_DIR = "outputs_hnsw_final"
ALPHA      = 1.5   # multiplier threshold (1.5 = recommended)
STRATEGY   = "majority"  # hapus hanya dari kelas lebih besar

CLASS_MAP = {
    0: "BENIGN", 1: "Bot", 2: "DDoS", 3: "DoS GoldenEye",
    4: "DoS Hulk", 5: "DoS Slowhttptest", 6: "DoS slowloris",
    7: "FTP-Patator", 8: "Heartbleed", 9: "Infiltration",
    10: "Port Scanning", 11: "SSH-Patator",
    12: "Web Attacks - Brute Force", 13: "Web Attacks - SQL Injection",
    14: "Web Attacks - XSS"
}


# ─── Load data ───────────────────────────────────────────────────────────────
print("Loading KDN outputs...")
distances_sq = np.load(f"{OUTPUT_DIR}/distances_hnsw.npy")
indices      = np.load(f"{OUTPUT_DIR}/indices_hnsw.npy")
y_enc        = np.load(f"{OUTPUT_DIR}/labels_y_encoded.npy")
y_raw        = np.load(f"{OUTPUT_DIR}/labels_y.npy", allow_pickle=True)
n            = len(y_enc)

# Kolom 0 = self, kolom 1 = NN ke-1
# FAISS IndexHNSWFlat return L2 SQUARED → sqrt untuk Euclidean
nn1_idx  = indices[:, 1]
nn1_dist = np.sqrt(distances_sq[:, 1].astype(np.float64))

class_names = np.array([CLASS_MAP[int(c)] for c in y_enc])
class_counts = Counter(y_raw)
print(f"  n={n:,}  classes={len(np.unique(y_enc))}")


# ─── Step 1: Within-class median 1-NN distance per kelas ────────────────────
# Ini jadi "referensi kerapatan" tiap kelas
# Hanya hitung untuk sample yang NN-nya sekelas dengan dirinya
print("\n[1] Menghitung within-class 1-NN distance per kelas...")

within_class_rows = []
for i in range(n):
    j = nn1_idx[i]
    if j != -1 and j < n and y_enc[j] == y_enc[i]:
        within_class_rows.append((y_enc[i], nn1_dist[i]))

df_within = pd.DataFrame(within_class_rows, columns=["class_enc", "dist"])
within_median = df_within.groupby("class_enc")["dist"].median()

print(f"\n  Median within-class 1-NN distance per kelas:")
print(f"  {'Kelas':<35} {'Median within-dist':>18}  {'Threshold (×{:.1f})'.format(ALPHA):>20}")
print(f"  {'-'*75}")

class_threshold = {}
for cid in sorted(within_median.index):
    cname  = CLASS_MAP[int(cid)]
    med    = within_median[cid]
    thresh = med * ALPHA
    class_threshold[int(cid)] = thresh
    print(f"  {cname:<35} {med:>18.4f}  {thresh:>20.4f}")


# ─── Step 2: Hitung global threshold (weighted median) ──────────────────────
# Untuk Tomek pair yang melibatkan 2 kelas berbeda, pakai threshold
# dari kelas yang "lebih rapat" (median within-dist lebih kecil)
# → lebih konservatif, tidak over-clean kelas sparse
print(f"\n[2] Global fallback threshold (median semua within-class dist):")
global_thresh = df_within["dist"].median() * ALPHA
print(f"  Global threshold = {global_thresh:.4f}  (alpha={ALPHA})")


# ─── Step 3: Temukan semua Tomek Links ──────────────────────────────────────
print("\n[3] Mencari Tomek Links (mutual 1-NN, beda kelas)...")

tomek_rows = []
for i in range(n):
    j = nn1_idx[i]
    if j == -1 or j >= n:
        continue
    if i < j and nn1_idx[j] == i and y_enc[i] != y_enc[j]:
        ci, cj = int(y_enc[i]), int(y_enc[j])
        # Threshold = min dari threshold dua kelas (lebih konservatif)
        t_i = class_threshold.get(ci, global_thresh)
        t_j = class_threshold.get(cj, global_thresh)
        pair_threshold = min(t_i, t_j)
        tomek_rows.append({
            "idx_i"          : i,
            "idx_j"          : j,
            "dist_euclidean" : nn1_dist[i],
            "class_i"        : CLASS_MAP[ci],
            "class_j"        : CLASS_MAP[cj],
            "threshold_i"    : round(t_i, 4),
            "threshold_j"    : round(t_j, 4),
            "pair_threshold" : round(pair_threshold, 4),
            "is_boundary"    : nn1_dist[i] <= pair_threshold,
        })

df_all  = pd.DataFrame(tomek_rows)
df_near = df_all[df_all["is_boundary"]].copy()
df_far  = df_all[~df_all["is_boundary"]].copy()

print(f"  Total Tomek Links             : {len(df_all):,}")
print(f"  Genuine boundary (diproses)   : {len(df_near):,}  (dist ≤ alpha × within-dist)")
print(f"  Isolated/sparse (diabaikan)   : {len(df_far):,}  (dist > alpha × within-dist)")


# ─── Step 4: Analisis per kelas ──────────────────────────────────────────────
print("\n[4] Breakdown Tomek pairs per kelas (semua vs boundary):")
print(f"  {'Kelas':<35} {'All pairs':>10} {'Boundary':>10} {'% Boundary':>12}")
print(f"  {'-'*70}")

for cid in sorted(np.unique(y_enc)):
    cname = CLASS_MAP[int(cid)]
    all_c = ((df_all.class_i == cname) | (df_all.class_j == cname)).sum()
    bnd_c = ((df_near.class_i == cname) | (df_near.class_j == cname)).sum()
    pct   = bnd_c / all_c * 100 if all_c > 0 else 0
    print(f"  {cname:<35} {all_c:>10,} {bnd_c:>10,} {pct:>11.1f}%")


# ─── Step 5: Tentukan sample yang dihapus ────────────────────────────────────
to_remove = set()
for _, row in df_near.iterrows():
    i, j   = int(row.idx_i), int(row.idx_j)
    ci, cj = row.class_i, row.class_j
    if class_counts[ci] >= class_counts[cj]:
        to_remove.add(i)
    else:
        to_remove.add(j)

to_remove_arr = np.array(sorted(to_remove), dtype=np.int64)

print(f"\n[5] Hasil penghapusan (strategy='{STRATEGY}', alpha={ALPHA}):")
print(f"  Sample dihapus  : {len(to_remove):,}")
print(f"  Sample tersisa  : {n - len(to_remove):,} / {n:,}")

if len(to_remove_arr) > 0:
    removed_cls = Counter([class_names[i] for i in to_remove_arr])
    print(f"\n  Breakdown kelas yang dihapus:")
    print(f"  {'Kelas':<35} {'Dihapus':>8} {'Total':>8} {'% dihapus':>10}")
    print(f"  {'-'*65}")
    for cls, cnt in sorted(removed_cls.items(), key=lambda x: -x[1]):
        total = class_counts[cls]
        print(f"  {cls:<35} {cnt:>8,} {total:>8,} {cnt/total*100:>9.2f}%")


# ─── Step 6: Perbandingan Standard vs Distance-Aware ─────────────────────────
to_remove_std = set()
for _, row in df_all.iterrows():
    i, j   = int(row.idx_i), int(row.idx_j)
    ci, cj = row.class_i, row.class_j
    if class_counts[ci] >= class_counts[cj]:
        to_remove_std.add(i)
    else:
        to_remove_std.add(j)

saved     = to_remove_std - to_remove
saved_cls = Counter([class_names[i] for i in saved])

print(f"\n[6] Standard vs Distance-Aware:")
print(f"  Standard Tomek  : -{len(to_remove_std):,} samples")
print(f"  Distance-Aware  : -{len(to_remove):,} samples  (alpha={ALPHA})")
print(f"  Diselamatkan    : +{len(saved):,} samples")
print(f"\n  Kelas yang paling banyak diselamatkan:")
for cls, cnt in saved_cls.most_common(8):
    print(f"    {cls:<35}: +{cnt:,}")


# ─── Step 7: Laporan Infiltration ────────────────────────────────────────────
print("\n" + "="*60)
print("LAPORAN INFILTRATION")
print("="*60)
inf_all  = df_all[(df_all.class_i == "Infiltration") | (df_all.class_j == "Infiltration")]
inf_near = df_near[(df_near.class_i == "Infiltration") | (df_near.class_j == "Infiltration")]
inf_cid  = [k for k, v in CLASS_MAP.items() if v == "Infiltration"][0]
inf_thresh = class_threshold.get(inf_cid, global_thresh)

print(f"  Within-class median dist  : {within_median.get(inf_cid, float('nan')):.4f}")
print(f"  Threshold (×{ALPHA})          : {inf_thresh:.4f}")
print(f"  Total Tomek pairs         : {len(inf_all)}")
if len(inf_all) > 0:
    print(f"  Pairs JAUH (diabaikan)    : {(~inf_all['is_boundary']).sum()}")
    print(f"  Pairs boundary (diproses) : {len(inf_near)}")
    print(f"\n  Distribusi jarak Infiltration pairs:")
    for p in [0, 25, 50, 75, 100]:
        pv = np.percentile(inf_all["dist_euclidean"], p)
        flag = " ← di atas threshold" if pv > inf_thresh else " ← di bawah threshold"
        print(f"    P{p:<3}: {pv:.4f}{flag}")
    print(f"\n  → Infiltration aman secara ALAMI karena threshold berbasis")
    print(f"    within-class distance, tanpa perlu 'protected class' yang diskriminatif.")


# ─── Step 8: Simpan ──────────────────────────────────────────────────────────
keep_mask = np.ones(n, dtype=bool)
if len(to_remove_arr) > 0:
    keep_mask[to_remove_arr] = False

np.save(f"{OUTPUT_DIR}/tomek_keep_mask.npy", keep_mask)
df_all.to_csv(f"{OUTPUT_DIR}/tomek_all_pairs.csv", index=False)
df_near.to_csv(f"{OUTPUT_DIR}/tomek_boundary_pairs.csv", index=False)

print(f"\n[OK] Tersimpan:")
print(f"  tomek_keep_mask.npy         : {keep_mask.sum():,} samples (True = keep)")
print(f"  tomek_all_pairs.csv         : {len(df_all):,} semua Tomek pairs")
print(f"  tomek_boundary_pairs.csv    : {len(df_near):,} genuine boundary pairs")
print(f"\n  Cara pakai:")
print(f"    mask = np.load('outputs_hnsw_final/tomek_keep_mask.npy')")
print(f"    X_clean, y_clean = X[mask], y[mask]")
