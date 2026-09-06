# -*- coding: utf-8 -*-
"""土壤相对湿度工具链验证（V-S1~V-S10）。

用法：
    PYTHONUTF8=1 python verify_soil_tools.py [51208|60513|all]

分段：
    51208 —— V-S1a~V-S7 + V-S9 + V-S10（不跑 gpt，复用旧 yuchuli 中间产物验证 Python 段；
             含旧品 bit-exact 复现与修复产品护栏）
    60513 —— V-S1b + V-S8 + V-S9 + V-S10（gpt S1+S2 全链实跑，耗时 1-2 小时量级）
    all   —— 两段都跑（默认）

旧 test/ 只读；新产物写 E:/YYR/turangshuifen/work/<日期>/。
样本口径：soim_values 为随机模拟数据（无实测值）——本验证的 R² 相关检查只做机械对照
    （bug 修复方向/指标有限性），不做精度声明。
"""
import glob
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import soil_tools as st

BASE = r"E:/YYR/turangshuifen"
AREA = "sujiatun"
SHP = r"E:/YYR/turangshuifen/行政区划/苏家屯区/data.shp"
OLD = os.path.join(BASE, "test", AREA)
WORK = os.path.join(BASE, "work")
CSV_DIR = os.path.join(OLD, "soim_values")

D51208, D60513 = "20251208", "20260513"

S1_CLIP_EXPECTED = set(st.S1_CLIP_FILES)
S2_CLIP_EXPECTED = set(st.S2_CLIP_FILES)
FEATURE_EXPECTED = S1_CLIP_EXPECTED | S2_CLIP_EXPECTED
# legacy 是验证对照物（复现旧品用），列入精确集合防止 .enp 等残留混入
RESULT_EXPECTED_51208 = {"feature_target.csv", "model.joblib", "metrics.json",
                         "soil_moisture_map.tif", "soil_moisture_map_legacy.tif"}
RESULT_EXPECTED_60513 = {"feature_target.csv", "model.joblib", "metrics.json",
                         "soil_moisture_map.tif"}

results = []


def check(name, ok, info=""):
    results.append((name, bool(ok), str(info)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {info}")


# ============================================================
# 辅助
# ============================================================
def file_set(folder, suffix=".tif"):
    return {f for f in os.listdir(folder) if f.endswith(suffix)} if os.path.isdir(folder) else set()


def grid_equal(a, b):
    return a.shape == b.shape and a.transform == b.transform and a.crs == b.crs


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_tree(root):
    snap = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            p = os.path.join(dirpath, f)
            stt = os.stat(p)
            snap[os.path.relpath(p, root)] = (stt.st_mtime_ns, stt.st_size, sha256_file(p))
    return snap


def csv_of(date):
    return os.path.join(CSV_DIR, f"soim_values{date}.csv")


def dirs_of(date):
    root = os.path.join(WORK, date)
    return {
        "root": root,
        "s1_yuchuli": os.path.join(root, "S1", "S1_yuchuli_1"),
        "s1_zt": os.path.join(root, "S1", "zhuan_tif_2"),
        "s1_clip": os.path.join(root, "S1", "clip"),
        "s2_yuchuli": os.path.join(root, "S2", "S2_yuchuli_1"),
        "s2_zt": os.path.join(root, "S2", "zhuan_tif_2"),
        "s2_mosaic": os.path.join(root, "S2", "mosaic_3"),
        "s2_clip": os.path.join(root, "S2", "clip"),
        "feature": os.path.join(root, "feature"),
        "result": os.path.join(root, "result"),
    }


def compare_tif_dir(old_dir, new_dir, tag):
    """逐文件：网格严格相等 + 数值 bit-exact（NaN 安全）。"""
    bad = []
    for f in sorted(file_set(old_dir)):
        p_old, p_new = os.path.join(old_dir, f), os.path.join(new_dir, f)
        if not os.path.isfile(p_new):
            bad.append(f"{tag}/{f} 新侧缺失")
            continue
        with rasterio.open(p_old) as a, rasterio.open(p_new) as b:
            if not grid_equal(a, b):
                bad.append(f"{tag}/{f} 网格不等")
            elif not np.array_equal(a.read(1), b.read(1), equal_nan=True):
                bad.append(f"{tag}/{f} 数值不等")
    return bad


# ============================================================
# V-S1：数据盘查（51208 / 60513）
# ============================================================
def v_s1(date):
    r = st.soil_inspect_data(date=date)
    old = r["stages"]["test"].get(date)
    if old is None:
        check(f"V-S1 {date} 旧链目录存在", False, "inspect 未发现 test 目录")
        return
    if date == D51208:
        exp = {"S1": {"data_0_zip": 1, "yuchuli_dim": 1, "zhuan_tif_2_tif": 9, "clip_tif": 9},
               "S2": {"data_0_zip": 1, "yuchuli_dim": 1, "zhuan_tif_2_tif": 11,
                      "mosaic_3_file": 0, "clip_tif": 11},
               "feature_tif": 20,
               "result_files": ["feature_target.csv", "soil_moisture_map.tif",
                                "soil_moisture_map.tif.enp"]}
    else:
        exp = {"S1": {"data_0_zip": 1, "yuchuli_dim": 0, "zhuan_tif_2_tif": 0, "clip_tif": 0},
               "S2": {"data_0_zip": 1, "yuchuli_dim": 0, "zhuan_tif_2_tif": 0,
                      "mosaic_3_file": 0, "clip_tif": 0},
               "feature_tif": 0, "result_files": []}
    ok = old == exp
    check(f"V-S1 {date} 旧链各阶段计数精确", ok, str(old) if not ok else "test 树与预期一致")

    if date == D51208:
        with rasterio.open(os.path.join(OLD, date, "result", "soil_moisture_map.tif")) as src:
            arr = src.read(1)
            nan_frac = float(np.isnan(arr).mean())
            v = arr[~np.isnan(arr)]
            ok = (src.shape == (2875, 7648) and src.dtypes[0] == "float32"
                  and str(src.crs) == "EPSG:4326" and src.nodata is None
                  and abs(float(v.min()) - 0.3074) < 0.0005
                  and abs(float(v.max()) - 0.3142) < 0.0005
                  and abs(float(v.mean()) - 0.3112) < 0.002
                  and abs(nan_frac - 0.526) < 0.02)
        check("V-S1 20251208 旧品属性", ok,
              f"shape={src.shape} {src.dtypes[0]} {src.crs} nodata={src.nodata} "
              f"{v.min():.4f}~{v.max():.4f} mean={v.mean():.4f} nan={nan_frac:.3f}")

    csvs = {c["file"]: c for c in r["csvs"]}
    ok = (csvs["soim_values2025042906.csv"]["rows"] == 101
          and csvs["soim_values2025051206.csv"]["rows"] == 101
          and csvs["soim_values20251208.csv"]["rows"] == 101
          and csvs["soim_values20260513.csv"]["rows"] == 101)
    check(f"V-S1 {date} 样本 CSV 清单", ok, "4 期各 101 行")


# ============================================================
# V-S2~V-S7：51208 Python 段复现 + 旧品复现 + 修复护栏
# ============================================================
def run_51208():
    print("=" * 60)
    print("===== 51208 段：Python 段复现 / 旧品 bit-exact / 修复产品 =====")
    print("=" * 60)
    d = dirs_of(D51208)
    csv = csv_of(D51208)

    # ---- V-S1a ----
    v_s1(D51208)

    # ---- 驱动：工具链（复用旧 yuchuli）----
    print("\n===== 驱动：51208 工具链 =====")
    r1 = st.soil_s1_features(os.path.join(OLD, D51208, "S1", "S1_yuchuli_1"), SHP,
                             d["s1_zt"], d["s1_clip"])
    print(f"  S1 features: clip={r1['clip_tif_count']} warns={r1['warns']}")
    r2 = st.soil_s2_features(os.path.join(OLD, D51208, "S2", "S2_yuchuli_1"), SHP,
                             d["s2_zt"], d["s2_mosaic"], d["s2_clip"])
    print(f"  S2 features: clip={r2['clip_tif_count']} warns={r2['warns']}")
    r3 = st.soil_assemble_features(d["s1_clip"], d["s2_clip"], csv,
                                   d["feature"], os.path.join(d["result"], "feature_target.csv"))
    print(f"  assemble: {r3['status']} valid={r3['valid_samples']} cols={r3['feature_columns']}")
    r4 = st.soil_train_model(os.path.join(d["result"], "feature_target.csv"), d["result"],
                             feature_dir=d["feature"])
    print(f"  train: {r4['status']} cv_rmse={r4['cv_rmse_mean']:.5f} cv_r2={r4['cv_r2_mean']:.4f}")
    r5 = st.soil_predict_map(d["feature"], os.path.join(d["result"], "model.joblib"),
                             os.path.join(d["result"], "soil_moisture_map_legacy.tif"),
                             mode="legacy", force=True)
    print(f"  predict legacy: {r5['shape']} {r5['value_min']:.4f}~{r5['value_max']:.4f} "
          f"nan={r5['nan_ratio']:.4f}")
    r6 = st.soil_predict_map(d["feature"], os.path.join(d["result"], "model.joblib"),
                             os.path.join(d["result"], "soil_moisture_map.tif"),
                             mode="correct", force=True)
    print(f"  predict correct: {r6['shape']} {r6['value_min']:.4f}~{r6['value_max']:.4f} "
          f"nan={r6['nan_ratio']:.4f} warns={r6['warns']}")

    # ---- V-S2：clip/zhuan_tif_2 bit-exact ----
    print("\n===== V-S2 Python 段复现 =====")
    bad = compare_tif_dir(os.path.join(OLD, D51208, "S1", "clip"), d["s1_clip"], "S1/clip")
    bad += compare_tif_dir(os.path.join(OLD, D51208, "S2", "clip"), d["s2_clip"], "S2/clip")
    for side in ("S1", "S2"):
        old_zt = os.path.join(OLD, D51208, side, "zhuan_tif_2")
        new_zt = d["s1_zt"] if side == "S1" else d["s2_zt"]
        for sub in sorted(file_set(old_zt, suffix="")):
            if os.path.isdir(os.path.join(old_zt, sub)):
                bad += compare_tif_dir(os.path.join(old_zt, sub), os.path.join(new_zt, sub),
                                       f"{side}/zhuan_tif_2/{sub}")
    check("V-S2 51208 clip+zhuan_tif_2 与旧品逐文件 bit-exact", not bad,
          f"{20 + 20} 文件" + (f" 不一致: {bad[:3]}" if bad else " 全一致"))
    check("V-S2 51208 S1 clip 文件集合 == 9 特征", file_set(d["s1_clip"]) == S1_CLIP_EXPECTED)
    check("V-S2 51208 S2 clip 文件集合 == 11 特征", file_set(d["s2_clip"]) == S2_CLIP_EXPECTED)

    # ---- V-S3：特征装配等价 ----
    print("\n===== V-S3 特征装配等价 =====")
    check("V-S3 51208 feature 文件集合 == 20 特征", file_set(d["feature"]) == FEATURE_EXPECTED)
    old_csv = pd.read_csv(os.path.join(OLD, D51208, "result", "feature_target.csv"))
    new_csv = pd.read_csv(os.path.join(d["result"], "feature_target.csv"))
    check("V-S3 51208 feature_target.csv 列名与旧品相等",
          list(old_csv.columns) == list(new_csv.columns), f"{old_csv.shape} vs {new_csv.shape}")
    check("V-S3 51208 feature_target.csv 逐值相等",
          np.array_equal(old_csv.values, new_csv.values), f"{new_csv.shape[0]}×{new_csv.shape[1]}")

    # ---- V-S4：训练确定性 ----
    print("\n===== V-S4 训练确定性 =====")
    model_p = os.path.join(d["result"], "model.joblib")
    metrics_p = os.path.join(d["result"], "metrics.json")
    bak_m, bak_j = model_p.replace(".joblib", "_a.joblib"), metrics_p.replace(".json", "_a.json")
    shutil.copy2(model_p, bak_m)
    shutil.copy2(metrics_p, bak_j)
    st.soil_train_model(os.path.join(d["result"], "feature_target.csv"), d["result"],
                        feature_dir=d["feature"], force=True)
    m_a, m_b = json.load(open(bak_j, encoding="utf-8")), json.load(open(metrics_p, encoding="utf-8"))
    check("V-S4 51208 两次训练 metrics.json 逐字段一致", m_a == m_b,
          f"cv_rmse={m_b['cv_rmse_mean']:.5f} cv_r2={m_b['cv_r2_mean']:.4f}")
    import joblib
    b_a, b_b = joblib.load(bak_m), joblib.load(model_p)
    X = np.random.RandomState(0).uniform(0, 1, (50, m_b["n_features"]))
    pa = b_a["model"].predict(b_a["scaler_X"].transform(X))
    pb = b_b["model"].predict(b_b["scaler_X"].transform(X))
    check("V-S4 51208 两次训练模型同输入预测 bit-exact", np.array_equal(pa, pb))
    grid_keys = {"n_estimators", "max_features", "max_depth", "min_samples_split", "min_samples_leaf"}
    legal = (set(m_b["best_params"]) == grid_keys
             and m_b["best_params"]["n_estimators"] in [50, 100, 200]
             and m_b["best_params"]["max_features"] in ["sqrt", "log2", None]
             and m_b["best_params"]["max_depth"] in [None, 10, 20, 30])
    check("V-S4 51208 best_params 合法", legal, str(m_b["best_params"]))
    check("V-S4 51208 特征重要性之和=1",
          abs(sum(m_b["feature_importances"]) - 1.0) < 1e-6, f"sum={sum(m_b['feature_importances']):.6f}")
    check("V-S4 51208 CV 指标有限", np.isfinite(m_b["cv_rmse_mean"]) and np.isfinite(m_b["cv_r2_mean"]))
    os.remove(bak_m)
    os.remove(bak_j)

    # ---- V-S5：旧品 bit-exact 复现 ----
    print("\n===== V-S5 旧品复现（legacy）=====")
    old_feat = [f for f in os.listdir(os.path.join(OLD, D51208, "feature")) if f.endswith(".tif")]
    new_feat = [f for f in os.listdir(d["feature"]) if f.endswith(".tif")]
    check("V-S5 前提：新旧 feature 目录 listdir 序一致（NTFS 排序稳定）", old_feat == new_feat,
          "序相同" if old_feat == new_feat else f"old={old_feat[:3]}... new={new_feat[:3]}...")
    with rasterio.open(os.path.join(OLD, D51208, "result", "soil_moisture_map.tif")) as a, \
         rasterio.open(os.path.join(d["result"], "soil_moisture_map_legacy.tif")) as b:
        oa, nb = a.read(1), b.read(1)
        mask_eq = np.array_equal(np.isnan(oa), np.isnan(nb))
        both = ~np.isnan(oa) & ~np.isnan(nb)
        maxd = float(np.max(np.abs(oa[both] - nb[both]))) if both.any() else float("nan")
    check("V-S5 51208 legacy 网格与旧品严格相等", grid_equal(a, b))
    check("V-S5 51208 legacy NaN 掩膜与旧品 array_equal", mask_eq)
    check("V-S5 51208 legacy 与旧品 bit-exact（max|Δ|≤1e-6）", maxd <= 1e-6, f"max|Δ|={maxd:.3e}")

    # ---- V-S6：bug 实证 ----
    print("\n===== V-S6 bug 实证（核心）=====")
    with rasterio.open(os.path.join(OLD, D51208, "result", "soil_moisture_map.tif")) as a, \
         rasterio.open(os.path.join(d["result"], "soil_moisture_map.tif")) as b:
        oa, ca = a.read(1), b.read(1)
        both = ~np.isnan(oa) & ~np.isnan(ca)
        maxd_c = float(np.max(np.abs(oa[both] - ca[both]))) if both.any() else float("nan")
        same_nan = float(np.mean(np.isnan(oa) == np.isnan(ca)))
    check("V-S6 51208 correct 与旧品显著不同（max|Δ|≥1e-4，旧品确用错位列序）",
          maxd_c >= 1e-4, f"max|Δ|={maxd_c:.4f}")
    check("V-S6 51208 新旧 NaN 掩膜大体一致（相同输入特征）", same_nan > 0.95, f"一致率={same_nan:.4f}")
    a_leg = st.soil_assess_model(model_p, os.path.join(d["result"], "feature_target.csv"),
                                 product_tif=os.path.join(d["result"], "soil_moisture_map_legacy.tif"),
                                 sample_csv=csv)
    a_cor = st.soil_assess_model(model_p, os.path.join(d["result"], "feature_target.csv"),
                                 product_tif=os.path.join(d["result"], "soil_moisture_map.tif"),
                                 sample_csv=csv)
    r2l = a_leg["sample_check"].get("r2")
    r2c = a_cor["sample_check"].get("r2")
    check("V-S6 51208 样本点 R²：correct ≥ legacy（机械对照：随机模拟样本+样本内回代，仅证列序修复方向）",
          r2c is not None and r2l is not None and r2c >= r2l - 0.01,
          f"legacy R²={r2l:.4f} correct R²={r2c:.4f}")

    # ---- V-S7：修复产品护栏 ----
    print("\n===== V-S7 修复产品护栏 =====")
    with rasterio.open(os.path.join(d["result"], "soil_moisture_map.tif")) as c:
        ca = c.read(1)
        nan_frac = float(np.isnan(ca).mean())
        v = ca[~np.isnan(ca)]
        check("V-S7 correct 网格与旧品相等", grid_equal(a, c))
        check("V-S7 correct dtype=float32 且 nodata=None（同旧口径）",
              c.dtypes[0] == "float32" and c.nodata is None)
        check("V-S7 correct 值域∈[0,1] 物理硬界",
              v.min() >= 0.0 and v.max() <= 1.0, f"{v.min():.4f}~{v.max():.4f}")
        check("V-S7 correct 值域∈[0.2,0.45] 软界", v.min() >= 0.2 and v.max() <= 0.45,
              f"{v.min():.4f}~{v.max():.4f}")
        check("V-S7 correct NaN 占比∈(0.3,0.8)（旧 52.6%±10pt）",
              0.3 < nan_frac < 0.8, f"nan={nan_frac:.4f}")
        diff = np.abs(oa[both] - ca[both])
        cc = float(np.corrcoef(oa[both], ca[both])[0, 1]) if both.sum() > 10 else float("nan")
        check("V-S7 correct vs 旧品 差图统计有限（mean|Δ|/corr 如实报告）",
              np.isfinite(diff.mean()) and np.isfinite(cc),
              f"mean|Δ|={diff.mean():.5f} corr={cc:.4f}")

    run_common(D51208, has_legacy=True)


# ============================================================
# V-S8：60513 gpt 全链
# ============================================================
def run_60513():
    print("=" * 60)
    print("===== 60513 段：gpt S1+S2 全链实跑 =====")
    print("=" * 60)
    d = dirs_of(D60513)
    csv = csv_of(D60513)

    v_s1(D60513)

    zips_s1 = sorted(glob.glob(os.path.join(OLD, D60513, "S1", "data_0", "*.zip")))
    zips_s2 = sorted(glob.glob(os.path.join(OLD, D60513, "S2", "data_0", "*.zip")))
    check("V-S8 60513 S1/S2 data_0 各 1 个 zip", len(zips_s1) == 1 and len(zips_s2) == 1,
          f"S1={[os.path.basename(z) for z in zips_s1]} S2={[os.path.basename(z) for z in zips_s2]}")
    if len(zips_s1) != 1 or len(zips_s2) != 1:
        print("[FAIL] V-S8 数据不全，跳过 60513 全链")
        return

    print("\n===== 驱动：60513 gpt 全链（S1 约 30-60 分钟，S2 约 30-90 分钟）=====")
    g1 = st.soil_s1_gpt(zips_s1[0], d["s1_yuchuli"])
    print(f"  S1 gpt: {g1['status']} {g1.get('duration_s', '')}s")
    g2 = st.soil_s2_gpt(zips_s2[0], d["s2_yuchuli"])
    print(f"  S2 gpt: {g2['status']} {g2.get('duration_s', '')}s")
    check("V-S8 60513 S1 gpt 成功", g1["status"] in ("success", "exists"), g1.get("note", ""))
    check("V-S8 60513 S2 gpt 成功", g2["status"] in ("success", "exists"), g2.get("note", ""))

    r1 = st.soil_s1_features(d["s1_yuchuli"], SHP, d["s1_zt"], d["s1_clip"])
    print(f"  S1 features: clip={r1['clip_tif_count']} warns={r1['warns']}")
    r2 = st.soil_s2_features(d["s2_yuchuli"], SHP, d["s2_zt"], d["s2_mosaic"], d["s2_clip"])
    print(f"  S2 features: clip={r2['clip_tif_count']} warns={r2['warns']}")
    check("V-S8 60513 S1 clip == 9 特征", file_set(d["s1_clip"]) == S1_CLIP_EXPECTED)
    check("V-S8 60513 S2 clip == 11 特征", file_set(d["s2_clip"]) == S2_CLIP_EXPECTED)
    check("V-S8 60513 sin/cos 在界（无 sin/cos 护栏告警）",
          not any("sin_rushejiao" in w for w in r1["warns"]), str(r1["warns"]))
    check("V-S8 60513 NDVI 在界（无 NDVI 护栏告警）",
          not any("NDVI" in w for w in r2["warns"]), str(r2["warns"]))
    with rasterio.open(os.path.join(d["s1_clip"], "rushejiao.tif")) as src:
        s1_grid = (src.shape, src.transform)
    with rasterio.open(os.path.join(d["s1_clip"], "sin_rushejiao.tif")) as src:
        vals = src.read(1)
        v = vals[~np.isnan(vals)]
        check("V-S8 60513 sin_rushejiao ∈ [-1,1]", v.size and float(v.min()) >= -1 and float(v.max()) <= 1,
              f"{float(v.min()):.4f}~{float(v.max()):.4f}")

    r3 = st.soil_assemble_features(d["s1_clip"], d["s2_clip"], csv,
                                   d["feature"], os.path.join(d["result"], "feature_target.csv"))
    print(f"  assemble: {r3['status']} raw={r3['raw_samples']} valid={r3['valid_samples']} "
          f"cols={r3['feature_columns']}")
    check("V-S8 60513 特征装配：20 列", r3["feature_columns"] == 20, f"cols={r3['feature_columns']}")
    check("V-S8 60513 特征装配：有效样本 50~101", 50 <= r3["valid_samples"] <= 101,
          f"valid={r3['valid_samples']}")

    r4 = st.soil_train_model(os.path.join(d["result"], "feature_target.csv"), d["result"],
                             feature_dir=d["feature"])
    print(f"  train: {r4['status']} cv_rmse={r4['cv_rmse_mean']:.5f} cv_r2={r4['cv_r2_mean']:.4f} "
          f"best={r4['best_params']}")
    check("V-S8 60513 训练成功且 CV 有限", r4["status"] in ("success", "exists")
          and np.isfinite(r4["cv_rmse_mean"]))

    r5 = st.soil_predict_map(d["feature"], os.path.join(d["result"], "model.joblib"),
                             os.path.join(d["result"], "soil_moisture_map.tif"),
                             mode="correct", force=True)
    print(f"  predict correct: {r5['shape']} {r5['value_min']}~{r5['value_max']} "
          f"nan={r5['nan_ratio']:.4f} warns={r5['warns']}")
    with rasterio.open(os.path.join(d["result"], "soil_moisture_map.tif")) as src:
        check("V-S8 60513 产品网格 == S1 clip 网格",
              (src.shape, src.transform) == s1_grid, f"{src.shape} @ {src.transform.a:.7f}")
        arr = src.read(1)
        v = arr[~np.isnan(arr)]
        nan_frac = float(np.isnan(arr).mean())
        check("V-S8 60513 产品值域∈[0,1] 硬界", v.size and v.min() >= 0 and v.max() <= 1,
              f"{float(v.min()):.4f}~{float(v.max()):.4f}")
        check("V-S8 60513 产品值域∈[0.2,0.45] 软界", v.size and v.min() >= 0.2 and v.max() <= 0.45,
              f"{float(v.min()):.4f}~{float(v.max()):.4f}")
        check("V-S8 60513 产品 NaN 占比∈(0,1)", 0.0 < nan_frac < 1.0, f"nan={nan_frac:.4f}")

    # 两期流程完整性（各期单独如实报告；无实测数据，不做时相对比——用户口径）
    a_51208 = st.soil_assess_model(os.path.join(WORK, D51208, "result", "model.joblib"),
                                   os.path.join(WORK, D51208, "result", "feature_target.csv"))
    a_60513 = st.soil_assess_model(os.path.join(d["result"], "model.joblib"),
                                   os.path.join(d["result"], "feature_target.csv"))
    print(f"  [各期指标] 51208: CV_RMSE={a_51208['cv']['rmse_mean']:.5f} "
          f"CV_R2={a_51208['cv']['r2_mean']:.4f} | 60513: CV_RMSE={a_60513['cv']['rmse_mean']:.5f} "
          f"CV_R2={a_60513['cv']['r2_mean']:.4f}（各期独立，不做对比解读）")
    check("V-S8 两期模型质量评估 CV R² 有限（各期如实报告，不做时相对比）",
          np.isfinite(a_51208["cv"]["r2_mean"]) and np.isfinite(a_60513["cv"]["r2_mean"]),
          f"60513 top3: {[n for n, _ in a_60513['feature_importance_top'][:3]]}")

    run_common(D60513, has_legacy=False)


# ============================================================
# V-S9 幂等 + V-S10 产物清单/JSON（两期共用）
# ============================================================
def run_common(date, has_legacy):
    print(f"\n===== V-S9 {date} 幂等 =====")
    d = dirs_of(date)
    csv = csv_of(date)
    before = snapshot_tree(d["root"])

    st.soil_inspect_data(date=date)
    if date == D60513:
        for z in sorted(glob.glob(os.path.join(OLD, date, "S1", "data_0", "*.zip"))):
            st.soil_s1_gpt(z, d["s1_yuchuli"])
        for z in sorted(glob.glob(os.path.join(OLD, date, "S2", "data_0", "*.zip"))):
            st.soil_s2_gpt(z, d["s2_yuchuli"])
        st.soil_s1_features(d["s1_yuchuli"], SHP, d["s1_zt"], d["s1_clip"])
        st.soil_s2_features(d["s2_yuchuli"], SHP, d["s2_zt"], d["s2_mosaic"], d["s2_clip"])
    else:
        st.soil_s1_features(os.path.join(OLD, date, "S1", "S1_yuchuli_1"), SHP, d["s1_zt"], d["s1_clip"])
        st.soil_s2_features(os.path.join(OLD, date, "S2", "S2_yuchuli_1"), SHP,
                            d["s2_zt"], d["s2_mosaic"], d["s2_clip"])
    st.soil_assemble_features(d["s1_clip"], d["s2_clip"], csv,
                              d["feature"], os.path.join(d["result"], "feature_target.csv"))
    st.soil_train_model(os.path.join(d["result"], "feature_target.csv"), d["result"],
                        feature_dir=d["feature"])
    st.soil_predict_map(d["feature"], os.path.join(d["result"], "model.joblib"),
                        os.path.join(d["result"], "soil_moisture_map.tif"), mode="correct")
    if has_legacy:
        st.soil_predict_map(d["feature"], os.path.join(d["result"], "model.joblib"),
                            os.path.join(d["result"], "soil_moisture_map_legacy.tif"), mode="legacy")
    st.soil_assess_model(os.path.join(d["result"], "model.joblib"),
                         os.path.join(d["result"], "feature_target.csv"))

    after = snapshot_tree(d["root"])
    check(f"V-S9 {date} 全部工具重跑后产物树 mtime+sha256 逐文件不变", before == after,
          f"{len(before)} 文件全部不变" if before == after else
          f"变化: {[k for k in before if k not in after or before[k] != after.get(k)][:5]}")

    print(f"\n===== V-S10 {date} 产物清单 + JSON =====")
    exp_result = RESULT_EXPECTED_51208 if has_legacy else RESULT_EXPECTED_60513
    check(f"V-S10 {date} S1 clip 精确集合（无 .enp 残留）", file_set(d["s1_clip"]) == S1_CLIP_EXPECTED)
    check(f"V-S10 {date} S2 clip 精确集合（无 .enp 残留）", file_set(d["s2_clip"]) == S2_CLIP_EXPECTED)
    check(f"V-S10 {date} feature 精确集合", file_set(d["feature"]) == FEATURE_EXPECTED)
    check(f"V-S10 {date} result 精确集合", file_set(d["result"], suffix="") == exp_result,
          str(sorted(file_set(d["result"], suffix=""))))

    payloads = [
        st.soil_inspect_data(date=date),
        st.soil_s1_features(d["s1_zt"].replace("zhuan_tif_2", "S1_yuchuli_1"), SHP,
                            d["s1_zt"], d["s1_clip"]) if date == D60513
        else st.soil_s1_features(os.path.join(OLD, date, "S1", "S1_yuchuli_1"), SHP,
                                 d["s1_zt"], d["s1_clip"]),
        st.soil_s2_features(d["s2_zt"].replace("zhuan_tif_2", "S2_yuchuli_1"), SHP,
                            d["s2_zt"], d["s2_mosaic"], d["s2_clip"]) if date == D60513
        else st.soil_s2_features(os.path.join(OLD, date, "S2", "S2_yuchuli_1"), SHP,
                                 d["s2_zt"], d["s2_mosaic"], d["s2_clip"]),
        st.soil_assemble_features(d["s1_clip"], d["s2_clip"], csv,
                                  d["feature"], os.path.join(d["result"], "feature_target.csv")),
        st.soil_train_model(os.path.join(d["result"], "feature_target.csv"), d["result"],
                            feature_dir=d["feature"]),
        st.soil_predict_map(d["feature"], os.path.join(d["result"], "model.joblib"),
                            os.path.join(d["result"], "soil_moisture_map.tif"), mode="correct"),
        st.soil_assess_model(os.path.join(d["result"], "model.joblib"),
                             os.path.join(d["result"], "feature_target.csv")),
    ]
    n_ok = 0
    for p in payloads:
        try:
            json.dumps(p, ensure_ascii=False, allow_nan=False)
            n_ok += 1
        except (TypeError, ValueError) as e:
            check(f"V-S10 {date} JSON 序列化失败", False, str(e))
    check(f"V-S10 {date} 全部工具返回 json.dumps(allow_nan=False) 直过",
          n_ok == len(payloads), f"{n_ok}/{len(payloads)}")


# ============================================================
# 汇总
# ============================================================
def summary():
    print("\n" + "=" * 60)
    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_pass
    print(f"共 {len(results)} 项，PASS {n_pass}，FAIL {n_fail}")
    for name, ok, info in results:
        if not ok:
            print(f"  [FAIL] {name}  {info}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    if phase in ("51208", "all"):
        run_51208()
    if phase in ("60513", "all"):
        run_60513()
    summary()
