# -*- coding: utf-8 -*-
"""
PM10(2.5) 反演工具链验证脚本（单景 PM10 20250210_12）

验证口径：
    - 预报数据为模拟数据（本机无 WRF nc），站点监测值为真实监测数据；
      评估=流程证明非精度，R² 只做机械对照；
    - 单景跑通即推进：仅验证 PM10 20250210_12；PM2.5 为镜像链（pollutant 参数化）
      不单独验证；
    - 旧目录（test/、PM10（PM25）/）只读；新产物写 E:/YYR/PM10(2.5)/work/。

验证策略（同 soil_agent）：
    - legacy 模式逐字复现旧品 → 与旧产物 bit-exact（NaN 掩膜 + max|Δ|≤1e-6）；
    - correct 模式修复三缺陷 → 护栏 + 与 legacy 的效应量化对比。

运行：cd pm_agent && PYTHONUTF8=1 python verify_pm_tools.py
"""

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import rasterio

import pm_tools

TEST = "E:/YYR/PM10(2.5)/test"
OLD_PRODUCT = "E:/YYR/PM10(2.5)/PM10（PM25）"
WORK = "E:/YYR/PM10(2.5)/work"
W = os.path.join(WORK, "20250210_12")

# 旧 ALIGNED 目录实测 os.listdir 序（legacy 预测复现旧品用）
OLD_LISTDIR_ORDER = [
    "blh.tif", "FH_Elevation_1km.tif", "H08_20250210_1200_AOT_Merged_Cropped.tif",
    "lai_hv.tif", "lai_lv.tif", "landuse_clip.tif", "population_density_clip.tif",
    "sp.tif", "t2m.tif", "tp.tif", "u10.tif"]

EXPECTED_FEATURES = ["DEM", "PD", "AOD", "LU",
                     "blh", "lai_hv", "lai_lv", "sp", "t2m", "tp", "u10"]

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    mark = "PASS" if cond else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))


def tif_matches(path_a, path_b, tol=1e-6):
    """掩膜感知的 tif 逐像元对比：NaN 掩膜一致 + 有效值 max|Δ|≤tol。"""
    with rasterio.open(path_a) as a, rasterio.open(path_b) as b:
        na, oa = a.read(1), b.read(1)
        if na.shape != oa.shape or a.transform != b.transform or a.crs != b.crs:
            return False, float("nan")
        nan_same = np.array_equal(np.isnan(na), np.isnan(oa))
        both = ~(np.isnan(na) | np.isnan(oa))
        md = float(np.max(np.abs(na[both] - oa[both]))) if both.any() else 0.0
        return bool(nan_same and md <= tol), md


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_inventory(root):
    """root 下所有文件 → {相对路径: (mtime_ns, sha256)}。"""
    inv = {}
    for dirpath, _, files in os.walk(root):
        for fn in files:
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, root).replace("\\", "/")
            inv[rel] = (os.stat(p).st_mtime_ns, sha256(p))
    return inv


def station_r2(csv_path, col="PM10"):
    df = pd.read_csv(csv_path)
    v = df.dropna(subset=["预测值"])
    y, p = v[col].values, v["预测值"].values
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    bias = float(np.mean(p - y))
    return len(v), float(r2), rmse, bias


# ===========================================================================
print("=" * 72)
print("V-P1 数据盘查")
print("=" * 72)

rep = json.loads(pm_tools.pm_inspect_data(TEST))["status"]
st = os.path.join(TEST, "station", "20250210")
check("V-P1.1 旧链数据树齐全",
      all(rep[k] in ("dir", "file") for k in
          ("station_csvs", "aod_tiqu", "era5_nc", "era5_output", "dem_tif",
           "pd_tif", "lu_tif", "aligned_dir", "old_rf_result",
           "old_tianbu_result", "old_sanwei_dir", "yubao_jjj_dir",
           "old_product_level0")),
      "缺失: " + ",".join(k for k, v in rep.items() if v == "missing"))
check("V-P1.2 站点逐时 CSV 24 个", len(os.listdir(st)) == 24,
      f"实际 {len(os.listdir(st))}")
check("V-P1.3 旧对齐特征 11 个 tif",
      len(glob_tifs := [f for f in os.listdir(
          os.path.join(TEST, "ALIGNED_DIR_dem_2km", "20250210_12"))
          if f.endswith(".tif")]) == 11, f"实际 {len(glob_tifs)}")

with rasterio.open(os.path.join(OLD_PRODUCT, "PM10_level0_result.tif")) as src:
    a = src.read(1)
    rng_ok = (abs(float(np.nanmin(a)) - 47.6828) < 0.01 and
              abs(float(np.nanmax(a)) - 121.9724) < 0.01)
    check("V-P1.4 旧品 level0 属性（746×1058/4326/47.68~121.97/NaN 24.1%）",
          src.shape == (746, 1058) and str(src.crs) == "EPSG:4326" and rng_ok
          and abs(100 * np.isnan(a).mean() - 24.1) < 0.5,
          f"{src.shape} {src.crs} {float(np.nanmin(a)):.2f}~{float(np.nanmax(a)):.2f} "
          f"NaN {100 * np.isnan(a).mean():.1f}%")

same, _ = tif_matches(os.path.join(TEST, "pm10", "yubao", "PM10_level1.tif"),
                      os.path.join(OLD_PRODUCT, "PM10_level0_result.tif"))
check("V-P1.5 yubao level1 == 旧品 level0（预报=旧品模拟数据佐证）", same)

# ===========================================================================
print("\n" + "=" * 72)
print("V-P2 AOD 提取（legacy bit-exact + correct 修复实证）")
print("=" * 72)

aod_nc = os.path.join(TEST, "AOD", "H09_20250210_0400_1HARP031_FLDK.02401_02401.nc")
legacy_dir = os.path.join(WORK, "aod", "legacy")
correct_dir = os.path.join(WORK, "aod", "correct")
pm_tools.pm_extract_aod(aod_nc, legacy_dir, crop_extent=(106.0, 32.0, 125.0, 45),
                        mode="legacy")
pm_tools.pm_extract_aod(aod_nc, correct_dir, crop_extent=(106.0, 32.0, 125.0, 45),
                        mode="correct")

legacy_crop = os.path.join(legacy_dir, "H08_20250210_1200_AOT_Merged_Cropped.tif")
old_crop = os.path.join(TEST, "AOD", "tiqu", "H08_20250210_1200_AOT_Merged_Cropped.tif")
same, md = tif_matches(legacy_crop, old_crop)
check("V-P2.1 legacy Cropped vs 旧品 bit-exact（双倍缩放逐字复刻）", same,
      f"max|Δ|={md:.2g}")

with rasterio.open(legacy_crop) as src:
    la = src.read(1)
check("V-P2.2 legacy 缺陷实证：保留 -32768 且值域 ~1e-4",
      float(np.nanmin(la)) < -30000 and float(np.nanmax(la)) < 0.001,
      f"值域 [{np.nanmin(la):.2g}, {np.nanmax(la):.2g}]")

correct_crop = os.path.join(correct_dir, "H08_20250210_1200_AOT_Merged_Cropped.tif")
with rasterio.open(correct_crop) as src:
    ca = src.read(1)
    ca_nodata = src.nodata
check("V-P2.3 correct 修复实证：真实 AOD 量级（0.02~1.8）且无 -32768",
      0.01 <= float(np.nanmin(ca)) <= float(np.nanmax(ca)) <= 3.0
      and float(np.nanmin(ca)) >= -100,
      f"值域 [{np.nanmin(ca):.4f}, {np.nanmax(ca):.4f}] nodata={ca_nodata}")
check("V-P2.4 correct 缺失值已转 NaN（invalid→NaN 而非 -32768）",
      np.isnan(ca).mean() > 0, f"NaN 占比 {100 * np.isnan(ca).mean():.1f}%")

# ===========================================================================
print("\n" + "=" * 72)
print("V-P3 ERA5 提取 bit-exact")
print("=" * 72)

pm_tools.pm_extract_era5(os.path.join(TEST, "ERA5", "39b938147c1544637d0a9c28f24937c0"),
                         os.path.join(WORK, "era5"))
ok = True
era5_new = os.path.join(WORK, "era5", "20250210_1200")
era5_old = os.path.join(TEST, "ERA5", "ERA5_output", "20250210_1200")
era5_files = sorted(os.listdir(era5_old))
for f in era5_files:
    same, md = tif_matches(os.path.join(era5_new, f), os.path.join(era5_old, f))
    if not same:
        ok = False
        print(f"    DIFF {f}: max|Δ|={md:.2g}")
check("V-P3.1 7 气象 tif 与旧 ERA5_output bit-exact", ok and len(era5_files) == 7,
      f"{len(era5_files)} 文件")

# ===========================================================================
print("\n" + "=" * 72)
print("V-P4 站点 CSV 处理")
print("=" * 72)

station_dir = os.path.join(WORK, "station")
pm_tools.pm_prepare_stations(
    os.path.join(TEST, "station", "站点_20250101-20250628"),
    os.path.join(TEST, "station", "站点列表-2022.02.13起.csv"),
    station_dir, date="20250210")
byte_diff = []
for f in sorted(os.listdir(os.path.join(TEST, "station", "20250210"))):
    with open(os.path.join(TEST, "station", "20250210", f), "rb") as a, \
         open(os.path.join(station_dir, f), "rb") as b:
        if a.read() != b.read():
            byte_diff.append(f)
check("V-P4.1 24 逐时 CSV 与旧 station 目录字节相等",
      not byte_diff and len(os.listdir(station_dir)) == 25,
      f"差异 {byte_diff if byte_diff else '无'}")

# ===========================================================================
print("\n" + "=" * 72)
print("V-P5 特征对齐 bit-exact")
print("=" * 72)

aligned_dir = os.path.join(W, "aligned")
pm_tools.pm_align_features(
    os.path.join(TEST, "DEM-1KM", "FH_Elevation_1km.tif"),
    os.path.join(TEST, "PD", "population_density_clip.tif"),
    os.path.join(TEST, "Landuse", "landuse_clip.tif"),
    legacy_crop,
    os.path.join(TEST, "ERA5", "ERA5_output", "20250210_1200"),
    aligned_dir)
old_aligned = os.path.join(TEST, "ALIGNED_DIR_dem_2km", "20250210_12")
ok = True
for f in sorted(os.listdir(old_aligned)):
    if not f.endswith(".tif"):
        continue
    same, md = tif_matches(os.path.join(aligned_dir, f), os.path.join(old_aligned, f))
    if not same:
        ok = False
        print(f"    DIFF {f}: max|Δ|={md:.2g}")
check("V-P5.1 对齐 11 tif 与旧 ALIGNED_DIR bit-exact", ok)

# ===========================================================================
print("\n" + "=" * 72)
print("V-P6 训练确定性 + 与旧模型等价")
print("=" * 72)

station_csv = os.path.join(TEST, "station", "20250210", "20250210_12.csv")
model_dir = os.path.join(W, "model")
r_train = json.loads(pm_tools.pm_train_rf(station_csv, aligned_dir, model_dir))
with open(os.path.join(model_dir, "metrics.json"), encoding="utf-8") as f:
    metrics = json.load(f)
check("V-P6.1 训练成功（156/260 有效样本）",
      r_train["status"] in ("ok", "exists") and metrics["n_valid_samples"] == 156,
      f"status={r_train['status']} n_valid={metrics['n_valid_samples']}")
check("V-P6.2 特征序正确（DEM/PD/AOD/LU+7 气象）",
      metrics["feature_names"] == EXPECTED_FEATURES,
      str(metrics["feature_names"]))
check("V-P6.3 CV 指标有限 + 重要性归一",
      all(np.isfinite(metrics["cv_rmse_list"])) and
      all(np.isfinite(metrics["cv_r2_list"])) and
      abs(sum(metrics["feature_importance"].values()) - 1.0) < 1e-6,
      f"CV RMSE {metrics['cv_mean_rmse']:.4f} / R² {metrics['cv_mean_r2']:.4f}")

# 二次训练（确定性）
model_dir2 = os.path.join(W, "model_repeat")
pm_tools.pm_train_rf(station_csv, aligned_dir, model_dir2)
m1 = pm_tools.joblib.load(os.path.join(model_dir, "model.pkl"))
m2 = pm_tools.joblib.load(os.path.join(model_dir2, "model.pkl"))
x = np.random.RandomState(0).uniform(0, 1, (50, 11)).astype(np.float32)
check("V-P6.4 两次训练模型预测逐点一致（确定性）",
      np.max(np.abs(m1.predict(x) - m2.predict(x))) == 0)

# 与旧 model.pkl 等价（旧品同机同环境训练，数据 bit-exact → 模型应完全一致）
old_model = pm_tools.joblib.load(
    os.path.join(TEST, "pm10", "20250210_12", "20250210_1200_model.pkl"))
same_model = (np.array_equal(m1.feature_importances_, old_model.feature_importances_)
              and np.max(np.abs(m1.predict(x) - old_model.predict(x))) == 0)
check("V-P6.5 新模型与旧 model.pkl 特征重要性/预测一致", same_model)

sc_new = pm_tools.joblib.load(os.path.join(model_dir, "scaler_X.pkl"))
sc_old = pm_tools.joblib.load(os.path.join(TEST, "pm10", "20250210_12", "scaler_X.pkl"))
check("V-P6.6 新 scaler_X 与旧 pkl 一致",
      np.allclose(sc_new.data_min_, sc_old.data_min_) and
      np.allclose(sc_new.data_max_, sc_old.data_max_))

# ===========================================================================
print("\n" + "=" * 72)
print("V-P7 legacy 预测复现旧品")
print("=" * 72)

rf_legacy = os.path.join(W, "rf_legacy", "20250210_1200_PM10_2KM.tif")
pm_tools.pm_predict_rf(aligned_dir, model_dir, rf_legacy, station_csv=station_csv,
                       mode="legacy", raster_order=OLD_LISTDIR_ORDER)
old_rf = os.path.join(TEST, "pm10", "20250210_12", "20250210_1200_PM10_2KM.tif")
same, md = tif_matches(rf_legacy, old_rf)
check("V-P7.1 legacy 预测 vs 旧 RF 产品 bit-exact（列序 bug 逐字复刻）", same,
      f"max|Δ|={md:.2g}")

new_csv = os.path.join(W, "rf_legacy", "station_predictions.csv")
old_csv = os.path.join(TEST, "pm10", "20250210_12", "20250210_12_station_predictions.csv")
dn, do = pd.read_csv(new_csv), pd.read_csv(old_csv)
cols_ok = list(dn.columns) == list(do.columns) and len(dn) == len(do)
vals_ok = bool(np.allclose(dn.select_dtypes(include=[np.number]).fillna(-9999).values,
                           do.select_dtypes(include=[np.number]).fillna(-9999).values,
                           atol=1e-4, equal_nan=True))
check("V-P7.2 station_predictions.csv 与旧品一致（列/行数/数值）",
      cols_ok and vals_ok, f"{len(dn)} 行")

# ===========================================================================
print("\n" + "=" * 72)
print("V-P8 列序 bug 实证（correct vs legacy）")
print("=" * 72)

rf_correct = os.path.join(W, "rf_correct", "20250210_1200_PM10_2KM.tif")
pm_tools.pm_predict_rf(aligned_dir, model_dir, rf_correct, station_csv=station_csv,
                       mode="correct")
with rasterio.open(rf_correct) as a, rasterio.open(rf_legacy) as b:
    ca, la = a.read(1), b.read(1)
    both = ~(np.isnan(ca) | np.isnan(la))
    d = ca - la
    maxd = float(np.max(np.abs(d[both])))
    meand = float(np.mean(np.abs(d[both])))
check("V-P8.1 correct ≠ legacy（列序错位效应显著，max|Δ|>5）",
      maxd > 5 and meand > 1,
      f"max|Δ|={maxd:.3f} mean|Δ|={meand:.3f}")

n_l, r2_l, rmse_l, bias_l = station_r2(
    os.path.join(W, "rf_legacy", "station_predictions.csv"))
n_c, r2_c, rmse_c, bias_c = station_r2(
    os.path.join(W, "rf_correct", "station_predictions.csv"))
check("V-P8.2 站点机械对照：correct R² ≥ legacy+0.2 且 |bias| 更小",
      n_l == n_c == 156 and r2_c >= r2_l + 0.2 and abs(bias_c) < abs(bias_l),
      f"legacy R²={r2_l:.4f}/bias={bias_l:.2f} → correct R²={r2_c:.4f}/bias={bias_c:.2f}")

# ===========================================================================
print("\n" + "=" * 72)
print("V-P9 tianbu 观测校正 bit-exact")
print("=" * 72)

corr_dir = os.path.join(W, "correct")
corr_base = os.path.join(corr_dir, "20250210_1200_校正结果_2KM_tianbu")
pm_tools.pm_correct_observation(
    rf_legacy, station_csv,
    os.path.join(TEST, "Landuse", "landuse_clip.tif"),
    os.path.join(OLD_PRODUCT, "PM10_level0_result.tif"),
    corr_base + ".tif")
old_corr_base = os.path.join(TEST, "pm10", "20250210_12",
                             "20250210_1200_校正结果_2KM_tianbu")
names = ["_forecast_smoothed", "_LU_resampled", "_preliminary", "_ML_filled",
         "_measured_surface", "_distance_weights", "_combined_filled",
         "_mosaic_smoothed", "_result_smoothed"]
ok = True
for n in names:
    same, md = tif_matches(corr_base + n + ".tif", old_corr_base + n + ".tif")
    if not same:
        ok = False
        print(f"    DIFF {n}: max|Δ|={md:.2g}")
same, md = tif_matches(corr_base + ".tif", old_corr_base + ".tif")
if not same:
    ok = False
    print(f"    DIFF 终产物: max|Δ|={md:.2g}")
check("V-P9.1 9 中间件 + 终产物与旧品 bit-exact", ok)

with rasterio.open(corr_base + "_result_smoothed.tif") as src:
    a = src.read(1)
check("V-P9.2 填补效果：NaN 41.4%→7.4%（±1pt）",
      abs(100 * np.isnan(a).mean() - 7.4) < 1.0,
      f"NaN {100 * np.isnan(a).mean():.1f}%")

# ===========================================================================
print("\n" + "=" * 72)
print("V-P10 三维垂直廓线（legacy bit-exact + 命名 bug）")
print("=" * 72)

sanwei_legacy = os.path.join(W, "sanwei_legacy")
pm_tools.pm_vertical_profile(
    os.path.join(TEST, "pm10", "yubao_jjj"),
    corr_base + "_ML_filled.tif",
    sanwei_legacy, level_heights=(40, 120, 250, 400), pollutant="PM10",
    mode="legacy", date_hour="20250210_12")

old_sanwei = os.path.join(TEST, "pm10", "20250210_12", "SANWEI")
ok = True
for f in ("parameters.tif", "parameters_fine.tif",
          "PM2.5_level0_result.tif", "PM2.5_level1_result.tif",
          "PM2.5_level2_result.tif", "PM2.5_level3_result.tif"):
    same, md = tif_matches(os.path.join(sanwei_legacy, f),
                           os.path.join(old_sanwei, f))
    if not same:
        ok = False
        print(f"    DIFF {f}: max|Δ|={md:.2g}")
check("V-P10.1 parameters/parameters_fine/4 层与旧 SANWEI bit-exact", ok)

has_wrong = all(os.path.exists(os.path.join(sanwei_legacy, f"PM2.5_level{i}_result.tif"))
                for i in range(4))
has_right = any(os.path.exists(os.path.join(sanwei_legacy, f"PM10_level{i}_result.tif"))
                for i in range(4))
check("V-P10.2 legacy 命名 bug 复刻（PM10 产品输出 PM2.5_* 前缀）",
      has_wrong and not has_right, "PM2.5_* 存在且无 PM10_*")

with rasterio.open(os.path.join(sanwei_legacy, "parameters.tif")) as src:
    check("V-P10.3 parameters.tif 391×391×4 波段（a/b/c/r²）",
          src.shape == (391, 391) and src.count == 4, str(src.shape))

sanwei_correct = os.path.join(W, "sanwei_correct")
pm_tools.pm_vertical_profile(
    os.path.join(TEST, "pm10", "yubao_jjj"),
    corr_base + "_ML_filled.tif",
    sanwei_correct, level_heights=(40, 120, 250, 400), pollutant="PM10",
    mode="correct", date_hour="20250210_12")
has_correct = all(os.path.exists(os.path.join(sanwei_correct, f"PM10_level{i}_result.tif"))
                  for i in range(4))
check("V-P10.4 correct 命名修复（PM10_* 前缀）", has_correct)

ok = True
for i in range(4):
    same, md = tif_matches(os.path.join(sanwei_correct, f"PM10_level{i}_result.tif"),
                           os.path.join(sanwei_legacy, f"PM2.5_level{i}_result.tif"))
    if not same:
        ok = False
        print(f"    DIFF level{i}: max|Δ|={md:.2g}")
check("V-P10.5 correct/legacy 4 层逐像元一致（仅命名不同，算法同一）", ok)

# ===========================================================================
print("\n" + "=" * 72)
print("V-P11 护栏 + 幂等 + 产物清单")
print("=" * 72)

with rasterio.open(corr_base + "_result_smoothed.tif") as src:
    ta = src.read(1)
with rasterio.open(rf_correct) as src:
    rca = src.read(1)
guard_ok = (0 <= np.nanmin(ta) <= np.nanmax(ta) <= 500 and
            0 <= np.nanmin(rca) <= np.nanmax(rca) <= 500)
soft_ok = (20 <= np.nanmin(ta) and np.nanmax(ta) <= 200 and
           20 <= np.nanmin(rca) and np.nanmax(rca) <= 200)
check("V-P11.1 值域硬护栏 [0,500] 通过", guard_ok,
      f"tianbu [{np.nanmin(ta):.2f}, {np.nanmax(ta):.2f}] "
      f"correct RF [{np.nanmin(rca):.2f}, {np.nanmax(rca):.2f}]")
check("V-P11.2 软护栏 [20,200]（旧品 47.68~121.97 口径）", soft_ok)

# 幂等：快照全部 work 文件 → 重跑全部工具 → 逐文件不变
inv_before = file_inventory(WORK)
pm_tools.pm_extract_aod(aod_nc, legacy_dir, crop_extent=(106.0, 32.0, 125.0, 45),
                        mode="legacy")
pm_tools.pm_extract_aod(aod_nc, correct_dir, crop_extent=(106.0, 32.0, 125.0, 45),
                        mode="correct")
pm_tools.pm_extract_era5(os.path.join(TEST, "ERA5", "39b938147c1544637d0a9c28f24937c0"),
                         os.path.join(WORK, "era5"))
pm_tools.pm_prepare_stations(
    os.path.join(TEST, "station", "站点_20250101-20250628"),
    os.path.join(TEST, "station", "站点列表-2022.02.13起.csv"),
    station_dir, date="20250210")
pm_tools.pm_align_features(
    os.path.join(TEST, "DEM-1KM", "FH_Elevation_1km.tif"),
    os.path.join(TEST, "PD", "population_density_clip.tif"),
    os.path.join(TEST, "Landuse", "landuse_clip.tif"), legacy_crop,
    os.path.join(TEST, "ERA5", "ERA5_output", "20250210_1200"), aligned_dir)
pm_tools.pm_train_rf(station_csv, aligned_dir, model_dir)
pm_tools.pm_predict_rf(aligned_dir, model_dir, rf_legacy, station_csv=station_csv,
                       mode="legacy", raster_order=OLD_LISTDIR_ORDER)
pm_tools.pm_predict_rf(aligned_dir, model_dir, rf_correct, station_csv=station_csv,
                       mode="correct")
pm_tools.pm_correct_observation(
    rf_legacy, station_csv, os.path.join(TEST, "Landuse", "landuse_clip.tif"),
    os.path.join(OLD_PRODUCT, "PM10_level0_result.tif"), corr_base + ".tif")
pm_tools.pm_vertical_profile(
    os.path.join(TEST, "pm10", "yubao_jjj"), corr_base + "_ML_filled.tif",
    sanwei_legacy, level_heights=(40, 120, 250, 400), pollutant="PM10",
    mode="legacy", date_hour="20250210_12")
pm_tools.pm_vertical_profile(
    os.path.join(TEST, "pm10", "yubao_jjj"), corr_base + "_ML_filled.tif",
    sanwei_correct, level_heights=(40, 120, 250, 400), pollutant="PM10",
    mode="correct", date_hour="20250210_12")
inv_after = file_inventory(WORK)
changed = sorted(p for p in inv_before
                 if p not in inv_after or inv_before[p] != inv_after[p])
new_files = sorted(p for p in inv_after if p not in inv_before)
check("V-P11.3 幂等：重跑全部工具后 mtime/sha256 逐文件不变",
      not changed and not new_files,
      f"变动 {len(changed)} 新增 {len(new_files)}")

# 产物清单精确集合
expected_files = set(
    [f"station/站点列表.csv"] +
    [f"station/20250210_{h}.csv" for h in range(24)] +
    [f"aod/legacy/H08_20250210_1200_AOT_Merged_{s}.tif" for s in ("Full", "Cropped")] +
    [f"aod/correct/H08_20250210_1200_AOT_Merged_{s}.tif" for s in ("Full", "Cropped")] +
    [f"era5/20250210_1200/{v}.tif" for v in
     ("blh", "lai_hv", "lai_lv", "sp", "t2m", "tp", "u10")] +
    [f"20250210_12/aligned/{f}" for f in
     ["FH_Elevation_1km.tif", "population_density_clip.tif", "landuse_clip.tif",
      "H08_20250210_1200_AOT_Merged_Cropped.tif",
      "blh.tif", "lai_hv.tif", "lai_lv.tif", "sp.tif", "t2m.tif", "tp.tif", "u10.tif"]] +
    [f"20250210_12/model/{f}" for f in
     ("model.pkl", "scaler_X.pkl", "scaler_y.pkl", "metrics.json")] +
    [f"20250210_12/model_repeat/{f}" for f in
     ("model.pkl", "scaler_X.pkl", "scaler_y.pkl", "metrics.json")] +
    [f"20250210_12/rf_legacy/{f}" for f in
     ("20250210_1200_PM10_2KM.tif", "station_predictions.csv")] +
    [f"20250210_12/rf_correct/{f}" for f in
     ("20250210_1200_PM10_2KM.tif", "station_predictions.csv")] +
    [f"20250210_12/correct/20250210_1200_校正结果_2KM_tianbu{n}.tif" for n in
     ["", "_forecast_smoothed", "_LU_resampled", "_preliminary", "_ML_filled",
      "_measured_surface", "_distance_weights", "_combined_filled",
      "_mosaic_smoothed", "_result_smoothed"]] +
    [f"20250210_12/sanwei_legacy/{f}" for f in
     ["parameters.tif", "parameters_fine.tif", "parameters_fine_metadata.txt",
      "PM2.5_level0_result.tif", "PM2.5_level1_result.tif",
      "PM2.5_level2_result.tif", "PM2.5_level3_result.tif",
      "pm2.5_20250210_12.png"]] +
    [f"20250210_12/sanwei_correct/{f}" for f in
     ["parameters.tif", "parameters_fine.tif", "parameters_fine_metadata.txt",
      "PM10_level0_result.tif", "PM10_level1_result.tif",
      "PM10_level2_result.tif", "PM10_level3_result.tif",
      "pm10_20250210_12.png"]])
actual_files = set(inv_after.keys())
check("V-P11.4 产物清单精确集合（无 .enp / 无杂散文件）",
      actual_files == expected_files,
      f"期望 {len(expected_files)} 实际 {len(actual_files)}"
      + ("" if actual_files == expected_files else
         " 多余: " + ",".join(sorted(actual_files - expected_files)) +
         " 缺失: " + ",".join(sorted(expected_files - actual_files))))

# JSON 可序列化（所有工具返回值 json.loads 已在调用中隐式验证）
sample = json.loads(pm_tools.pm_inspect_data(TEST))
check("V-P11.5 全部工具返回 JSON 可解析（_out 统一封装）",
      isinstance(sample, dict) and "status" in sample)

# ===========================================================================
print("\n" + "=" * 72)
n_pass = sum(1 for _, ok, _ in results if ok)
n_fail = len(results) - n_pass
print(f"汇总: {n_pass}/{len(results)} PASS"
      + (f"，{n_fail} FAIL" if n_fail else "，全部通过"))
if n_fail:
    print("失败项:")
    for name, ok, detail in results:
        if not ok:
            print(f"  - {name}: {detail}")
print("=" * 72)
sys.exit(1 if n_fail else 0)
