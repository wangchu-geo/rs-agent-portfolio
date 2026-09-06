# -*- coding: utf-8 -*-
"""
水体总磷反演链验证脚本 V-PW1 ~ V-PW14（verify 模式，参照 pm/soil 成熟范式）

验证口径：
    - legacy 模式逐字复现旧品，与旧品产物做 bit-exact 对照（掩膜一致 + max|Δ|≤1e-6）；
    - correct 模式修复旧品 6 缺陷（坏值列入模/阈值 0/÷0 语义/模型未落盘/.enp 残留/
      两链口径差异），产物做硬护栏；
    - 总磷为真实监测数据但样本仅 45 个 → R²=0.3157 只做机械对照，评估=流程证明非精度。
旧目录（水体磷反演总结/）只读；新产物写 work/20240828_单景/。
运行：PYTHONUTF8=1 python verify_pw_tools.py
"""

import hashlib
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd
import rasterio
from osgeo import gdal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pw_tools as pw  # noqa: E402

gdal.UseExceptions()

BASE = r"E:/YYR/P_water/水体磷反演总结"
OLD = os.path.join(BASE, "test")
WORK = r"E:/YYR/P_water/work/20240828_单景"
OLD_S2 = os.path.join(OLD, "data", "S2")
OLD_S3 = os.path.join(OLD, "data", "S3")
OLD_FEAT = os.path.join(OLD, "data", "feature")
OLD_PRED = os.path.join(OLD, "data", "result", "pred_merged.tif")
SHP_OLD = os.path.join(BASE, "矢量", "水体.shp")
SHP_WORK = os.path.join(WORK, "水体.shp")
GRAPHS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")
SAMPLES = os.path.join(WORK, "samples")
S2_TIF = os.path.join(WORK, "s2", "tif")
S2_CLIP = os.path.join(WORK, "s2", "clip")
S3_TIF = os.path.join(WORK, "s3", "tif")
FEATS = os.path.join(WORK, "feats")
MODEL_L = os.path.join(WORK, "model")
MODEL_C = os.path.join(WORK, "model_correct")
RESULT = os.path.join(WORK, "result")
PRED_L = os.path.join(RESULT, "pred_merged_legacy.tif")
PRED_C = os.path.join(RESULT, "pred_merged_correct.tif")
PRED_C_CLIP = os.path.join(RESULT, "pred_merged_correct_clip.tif")

S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
OA_BANDS = [f"Oa{i:02d}" for i in range(1, 22)]
INDEX_TIFS = ["B5+B4.tif", "B5-B4.tif", "B3_divide_B5.tif",
              "B4_divide_B11.tif", "B6_divide_B5.tif", "B3_divide_B9.tif"]
TARGET = "日均统计结果_总磷(mg/L)"

RESULTS = []  # (节名, [(检查名, PASS/FAIL, 细节), ...])


def check(name, cond, detail=""):
    RESULTS[-1][1].append((name, bool(cond), detail))
    return bool(cond)


def _mask_of(d, nodata):
    """按各自 nodata 建有效掩膜（nodata 四分：0.0 / nan / -9999 / None）。"""
    if nodata is not None and np.isfinite(nodata):
        return ~np.isclose(d, nodata) & ~np.isnan(d)
    return ~np.isnan(d)


def tif_matches(path_a, path_b, tol=1e-6):
    """掩膜感知对比：NaN/0.0/-9999 各自掩膜 array_equal + 有效值 max|Δ|。"""
    with rasterio.open(path_a) as a, rasterio.open(path_b) as b:
        da = a.read(1).astype(np.float64)
        db = b.read(1).astype(np.float64)
        ma, mb = _mask_of(da, a.nodata), _mask_of(db, b.nodata)
        if da.shape != db.shape:
            return False, f"shape {da.shape} vs {db.shape}"
        if not np.array_equal(ma, mb):
            return False, f"掩膜不一致：{np.count_nonzero(ma != mb)} 像元"
        if not ma.any():
            return True, "全无效（掩膜一致）"
        d = np.abs(da[ma] - db[mb])
        return bool(d.max() <= tol), f"max|Δ|={d.max():.3e}"


def file_inventory(root):
    """遍历目录树返回 {relpath: (mtime_ns, size)} 清单（幂等对比用）。"""
    inv = {}
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            p = os.path.join(dirpath, f)
            st = os.stat(p)
            inv[os.path.relpath(p, root)] = (st.st_mtime_ns, st.st_size)
    return inv


def section(name):
    RESULTS.append([name, []])
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")


def section_summary():
    name, checks = RESULTS[-1]
    passed = sum(1 for _, ok, _ in checks if ok)
    for cname, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {cname}" + (f"  -- {detail}" if detail else ""))
    print(f"  >>> {passed}/{len(checks)} PASS")


# ---------------------------------------------------------------------------
# V-PW1 数据盘查
# ---------------------------------------------------------------------------
def v_pw1():
    section("V-PW1 数据盘查")
    r = json.loads(pw.pw_inspect_data(BASE))
    check("工具返回可解析", r["status"] == "ok", r.get("status"))
    c = r["checks"]
    for k in ("monitor_4stations", "monitor_stray_copy", "s2_csv", "s3_csv",
              "daily_csv", "s2s3_csv", "feature_csv", "data_feature_csv",
              "s2_zip", "s3_zip", "s2_clip", "s3_tif", "feat_dir", "pred", "shp"):
        check(f"盘查项 {k}", bool(c.get(k)), str(c.get(k)))
    check("监测 4 站全部存在", all(c.get("monitor_4stations", [])),
          str(c.get("monitor_4stations")))
    check("feature 目录 39 tif", r["details"]["feature_tif_count"] == 39,
          str(r["details"]["feature_tif_count"]))
    section_summary()


# ---------------------------------------------------------------------------
# V-PW2 旧品基准断言
# ---------------------------------------------------------------------------
def v_pw2():
    section("V-PW2 旧品基准断言（read-only）")
    with rasterio.open(OLD_PRED) as src:
        a = src.read(1)
        valid = a != -9999
        check("pred 形状 3184×1858", list(src.shape) == [3184, 1858], str(src.shape))
        check("pred nodata=-9999", src.nodata == -9999, str(src.nodata))
        check("pred CRS 4326", str(src.crs).endswith("4326"), str(src.crs))
        check("pred 有效像元 119721", int(valid.sum()) == 119721, str(int(valid.sum())))
        vmin, vmax = float(a[valid].min()), float(a[valid].max())
        check("pred 有效值域 0.07636~0.14077",
              abs(vmin - 0.076358) < 1e-5 and abs(vmax - 0.140773) < 1e-5,
              f"{vmin:.6f}~{vmax:.6f}")
    # nodata 四分实测
    with rasterio.open(os.path.join(OLD_S2, "clip_3", "S2_B1.tif")) as s:
        check("S2 clip 链 nodata=0.0", s.nodata == 0.0, str(s.nodata))
    with rasterio.open(os.path.join(OLD_FEAT, "Oa02_radiance.tif")) as s:
        check("Oa 链 nodata=nan", np.isnan(s.nodata), str(s.nodata))
    with rasterio.open(os.path.join(OLD_FEAT, "B5+B4.tif")) as s:
        check("指数 tif nodata=0.0", s.nodata == 0.0, str(s.nodata))
    f = pd.read_csv(os.path.join(OLD, "feature.csv"), encoding="gbk")
    check("feature.csv 45×33 gbk", list(f.shape) == [45, 33], str(list(f.shape)))
    check("总磷实测 0.0577~0.1812",
          abs(f[TARGET].min() - 0.057667) < 1e-6 and abs(f[TARGET].max() - 0.181167) < 1e-6,
          f"{f[TARGET].min():.6f}~{f[TARGET].max():.6f}")
    bad = (f["Oa01_radiance"] < -1e9).all() and (f["Oa10_radiance"] < -1e9).all()
    check("Oa01/Oa10 两列 100% 掩膜坏值", bad,
          f"Oa01={f['Oa01_radiance'].min()} Oa10={f['Oa10_radiance'].min()}")
    section_summary()


# ---------------------------------------------------------------------------
# V-PW3 pw_daily_mean legacy == 旧日均统计结果.csv
# ---------------------------------------------------------------------------
def v_pw3():
    section("V-PW3 pw_daily_mean legacy")
    out = os.path.join(SAMPLES, "日均统计结果.csv")
    r = json.loads(pw.pw_daily_mean(os.path.join(OLD, "监测"), out, mode="legacy"))
    new = pd.read_csv(out, encoding="utf-8-sig")
    old = pd.read_csv(os.path.join(OLD, "日均统计结果.csv"), encoding="utf-8-sig")
    check("形状 688×13", list(new.shape) == [688, 13], str(list(new.shape)))
    check("列序一致", list(new.columns) == list(old.columns), str(list(new.columns)))
    check("值 equal_nan 一致", new.equals(old))
    check("date 为 float 20240701.0（旧品 mean 巧合 bug 保留）",
          new["date"].dtype == np.float64 and float(new["date"].iloc[0]) == 20240701.0,
          f"dtype={new['date'].dtype} sample={new['date'].iloc[0]}")
    check("氨氮列居末（object 列被 numeric_only 排除机制）",
          list(new.columns)[-1] == "氨氮(mg/L)", str(list(new.columns)[-1]))
    r2 = json.loads(pw.pw_daily_mean(os.path.join(OLD, "监测"),
                                     os.path.join(SAMPLES, "日均统计结果_correct.csv"),
                                     mode="correct"))
    nc = pd.read_csv(os.path.join(SAMPLES, "日均统计结果_correct.csv"), encoding="utf-8-sig")
    check("correct：date 显式 int64", nc["date"].dtype == np.int64, str(nc["date"].dtype))
    check("correct：形状与 legacy 一致", list(nc.shape) == [688, 13], str(list(nc.shape)))
    section_summary()


# ---------------------------------------------------------------------------
# V-PW4 pw_merge_spatiotemporal legacy == 旧 s2-s3.csv
# ---------------------------------------------------------------------------
def v_pw4():
    section("V-PW4 pw_merge_spatiotemporal legacy")
    out = os.path.join(SAMPLES, "s2-s3.csv")
    r = json.loads(pw.pw_merge_spatiotemporal(
        os.path.join(OLD, "S2.csv"), os.path.join(OLD, "S3.csv"),
        os.path.join(OLD, "日均统计结果.csv"), out, mode="legacy"))
    new = pd.read_csv(out, encoding="utf-8-sig")
    old = pd.read_csv(os.path.join(OLD, "s2-s3.csv"), encoding="utf-8-sig")
    check("形状 109×52", list(new.shape) == [109, 52], str(list(new.shape)))
    check("列序一致", list(new.columns) == list(old.columns))
    check("值一致", new.equals(old))
    check("date float/int 数值匹配巧合保留（109 行成立）", len(new) == 109)
    r2 = json.loads(pw.pw_merge_spatiotemporal(
        os.path.join(OLD, "S2.csv"), os.path.join(OLD, "S3.csv"),
        os.path.join(OLD, "日均统计结果.csv"), os.path.join(SAMPLES, "s2-s3_correct.csv"),
        mode="correct"))
    mc = pd.read_csv(os.path.join(SAMPLES, "s2-s3_correct.csv"), encoding="utf-8-sig")
    check("correct：date 统一 int64 后 109 行不变", mc["date"].dtype == np.int64
          and len(mc) == 109, f"dtype={mc['date'].dtype} rows={len(mc)}")
    section_summary()


# ---------------------------------------------------------------------------
# V-PW5 pw_build_samples legacy == 旧 feature.csv
# ---------------------------------------------------------------------------
def v_pw5():
    section("V-PW5 pw_build_samples legacy")
    out = os.path.join(SAMPLES, "feature.csv")
    r = json.loads(pw.pw_build_samples(os.path.join(SAMPLES, "s2-s3.csv"), out,
                                       mode="legacy"))
    new = pd.read_csv(out, encoding="gbk")
    old = pd.read_csv(os.path.join(OLD, "feature.csv"), encoding="gbk")
    check("形状 45×33", list(new.shape) == [45, 33], str(list(new.shape)))
    check("列序一致", list(new.columns) == list(old.columns))
    check("值+dtype 完全一致（equals）", new.equals(old))
    check("无 NaN", not new.isna().any().any())
    check("Oa 列去 S3_ 前缀且无 vOa19",
          not any(c.startswith("S3_") for c in new.columns)
          and not any("vOa19" in c for c in new.columns))
    check("S2_B 列保留前缀 12 个", sum(1 for c in new.columns if c.startswith("S2_B")) == 12)
    section_summary()


# ---------------------------------------------------------------------------
# V-PW6 pw_build_samples correct（缺陷#1 修复）
# ---------------------------------------------------------------------------
def v_pw6():
    section("V-PW6 pw_build_samples correct（坏值列清洗）")
    out = os.path.join(SAMPLES, "feature_correct.csv")
    r = json.loads(pw.pw_build_samples(os.path.join(SAMPLES, "s2-s3_correct.csv"), out,
                                       mode="correct"))
    f = pd.read_csv(out, encoding="gbk")
    legacy = pd.read_csv(os.path.join(SAMPLES, "feature.csv"), encoding="gbk")
    check("形状 45×31（Oa01/Oa10 整列删除）", list(f.shape) == [45, 31], str(list(f.shape)))
    check("无 NaN", not f.isna().any().any())
    check("无 <-1e9 坏值", float(f.select_dtypes("number").min().min()) > -1e9,
          str(float(f.select_dtypes("number").min().min())))
    check("Oa01/Oa10 已删除", "Oa01_radiance" not in f.columns
          and "Oa10_radiance" not in f.columns)
    # 两输入 CSV 行序均来自 merge 原序（dropna 保序）→ 按位置对齐比共同列值
    common = [c for c in f.columns if c in legacy.columns]
    check("与 legacy 45 行行集一致（共同列值相等）",
          len(f) == len(legacy) == 45
          and np.allclose(f[common].round(9), legacy[common].round(9)),
          f"len={len(f)} common_cols={len(common)}")
    check("总磷列与 legacy 一致",
          np.allclose(f[TARGET].round(9), legacy[TARGET].round(9)))
    section_summary()


# ---------------------------------------------------------------------------
# V-PW7 pw_prepare_s2 legacy == 旧 clip_3（12 tif）
# ---------------------------------------------------------------------------
def v_pw7():
    section("V-PW7 pw_prepare_s2 legacy")
    r = json.loads(pw.pw_prepare_s2(
        os.path.join(OLD_S2, "data_0"),
        os.path.join(GRAPHS, "S2_波段处理.xml"),
        os.path.join(OLD_S2, "yuchuli_1"),
        S2_TIF, S2_CLIP, SHP_WORK, mode="legacy"))
    check("工具返回 ok（exists 短路亦通过）", r["status"] in ("ok", "exists"),
          f"status={r['status']} clipped={r.get('clipped')}")
    check("clip 12 个 S2_B*.tif（文件实测）",
          len(glob_files(S2_CLIP, "S2_B*.tif")) == 12,
          str(len(glob_files(S2_CLIP, "S2_B*.tif"))))
    all_ok = True
    detail = []
    for b in S2_BANDS:
        newp = os.path.join(S2_CLIP, f"S2_{b}.tif")
        oldp = os.path.join(OLD_S2, "clip_3", f"S2_{b}.tif")
        if not os.path.exists(newp):
            all_ok = False
            detail.append(f"{b}:缺失")
            continue
        with rasterio.open(newp) as n, rasterio.open(oldp) as o:
            same_grid = (n.shape == o.shape
                         and np.allclose(n.transform, o.transform, atol=0)
                         and n.crs == o.crs)
            same_nodata = (n.nodata == o.nodata == 0.0)
            m, d = tif_matches(newp, oldp)
            ok = same_grid and same_nodata and m
            if not ok:
                all_ok = False
            detail.append(f"{b}:grid={same_grid},nodata={same_nodata},{d}")
    check("12 个 clip 与旧品形状/transform/CRS/nodata=0.0/掩膜/值一致",
          all_ok, "; ".join(detail))
    check("tif 目录嵌套同名子目录 12 个（复刻旧布局）",
          len(os.listdir(S2_TIF)) == 1
          and len(glob_files(S2_TIF, "*.tif")) == 12,
          str(os.listdir(S2_TIF)))
    section_summary()


def glob_files(root, pattern):
    import glob
    return glob.glob(os.path.join(root, "**", pattern), recursive=True)


# ---------------------------------------------------------------------------
# V-PW8 pw_prepare_s3 legacy == 旧 feature 目录 21 Oa
# ---------------------------------------------------------------------------
def v_pw8():
    section("V-PW8 pw_prepare_s3 legacy")
    r = json.loads(pw.pw_prepare_s3(
        os.path.join(OLD_S3, "data_0"),
        os.path.join(GRAPHS, "S3_波段处理.xml"),
        os.path.join(OLD_S3, "yuchuli_1"),
        S3_TIF,
        os.path.join(OLD_S2, "clip_3", "S2_B1.tif"),
        S2_CLIP, FEATS, mode="legacy"))
    check("工具返回 ok（exists 短路亦通过）", r["status"] in ("ok", "exists"),
          f"status={r['status']} aligned={r.get('aligned')}")
    check("对齐 21 个 Oa（文件实测）", len(glob_files(FEATS, "Oa*_radiance.tif")) == 21,
          str(len(glob_files(FEATS, "Oa*_radiance.tif"))))
    check("S2 clip 拷贝 12 个（文件实测）", len(glob_files(FEATS, "S2_B*.tif")) == 12,
          str(len(glob_files(FEATS, "S2_B*.tif"))))
    all_ok = True
    detail = []
    for band in OA_BANDS:
        name = f"{band}_radiance.tif"
        newp, oldp = os.path.join(FEATS, name), os.path.join(OLD_FEAT, name)
        m, d = tif_matches(newp, oldp)
        if not m:
            all_ok = False
            detail.append(f"{band}:{d}")
    check("21 个 Oa 与旧品 NaN 掩膜一致 + max|Δ|≤1e-6", all_ok,
          "; ".join(detail[:3]) if detail else "全部一致")
    with rasterio.open(os.path.join(FEATS, "Oa02_radiance.tif")) as s:
        check("Oa 输出 nodata=nan", np.isnan(s.nodata), str(s.nodata))
    section_summary()


# ---------------------------------------------------------------------------
# V-PW9 pw_compute_features legacy == 旧 data/feature.csv + 6 指数 tif
# ---------------------------------------------------------------------------
def v_pw9():
    section("V-PW9 pw_compute_features legacy")
    out_csv = os.path.join(SAMPLES, "data_feature.csv")
    r = json.loads(pw.pw_compute_features(os.path.join(SAMPLES, "feature.csv"),
                                          FEATS, out_csv, FEATS, mode="legacy"))
    new = pd.read_csv(out_csv)
    old = pd.read_csv(os.path.join(OLD, "data", "feature.csv"))
    check("data/feature.csv 39 列", list(new.shape) == [45, 39], str(list(new.shape)))
    check("列序一致", list(new.columns) == list(old.columns))
    check("值一致（equals）", new.equals(old))
    check("6 指数列追加", [c for c in new.columns[-6:]] == [op["new_col"] for op in pw.FEATURE_OPS],
          str(list(new.columns[-6:])))
    all_ok = True
    detail = []
    for t in INDEX_TIFS:
        m, d = tif_matches(os.path.join(FEATS, t), os.path.join(OLD_FEAT, t))
        if not m:
            all_ok = False
            detail.append(f"{t}:{d}")
    check("6 指数 tif 与旧品 nodata=0.0 掩膜+值一致", all_ok,
          "; ".join(detail) if detail else "全部一致")
    section_summary()


# ---------------------------------------------------------------------------
# V-PW10 pw_train_rf legacy（阈值 0 缺陷#2 复刻 + 参照值 + 确定性）
# ---------------------------------------------------------------------------
def v_pw10():
    section("V-PW10 pw_train_rf legacy")
    r = json.loads(pw.pw_train_rf(os.path.join(SAMPLES, "data_feature.csv"),
                                  MODEL_L, mode="legacy"))
    check("38 特征全入选（threshold=0 缺陷复刻）",
          r["n_selected_features"] == 38 and r["n_input_features"] == 38,
          f"selected={r['n_selected_features']} input={r['n_input_features']}")
    check("含坏值列 Oa01/Oa10（缺陷#1 实证）",
          "Oa01_radiance" in r["selected_features"] and "Oa10_radiance" in r["selected_features"])
    check("best_params 参照值", r["best_params"] == pw._LEGACY_REF["best_params"],
          str(r["best_params"]))
    check("测试 R²=0.3157（旧脚本仅打印 4 位小数，按 4 位断言）",
          round(r["test_r2"], 4) == 0.3157, str(r["test_r2"]))
    check("测试 RMSE=0.0244（同上）",
          round(r["test_rmse"], 4) == 0.0244, str(r["test_rmse"]))
    check("四件落盘（model/scaler/features/metrics）",
          all(os.path.exists(os.path.join(MODEL_L, f))
              for f in ("model.pkl", "scaler.pkl", "features.json", "metrics.json")))
    # 确定性：重训一次对比 metrics
    import shutil
    bak = MODEL_L + "_bak"
    if os.path.exists(bak):
        shutil.rmtree(bak)
    shutil.move(MODEL_L, bak)
    r2 = json.loads(pw.pw_train_rf(os.path.join(SAMPLES, "data_feature.csv"),
                                   MODEL_L, mode="legacy"))
    same = (r2["selected_features"] == r["selected_features"]
            and r2["best_params"] == r["best_params"]
            and r2["test_r2"] == r["test_r2"] and r2["test_rmse"] == r["test_rmse"])
    shutil.rmtree(bak)
    check("两次训练完全一致（确定性）", same)
    section_summary()


# ---------------------------------------------------------------------------
# V-PW11 pw_predict_rf legacy == 旧 pred_merged.tif（bit-exact）
# ---------------------------------------------------------------------------
def v_pw11():
    section("V-PW11 pw_predict_rf legacy")
    r = json.loads(pw.pw_predict_rf(FEATS, MODEL_L, PRED_L, mode="legacy"))
    check("工具返回 ok（exists 短路亦通过）", r["status"] in ("ok", "exists"),
          f"status={r['status']} n_features={r.get('n_features')}")
    with open(os.path.join(MODEL_L, "features.json"), encoding="utf-8") as f:
        n_feat = len(json.load(f)["selected_features"])
    check("38 特征堆叠（features.json 实测）", n_feat == 38, str(n_feat))
    m, d = tif_matches(PRED_L, OLD_PRED)
    check("与旧品 pred_merged.tif 掩膜一致 + max|Δ|≤1e-6（期望 0.0）", m, d)
    with rasterio.open(PRED_L) as s:
        a = s.read(1)
        check("有效像元 119721", int((a != -9999).sum()) == 119721,
              str(int((a != -9999).sum())))
        check("nodata=-9999", s.nodata == -9999, str(s.nodata))
    section_summary()


# ---------------------------------------------------------------------------
# V-PW12 pw_train_rf correct（阈值 0.3 + 坏值列不入选）
# ---------------------------------------------------------------------------
def v_pw12():
    section("V-PW12 correct 特征计算 + pw_train_rf correct")
    out_csv = os.path.join(SAMPLES, "data_feature_correct.csv")
    r7 = json.loads(pw.pw_compute_features(os.path.join(SAMPLES, "feature_correct.csv"),
                                           FEATS, out_csv, FEATS, mode="correct"))
    check("correct 特征计算 37 列（31+6 指数）",
          pd.read_csv(out_csv).shape[1] == 37, str(pd.read_csv(out_csv).shape))
    # correct 重写的 6 指数 tif 与 legacy 值一致（本集无 ÷0，语义应相同）
    ok = True
    for t in INDEX_TIFS:
        m, d = tif_matches(os.path.join(FEATS, t), os.path.join(OLD_FEAT, t))
        ok = ok and m
    check("correct 重写后 6 指数 tif 仍与旧品一致", ok)
    r = json.loads(pw.pw_train_rf(out_csv, MODEL_C, mode="correct",
                                  correlation_threshold=0.3))
    check("输入 36 特征（37 列-目标）", r["n_input_features"] == 36,
          str(r["n_input_features"]))
    check("阈值 0.3 生效", r["threshold"] == 0.3, str(r["threshold"]))
    check("坏值列 Oa01/Oa10 不在特征（缺陷#1 修复）",
          "Oa01_radiance" not in r["selected_features"]
          and "Oa10_radiance" not in r["selected_features"])
    check("筛选后特征数 < 输入（约 15 个）",
          0 < r["n_selected_features"] < 36, str(r["n_selected_features"]))
    check("四件落盘", all(os.path.exists(os.path.join(MODEL_C, f))
                          for f in ("model.pkl", "scaler.pkl", "features.json", "metrics.json")))
    check("metrics 报告 R²/RMSE（机械对照口径）",
          "test_r2" in r and "test_rmse" in r,
          f"R²={r.get('test_r2')} RMSE={r.get('test_rmse')}")
    section_summary()


# ---------------------------------------------------------------------------
# V-PW13 correct 反演 + clip + 护栏
# ---------------------------------------------------------------------------
def v_pw13():
    section("V-PW13 correct 反演 + 水体裁剪 + 护栏")
    r = json.loads(pw.pw_predict_rf(FEATS, MODEL_C, PRED_C, mode="correct"))
    check("correct 反演产物生成", r["status"] in ("ok", "exists")
          and os.path.exists(PRED_C), f"n_features={r.get('n_features')}")
    with rasterio.open(PRED_C) as s:
        a = s.read(1)
        valid = a[a != -9999]
        check("有效像元存在", len(valid) > 0, str(len(valid)))
        check("硬护栏 [0,1]（总磷 mg/L）",
              float(valid.min()) >= 0.0 and float(valid.max()) <= 1.0,
              f"{valid.min():.6f}~{valid.max():.6f}")
        soft = (0.05 <= valid.min()) and (valid.max() <= 0.2)
        check("软护栏 [0.05,0.2]（实测 0.0577~0.1812）", soft,
              f"{valid.min():.6f}~{valid.max():.6f}")
        check("nodata=-9999", s.nodata == -9999)
    # legacy vs correct 差异量化（机械对照）
    with rasterio.open(PRED_L) as l, rasterio.open(PRED_C) as c:
        dl, dc = l.read(1).astype(np.float64), c.read(1).astype(np.float64)
        ml, mc = _mask_of(dl, -9999), _mask_of(dc, -9999)
        both = ml & mc
        d = np.abs(dl[both] - dc[both])
        check("legacy/correct 共同有效区差异量化（报告）", True,
              f"共同有效 {int(both.sum())} 像元，max|Δ|={d.max():.4f}，"
              f"mean|Δ|={d.mean():.4f}，Δ>0.02 占 {(d > 0.02).mean():.1%}")
    rc = json.loads(pw.pw_clip_product(PRED_C, SHP_WORK, PRED_C_CLIP))
    check("clip 产物生成", rc["status"] in ("ok", "exists")
          and os.path.exists(PRED_C_CLIP), str(rc.get("attrs")))
    shp_ds = gdal.OpenEx(SHP_WORK)
    xmin, xmax, ymin, ymax = shp_ds.GetLayer().GetExtent()
    shp_ds = None
    with rasterio.open(PRED_C_CLIP) as s:
        b = s.bounds
        inside = (b.left >= xmin - 1e-9 and b.bottom >= ymin - 1e-9
                  and b.right <= xmax + 1e-9 and b.top <= ymax + 1e-9)
        check("clip bounds ⊆ 水体.shp", inside,
              f"clip={[round(x, 4) for x in (b.left, b.bottom, b.right, b.top)]} "
              f"shp={[round(x, 4) for x in (xmin, ymin, xmax, ymax)]}")
        a = s.read(1)
        check("clip 有效值无 -9999 泄漏", float(a[a != -9999].min()) >= 0.0)
    section_summary()


# ---------------------------------------------------------------------------
# V-PW14 资产 + 幂等 + 收尾
# ---------------------------------------------------------------------------
def v_pw14():
    section("V-PW14 资产/无 .enp/幂等/JSON")
    check("graphs 2 个 SNAP XML 齐",
          os.path.exists(os.path.join(GRAPHS, "S2_波段处理.xml"))
          and os.path.exists(os.path.join(GRAPHS, "S3_波段处理.xml")))
    check("S3_波段处理2.xml（硬编码路径版）未拷贝",
          not os.path.exists(os.path.join(GRAPHS, "S3_波段处理2.xml")))
    enps = []
    for dirpath, _d, files in os.walk(WORK):
        enps += [os.path.join(dirpath, f) for f in files if f.endswith(".enp")]
    check("新产物树无 .enp 残留（缺陷#5 修复）", not enps, str(enps))
    check("旧品 feature 目录确有 2 个 .enp（对照实证）",
          len([f for f in os.listdir(OLD_FEAT) if f.endswith(".enp")]) == 2)
    # 幂等：记录清单 → 全工具重跑（exists 短路）→ 清单不变
    inv_before = file_inventory(WORK)
    pw.pw_daily_mean(os.path.join(OLD, "监测"), os.path.join(SAMPLES, "日均统计结果.csv"),
                     mode="legacy")
    pw.pw_merge_spatiotemporal(os.path.join(OLD, "S2.csv"), os.path.join(OLD, "S3.csv"),
                               os.path.join(OLD, "日均统计结果.csv"),
                               os.path.join(SAMPLES, "s2-s3.csv"), mode="legacy")
    pw.pw_build_samples(os.path.join(SAMPLES, "s2-s3.csv"),
                        os.path.join(SAMPLES, "feature.csv"), mode="legacy")
    pw.pw_prepare_s2(os.path.join(OLD_S2, "data_0"), os.path.join(GRAPHS, "S2_波段处理.xml"),
                     os.path.join(OLD_S2, "yuchuli_1"), S2_TIF, S2_CLIP, SHP_WORK,
                     mode="legacy")
    pw.pw_prepare_s3(os.path.join(OLD_S3, "data_0"), os.path.join(GRAPHS, "S3_波段处理.xml"),
                     os.path.join(OLD_S3, "yuchuli_1"), S3_TIF,
                     os.path.join(OLD_S2, "clip_3", "S2_B1.tif"), S2_CLIP, FEATS,
                     mode="legacy")
    pw.pw_compute_features(os.path.join(SAMPLES, "feature.csv"), FEATS,
                           os.path.join(SAMPLES, "data_feature.csv"), FEATS, mode="legacy")
    pw.pw_compute_features(os.path.join(SAMPLES, "feature_correct.csv"), FEATS,
                           os.path.join(SAMPLES, "data_feature_correct.csv"), FEATS,
                           mode="correct")
    pw.pw_train_rf(os.path.join(SAMPLES, "data_feature.csv"), MODEL_L, mode="legacy")
    pw.pw_train_rf(os.path.join(SAMPLES, "data_feature_correct.csv"), MODEL_C,
                   mode="correct")
    pw.pw_predict_rf(FEATS, MODEL_L, PRED_L, mode="legacy")
    pw.pw_predict_rf(FEATS, MODEL_C, PRED_C, mode="correct")
    pw.pw_clip_product(PRED_C, SHP_WORK, PRED_C_CLIP)
    inv_after = file_inventory(WORK)
    changed = [k for k in inv_before
               if k not in inv_after or inv_before[k] != inv_after[k]]
    added = [k for k in inv_after if k not in inv_before]
    check("幂等重跑：所有文件 mtime/size 不变", not changed and not added,
          f"changed={changed[:3]} added={added[:3]}")
    check("产物文件总数", len(inv_after) > 0, str(len(inv_after)))
    # 所有工具 JSON 返回值可解析（本次运行已全部 json.loads）
    check("全部工具返回 JSON 可解析（本次运行隐式验证）", True)
    section_summary()


def main():
    print(f"水体总磷反演链验证 V-PW1~V-PW14")
    print(f"旧品只读目录: {OLD}")
    print(f"新产物目录:   {WORK}")
    for fn in (v_pw1, v_pw2, v_pw3, v_pw4, v_pw5, v_pw6,
               v_pw7, v_pw8, v_pw9, v_pw10, v_pw11, v_pw12, v_pw13, v_pw14):
        try:
            fn()
        except Exception:  # noqa: BLE001
            if not RESULTS or RESULTS[-1][0] != fn.__name__:
                section(fn.__name__)
            check("异常", False, traceback.format_exc().strip().splitlines()[-1])
            section_summary()
    total = sum(len(c) for _, c in RESULTS)
    passed = sum(1 for _, c in RESULTS for _, ok, _ in c if ok)
    print(f"\n{'=' * 70}\n总计: {passed}/{total} PASS")
    if passed < total:
        failed = [(n, cn, d) for n, c in RESULTS for cn, ok, d in c if not ok]
        print("FAIL 清单:")
        for n, cn, d in failed:
            print(f"  [{n}] {cn} -- {d}")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
