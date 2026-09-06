"""
跨年 NDVI 处理链驱动 + 联合分析（V4）
对两个哨兵2日期（20241231 / 20250425）跑完整 NDVI 链：
    list → snap_preprocess_date → convert_date_to_tif → apply_cloud_mask_date
    → mosaic_date → clip_date_to_aoi
再补：跨年 NDVI 对比 ×2 + 新配对 analyze_lst_ndvi ×2（用 V3 已产出的 LST clip）

与 verify_crossyear.py 的关系：V3 管 LST 域，本脚本管 NDVI 域与联合分析。
每一步失败即中止并打印原因（确定性驱动；Agent 编排能力由问题文件驱动的交互入口单独验证）。

运行：PYTHONUTF8=1 python run_ndvi_crossyear.py
SNAP 预处理每景 2-6 分钟，两个日期共 6 景，全程约 25-50 分钟。
"""
import os
import sys

import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_tools import (
    apply_cloud_mask_date,
    clip_date_to_aoi,
    compare_ndvi_dates,
    convert_date_to_tif,
    list_available_scenes,
    mosaic_date,
    snap_preprocess_date,
)
from lst_tools import analyze_lst_ndvi

DATA_DIR = "E:/YYR/NDVI/multi_date"
WORK = os.path.join(DATA_DIR, "work")
COMPARE_DIR = os.path.join(DATA_DIR, "compare")
LST_JOINT = "E:/YYR/LST/multi_date/joint"
SHP = "E:/YYR/NDVI/vector/1/1.shp"
XML = "E:/YYR/NDVI/NDVI_COLUD_MASK.xml"
NEW_DATES = ("20241231", "20250425")
OLD_CLIPS = {
    "20251228": "E:/YYR/NDVI/multi_date/work/20251228/clip/NDVI_20251228_clip.tif",
    "20260510": "E:/YYR/NDVI/multi_date/work/20260510/clip/NDVI_20260510_clip.tif",
}
LST_CLIPS = {
    "20241231": "E:/YYR/LST/multi_date/work/20241231/clip/LST_20241231_clip.tif",
    "20250430": "E:/YYR/LST/multi_date/work/20250430/clip/LST_20250430_clip.tif",
}

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""),
          flush=True)


def fail(msg):
    print(f"\n[中止] {msg}", flush=True)
    sys.exit(1)


# ------------------------------------------------------------
# V4.1 场景发现
# ------------------------------------------------------------
disc = list_available_scenes(DATA_DIR)
check("V4.1 场景发现：4 日期齐全",
      disc["dates"] == ["20241231", "20250425", "20251228", "20260510"],
      str(disc["dates"]))

# ------------------------------------------------------------
# V4.2/V4.3 NDVI 全链（新日期）
# ------------------------------------------------------------
new_clips = {}
for date in NEW_DATES:
    w = os.path.join(WORK, date)
    print(f"\n===== 处理 {date} =====", flush=True)
    r_pre = snap_preprocess_date(disc["scenes_by_date"][date], XML,
                                 os.path.join(w, "yuchuli"))
    n_fail = sum(1 for x in r_pre["results"] if x["status"] != "success")
    if n_fail:
        fail(f"{date} 预处理失败 {n_fail} 景：{r_pre}")
    print(f"  预处理完成：{r_pre['success']}/{r_pre['total']}", flush=True)

    r_conv = convert_date_to_tif(os.path.join(w, "yuchuli"),
                                 os.path.join(w, "tif"))
    if r_conv["total"] == 0 or not all(x["status"] == "success"
                                       for x in r_conv["results"]):
        fail(f"{date} 转 tif 失败：{r_conv}")
    print("  转 tif 完成", flush=True)

    r_mask = apply_cloud_mask_date(os.path.join(w, "tif"),
                                   os.path.join(w, "quyun"))
    if r_mask["total"] == 0 or not all(x["status"] == "success"
                                       for x in r_mask["results"]):
        fail(f"{date} 去云失败：{r_mask}")
    print("  去云完成", flush=True)

    r_mos = mosaic_date(os.path.join(w, "quyun"), os.path.join(w, "mosaic"),
                        date=date)
    if not r_mos.get("mosaic_path"):
        fail(f"{date} 镶嵌失败：{r_mos}")
    print(f"  镶嵌完成：{r_mos['mosaic_path']}", flush=True)

    r_clip = clip_date_to_aoi(r_mos["mosaic_path"], SHP,
                              os.path.join(w, "clip"))
    new_clips[date] = r_clip["clip_path"]
    print(f"  裁剪完成：{new_clips[date]}", flush=True)

# 两期新 clip 与旧 clip 网格一致（AOI 固定，10m 对齐）
grids = {}
for p in list(new_clips.values()) + list(OLD_CLIPS.values()):
    with rasterio.open(p) as src:
        grids[p] = (src.width, src.height, src.transform, src.crs)
check("V4.2 四期 NDVI clip 网格完全相等（6159×5268）",
      len(set(grids.values())) == 1,
      f"{grids[new_clips[NEW_DATES[0]]][0]}x{grids[new_clips[NEW_DATES[0]]][1]}")

for date in NEW_DATES:
    with rasterio.open(new_clips[date]) as src:
        a = src.read(1).astype(np.float32)
        nodata = src.nodata
    valid = a[a != nodata]
    valid = valid[(valid >= -1) & (valid <= 1)]
    check(f"V4.2 {date} clip 值域/有效占比合理",
          0 < valid.mean() < 1 and 0.2 < valid.size / a.size < 0.5,
          f"mean={valid.mean():.3f} valid={valid.size / a.size:.4f}")

# ------------------------------------------------------------
# V4.4 跨年 NDVI 对比 ×2（同季节跨年，Δ 应远小于季节间 +0.356）
# ------------------------------------------------------------
for tag, (d1, d2) in (("冬季对", ("20241231", "20251228")),
                      ("春季对", ("20250425", "20260510"))):
    r = compare_ndvi_dates(new_clips[d1], OLD_CLIPS[d2], COMPARE_DIR)
    ok_prod = all(os.path.isfile(r[k]) for k in
                  ("diff_path", "class_path", "change_map_png", "area_chart_png"))
    check(f"V4.4 {tag} 四产物齐全", ok_prod)
    cs = r["classes"]
    rv = {k: cs[k]["ratio_of_valid"] for k in ("改善", "稳定", "退化")}
    check(f"V4.4 {tag} Δmean 远小于季节间(+0.356) 且分类混合",
          abs(r["delta_stats"]["mean"]) < 0.2
          and rv["改善"] < 0.95 and rv["退化"] < 0.95 and rv["稳定"] > 0.05,
          f"Δmean={r['delta_stats']['mean']:+.3f} "
          f"改善={rv['改善']:.3f} 稳定={rv['稳定']:.3f} 退化={rv['退化']:.3f}")

# ------------------------------------------------------------
# V4.5 新配对联合分析 ×2（LST 20241231↔NDVI 20241231 差0天；
#                          LST 20250430↔NDVI 20250425 差5天）
# ------------------------------------------------------------
for lst_date, ndvi_date in (("20241231", "20241231"), ("20250430", "20250425")):
    r = analyze_lst_ndvi(LST_CLIPS[lst_date], new_clips[ndvi_date], LST_JOINT)
    check(f"V4.5 联合分析 {lst_date}↔{ndvi_date} n 百万级且 r∈[-1,1]",
          r["n"] > 500_000 and -1 <= r["pearson_r"] <= 1,
          f"n={r['n']:,} r={r['pearson_r']:.3f}")
    check(f"V4.5 散点 PNG 非空白",
          os.path.isfile(r["scatter_png"])
          and np.asarray(__import__("PIL.Image").Image.open(
              r["scatter_png"]).convert("RGB")).sum(axis=2).min() < 700)

print(f"\n{'=' * 50}")
print(f"共 {len(results)} 项，PASS {sum(results)}，FAIL {len(results) - sum(results)}")
sys.exit(0 if all(results) else 1)
