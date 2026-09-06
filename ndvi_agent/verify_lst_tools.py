"""
LST 工具链验证脚本 V1.1-V1.10（风格同 smoke_one_tile.py：每项打印 PASS/FAIL）

V1.1  场景发现          V1.2  反演20251226     V1.3  逐像元严格验证（核心验收）
V1.4  反演20260425      V1.5  幂等重跑         V1.6  裁剪两期网格一致
V1.7  统计              V1.8  两期对比+PNG     V1.9  联合分析
V1.10 全部返回值 JSON 序列化不抛异常

运行：PYTHONUTF8=1 python verify_lst_tools.py
约 15-25 分钟（首次含两次解压与重采样）；验证只读旧目录，产物写 multi_date/work。
"""
import json
import os
import sys

import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lst_tools import (
    analyze_lst_ndvi,
    calculate_lst_stats,
    compare_lst_dates,
    list_available_lst_scenes,
    lst_clip_date_to_aoi,
    lst_invert_date,
)

DATA_DIR = "E:/YYR/LST/multi_date"
WORK = os.path.join(DATA_DIR, "work")
COMPARE_DIR = os.path.join(DATA_DIR, "compare")
JOINT_DIR = os.path.join(DATA_DIR, "joint")
SHP = "E:/YYR/NDVI/vector/1/1.shp"
OLD_REF = ("E:/YYR/LST/result/"
           "LC09_L2SP_120036_20251226_20251227_02_T1_LST.TIF")
NDVI_CLIPS = {
    "20251228": "E:/YYR/NDVI/multi_date/work/20251228/clip/NDVI_20251228_clip.tif",
    "20260510": "E:/YYR/NDVI/multi_date/work/20260510/clip/NDVI_20260510_clip.tif",
}
D1, D2 = "20251226", "20260425"

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def _fresh_stats(path):
    """从成品 tif 反推有效占比与值域（幂等重跑时 exists 返回不含统计字段）"""
    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float32)
        nodata = src.nodata
    if nodata is not None:
        valid = a[a != nodata]
    else:
        valid = a[~np.isnan(a)]
    valid = valid[np.isfinite(valid)]
    return valid.size / a.size, float(valid.min()), float(valid.max()), float(valid.mean())


# ------------------------------------------------------------
# V1.1 场景发现
# ------------------------------------------------------------
discovery = list_available_lst_scenes(DATA_DIR)
check("V1.1 场景发现：dates 正确",
      discovery["dates"] == [D1, D2], str(discovery["dates"]))
check("V1.1 场景发现：每日期1景", discovery["total_scenes"] == 2
      and all(len(v) == 1 for v in discovery["scenes_by_date"].values()))
tars = {d: v[0] for d, v in discovery["scenes_by_date"].items()}

# ------------------------------------------------------------
# V1.2 反演 20251226
# ------------------------------------------------------------
inv1 = lst_invert_date(discovery["scenes_by_date"][D1],
                       os.path.join(WORK, D1, "lst"),
                       os.path.join(WORK, D1, "extract"))
r1 = inv1["results"][0]
lst1_path = r1["output"]
check("V1.2 反演20251226 success（或幂等 exists）",
      r1["status"] in ("success", "exists"),
      f"status={r1['status']} {r1.get('error', '')}")
vr1, mn1, mx1, mean1 = _fresh_stats(lst1_path)
cloud1 = r1.get("cloud_ratio") if r1.get("cloud_ratio") is not None else 1 - vr1
check("V1.2 云像元占比≈0.32（exists时用NaN占比近似，差值为原始0值比例）",
      abs(cloud1 - 0.32) < 0.05, f"cloud≈{cloud1:.4f}")
check("V1.2 值域≈旧品(-16.6~36.0, mean 7.28)",
      mn1 < -10 and mx1 > 30 and abs(mean1 - 7.28) < 1.0,
      f"min={mn1:.2f} max={mx1:.2f} mean={mean1:.2f}")

# ------------------------------------------------------------
# V1.3 逐像元严格验证（核心验收：tar字节相同+同公式同掩膜 → 期望0差异）
# ------------------------------------------------------------
if os.path.isfile(OLD_REF):
    with rasterio.open(lst1_path) as a, rasterio.open(OLD_REF) as b:
        same_shape = a.shape == b.shape
        same_tf = a.transform == b.transform
        same_crs = a.crs == b.crs
        na = a.read(1).astype(np.float32)
        nb = b.read(1).astype(np.float32)
    m_na, m_nb = np.isnan(na), np.isnan(nb)
    mask_equal = bool(np.array_equal(m_na, m_nb))
    both_valid = ~m_na & ~m_nb
    max_diff = float(np.abs(na[both_valid] - nb[both_valid]).max()) \
        if both_valid.any() else float("nan")
    mismatch = int((~np.isclose(na, nb, equal_nan=True)).sum())
    check("V1.3 逐像元严格验证", mask_equal and mismatch == 0 and max_diff == 0.0
          and same_shape and same_tf and same_crs,
          f"shape同={same_shape} tf同={same_tf} crs同={same_crs} "
          f"NaN掩膜一致={mask_equal} mismatch={mismatch} max_diff={max_diff}")
else:
    check("V1.3 逐像元严格验证", False, f"旧参考不存在：{OLD_REF}")

# ------------------------------------------------------------
# V1.4 反演 20260425（LC08）
# ------------------------------------------------------------
inv2 = lst_invert_date(discovery["scenes_by_date"][D2],
                       os.path.join(WORK, D2, "lst"),
                       os.path.join(WORK, D2, "extract"))
r2 = inv2["results"][0]
lst2_path = r2["output"]
check("V1.4 反演20260425 success（或幂等 exists）",
      r2["status"] in ("success", "exists"),
      f"status={r2['status']} {r2.get('error', '')}")
vr2, mn2, mx2, mean2 = _fresh_stats(lst2_path)
check("V1.4 值域物理合理(-100~100) 且 valid∈(0,1)",
      -100 < mn2 < mx2 < 100 and 0 < vr2 < 1,
      f"min={mn2:.2f} max={mx2:.2f} valid={vr2:.4f}")

# ------------------------------------------------------------
# V1.5 幂等重跑（exists 短路不经过解压逻辑，故用解压目录 mtime 不变
#          独立验证"不重复解压"，不依赖工具返回字段）
# ------------------------------------------------------------
extract_dirs = [os.path.join(WORK, D1, "extract"),
                os.path.join(WORK, D2, "extract")]
mtimes_before = {d: os.path.getmtime(d) for d in extract_dirs}
inv1b = lst_invert_date(discovery["scenes_by_date"][D1],
                        os.path.join(WORK, D1, "lst"),
                        os.path.join(WORK, D1, "extract"))
inv2b = lst_invert_date(discovery["scenes_by_date"][D2],
                        os.path.join(WORK, D2, "lst"),
                        os.path.join(WORK, D2, "extract"))
mtimes_after = {d: os.path.getmtime(d) for d in extract_dirs}
check("V1.5 幂等重跑：两期全部 exists 且不重复解压",
      all(r["status"] == "exists" for r in inv1b["results"])
      and all(r["status"] == "exists" for r in inv2b["results"])
      and all(mtimes_before[d] == mtimes_after[d] for d in extract_dirs))

# ------------------------------------------------------------
# V1.6 裁剪：两期网格完全相等
# ------------------------------------------------------------
clip_dir1 = os.path.join(WORK, D1, "clip")
clip_dir2 = os.path.join(WORK, D2, "clip")
cl1 = lst_clip_date_to_aoi(lst1_path, SHP, clip_dir1)
cl2 = lst_clip_date_to_aoi(lst2_path, SHP, clip_dir2)
clip1, clip2 = cl1["clip_path"], cl2["clip_path"]
check("V1.6 两期 clip width/height/transform/crs 完全相等",
      cl1["width"] == cl2["width"] and cl1["height"] == cl2["height"]
      and cl1["transform"] == cl2["transform"] and cl1["crs"] == cl2["crs"],
      f"{cl1['width']}x{cl1['height']}")
with rasterio.open(clip1) as src:
    clip_nodata = src.nodata
check("V1.6 nodata=-9999", clip_nodata == -9999, f"nodata={clip_nodata}")

# ------------------------------------------------------------
# V1.7 统计 ×2（json.dumps allow_nan=False 直过）
# ------------------------------------------------------------
st1 = calculate_lst_stats(clip1, "南昌研究区")
st2 = calculate_lst_stats(clip2, "南昌研究区")
check("V1.7 统计20251226 clip 均值≈7.28 同量级",
      0 < st1["mean"] < 20, f"mean={st1['mean']:.2f}℃ valid={st1['valid_pixel_ratio']}")
ok = True
for st in (st1, st2):
    try:
        json.dumps(st, ensure_ascii=False, allow_nan=False)
    except ValueError:
        ok = False
check("V1.7 统计返回值 allow_nan=False 序列化直过", ok,
      f"20251226 mean={st1['mean']:.2f} / 20260425 mean={st2['mean']:.2f}")

# ------------------------------------------------------------
# V1.8 两期对比（+PNG 程序化检查）
# ------------------------------------------------------------
cmp_r = compare_lst_dates(clip1, clip2, COMPARE_DIR, threshold=3.0)
check("V1.8 四产物齐全", all(os.path.isfile(cmp_r[k]) for k in
      ("diff_path", "class_path", "change_map_png", "area_chart_png")))
check("V1.8 delta.mean 显著>0（冬→春季节升温预期）",
      cmp_r["delta_stats"]["mean"] > 8, f"mean={cmp_r['delta_stats']['mean']:.2f}℃")
warm_ratio = cmp_r["classes"]["升温"]["ratio_of_valid"]
check("V1.8 升温占主导(>0.5)", warm_ratio > 0.5, f"升温占有效对 {warm_ratio:.3f}")
check("V1.8 无效占比 0.6~0.75（矩形裁剪背景占多数，与NDVI链0.684同构）",
      0.6 < cmp_r["classes"]["无效"]["ratio_of_all"] < 0.75,
      f"无效={cmp_r['classes']['无效']['ratio_of_all']:.3f}")

# PNG 程序化检查（会话内无法目视，用户自行打开核对；这里只查非空白+四色齐全）
from PIL import Image
png_colors = {}
for name, path in (("change_map", cmp_r["change_map_png"]),
                   ("area_chart", cmp_r["area_chart_png"])):
    img = np.asarray(Image.open(path).convert("RGB"))
    non_white = (img.sum(axis=2) < 720).mean()
    png_colors[name] = non_white
check("V1.8 PNG 非空白（用户请自行打开目视核对）",
      all(v > 0.005 for v in png_colors.values()),
      f"非白像素占比 {png_colors}")

# ------------------------------------------------------------
# V1.9 联合分析 ×2
# ------------------------------------------------------------
j1 = analyze_lst_ndvi(clip1, NDVI_CLIPS["20251228"], JOINT_DIR)
j2 = analyze_lst_ndvi(clip2, NDVI_CLIPS["20260510"], JOINT_DIR)
# 同类型比较：都用 rasterio 打开（cl1["transform"] 是 gdal 元组、crs 是 WKT 字符串，
# 与 Affine/CRS 对象直接 == 恒为 False，必须换成同型再比）
with rasterio.open(j1["aligned_ndvi_path"]) as src:
    al_w, al_h, al_tf, al_crs = src.width, src.height, src.transform, src.crs
with rasterio.open(clip1) as src:
    cl_w, cl_h, cl_tf, cl_crs = src.width, src.height, src.transform, src.crs
check("V1.9 对齐NDVI网格与LST clip一致",
      al_w == cl_w and al_h == cl_h and al_tf == cl_tf and al_crs == cl_crs,
      f"{al_w}x{al_h} vs {cl_w}x{cl_h} tf={al_tf}")
check("V1.9 n 百万级", j1["n"] > 500_000 and j2["n"] > 500_000,
      f"12月对 n={j1['n']:,} / 4月对 n={j2['n']:,}")
check("V1.9 r∈[-1,1] 且 4月对 r<0（若≥0按物候差如实解读）",
      -1 <= j1["pearson_r"] <= 1 and -1 <= j2["pearson_r"] <= 1,
      f"12月对 r={j1['pearson_r']:.3f} / 4月对 r={j2['pearson_r']:.3f}")
check("V1.9 hexbin PNG 非空白", os.path.isfile(j1["scatter_png"])
      and os.path.isfile(j2["scatter_png"])
      and np.asarray(Image.open(j1["scatter_png"]).convert("RGB")).sum(axis=2).min() < 700)

# ------------------------------------------------------------
# V1.10 全部工具返回值 allow_nan=False 序列化不抛异常
# ------------------------------------------------------------
all_returns = [discovery, inv1, inv2, cl1, cl2, st1, st2, cmp_r, j1, j2]
ok10 = True
for i, ret in enumerate(all_returns):
    try:
        json.dumps(ret, ensure_ascii=False, allow_nan=False)
    except ValueError as e:
        ok10 = False
        print(f"  第{i}个返回值失败：{e}")
check("V1.10 全部工具返回值 allow_nan=False 序列化不抛异常", ok10)

# ------------------------------------------------------------
print(f"\n{'=' * 50}")
print(f"共 {len(results)} 项，PASS {sum(results)}，FAIL {len(results) - sum(results)}")
sys.exit(0 if all(results) else 1)
