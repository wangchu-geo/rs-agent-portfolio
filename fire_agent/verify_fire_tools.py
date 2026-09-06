"""
verify_fire_tools.py —— 火点反演链全量验证（V-F1~V-F9）

驱动 3 时次全链（提取→云掩膜→火点识别→统计）后逐项验证：
    旧 result/ 只读参照（0200/0300 有旧品；0400 无参照品，走物理合理性 V-F7）。
    新产物写 E:/YYR/fire/work/。

判据总览：
    V-F1 场景发现配对        V-F2 提取等价（网格/仿射/B01/CLTYPE）
    V-F3 云掩膜两级等价      V-F4 火点掩膜（属性/三层判定/边界证据/总量重合）
    V-F5 幂等重跑            V-F6 产物清单
    V-F7 0400 物理验证       V-F8 统计工具
    V-F9 JSON 序列化

运行：PYTHONUTF8=1 python verify_fire_tools.py
全程约 20-40 分钟（3 时次提取为主）。
"""
import hashlib
import json
import os
import sys

import numpy as np
import rasterio
from scipy.ndimage import correlate, label as sci_label

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fire_tools import (  # noqa: E402
    EXPECTED_FILES,
    _landuse_to_scene_grid,
    apply_fire_cloud_mask,
    calculate_fire_stats,
    generate_fire_mask,
    himawari_extract_bands,
    list_available_himawari_scenes,
)

DATA = "E:/YYR/fire/Himawari"
WORK = "E:/YYR/fire/work"
SHP = "E:/YYR/fire/data/china2/china2.shp"
LANDUSE = "E:/YYR/fire/data/landuse.tif"
SLOTS = ["20250415_0200", "20250415_0300", "20250415_0400"]
OLD = {"20250415_0200": "E:/YYR/fire/result/NC_H09_20250415_0200",
       "20250415_0300": "E:/YYR/fire/result/NC_H09_20250415_0300"}
BANDS = ("07", "14")
CLOUD_SUFFIXES = ("cloud_masked", "cloud_masked_cleaned")

results = []
tool_returns = []  # V-F9：全部工具返回值做 JSON 序列化测试


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}"
          + (f"  {detail}" if detail else ""), flush=True)


def fail(msg):
    print(f"\n[中止] {msg}", flush=True)
    sys.exit(1)


def scene_dir_of(slot):
    return os.path.join(WORK, f"NC_H09_{slot}")


def load_arr(path):
    with rasterio.open(path) as src:
        return src.read(1), src.meta.copy(), src.transform, src.crs


# ============================================================
# V-F1 场景发现
# ============================================================
disc = list_available_himawari_scenes(DATA)
check("V-F1 发现 3 时次且精确配对",
      disc["slots"] == SLOTS and disc["total_slots"] == 3 and not disc["unparsed"],
      str(disc["slots"]) + (f" unparsed={disc['unparsed']}"
                            if disc["unparsed"] else ""))

# ============================================================
# 驱动：3 时次全链 + 0200 B01 可见光路径抽查
# ============================================================
print("\n===== 驱动：全链处理 3 时次 =====", flush=True)
for slot in SLOTS:
    r21 = disc["scenes_by_slot"][slot]["r21"]
    clp = disc["scenes_by_slot"][slot]["clp"]
    r = himawari_extract_bands(r21, clp, WORK, SHP, bands=(7, 14))
    tool_returns.append(r)
    n_err = sum(1 for b in r["band_results"] if b["status"] == "error")
    n_ok = sum(1 for b in r["band_results"] if b["status"] in ("success", "exists"))
    print(f"  提取 {slot}: 波段 {n_ok}/2, CLTYPE={r['cltype']['status']}", flush=True)
    if n_err or r["cltype"]["status"] != "success" and \
            r["cltype"]["status"] != "exists":
        fail(f"{slot} 提取失败：{r}")

    r2 = apply_fire_cloud_mask(scene_dir_of(slot))
    tool_returns.append(r2)
    print(f"  云掩膜 {slot}: {r2['status']}", flush=True)
    if r2["status"] not in ("success", "exists"):
        fail(f"{slot} 云掩膜失败：{r2}")

    r3 = generate_fire_mask(scene_dir_of(slot), LANDUSE)
    tool_returns.append(r3)
    print(f"  火点 {slot}: {r3['status']} fire_pixels={r3.get('fire_pixels')}",
          flush=True)
    if r3["status"] not in ("success", "exists"):
        fail(f"{slot} 火点识别失败：{r3}")

    r4 = calculate_fire_stats(r3["fire_tif"], region_name=slot)
    tool_returns.append(r4)

r_b01 = himawari_extract_bands(disc["scenes_by_slot"]["20250415_0200"]["r21"],
                               disc["scenes_by_slot"]["20250415_0200"]["clp"],
                               WORK, SHP, bands=(1,))
tool_returns.append(r_b01)
print("  提取 0200 B01（可见光路径抽查）:", r_b01["band_results"][0]["status"],
      flush=True)

# ============================================================
# V-F2 提取等价（0200/0300 有旧品参照）
# ============================================================
print("\n===== V-F2 提取等价 =====", flush=True)
for slot in ("20250415_0200", "20250415_0300"):
    old_dir, new_dir = OLD[slot], scene_dir_of(slot)
    for band in BANDS:
        a, _, at, ac = load_arr(os.path.join(old_dir, f"H09_B{band}.tif"))
        b, _, bt, bc = load_arr(os.path.join(new_dir, f"H09_B{band}.tif"))
        check(f"V-F2a {slot} B{band} 网格与旧品严格相等",
              (a.shape, at, ac) == (b.shape, bt, bc),
              f"{b.shape[1]}x{b.shape[0]}")
        d = (a.astype(np.float64) - 273.15) * 100.0 - b.astype(np.float64)
        d_fin = d[np.isfinite(d)]
        check(f"V-F2b {slot} B{band} 仿射等价 max|d|≤0.005",
              d_fin.size > 0 and np.abs(d_fin).max() <= 0.005,
              f"max|d|={np.abs(d_fin).max():.6f}")
        check(f"V-F2b {slot} B{band} 仿射等价 mean|d|≤1.5e-3",
              np.abs(d_fin).mean() <= 1.5e-3,
              f"mean|d|={np.abs(d_fin).mean():.2e}")
        check(f"V-F2b {slot} B{band} NaN 掩膜 array_equal",
              np.array_equal(np.isnan(a), np.isnan(b)))
        exp = (a.astype(np.float64) - 273.15) * 100.0
        exp_fin = exp[np.isfinite(exp)]
        b_fin = b[np.isfinite(b)]
        check(f"V-F2b {slot} B{band} 值域与仿射推得旧品一致(±0.5K)",
              abs(b_fin.min() - exp_fin.min()) <= 0.5
              and abs(b_fin.max() - exp_fin.max()) <= 0.5,
              f"新 {b_fin.min():.2f}~{b_fin.max():.2f}K")
    a_cl, _, at, ac = load_arr(os.path.join(old_dir, "H09_CLTYPE.tif"))
    b_cl, _, bt, bc = load_arr(os.path.join(new_dir, "H09_CLTYPE.tif"))
    check(f"V-F2a {slot} CLTYPE 网格与旧品严格相等",
          (a_cl.shape, at, ac) == (b_cl.shape, bt, bc),
          f"{b_cl.shape[1]}x{b_cl.shape[0]}")
    check(f"V-F2d {slot} CLTYPE 严格 bit-exact",
          np.array_equal(a_cl, b_cl, equal_nan=True))

# V-F2c：B01 可见光路径（albedo/cos(SOZ)）bit-exact
a1, _, a1t, a1c = load_arr(os.path.join(OLD["20250415_0200"], "H09_B01.tif"))
b1, _, b1t, b1c = load_arr(os.path.join(scene_dir_of("20250415_0200"),
                                        "H09_B01.tif"))
check("V-F2c 0200 B01 网格与旧品严格相等", (a1.shape, a1t, a1c) == (b1.shape, b1t, b1c))
check("V-F2c 0200 B01 可见光路径 bit-exact",
      np.array_equal(a1, b1, equal_nan=True),
      f"值域 {np.nanmin(b1):.3f}~{np.nanmax(b1):.3f}")
# B01 仅为可见光路径抽查产物，删除以保持 EXPECTED_FILES 严格清单（V-F6）
os.remove(os.path.join(scene_dir_of("20250415_0200"), "H09_B01.tif"))

# ============================================================
# V-F3 云掩膜两级等价
# ============================================================
print("\n===== V-F3 云掩膜等价 =====", flush=True)
for slot in ("20250415_0200", "20250415_0300"):
    old_dir, new_dir = OLD[slot], scene_dir_of(slot)
    for band in BANDS:
        for suffix in CLOUD_SUFFIXES:
            a, _, _, _ = load_arr(os.path.join(old_dir, f"H09_B{band}_{suffix}.tif"))
            b, _, _, _ = load_arr(os.path.join(new_dir, f"H09_B{band}_{suffix}.tif"))
            # 云掩膜本身无算术：NaN 放置应逐像元一致；非 NaN 值只差量纲仿射
            # （旧品 276K 压缩量纲 vs 新物理 K），按 V-F2b 同容差验证
            check(f"V-F3 {slot} B{band}_{suffix} NaN 掩膜（云放置）array_equal",
                  np.array_equal(np.isnan(a), np.isnan(b)))
            fin = np.isfinite(a) & np.isfinite(b)
            d = (a[fin].astype(np.float64) - 273.15) * 100.0 \
                - b[fin].astype(np.float64)
            check(f"V-F3 {slot} B{band}_{suffix} 非 NaN 值仿射等价 max|d|≤0.005",
                  fin.sum() > 0 and np.abs(d).max() <= 0.005,
                  f"max|d|={np.abs(d).max():.6f}")

# ============================================================
# V-F4 火点掩膜（属性 / 三层判定 / 边界证据 / 总量重合）
# ============================================================
print("\n===== V-F4 火点掩膜 =====", flush=True)
tier_notes = {}


def fire_cond_stats(b07, b14):
    """与 generate_fire_mask 相同的 float32 统计链，返回四条件与统计值。"""
    vm = np.isfinite(b07) & np.isfinite(b14)
    b07n = np.where(vm, b07, np.nan)
    b14n = np.where(vm, b14, np.nan)
    m07, s07 = float(np.nanmean(b07n)), float(np.nanstd(b07n))
    m14, s14 = float(np.nanmean(b14n)), float(np.nanstd(b14n))
    diff = b07n - b14n
    md, sd = float(np.nanmean(diff)), float(np.nanstd(diff))
    c1 = diff > md
    c2 = b07n > (m07 + 2.8 * sd)
    c3 = diff > (md + 2.5 * sd)
    c4 = b14n > (m14 + 2 * s14)
    stats = dict(m07=m07, s07=s07, m14=m14, s14=s14, md=md, sd=sd)
    return ((c1 & c2) | c3 | c4), diff, stats


for slot in ("20250415_0200", "20250415_0300"):
    old_dir, new_dir = OLD[slot], scene_dir_of(slot)
    old_mask, _, _, _ = load_arr(os.path.join(old_dir, "H09_fire_point.tif"))
    old_fire = (old_mask == 1)
    new_mask, meta, ft, fc = load_arr(os.path.join(new_dir, "H09_fire_point.tif"))
    new_fire = (new_mask == 1)
    old_count, new_count = int(old_fire.sum()), int(new_fire.sum())

    with rasterio.open(os.path.join(old_dir, "H09_fire_point.tif")) as src:
        old_grid = (src.width, src.height, src.transform, src.crs, src.dtypes[0],
                    src.nodata)
    check(f"V-F4a {slot} 火点 tif 属性一致（网格/dtype/nodata）",
          (meta["width"], meta["height"], ft, fc, meta["dtype"],
           meta["nodata"]) == old_grid,
          f"{meta['width']}x{meta['height']} {meta['dtype']} nodata={meta['nodata']}")
    vals = np.unique(new_mask)
    check(f"V-F4a {slot} 火点值域 ⊆ {{0,1}}", set(vals.tolist()) <= {0, 1},
          str(vals.tolist()))

    dmask = new_fire != old_fire
    n_diff = int(dmask.sum())
    check(f"V-F4a {slot} 新旧火点数记录（old={old_count}, new={new_count}, "
          f"diff={n_diff}）", True)

    # V-F4c 边界证据（仅 diff>0 时有意义）
    if n_diff > 0:
        b07, _, _, _ = load_arr(os.path.join(new_dir,
                                             "H09_B07_cloud_masked_cleaned.tif"))
        b14, _, _, _ = load_arr(os.path.join(new_dir,
                                             "H09_B14_cloud_masked_cleaned.tif"))
        _, diff_arr, st = fire_cond_stats(b07, b14)
        dist = np.abs(np.stack([
            diff_arr - st["md"],
            b07 - (st["m07"] + 2.8 * st["sd"]),
            diff_arr - (st["md"] + 2.5 * st["sd"]),
            b14 - (st["m14"] + 2 * st["s14"]),
        ])).min(axis=0)
        boundary_frac = float((dist[dmask] <= 0.05).mean())
        nb = correlate(dmask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8),
                       mode="constant")
        cluster_frac = float((nb[dmask] >= 2).mean())
        check(f"V-F4c {slot} 边界证据：≥95% 差异像元距某阈值 ≤0.05K",
              boundary_frac >= 0.95, f"{boundary_frac:.3f}")
        check(f"V-F4c {slot} 8 邻域聚集度 ≥0.5", cluster_frac >= 0.5,
              f"{cluster_frac:.3f}")
    else:
        boundary_frac = cluster_frac = 1.0

    # V-F4b 三层判定
    pass_t = max(20, int(old_count * 0.05))
    note_t = int(old_count * 0.10)
    if n_diff <= pass_t:
        tier, tier_detail = "PASS", f"diff={n_diff} ≤ {pass_t}"
    elif n_diff <= note_t and boundary_frac >= 0.95 and cluster_frac >= 0.5:
        tier = "PASS-WITH-NOTE"
        tier_detail = (f"diff={n_diff} ≤ {note_t} 且边界证据通过 → "
                       "float32 统计阈值平移的边界像元翻转")
    else:
        tier = "FAIL"
        tier_detail = (f"diff={n_diff} 超线或证据不过（boundary={boundary_frac:.3f}"
                       f" cluster={cluster_frac:.3f}）")
    tier_notes[slot] = (n_diff, tier, tier_detail)
    check(f"V-F4b {slot} 三层判定（旧={old_count} 新={new_count}）",
          tier != "FAIL", tier_detail)

    # V-F4d 总量与重合
    inter = int((new_fire & old_fire).sum())
    union = int((new_fire | old_fire).sum())
    iou = inter / union if union else 0.0
    check(f"V-F4d {slot} 新火点数 ∈ 旧数±10%",
          abs(new_count - old_count) <= 0.10 * old_count,
          f"{new_count} vs {old_count}")
    check(f"V-F4d {slot} IoU≥0.9", iou >= 0.9, f"IoU={iou:.4f}")

# ============================================================
# V-F5 幂等重跑（0200）
# ============================================================
print("\n===== V-F5 幂等 =====", flush=True)


def snapshot(scene_dir):
    out = {}
    for f in EXPECTED_FILES:
        p = os.path.join(scene_dir, f)
        st = os.stat(p)
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out[f] = (st.st_mtime_ns, h.hexdigest())
    return out


s0200 = scene_dir_of("20250415_0200")
before = snapshot(s0200)
r21 = disc["scenes_by_slot"]["20250415_0200"]["r21"]
clp = disc["scenes_by_slot"]["20250415_0200"]["clp"]
r = himawari_extract_bands(r21, clp, WORK, SHP, bands=(7, 14))
tool_returns.append(r)
r2 = apply_fire_cloud_mask(s0200)
tool_returns.append(r2)
r3 = generate_fire_mask(s0200, LANDUSE)
tool_returns.append(r3)
skipped = (r["cltype"]["status"] == "exists"
           and all(b["status"] == "exists" for b in r["band_results"])
           and r2["status"] == "exists" and r3["status"] == "exists")
check("V-F5 0200 四工具重跑全部 exists 跳过", skipped)
after = snapshot(s0200)
check("V-F5 0200 产物 mtime + sha256 逐文件不变", before == after)

# ============================================================
# V-F6 产物清单
# ============================================================
print("\n===== V-F6 产物清单 =====", flush=True)
for slot in SLOTS:
    files = set(os.listdir(scene_dir_of(slot)))
    check(f"V-F6 {slot} 文件集合 == EXPECTED_FILES（无残留混入）",
          files == EXPECTED_FILES,
          "" if files == EXPECTED_FILES else
          f"多 {sorted(files - EXPECTED_FILES)} 缺 {sorted(EXPECTED_FILES - files)}")

# ============================================================
# V-F7 0400 全链物理验证（无旧品参照）
# ============================================================
print("\n===== V-F7 0400 物理验证 =====", flush=True)
s0400 = scene_dir_of("20250415_0400")
b07, _, _, _ = load_arr(os.path.join(s0400, "H09_B07_cloud_masked_cleaned.tif"))
b14, _, _, _ = load_arr(os.path.join(s0400, "H09_B14_cloud_masked_cleaned.tif"))
b07_fin, b14_fin = b07[np.isfinite(b07)], b14[np.isfinite(b14)]
check("V-F7 0400 B07 值域∈[230,370]K",
      b07_fin.min() >= 230 and b07_fin.max() <= 370,
      f"{b07_fin.min():.2f}~{b07_fin.max():.2f}")
check("V-F7 0400 B14 值域∈[170,330]K",
      b14_fin.min() >= 170 and b14_fin.max() <= 330,
      f"{b14_fin.min():.2f}~{b14_fin.max():.2f}")

mask40, meta40, t40, c40 = load_arr(os.path.join(s0400, "H09_fire_point.tif"))
n40 = int((mask40 == 1).sum())
check("V-F7 0400 火点数∈[500,8000]", 500 <= n40 <= 8000, str(n40))

# 逐像元验证：每个火点必须满足四条件至少一项 + 林地 + 有效
cond40, _, _ = fire_cond_stats(b07, b14)
with rasterio.open(os.path.join(s0400, "H09_B07_cloud_masked_cleaned.tif")) as src:
    lu = _landuse_to_scene_grid(LANDUSE, s0400, src.crs, src.transform,
                                src.bounds, src.height, src.width)
fire40 = mask40 == 1
lu_ok = (lu >= 20) & (lu < 25)
check("V-F7 0400 逐像元：每个火点满足四条件至少一项（100%）",
      bool(np.all(cond40[fire40])), f"{n40} 像元")
check("V-F7 0400 逐像元：每个火点位于林地类且值有效",
      bool(np.all(lu_ok[fire40])) and bool(np.all(np.isfinite(b07[fire40]))))

# 网格与 0200 新场景一致；火点 bbox 中心与 0200 差 ≤0.5°
mask20, _, _, _ = load_arr(os.path.join(scene_dir_of("20250415_0200"),
                                        "H09_fire_point.tif"))
_, _, t20, c20 = load_arr(os.path.join(scene_dir_of("20250415_0200"),
                                       "H09_B07.tif"))
check("V-F7 0400 网格与 0200 新场景严格相等",
      (meta40["width"], meta40["height"], t40, c40)
      == (mask20.shape[1], mask20.shape[0], t20, c20))
cx, cy = {}, {}
for slot, tag in (("20250415_0200", "0200"), ("20250415_0400", "0400")):
    m = mask20 if slot.endswith("0200") else mask40
    rs, cs = np.nonzero(m == 1)
    with rasterio.open(os.path.join(scene_dir_of(slot), "H09_B07.tif")) as src:
        gt = src.transform
    cx[tag] = gt.c + (cs.min() + (cs.max() - cs.min()) / 2) * gt.a
    cy[tag] = gt.f + (rs.min() + (rs.max() - rs.min()) / 2) * gt.e
check("V-F7 0400 火点 bbox 中心与 0200 差 ≤0.5°",
      abs(cx["0400"] - cx["0200"]) <= 0.5 and abs(cy["0400"] - cy["0200"]) <= 0.5,
      f"Δlon={cx['0400']-cx['0200']:.3f} Δlat={cy['0400']-cy['0200']:.3f}")

# ============================================================
# V-F8 统计工具
# ============================================================
print("\n===== V-F8 统计工具 =====", flush=True)
stats_returns = [r for r in tool_returns if "top5_clusters" in r]
for r in stats_returns:
    slot = r["region"]
    m, _, gt, _ = load_arr(r["fire_tif"])
    fire = m == 1
    n_ind = int(fire.sum())
    check(f"V-F8 {slot} fire_pixels 与独立重算 exact",
          r["fire_pixels"] == n_ind, f"{r['fire_pixels']}")
    check(f"V-F8 {slot} 占比∈[0.0005,0.002]",
          0.0005 <= r["fire_ratio"] <= 0.002, f"{r['fire_ratio']:.6f}")
    # 独立面积公式（float64，逐像元）交叉验证
    rows, cols = np.nonzero(fire)
    res_x, res_y = abs(gt.a), abs(gt.e)
    lats = gt.f + (rows + 0.5) * gt.e
    area_ind = float((res_x * 111.32 * np.cos(np.deg2rad(lats))
                      * res_y * 111.32).sum())
    check(f"V-F8 {slot} 面积独立公式交叉 |Δ|≤0.1%",
          area_ind > 0 and abs(area_ind - r["fire_area_km2"]) / area_ind <= 0.001,
          f"tool={r['fire_area_km2']:.1f} ind={area_ind:.1f}")
    # bbox 像素四至 exact
    if r["bbox"] is None:
        check(f"V-F8 {slot} bbox 像素四至 exact", False, "无火点（bbox=None）")
    else:
        rmin, rmax = int(rows.min()), int(rows.max())
        cmin, cmax = int(cols.min()), int(cols.max())
        check(f"V-F8 {slot} bbox 像素四至 exact",
              r["bbox"]["pixels"] == [rmin, rmax, cmin, cmax])
    # 连通簇数独立重算 + top5 单调 + 总量约束
    labeled, n_cl = sci_label(fire)
    sizes = np.bincount(labeled.ravel())[1:]
    order = np.argsort(sizes)[::-1]
    t5 = [int(sizes[i]) for i in order[:5]]
    check(f"V-F8 {slot} 簇数独立重算 exact", r["clusters"] == int(n_cl),
          str(r["clusters"]))
    check(f"V-F8 {slot} top-5 簇面积单调且合计≤火点总数",
          t5 == sorted(t5, reverse=True)
          and [x["size_pixels"] for x in r["top5_clusters"]] == t5
          and sum(t5) <= n_ind,
          str(t5))

# ============================================================
# V-F9 JSON 序列化
# ============================================================
print("\n===== V-F9 JSON 序列化 =====", flush=True)
n_json_ok = 0
for i, ret in enumerate(tool_returns):
    try:
        json.dumps(ret, ensure_ascii=False, allow_nan=False)
        n_json_ok += 1
    except (TypeError, ValueError) as e:
        check(f"V-F9 工具返回 #{i} 序列化失败", False, str(e))
check(f"V-F9 全部 {len(tool_returns)} 个工具返回 json.dumps(allow_nan=False) 直过",
      n_json_ok == len(tool_returns), f"{n_json_ok}/{len(tool_returns)}")

# ============================================================
print(f"\n{'=' * 60}")
print(f"共 {len(results)} 项，PASS {sum(results)}，FAIL {len(results) - sum(results)}")
print("V-F4 三层判定实测：",
      "; ".join(f"{s}: diff={v[0]} → {v[1]}" for s, v in tier_notes.items()))
sys.exit(0 if all(results) else 1)
