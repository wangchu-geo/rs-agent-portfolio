"""
多时相 NDVI 反演与对比 —— 管线工具集
目标：把旧的一次性管线 NDVI_CLOUD_MASK.py 拆成 Agent 可调用的工具，
      每个工具对应旧管线的一步，按 (输入, 输出, 时相) 参数化

与旧管线五步的对应关系：
    旧: gpt预处理 → img转tif → 去云 → 镶嵌 → 裁剪
    新: snap_preprocess_date → convert_date_to_tif → apply_cloud_mask_date
        → mosaic_date → clip_date_to_aoi
    外加: list_available_scenes（场景发现）与 compare_ndvi_dates（两期对比）

设计原则：
    1. 工具按时相批量：一次调用处理该日期的全部景，返回逐景状态
    2. 数据路径全部由模型显式传入，工具内部只固定命名规则
    3. SNAP 13 可直接读取 .SAFE.zip 压缩包，无需解压
    4. 所有返回值经 _json_safe 清洗（NaN → null），保证是合法 JSON
"""
import json
import os
import re
import shutil
import subprocess

import numpy as np
import rasterio

# matplotlib 必须在使用 pyplot 之前切换到无界面后端（Agent 循环里没有 GUI 窗口）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from osgeo import gdal

# ============================================================
# 常量：环境相关配置
# ============================================================
GPT_EXE = r"E:/ruanjian/SNAP/esa-snap/bin/gpt.exe"
GPT_TIMEOUT = 3600                      # 单景 gpt 处理超时（秒）
NODATA_RAW = -9999.0                    # 裁剪后背景像元值（与 ndvi_tools 的统计兼容）
DATE_RE = r"S2[AB]_MSIL2A_(\d{8})T"     # 从哨兵2场景名提取日期
GDAL_CREATION = ["BIGTIFF=IF_SAFER", "TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=2"]

# 变化分级：改善/稳定/退化 + 无效
CLASS_NAMES = {0: "无效", 1: "改善", 2: "稳定", 3: "退化"}
# 颜色全部来自 dataviz 调色板文档（经验证器校验）：
#   蓝↔红 = 官方发散对（色盲 ΔE 21.6 通过）；稳定 = 基线灰；无效 = 发散中性灰 + 影线
CLASS_COLORS = {0: "#f0efec", 1: "#2a78d6", 2: "#c3c2b7", 3: "#e34948"}

# ============================================================
# 通用辅助函数
# ============================================================
def _json_safe(obj):
    """递归清洗：np.nan/np.inf → None、np 标量 → python 标量、ndarray → list。

    为什么必须有这一步：json.dumps 不接受 NaN（非法 JSON）和 numpy 类型，
    工具返回值是要回传给大模型的 JSON 字符串，清洗后才不会在序列化时报错。
    """
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def _scene_date(scene_name: str) -> str:
    """从哨兵2场景名提取成像日期（如 S2B_MSIL2A_20260510T... → 20260510）"""
    m = re.search(DATE_RE, scene_name)
    if not m:
        raise ValueError(f"无法从名称解析成像日期：{scene_name}")
    return m.group(1)


def _setup_chinese_font():
    """matplotlib 中文字体：微软雅黑（Windows 系统无衬线字体，与调色板规范一致）"""
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 工具 1：场景发现
# ============================================================
def list_available_scenes(data_dir: str) -> dict:
    """
    扫描影像目录，从哨兵2 SAFE 文件名中解析成像日期，按日期分组

    兼容两种形态（SNAP 13 可直接读取压缩包，推荐不压缩包直接传 gpt）：
      1) .SAFE.zip 压缩包文件
      2) 已解压的 .SAFE 目录

    返回：
        {'dates': [按日期排序], 'scenes_by_date': {日期: [景路径列表]},
         'total_scenes': 景数, 'hint': 使用提示}
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"找不到影像目录：{data_dir}")

    scenes = []
    for name in sorted(os.listdir(data_dir)):
        full = os.path.join(data_dir, name)
        if name.endswith(".SAFE.zip") and os.path.isfile(full):
            scenes.append(full)
        elif name.endswith(".SAFE") and os.path.isdir(full):
            scenes.append(full)

    if not scenes:
        raise FileNotFoundError(f"{data_dir} 下没有找到 .SAFE.zip 影像或 .SAFE 目录")

    scenes_by_date = {}
    for path in scenes:
        date = _scene_date(os.path.basename(path))
        scenes_by_date.setdefault(date, []).append(path)

    return _json_safe({
        "dates": sorted(scenes_by_date),
        "scenes_by_date": scenes_by_date,
        "total_scenes": len(scenes),
        "hint": "把 scenes_by_date[日期] 的路径列表原样传给 snap_preprocess_date 的 scene_paths",
    })


# ============================================================
# 工具 2：SNAP gpt 预处理（旧管线第 1 步）
# ============================================================
def _run_gpt_one(scene_path: str, xml_path: str, out_dim: str) -> dict:
    """单景 gpt 调用：list 参数直传（不用 shell，规避 cmd 引号/%转义坑），
    输出用 utf-8 容错捕获，超时保护。gpt 首景有约 1-2 分钟 auxdata 预热属正常。"""
    cmd = [GPT_EXE, xml_path, f"-Pinput={scene_path}", f"-Poutput={out_dim}"]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=GPT_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return {"tile": os.path.basename(scene_path), "status": "timeout",
                "log_tail": f"处理超过 {GPT_TIMEOUT} 秒被终止"}
    except Exception as e:
        return {"tile": os.path.basename(scene_path), "status": "error",
                "error": str(e)}

    if proc.returncode == 0:
        return {"tile": os.path.basename(scene_path), "status": "success",
                "output_dim": out_dim, "log_tail": proc.stdout[-800:]}
    return {"tile": os.path.basename(scene_path), "status": "failed",
            "returncode": proc.returncode,
            "log_tail": (proc.stdout + proc.stderr)[-800:]}


def snap_preprocess_date(scene_paths: list, xml_path: str, output_dir: str) -> dict:
    """
    调用 SNAP gpt 对某日期的一批影像执行 XML 流程：
    NDVI 波段运算 + 云影掩膜（SCL 3/8/9/10），输出 BEAM-DIMAP。

    每景约 2-6 分钟；SNAP 13 直接读取 .SAFE.zip，无需解压。
    输出目录中生成 <影像名>.dim 与 <影像名>.data。
    """
    if not xml_path or not os.path.isfile(xml_path):
        raise FileNotFoundError(f"找不到 gpt 流程 XML：{xml_path}")
    if not scene_paths:
        raise ValueError("scene_paths 为空：请传入 list_available_scenes 返回的路径列表")
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for scene in sorted(scene_paths):
        scene_name = os.path.basename(scene)
        out_dim = os.path.join(output_dir, scene_name + ".dim")
        # 覆盖语义：已存在同名结果先删除再重跑，避免 gpt 报"目标已存在"
        data_dir = out_dim[:-4] + ".data"
        for stale in (out_dim, data_dir):
            if os.path.isdir(stale):
                shutil.rmtree(stale)
            elif os.path.isfile(stale):
                os.remove(stale)
        results.append(_run_gpt_one(scene, xml_path, out_dim))

    return _json_safe({
        "total": len(scene_paths),
        "success": sum(1 for r in results if r["status"] == "success"),
        "output_dir": output_dir,
        "results": results,
    })


# ============================================================
# 工具 3：BEAM-DIMAP 转 GeoTIFF（旧管线第 2 步）
# ============================================================
def convert_date_to_tif(dim_dir: str, output_dir: str) -> dict:
    """
    把 BEAM-DIMAP 结果（<场景名>.data 目录中的 NDVI.img、cloud_shadow_mask.img）
    转成 GeoTIFF，保留原坐标系/仿射变换/数据类型。

    输出结构：output_dir/<场景名>.data/{NDVI.tif, cloud_shadow_mask.tif}
    """
    if not os.path.isdir(dim_dir):
        raise FileNotFoundError(f"找不到 gpt 输出目录：{dim_dir}")
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for sub in sorted(os.listdir(dim_dir)):
        data_dir = os.path.join(dim_dir, sub)
        if not (sub.endswith(".data") and os.path.isdir(data_dir)):
            continue
        imgs = sorted(f for f in os.listdir(data_dir) if f.endswith(".img"))
        if not imgs:
            results.append({"tile": sub, "status": "failed",
                            "error": "该目录中没有 .img 波段文件"})
            continue
        files = []
        for img in imgs:
            src_path = os.path.join(data_dir, img)
            out_sub = os.path.join(output_dir, sub)
            os.makedirs(out_sub, exist_ok=True)
            out_path = os.path.join(out_sub, img[:-4] + ".tif")
            with rasterio.open(src_path) as src:
                profile = src.profile
                profile.update(driver="GTiff")
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(src.read())
            files.append(out_path)
        results.append({"tile": sub, "status": "success", "files": files})

    return _json_safe({"total": len(results), "output_dir": output_dir,
                       "results": results})


# ============================================================
# 工具 4：去云掩膜（旧管线第 3 步）
# ============================================================
def apply_cloud_mask_date(tif_dir: str, output_dir: str) -> dict:
    """
    去云：掩膜为 0（云/云影/SCL 无效）的像元置 NaN，
    超出 [-1,1] 理论范围的异常值也置 NaN，输出 float32 产品，
    并统计每景被掩掉的云像元占比。
    """
    if not os.path.isdir(tif_dir):
        raise FileNotFoundError(f"找不到 tif 目录：{tif_dir}")
    os.makedirs(output_dir, exist_ok=True)

    results = []
    for sub in sorted(os.listdir(tif_dir)):
        sub_dir = os.path.join(tif_dir, sub)
        if not (sub.endswith(".data") and os.path.isdir(sub_dir)):
            continue
        ndvi_tif = os.path.join(sub_dir, "NDVI.tif")
        mask_tif = os.path.join(sub_dir, "cloud_shadow_mask.tif")
        if not (os.path.isfile(ndvi_tif) and os.path.isfile(mask_tif)):
            results.append({"tile": sub, "status": "failed",
                            "error": "缺少 NDVI.tif 或 cloud_shadow_mask.tif"})
            continue

        with rasterio.open(ndvi_tif) as src:
            ndvi = src.read(1).astype(np.float32)
            profile = src.profile
        with rasterio.open(mask_tif) as src:
            mask = src.read(1)

        ndvi[mask == 0] = np.nan
        ndvi[(ndvi < -1.0) | (ndvi > 1.0)] = np.nan

        cloud_ratio = float((mask == 0).sum()) / mask.size
        out_path = os.path.join(output_dir, sub + "_NDVI_processed.tif")
        profile.update(driver="GTiff", nodata=float("nan"))
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(ndvi, 1)
        results.append({"tile": sub, "status": "success", "output": out_path,
                        "cloud_pixel_ratio": round(cloud_ratio, 4)})

    return _json_safe({"total": len(results), "output_dir": output_dir,
                       "results": results})


# ============================================================
# 工具 5：镶嵌（旧管线第 4 步）
# ============================================================
def mosaic_date(processed_dir: str, output_dir: str, date: str = "") -> dict:
    """
    把某日期全部去云 NDVI 镶嵌为单幅 GeoTIFF。

    用 VRT 虚拟镶嵌 + Translate 流式写入（单景约 482MB，整幅载入内存会爆，
    必须走块级流式，不能 rasterio.merge 进 numpy）。
    """
    if not os.path.isdir(processed_dir):
        raise FileNotFoundError(f"找不到去云结果目录：{processed_dir}")

    tif_list = []
    for root, _, files in os.walk(processed_dir):
        for f in sorted(files):
            if f.endswith("_NDVI_processed.tif"):
                tif_list.append(os.path.join(root, f))
    if not tif_list:
        raise FileNotFoundError(f"{processed_dir} 下没有找到 *_NDVI_processed.tif")

    if not date:
        m = re.search(r"(\d{8})", os.path.basename(tif_list[0]))
        date = m.group(1) if m else "unknown"

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"NDVI_{date}_mosaic.tif")
    vrt_path = "/vsimem/ndvi_mosaic.vrt"
    gdal.BuildVRT(vrt_path, tif_list,
                  options=gdal.BuildVRTOptions(resampleAlg="nearest"))
    gdal.Translate(out_path, vrt_path, outputType=gdal.GDT_Float32,
                   creationOptions=GDAL_CREATION)
    gdal.Unlink(vrt_path)

    ds = gdal.Open(out_path)
    if ds is None:
        raise RuntimeError(f"镶嵌失败：{out_path}")
    w, h = ds.RasterXSize, ds.RasterYSize
    crs = ds.GetProjection()
    ds = None
    return _json_safe({
        "mosaic_path": out_path, "tiles": len(tif_list),
        "width": w, "height": h, "crs": crs,
        "size_mb": round(os.path.getsize(out_path) / 1024 / 1024, 1),
    })


# ============================================================
# 工具 6：按研究区裁剪（旧管线第 5 步）
# ============================================================
def clip_date_to_aoi(mosaic_tif: str, shp_path: str, output_dir: str) -> dict:
    """
    用研究区矢量裁剪镶嵌结果。背景置 -9999（ndvi_tools 统计兼容该 nodata），
    矢量与栅格坐标系不一致时 gdal.Warp 自动重投影。
    """
    if not os.path.isfile(mosaic_tif):
        raise FileNotFoundError(f"找不到镶嵌结果：{mosaic_tif}")
    if not os.path.isfile(shp_path):
        raise FileNotFoundError(f"找不到裁剪矢量：{shp_path}")

    m = re.search(r"(\d{8})", os.path.basename(mosaic_tif))
    date = m.group(1) if m else "unknown"

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"NDVI_{date}_clip.tif")

    # 注意：GDAL 3.6 的 Python 绑定没有 config_options 上下文管理器，
    # 用 SetConfigOption + finally 恢复（对齐旧管线里 GDALWARP_IGNORE_BAD_CUTLINE=YES 的做法）
    gdal.SetConfigOption("GDALWARP_IGNORE_BAD_CUTLINE", "YES")
    try:
        ds = gdal.Warp(out_path, mosaic_tif, options=gdal.WarpOptions(
            cutlineDSName=shp_path,
            cropToCutline=True,
            dstNodata=NODATA_RAW,
            resampleAlg=gdal.GRA_NearestNeighbour,
            multithread=True,
            creationOptions=GDAL_CREATION,
        ))
    finally:
        gdal.SetConfigOption("GDALWARP_IGNORE_BAD_CUTLINE", None)
    if ds is None:
        raise RuntimeError(
            f"裁剪失败：{mosaic_tif}（请检查矢量与栅格的范围是否相交）")

    gt = ds.GetGeoTransform()
    w, h = ds.RasterXSize, ds.RasterYSize
    crs = ds.GetProjection()
    xmin, ymax = gt[0], gt[3]
    xmax, ymin = gt[0] + w * gt[1], gt[3] + h * gt[5]
    ds = None
    return _json_safe({
        "clip_path": out_path, "crs": crs, "width": w, "height": h,
        "transform": gt, "extent": [xmin, ymax, xmax, ymin],
    })


# ============================================================
# 工具 7：两期对比
# ============================================================
def compare_ndvi_dates(date1_path: str, date2_path: str, output_dir: str,
                       threshold: float = 0.1) -> dict:
    """
    两期 NDVI 裁剪产品对比（date1=较早，date2=较晚，Δ = 晚 − 早）：

    1. 把较早日期无条件重采样到较晚日期的精确网格（不假设两期网格一致）
    2. 逐像元求差并分类：Δ>=0.1 改善 / |Δ|<0.1 稳定 / Δ<=-0.1 退化 / 无效
    3. 输出差值 tif、分类 tif、变化分类图 PNG、面积占比图 PNG
    4. 返回各类像元数、面积（公顷）与占比统计

    面积换算：10m 像元 = 100 m² = 0.01 公顷
    """
    for p in (date1_path, date2_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"找不到 NDVI 结果：{p}")
    os.makedirs(output_dir, exist_ok=True)

    # ---------- 日期标签（从文件名提取，失败用占位名） ----------
    m1 = re.search(r"(\d{8})", os.path.basename(date1_path))
    m2 = re.search(r"(\d{8})", os.path.basename(date2_path))
    d1 = m1.group(1) if m1 else "date1"
    d2 = m2.group(1) if m2 else "date2"

    # ---------- 第一步：网格对齐（模板 = 较晚日期） ----------
    with rasterio.open(date2_path) as tpl:
        gt, crs, w, h = tpl.transform, tpl.crs, tpl.width, tpl.height
        profile = tpl.profile
    xmin, ymax = gt[2], gt[5]
    xmax, ymin = gt[2] + w * gt[0], gt[5] + h * gt[4]

    aligned1 = os.path.join(output_dir, "__aligned_date1.tif")
    warp_ds = gdal.Warp(aligned1, date1_path, options=gdal.WarpOptions(
        format="GTiff", dstSRS=crs.to_wkt(),
        outputBounds=(xmin, ymin, xmax, ymax), width=w, height=h,
        dstNodata=float("nan"), resampleAlg=gdal.GRA_NearestNeighbour,
        creationOptions=GDAL_CREATION))
    if warp_ds is None:
        raise RuntimeError("网格对齐失败（两期影像范围可能完全不相交）")
    warp_ds = None

    # ---------- 第二步：nodata 统一为 NaN 后差分 ----------
    with rasterio.open(aligned1) as src:
        a1 = src.read(1).astype(np.float32)
        n1 = src.nodata
    with rasterio.open(date2_path) as src:
        a2 = src.read(1).astype(np.float32)
        n2 = src.nodata
    # 关键：不先把 -9999 换成 NaN，差分会把背景算成 Δ≈0.999 的虚假"改善"
    if n1 is not None:
        a1[a1 == n1] = np.nan
    if n2 is not None:
        a2[a2 == n2] = np.nan

    valid = ~np.isnan(a1) & ~np.isnan(a2)
    delta = np.where(valid, a2 - a1, np.nan)
    cls = np.zeros(delta.shape, dtype="uint8")
    cls[valid & (delta >= threshold)] = 1
    cls[valid & (np.abs(delta) < threshold)] = 2
    cls[valid & (delta <= -threshold)] = 3

    # ---------- 第三步：写差值 / 分类栅格 ----------
    diff_path = os.path.join(output_dir, f"NDVI_diff_{d1}_vs_{d2}.tif")
    class_path = os.path.join(output_dir, f"NDVI_class_{d1}_vs_{d2}.tif")

    profile.update(driver="GTiff", count=1, dtype="float32",
                   nodata=float("nan"), compress="deflate",
                   tiled=True, bigtiff="IF_SAFER")
    with rasterio.open(diff_path, "w", **profile) as dst:
        dst.write(delta, 1)

    profile.update(dtype="uint8", nodata=0)
    with rasterio.open(class_path, "w", **profile) as dst:
        dst.write(cls, 1)

    # ---------- 第四步：分级统计 ----------
    total_pairs = cls.size
    valid_pairs = int(valid.sum())
    per_class = {}
    for code, name in CLASS_NAMES.items():
        pixels = int((cls == code).sum())
        area_ha = pixels * 0.01
        per_class[name] = {
            "pixels": pixels,
            "area_ha": round(area_ha, 2),
            "ratio_of_all": round(pixels / total_pairs, 4) if total_pairs else 0.0,
            # 无效类不是有效对的子集，占有效对比例无意义 → None（其余三类之和为 1）
            "ratio_of_valid": (round(pixels / valid_pairs, 4)
                               if (valid_pairs and code != 0) else None),
        }
    dv = delta[valid]
    delta_stats = {
        "mean": float(dv.mean()), "std": float(dv.std()),
        "min": float(dv.min()), "max": float(dv.max()),
        "valid_pairs": valid_pairs,
    }

    # ---------- 第五步：出图 ----------
    _setup_chinese_font()
    change_map_png = _plot_change_map(cls, gt, per_class, d1, d2, output_dir)
    area_chart_png = _plot_area_chart(per_class, d1, d2, output_dir)

    # 中间对齐产物用完即删，保持输出目录干净
    os.remove(aligned1)

    return _json_safe({
        "date1": d1, "date2": d2, "threshold": threshold,
        "diff_path": diff_path, "class_path": class_path,
        "change_map_png": change_map_png, "area_chart_png": area_chart_png,
        "classes": per_class, "delta_stats": delta_stats,
        "note": "面积按10m像元=0.01公顷换算；无效=任一日期无有效像元（云/影/背景）",
    })


def _plot_change_map(cls, gt, per_class, d1, d2, output_dir,
                     class_names=CLASS_NAMES, class_colors=CLASS_COLORS,
                     product="NDVI", file_prefix="change_map") -> str:
    """图1：变化分类图。三类变化色填图，无效类留白（图例用影线标识）。

    颜色取自调色板文档：蓝↔红为官方发散对（色盲验证 ΔE 21.6 通过），
    稳定用基线灰 —— 中性色在分类中"后退"，让变化类突出。
    class_names/class_colors/product/file_prefix 供 LST 对比工具复用
    （默认值与原 NDVI 行为逐字节一致）。
    """
    h, w = cls.shape
    xmin, ymax = gt[2], gt[5]
    xmax, ymin = gt[2] + w * gt[0], gt[5] + h * gt[4]

    masked = np.ma.masked_where(cls == 0, cls)
    cmap = ListedColormap([class_colors[1], class_colors[2], class_colors[3]])
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.imshow(masked, cmap=cmap, norm=norm, interpolation="nearest",
              extent=(xmin, xmax, ymin, ymax))
    ax.set_xlabel("东向坐标 (m)")
    ax.set_ylabel("北向坐标 (m)")
    ax.set_title(f"{product}变化分类图（{d1} → {d2}）", fontsize=14)

    legend_patches = [
        Patch(facecolor=class_colors[1],
              label=f"{class_names[1]}（{per_class[class_names[1]]['ratio_of_all'] * 100:.1f}%）"),
        Patch(facecolor=class_colors[2],
              label=f"{class_names[2]}（{per_class[class_names[2]]['ratio_of_all'] * 100:.1f}%）"),
        Patch(facecolor=class_colors[3],
              label=f"{class_names[3]}（{per_class[class_names[3]]['ratio_of_all'] * 100:.1f}%）"),
        Patch(facecolor=class_colors[0], edgecolor="#898781", hatch="///",
              label=f"{class_names[0]}（{per_class[class_names[0]]['ratio_of_all'] * 100:.1f}%）"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", framealpha=0.9)

    png_path = os.path.join(output_dir, f"{file_prefix}_{d1}_vs_{d2}.png")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


def _plot_area_chart(per_class, d1, d2, output_dir,
                     class_names=CLASS_NAMES, class_colors=CLASS_COLORS,
                     order=(0, 3, 2, 1), product="NDVI",
                     file_prefix="area_proportion") -> str:
    """图2：面积占比横向条形图。四类颜色与图1一致（颜色跟随实体），
    条端直接标注百分比，刻度标签带面积公顷数。
    class_names/class_colors/order/product/file_prefix 供 LST 对比工具复用
    （默认值与原 NDVI 行为逐字节一致）。"""
    labels = [class_names[i] for i in order]    # barh 自底向上画 → 首个变化类在顶部
    values = [per_class[name]["ratio_of_all"] * 100 for name in labels]
    colors = [class_colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    bars = ax.barh(labels, values, height=0.55, color=colors,
                   edgecolor="#fcfcfb", linewidth=1.0)
    max_val = max(values) if values else 1.0
    for i, (bar, val) in enumerate(zip(bars, values)):
        if order[i] == 0:
            bar.set_hatch("///")   # 浅灰类加影线，避免与背景混淆
        ax.text(bar.get_width() + max_val * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", color="#0b0b0b")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(
        [f"{name}（{per_class[name]['area_ha']:,.0f} 公顷）" for name in labels])
    ax.set_xlabel("面积占比 (%)")
    ax.set_title(f"{product}变化类型面积占比（{d1} → {d2}）", fontsize=14)
    ax.set_xlim(0, max_val * 1.15)

    ax.xaxis.grid(True, color="#e1e0d9", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")

    png_path = os.path.join(output_dir, f"{file_prefix}_{d1}_vs_{d2}.png")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return png_path
