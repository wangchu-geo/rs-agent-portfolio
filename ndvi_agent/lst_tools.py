"""
Landsat 地表温度（LST）反演 —— 管线工具集
把旧脚本 E:/YYR/LST/LST2.py 拆成 Agent 可调用的工具，并补齐旧脚本缺失的
裁剪/统计/对比/联合分析能力。

与旧脚本的对应关系：
    旧: process_tar_files → extract_tar → process_scene_directory
        （找 B10 → ST_B10 线性换算 K→℃ → QA 启发式掩膜 → 写 LST.TIF）
    新: lst_invert_date（解压+换算+掩膜一体，按时相批量）
    新增: list_available_lst_scenes / lst_clip_date_to_aoi /
          calculate_lst_stats / compare_lst_dates / analyze_lst_ndvi

设计原则（与 pipeline_tools 一致）：
    1. 工具按时相批量：一次调用处理该日期全部景，返回逐景状态
    2. 云掩膜口径与 6 月历史产品严格一致（旧脚本启发式 qa<2 或 qa>22000，
       用户已确认保留——换取与新链逐像元严格可验证性）
    3. 数据路径全部由模型显式传入，工具内部只固定命名规则
    4. 所有返回值经 _json_safe 清洗（NaN → null）
"""
import os
import re
import tarfile

import numpy as np
import rasterio

from osgeo import gdal, gdalconst

from pipeline_tools import (
    GDAL_CREATION,
    NODATA_RAW,
    _json_safe,
    _plot_area_chart,
    _plot_change_map,
    _setup_chinese_font,
)

# pipeline_tools 导入时已执行 matplotlib.use("Agg")，此处可直接取 pyplot
import matplotlib.pyplot as plt

# ============================================================
# 常量
# ============================================================
# 从 Landsat C2 L2SP 文件名解析获取日期：
#   LC09_L2SP_120036_20251226_20251227_02_T1.tar
#   卫星标识(2-3位) 条带       ^^^^^^^^ 获取日期（取第一个8位日期） ^^^^^^^^ 处理日期
LST_TAR_RE = re.compile(r"L\w{2,3}_L2SP_\d{6}_(\d{8})_")

# ST_B10 线性换算（与旧脚本逐字一致；L2SP 的 ST_B10 已是地表温度波段，K → ℃）
LST_SCALE, LST_OFFSET_K, LST_C0 = 0.00341802, 149.0, 273.15
# QA_PIXEL 启发式云掩膜阈值（旧脚本原样保留）
QA_CLEAR_MIN, QA_CLOUD_MAX = 2, 22000
# 30m 像元 = 900 m² = 0.09 公顷（与 NDVI 10m 的 0.01 不同！）
LST_PIXEL_HA = 0.09

# ΔLST 变化分级：红=升温、蓝=降温（复用已验证调色板，与 NDVI 的红退化/蓝改善语义对齐）
LST_CLASS_NAMES = {0: "无效", 1: "升温", 2: "稳定", 3: "降温"}
LST_CLASS_COLORS = {0: "#f0efec", 1: "#e34948", 2: "#c3c2b7", 3: "#2a78d6"}

# 温度分级（阈值从高到低，与 ndvi_tools.NDVI_LEVELS 结构同构）
LST_LEVELS = [
    (30.0, "炎热", "城市/强受热地表"),
    (20.0, "暖热", "春秋暖区"),
    (10.0, "温和", "过渡带"),
    (0.0, "寒冷", "冬季低温区"),
    (float("-inf"), "低温", "积雪或极寒地表"),
]


# ============================================================
# 通用辅助函数
# ============================================================
def _lst_tar_date(tar_path: str) -> str:
    """从 tar 文件名解析获取日期（LC09_L2SP_120036_20251226_... → 20251226）"""
    m = LST_TAR_RE.search(os.path.basename(tar_path))
    if not m:
        raise ValueError(f"无法从文件名解析获取日期：{tar_path}")
    return m.group(1)


def _extract_tar_if_needed(tar_path: str, extract_root: str):
    """解压 tar 到 extract_root/<tar名去.tar>/，已解压则复用。

    单代码路径：无条件 extractall 到该子目录，同时覆盖平铺 tar（成员直接落下）
    与带目录结构的 tar（落进子目录，os.walk 照样能找到）——
    比旧脚本猜测顶层目录的 if/else 更稳。
    返回 (场景目录, 是否复用)。
    """
    scene_dir = os.path.join(extract_root, os.path.basename(tar_path)[:-4])
    os.makedirs(scene_dir, exist_ok=True)

    for root, _, files in os.walk(scene_dir):
        if any(f.endswith("_ST_B10.TIF") for f in files):
            return scene_dir, True

    with tarfile.open(tar_path, "r:*") as tar:
        tar.extractall(path=scene_dir)
    return scene_dir, False


# ============================================================
# 工具 1：场景发现
# ============================================================
def list_available_lst_scenes(data_dir: str) -> dict:
    """
    扫描影像目录，从 Landsat C2 L2SP tar 文件名解析获取日期，按日期分组

    返回：
        {'dates': [按日期排序], 'scenes_by_date': {日期: [tar路径列表]},
         'total_scenes': 景数, 'unparsed': [解析失败的文件], 'hint': 使用提示}
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"找不到影像目录：{data_dir}")

    scenes_by_date = {}
    unparsed = []
    for name in sorted(os.listdir(data_dir)):
        if not name.endswith(".tar"):
            continue
        full = os.path.join(data_dir, name)
        if not os.path.isfile(full):
            continue
        m = LST_TAR_RE.search(name)
        if not m:
            unparsed.append(name)   # 单个坏文件不炸掉整个发现工具
            continue
        scenes_by_date.setdefault(m.group(1), []).append(full)

    if not scenes_by_date:
        raise FileNotFoundError(f"{data_dir} 下没有找到可解析日期的 Landsat *.tar 影像")

    return _json_safe({
        "dates": sorted(scenes_by_date),
        "scenes_by_date": scenes_by_date,
        "total_scenes": sum(len(v) for v in scenes_by_date.values()),
        "unparsed": unparsed,
        "hint": "把 scenes_by_date[日期] 的路径列表原样传给 lst_invert_date 的 scene_paths",
    })


# ============================================================
# 工具 2：LST 反演（旧脚本主流程的批量化）
# ============================================================
def lst_invert_date(scene_paths: list, output_dir: str, extract_dir: str = "") -> dict:
    """
    对某日期的一批 Landsat C2 L2SP tar 执行：解压 → ST_B10 线性换算（K→℃）
    → QA_PIXEL 启发式云掩膜，输出 Float32 地表温度产品 LST_<日期>.tif（nodata=NaN）。

    云掩膜口径与 6 月历史产品严格一致（qa<2 或 qa>22000 置 NaN）。
    单景约 2-4 分钟（首次含解压）；已存在的产品与解压目录自动跳过（幂等）。
    """
    if not scene_paths:
        raise ValueError("scene_paths 为空：请传入 list_available_lst_scenes 返回的路径列表")
    os.makedirs(output_dir, exist_ok=True)
    if not extract_dir:
        extract_dir = os.path.join(os.path.dirname(os.path.abspath(output_dir)), "extract")
    os.makedirs(extract_dir, exist_ok=True)

    results = []
    for tar_path in sorted(scene_paths):
        if not os.path.isfile(tar_path):
            results.append({"scene": os.path.basename(tar_path), "status": "failed",
                            "error": f"文件不存在：{tar_path}"})
            continue
        try:
            results.append(_invert_one_scene(tar_path, output_dir, extract_dir))
        except Exception as e:
            # per-scene 失败不炸整个批次（镜像 snap_preprocess_date 的模式）
            results.append({"scene": os.path.basename(tar_path), "status": "failed",
                            "error": str(e)})

    return _json_safe({
        "total": len(scene_paths),
        "success": sum(1 for r in results if r["status"] == "success"),
        "output_dir": output_dir,
        "extract_dir": extract_dir,
        "results": results,
    })


def _invert_one_scene(tar_path: str, output_dir: str, extract_dir: str) -> dict:
    scene_name = os.path.basename(tar_path)[:-4]   # 去 .tar
    date = _lst_tar_date(tar_path)
    out_path = os.path.join(output_dir, f"LST_{date}.tif")

    # 幂等短路：产品已存在直接返回（Agent 纠错重试代价极小）
    if os.path.isfile(out_path):
        return {"scene": scene_name, "date": date, "status": "exists",
                "output": out_path}

    scene_dir, reused = _extract_tar_if_needed(tar_path, extract_dir)

    # 找 ST_B10 与 QA_PIXEL
    b10_path = None
    qa_path = None
    for root, _, files in os.walk(scene_dir):
        for f in files:
            if f.endswith("_ST_B10.TIF"):
                b10_path = os.path.join(root, f)
            elif f.endswith("_QA_PIXEL.TIF"):
                qa_path = os.path.join(root, f)
    if not b10_path:
        raise FileNotFoundError(
            f"未找到 *_ST_B10.TIF（本工具仅适用 Collection 2 L2SP 产品）：{scene_dir}")

    # ---- 换算（逐字复刻旧脚本 calculate_lst）----
    ds = gdal.Open(b10_path)
    if ds is None:
        raise ValueError(f"无法打开文件: {b10_path}")
    b10 = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    lst = np.where(b10 != 0, LST_SCALE * b10 + LST_OFFSET_K - LST_C0, np.nan)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()

    # ---- QA 掩膜（逐字复刻旧脚本 apply_cloud_mask；重采样分支保留）----
    cloud_ratio = None
    mask_note = None
    if qa_path:
        qa_ds = gdal.Open(qa_path)
        if qa_ds is None:
            mask_note = "QA 文件无法打开，未应用云掩膜（旧脚本同语义）"
        elif (qa_ds.RasterXSize != ds.RasterXSize) or (qa_ds.RasterYSize != ds.RasterYSize):
            # 分辨率不匹配：MEM 数据集 + 最近邻重采样（本次实测不触发，分支保留）
            mem = gdal.GetDriverByName("MEM").Create(
                "", ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_UInt16)
            mem.SetGeoTransform(gt)
            mem.SetProjection(proj)
            gdal.ReprojectImage(qa_ds, mem, qa_ds.GetProjection(), proj,
                                gdalconst.GRA_NearestNeighbour)
            qa = mem.GetRasterBand(1).ReadAsArray()
            mem = None
            cloud_mask = (qa < QA_CLEAR_MIN) | (qa > QA_CLOUD_MAX)
            lst[cloud_mask] = np.nan
            cloud_ratio = float(cloud_mask.sum()) / cloud_mask.size
        else:
            qa = qa_ds.GetRasterBand(1).ReadAsArray()
            cloud_mask = (qa < QA_CLEAR_MIN) | (qa > QA_CLOUD_MAX)
            lst[cloud_mask] = np.nan
            cloud_ratio = float(cloud_mask.sum()) / cloud_mask.size
        qa_ds = None
    else:
        mask_note = "未找到 QA_PIXEL，未应用云掩膜（旧脚本同语义）"

    # ---- 有效值统计 ----
    valid = np.isfinite(lst)
    valid_ratio = float(valid.sum()) / lst.size
    lv = lst[valid]
    lst_min, lst_max, lst_mean = ((float(lv.min()), float(lv.max()), float(lv.mean()))
                                  if lv.size else (None, None, None))

    # ---- 写 GeoTIFF（Float32，nodata=NaN，与旧脚本同型）----
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(out_path, ds.RasterXSize, ds.RasterYSize, 1,
                           gdal.GDT_Float32, options=GDAL_CREATION)
    dst_ds.SetGeoTransform(gt)
    dst_ds.SetProjection(proj)
    band = dst_ds.GetRasterBand(1)
    band.WriteArray(lst)
    band.SetNoDataValue(float("nan"))
    band.FlushCache()
    dst_ds = None
    ds = None

    return {
        "scene": scene_name, "date": date, "status": "success",
        "reuse_extracted": reused, "output": out_path,
        "cloud_ratio": round(cloud_ratio, 4) if cloud_ratio is not None else None,
        "valid_ratio": round(valid_ratio, 4),
        "lst_min": lst_min, "lst_max": lst_max, "lst_mean": lst_mean,
        "width": int(b10.shape[1]), "height": int(b10.shape[0]), "crs": proj,
        "note": mask_note or "云掩膜为旧脚本启发式（QA<2 或 >22000），"
                            "与历史产品口径一致，可能残留薄云/漏检云影",
    }


# ============================================================
# 工具 3：按研究区裁剪（旧脚本缺失，本次补齐）
# ============================================================
def lst_clip_date_to_aoi(lst_tif: str, shp_path: str, output_dir: str) -> dict:
    """
    用研究区矢量裁剪整景 LST 产品（与 NDVI 链共用同一矢量）。
    背景置 -9999（calculate_lst_stats 兼容该 nodata），
    矢量与栅格坐标系不一致时 gdal.Warp 自动重投影。
    """
    if not os.path.isfile(lst_tif):
        raise FileNotFoundError(f"找不到 LST 产品：{lst_tif}")
    if not os.path.isfile(shp_path):
        raise FileNotFoundError(f"找不到裁剪矢量：{shp_path}")

    m = re.search(r"(\d{8})", os.path.basename(lst_tif))
    date = m.group(1) if m else "unknown"

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"LST_{date}_clip.tif")

    # 注意：GDAL 3.6 的 Python 绑定没有 config_options 上下文管理器，
    # 用 SetConfigOption + finally 恢复（与 pipeline_tools.clip_date_to_aoi 同写法）
    gdal.SetConfigOption("GDALWARP_IGNORE_BAD_CUTLINE", "YES")
    try:
        ds = gdal.Warp(out_path, lst_tif, options=gdal.WarpOptions(
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
            f"裁剪失败：{lst_tif}（请检查矢量与栅格的范围是否相交）")

    gt = ds.GetGeoTransform()
    w, h = ds.RasterXSize, ds.RasterYSize
    crs = ds.GetProjection()
    xmin, ymax = gt[0], gt[3]
    xmax, ymin = gt[0] + w * gt[1], gt[3] + h * gt[5]
    ds = None
    return _json_safe({
        "clip_path": out_path, "crs": crs, "width": w, "height": h,
        "transform": gt, "extent": [xmin, ymax, xmax, ymin],
        "nodata": NODATA_RAW,
    })


# ============================================================
# 工具 4：LST 统计（镜像 ndvi_tools.calculate_ndvi_stats）
# ============================================================
def calculate_lst_stats(lst_path: str, region_name: str = "") -> dict:
    """
    读取本地 LST 栅格（.tif，已完成反演与裁剪的产品），计算
    均值/最大值/最小值/标准差/有效像元占比，以及各温度等级的面积占比。

    温度分级：>=30 炎热；20-30 暖热；10-20 温和；0-10 寒冷；<0 低温。
    """
    if not os.path.exists(lst_path):
        raise FileNotFoundError(f"找不到LST文件：{lst_path}")

    with rasterio.open(lst_path) as src:
        data = src.read(1).astype(np.float32)

        # ---------- 有效像元筛选（兼容 -9999 与 NaN 两种 nodata） ----------
        nodata = src.nodata
        if nodata is not None:
            valid = data[data != nodata]
        else:
            valid = data[~np.isnan(data)]
        valid = valid[np.isfinite(valid)]

        # 过滤超出地表温度物理范围 [-100, 100] ℃ 的异常值
        valid = valid[(valid >= -100.0) & (valid <= 100.0)]

        total_pixels = data.size
        valid_pixels = valid.size
        valid_ratio = valid_pixels / total_pixels if total_pixels > 0 else 0.0

        # ---------- 统计指标 ----------
        stats = {
            'region_name': region_name,
            'mean': float(valid.mean()),
            'max': float(valid.max()),
            'min': float(valid.min()),
            'std': float(valid.std()),
            'unit': '℃',
            'valid_pixel_ratio': round(valid_ratio, 4),
            'valid_pixels': int(valid_pixels),
            'total_pixels': int(total_pixels),
            'path': lst_path,
        }

        # ---------- 温度等级面积占比 ----------
        # 区间划分：[30,∞) [20,30) [10,20) [0,10) (-∞,0)，互不重叠
        coverage = {}
        prev_threshold = None
        for threshold, name, description in LST_LEVELS:
            if prev_threshold is None:
                mask = valid >= threshold          # 最高档：>= 30
            elif threshold == float("-inf"):
                mask = valid < prev_threshold      # 最低档：< 0
            else:
                mask = (valid >= threshold) & (valid < prev_threshold)
            ratio = float(mask.sum()) / valid_pixels if valid_pixels > 0 else 0.0
            coverage[name] = {'ratio': round(ratio, 4), 'description': description}
            prev_threshold = threshold
        stats['coverage'] = coverage

        # ---------- 数据质量提示（与 NDVI 统计同款专业提示） ----------
        if valid_ratio < 0.6:
            stats['note'] = (
                "有效像元占比偏低。注意：若该栅格经过矢量裁剪，"
                "矩形范围中研究区边界外的背景像元会被标记为nodata，"
                "这是正常现象，不代表数据质量问题；统计结果仅代表研究区内部。"
            )
        else:
            stats['note'] = "有效像元占比正常，统计结果可靠。"

    return stats


# ============================================================
# 工具 5：两期 LST 对比（镜像 compare_ndvi_dates）
# ============================================================
def compare_lst_dates(date1_path: str, date2_path: str, output_dir: str,
                      threshold: float = 3.0) -> dict:
    """
    两期 LST 裁剪产品对比（date1=较早，date2=较晚，Δ = 晚 − 早，单位℃）：

    1. 把较早日期无条件重采样到较晚日期的精确网格（不假设两期网格一致）
    2. 逐像元求差并分类：Δ>=3 升温 / |Δ|<3 稳定 / Δ<=-3 降温 / 无效
    3. 输出差值 tif、分类 tif、变化分类图 PNG、面积占比图 PNG
    4. 返回各类像元数、面积（公顷）与占比统计

    面积换算：30m 像元 = 900 m² = 0.09 公顷
    """
    for p in (date1_path, date2_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"找不到 LST 结果：{p}")
    os.makedirs(output_dir, exist_ok=True)

    # ---------- 日期标签 ----------
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
    # 关键：不先把 -9999 换成 NaN，差分会把背景算成 Δ≈±9999 的虚假"升温/降温"
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
    diff_path = os.path.join(output_dir, f"LST_diff_{d1}_vs_{d2}.tif")
    class_path = os.path.join(output_dir, f"LST_class_{d1}_vs_{d2}.tif")

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
    for code, name in LST_CLASS_NAMES.items():
        pixels = int((cls == code).sum())
        area_ha = pixels * LST_PIXEL_HA
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

    # ---------- 第五步：出图（复用 pipeline_tools 的参数化 helper） ----------
    _setup_chinese_font()
    change_map_png = _plot_change_map(
        cls, gt, per_class, d1, d2, output_dir,
        class_names=LST_CLASS_NAMES, class_colors=LST_CLASS_COLORS,
        product="LST", file_prefix="lst_change_map")
    area_chart_png = _plot_area_chart(
        per_class, d1, d2, output_dir,
        class_names=LST_CLASS_NAMES, class_colors=LST_CLASS_COLORS,
        order=(0, 3, 2, 1), product="LST", file_prefix="lst_area_proportion")

    # 中间对齐产物用完即删，保持输出目录干净
    os.remove(aligned1)

    return _json_safe({
        "date1": d1, "date2": d2, "threshold": threshold, "unit": "℃",
        "diff_path": diff_path, "class_path": class_path,
        "change_map_png": change_map_png, "area_chart_png": area_chart_png,
        "classes": per_class, "delta_stats": delta_stats,
        "note": "面积按30m像元=0.09公顷换算；无效=任一日期无有效像元（云/影/背景）。"
                "局限：单景LST受两日瞬时天气差异影响，跨年对比若Δ空间均匀"
                "（std小）通常由天气主导而非地表变化，剥离天气需多景平均"
                "或天气匹配（本工具不提供）",
    })


# ============================================================
# 工具 6：LST-NDVI 联合相关分析
# ============================================================
def analyze_lst_ndvi(lst_path: str, ndvi_path: str, output_dir: str) -> dict:
    """
    LST-NDVI 联合相关分析：

    1. NDVI 10m 产品用平均聚合（average）重采样到 LST 30m 的精确网格
       （3×3 窗口，nodata 不参与均值——降采样的科学选择；
        把 LST 升采样到 10m 是假精度，不做）
    2. 统一无效像元（-9999/NaN）与物理范围（NDVI∈[-1,1]，LST∈[-100,100]）
    3. 计算有效像元对的 Pearson 相关系数与线性回归（x=NDVI，y=LST℃）
    4. 输出密度散点图 PNG（hexbin + 回归线）与 30m 对齐 NDVI 副产品

    注意：相关性不构成因果结论；两期影像存在日期差时 r 只能作近似同期关系解读。
    """
    for p in (lst_path, ndvi_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"找不到输入影像：{p}")
    os.makedirs(output_dir, exist_ok=True)

    m_l = re.search(r"(\d{8})", os.path.basename(lst_path))
    m_n = re.search(r"(\d{8})", os.path.basename(ndvi_path))
    ldate = m_l.group(1) if m_l else "lst"
    ndate = m_n.group(1) if m_n else "ndvi"

    # ---------- 第一步：NDVI 重采样到 LST 网格（模板 = LST） ----------
    with rasterio.open(lst_path) as src:
        gt, crs, w, h = src.transform, src.crs, src.width, src.height
    xmin, ymax = gt[2], gt[5]
    xmax, ymin = gt[2] + w * gt[0], gt[5] + h * gt[4]

    # 对齐 NDVI 保留为正式副产品（30m），便于复检与后续分析
    aligned_ndvi = os.path.join(output_dir, f"NDVI_{ndate}_30m.tif")
    warp_ds = gdal.Warp(aligned_ndvi, ndvi_path, options=gdal.WarpOptions(
        format="GTiff", dstSRS=crs.to_wkt(),
        outputBounds=(xmin, ymin, xmax, ymax), width=w, height=h,
        dstNodata=float("nan"), resampleAlg=gdal.GRA_Average,
        creationOptions=GDAL_CREATION))
    if warp_ds is None:
        raise RuntimeError("NDVI 重采样失败（两期影像范围可能完全不相交）")
    warp_ds = None

    # ---------- 第二步：nodata 统一 + 物理范围过滤 ----------
    with rasterio.open(lst_path) as src:
        l_arr = src.read(1).astype(np.float32)
        nl = src.nodata
    with rasterio.open(aligned_ndvi) as src:
        v_arr = src.read(1).astype(np.float32)
        nv = src.nodata
    if nl is not None:
        l_arr[l_arr == nl] = np.nan
    if nv is not None:
        v_arr[v_arr == nv] = np.nan
    l_arr = np.where((l_arr >= -100.0) & (l_arr <= 100.0), l_arr, np.nan)
    v_arr = np.where((v_arr >= -1.0) & (v_arr <= 1.0), v_arr, np.nan)

    valid = np.isfinite(l_arr) & np.isfinite(v_arr)
    n = int(valid.sum())
    if n < 2:
        raise RuntimeError(f"有效像元对不足（{n}），无法计算相关")
    lv, vv = l_arr[valid], v_arr[valid]

    # ---------- 第三步：Pearson r + 线性回归（x=NDVI，y=LST） ----------
    r = float(np.corrcoef(vv, lv)[0, 1])
    slope, intercept = np.polyfit(vv, lv, 1)
    slope, intercept = float(slope), float(intercept)
    lst_mean = float(lv.mean())
    ndvi_mean = float(vv.mean())

    # ---------- 第四步：hexbin 密度散点 PNG（有效对百万级，必须聚合） ----------
    _setup_chinese_font()
    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    hb = ax.hexbin(vv, lv, gridsize=120, mincnt=1, cmap="Blues")
    fig.colorbar(hb, ax=ax, label="像元对数")

    xspan = np.array([float(vv.min()), float(vv.max())])
    ax.plot(xspan, slope * xspan + intercept, color="#e34948", lw=2)
    ax.text(0.03, 0.95,
            f"r = {r:.3f}（n = {n:,}）\ny = {slope:.2f}x + {intercept:.1f}",
            transform=ax.transAxes, va="top", fontsize=11,
            bbox=dict(facecolor="#fcfcfb", edgecolor="#c3c2b7", alpha=0.9))
    ax.set_xlabel("NDVI")
    ax.set_ylabel("LST (℃)")
    ax.set_title(f"LST-NDVI 相关散点（{ldate} LST vs {ndate} NDVI）", fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")

    scatter_png = os.path.join(output_dir, f"lst_ndvi_scatter_{ldate}_{ndate}.png")
    fig.savefig(scatter_png, bbox_inches="tight")
    plt.close(fig)

    return _json_safe({
        "lst_path": lst_path, "ndvi_path": ndvi_path,
        "lst_date": ldate, "ndvi_date": ndate,
        "aligned_ndvi_path": aligned_ndvi,
        "n": n, "pearson_r": round(r, 4),
        "slope": round(slope, 4), "intercept": round(intercept, 4),
        "lst_mean": round(lst_mean, 2), "ndvi_mean": round(ndvi_mean, 4),
        "scatter_png": scatter_png,
        "interpretation_hint": (
            "r<0 的常见解释是植被蒸散降温效应（NDVI 越高地表越凉）；"
            "本工具不推断因果，且两期影像存在日期差时 r 只能作为近似同期关系解读"
        ),
    })
