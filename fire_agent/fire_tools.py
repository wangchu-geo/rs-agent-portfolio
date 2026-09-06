"""
fire_tools.py —— Himawari-9 森林火点反演工具链（5 工具）

与旧脚本 E:/YYR/fire/脚本/fire.py 的对应关系：
    本工具                          旧函数（fire.py）
    list_available_himawari_scenes   无（新发现工具，替代 main 中手工拼文件名）
    himawari_extract_bands           process_himawari_nc + process_cltype_nc
    apply_fire_cloud_mask            process_cloud_masking + resample_cltype
                                     + process_band + clean_small_objects
    generate_fire_mask               generate_fire_mask
    calculate_fire_stats             无（新统计工具）

量纲修正（2026-08-31 实证）：
    旧脚本 convert_brightness_temperature = raw*0.01+273.15 套在 xarray 已解码的
    开尔文值上（NC 中 tbb_XX 解码后即为 K：tbb_07 实测 241.7~363.8K、tbb_14
    181.1~320.0K，attrs units='K'），把 240~364K 压缩到 276.1~276.3K
    （旧品 H09_B07.tif 实测值域）。本工具直接取解码 K 值，不复刻二次换算。
    火点四条件均为均值/标准差/亮温差形式，对仿射变换不变，故量纲修正不改变
    火点检测结果（verify_fire_tools.py V-F4 实证）。

云掩膜口径与旧脚本严格一致：CLTYPE>0 或 NaN 视为云（0=晴空，1-10 各类云，
255 已转 NaN），B07/B14 掩膜后再去除 <12 像元的离散小斑块。
保真要求：CLTYPE 重采样必须 bilinear（分类数据插值是旧口径，不得改 nearest）。

运行环境：Python 3.9（numpy 2.0.2 / rasterio 1.4.3(GDAL 3.6.2) /
xarray 2024.7.0）。注意 ds['start_time'] 等时间变量解码有 int64 cast 错误，
一律 decode_times=False，时次从文件名正则解析。
"""
import os
import re
import sys

import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr
from rasterio.mask import mask as rio_mask
from rasterio.transform import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject
from scipy.ndimage import label as scipy_label
from shapely.ops import unary_union

# 复用 NDVI 链的公共 helper（同一仓库兄弟目录；文件即模块，无 __init__.py）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "ndvi_agent"))
from pipeline_tools import _json_safe  # noqa: E402

# ------------------------------------------------------------
# 常量与正则
# ------------------------------------------------------------
# 文件名：NC_H09_<日期8位>_<UTC时次4位>_<产品>_FLDK.<网格>_<网格>.nc
R21_RE = re.compile(r"NC_H09_(\d{8})_(\d{4})_R21_FLDK\.06001_06001\.nc$")
CLP_RE = re.compile(r"NC_H09_(\d{8})_(\d{4})_L2CLP010_FLDK\.02401_02401\.nc$")

VISIBLE_BANDS = range(1, 7)   # 通道1-6：albedo_0X（请求时除以 cos(SOZ)）
IR_BANDS = range(7, 17)       # 通道7-16：tbb_XX（解码后即为 K）
CLOUD_MIN_SIZE = 12           # 云掩膜后去除 <12 像元小斑块（旧脚本默认）
LANDUSE_FOREST = (20, 25)     # 林地类条件 landuse>=20 且 <25（实测类值 21-24）

# 单时次产物精确文件名集合（禁 glob 通配——旧 result/ 里有 .enp 残留文件）
EXPECTED_FILES = {
    "H09_CLTYPE.tif",
    "H09_B07.tif", "H09_B14.tif",
    "H09_B07_cloud_masked.tif", "H09_B14_cloud_masked.tif",
    "H09_B07_cloud_masked_cleaned.tif", "H09_B14_cloud_masked_cleaned.tif",
    "H09_landuse_reprj.tif",
    "H09_fire_point.tif",
}


class KelvinGuardrailError(RuntimeError):
    """tbb 解码值不满足物理 K 值域/离散度护栏（疑似解码行为变化或二次换算复演）。"""


# ------------------------------------------------------------
# 私有 helper
# ------------------------------------------------------------
def _slot_from_filename(path):
    """从 NC 文件名解析时次 'YYYYMMDD_HHMM'；解析失败返回 None。"""
    name = os.path.basename(path)
    for rx in (R21_RE, CLP_RE):
        m = rx.match(name)
        if m:
            return f"{m.group(1)}_{m.group(2)}"
    return None


def _grid_transform(ds):
    """
    等经纬度网格仿射变换（逐字复刻 fire.py 148-150/181-183 行）。
    注意：分辨率用 /len(lon) 而非 /(len(lon)-1)——这是与旧品网格逐像元对齐的
    前提，不得"顺手修正"。
    """
    lon = ds["longitude"].values
    lat = ds["latitude"].values
    res_x = (lon.max() - lon.min()) / len(lon)
    res_y = (lat.max() - lat.min()) / len(lat)
    return Affine(res_x, 0, lon.min(), 0, -res_y, lat.max())


def _rio_profile(width, height, dtype, transform, nodata):
    """rasterio 写参（压缩选项与 pipeline_tools.GDAL_CREATION 等价）。"""
    return {
        "driver": "GTiff", "width": width, "height": height, "count": 1,
        "dtype": dtype, "crs": "EPSG:4326", "transform": transform,
        "nodata": nodata, "compress": "deflate", "tiled": True,
        "predictor": 2, "bigtiff": "IF_SAFER",
    }


def _write_tif(path, array, transform):
    """写全盘 float32 GeoTIFF（nodata=NaN，与旧 save_tiff 同型）。"""
    h, w = array.shape
    profile = _rio_profile(w, h, "float32", transform, np.nan)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(np.float32), 1)


def _mask_with_china2(tif_path, shp_path):
    """
    用中国东部及南部矢量掩膜并原地覆盖（对应 fire.py mask_tiff_with_shapefile）。
    crop=True、nodata=NaN、all_touched=False，与旧口径一致。
    """
    gdf = gpd.read_file(shp_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", inplace=False)
    with rasterio.open(tif_path) as src:
        vector_proj = gdf.to_crs(src.crs)
        geometry_union = [unary_union(vector_proj.geometry)]
        out_image, out_transform = rio_mask(
            src, geometry_union, crop=True, nodata=np.nan,
            filled=True, all_touched=False)
        out_meta = src.meta.copy()
    out_meta.update({
        "height": out_image.shape[1], "width": out_image.shape[2],
        "transform": out_transform, "nodata": np.nan,
    })
    with rasterio.open(tif_path, "w", **out_meta) as dest:
        dest.write(out_image)


def _file_value_stats(path):
    """从成品 tif 反推值域与有效占比（裁剪后中国区域）。"""
    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float32)
    fin = a[np.isfinite(a)]
    ratio = float(fin.size) / a.size
    if fin.size == 0:
        return None, None, ratio
    return round(float(fin.min()), 3), round(float(fin.max()), 3), round(ratio, 4)


def _check_kelvin_guardrail(band, data):
    """
    旧量纲 bug 的防复演护栏（2026-08-31 全盘实测校准）：
    物理 K 值动态范围大（tbb_07 std=15.4K、tbb_14 std=22.7K，全盘有限值占比
    100%，无填充值）；而旧脚本 ×0.01+273.15 会把动态范围压缩到 ~0.05K、
    均值 276K 附近；xarray 若返回原始计数（如 27500）则远超 500 上限。
    """
    arr = np.asarray(data, dtype=np.float64)
    fin = arr[np.isfinite(arr)]
    if fin.size < 1000:
        raise KelvinGuardrailError(
            f"tbb_{band:02d} 有效值过少（{fin.size}），无法护栏校验")
    vmin, vmax = float(fin.min()), float(fin.max())
    vstd = float(fin.std())
    if not (120.0 < vmin and vmax < 500.0):
        raise KelvinGuardrailError(
            f"tbb_{band:02d} 解码值域 {vmin:.1f}~{vmax:.1f} 超出物理 K 护栏 "
            "(120,500)，疑似 xarray 解码行为变化（如返回原始计数），停止产出")
    if vstd < 1.0:
        raise KelvinGuardrailError(
            f"tbb_{band:02d} 动态范围过窄（std={vstd:.3f}K），疑似复演了旧脚本 "
            "×0.01+273.15 二次换算压缩，停止产出")


def _resample_cltype_to_ref(cltype_path, ref_transform, ref_shape, ref_crs):
    """
    CLTYPE（5km 网格）bilinear 重采样到 B07（2km）参考网格（fire.py resample_cltype）。
    保真要求：分类数据也必须 bilinear（旧口径，不得改 nearest）——云条件
    (cl>0)|isnan 对权重 ulp 级波动免疫，与旧品逐像元一致依赖此特性。
    """
    with rasterio.open(cltype_path) as src:
        src_data = src.read(1).astype(np.float32)
        src_nodata = src.nodata
    if src_nodata is not None:  # NaN==NaN 恒 False，逐字保留旧口径
        src_data = np.where(src_data == src_nodata, np.nan, src_data)
    data_resampled = np.full(ref_shape, np.nan, dtype=np.float32)
    reproject(
        source=src_data, destination=data_resampled,
        src_transform=src.transform, src_crs=src.crs,
        dst_transform=ref_transform, dst_crs=ref_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan, dst_nodata=np.nan)
    return data_resampled


def _clean_small_objects(input_path, output_path, min_size=CLOUD_MIN_SIZE):
    """
    去除 <min_size 像元的离散连通斑块（对应 fire.py clean_small_objects）。
    对 NaN/nodata 以外且值 !=0 的区域做 4 连通标记，小斑块置 nodata。
    """
    with rasterio.open(input_path) as src:
        data = src.read(1)
        profile = src.profile
        nodata = src.nodata
    if nodata is None:
        nodata = np.nan
        profile.update(nodata=nodata, dtype="float32")
        data = data.astype("float32")
    nan_mask = np.isnan(data) if np.isnan(nodata) else (data == nodata)
    valid_mask = ~nan_mask & (data != 0)
    labeled_array, num_features = scipy_label(valid_mask)
    cleaned_data = data.copy()
    for i in range(1, num_features + 1):
        region = (labeled_array == i)
        if np.sum(region) < min_size:
            cleaned_data[region] = nodata
    cleaned_data[nan_mask] = nodata
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(cleaned_data, 1)


def _landuse_to_scene_grid(landuse_path, scene_dir, crs, transform,
                           bounds, height, width):
    """
    土地利用数据两级重投影到场景网格（逐字复刻 fire.py 458-533 行）：
    ① file-based 重投影到场景经纬度范围（nearest）→ 持久化中间产物
      H09_landuse_reprj.tif（旧脚本用 tempfile 且异常时泄漏，此处改为场景内
      持久中间产物，语义一致、可调试、天然幂等）
    ② 矩形 bbox mask(crop=True) 精确裁剪；若形状与场景不符，二次数组级
      reproject（nearest）到 (height,width)——注意旧代码此处 src_transform
      用裁剪前文件 transform，逐字保留
    """
    reprj_path = os.path.join(scene_dir, "H09_landuse_reprj.tif")
    if not os.path.isfile(reprj_path):
        with rasterio.open(landuse_path) as landuse_src:
            dst_transform, dst_width, dst_height = calculate_default_transform(
                landuse_src.crs, crs, landuse_src.width, landuse_src.height,
                *bounds)
            kwargs = {
                "driver": "GTiff", "height": dst_height, "width": dst_width,
                "count": 1, "dtype": landuse_src.dtypes[0], "crs": crs,
                "transform": dst_transform, "nodata": landuse_src.nodata,
            }
        with rasterio.open(reprj_path, "w", **kwargs) as dst:
            with rasterio.open(landuse_path) as landuse_src:
                reproject(
                    source=rasterio.band(landuse_src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=landuse_src.transform,
                    src_crs=landuse_src.crs,
                    dst_transform=dst_transform, dst_crs=crs,
                    resampling=Resampling.nearest)
    with rasterio.open(reprj_path) as reprojected_landuse:
        landuse_data, _ = rio_mask(
            reprojected_landuse,
            [{"type": "Polygon", "coordinates": [[
                (bounds.left, bounds.bottom), (bounds.left, bounds.top),
                (bounds.right, bounds.top), (bounds.right, bounds.bottom),
                (bounds.left, bounds.bottom)]]}],
            crop=True)
        if landuse_data.shape[1:] != (height, width):
            resampled_landuse = np.zeros((1, height, width),
                                         dtype=landuse_data.dtype)
            reproject(landuse_data, resampled_landuse,
                      src_transform=reprojected_landuse.transform,
                      dst_transform=transform, src_crs=crs, dst_crs=crs,
                      resampling=Resampling.nearest)
            landuse_arr = resampled_landuse[0]
        else:
            landuse_arr = landuse_data[0]
    return landuse_arr


def _count_fire_pixels(path):
    with rasterio.open(path) as src:
        return int((src.read(1) == 1).sum())


# ------------------------------------------------------------
# 工具 1：场景发现
# ------------------------------------------------------------
def list_available_himawari_scenes(data_dir):
    """
    扫描目录，从 Himawari-9 NC 文件名解析时次（YYYYMMDD_HHMM），按槽位配对
    R21 辐射产品与 L2CLP010 云产品。这是火点链的第一步。
    只识别 NC_H09_*.nc；解析失败的进 unparsed，不抛异常。
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"目录不存在：{data_dir}")
    slots = {}
    unparsed = []
    for name in sorted(os.listdir(data_dir)):
        if not name.endswith(".nc"):
            continue
        path = os.path.join(data_dir, name)
        m = R21_RE.match(name)
        if m:
            slot = f"{m.group(1)}_{m.group(2)}"
            slots.setdefault(slot, {"r21": None, "clp": None})["r21"] = path
            continue
        m = CLP_RE.match(name)
        if m:
            slot = f"{m.group(1)}_{m.group(2)}"
            slots.setdefault(slot, {"r21": None, "clp": None})["clp"] = path
            continue
        unparsed.append(name)
    missing = [s for s, p in slots.items() if p["r21"] is None or p["clp"] is None]
    hint = ("把 scenes_by_slot[时次] 的 r21/clp 路径传给 himawari_extract_bands。"
            + (f" 注意：时次 {missing} 缺少 R21 或 L2CLP 配对，无法提取。"
               if missing else ""))
    return {
        "slots": sorted(slots),
        "scenes_by_slot": {s: slots[s] for s in sorted(slots)},
        "total_slots": len(slots),
        "unparsed": unparsed,
        "hint": hint,
    }


# ------------------------------------------------------------
# 工具 2：波段与云类型提取
# ------------------------------------------------------------
def himawari_extract_bands(r21_path, clp_path, output_dir, shp_path,
                           bands=(7, 14)):
    """
    提取 R21 波段与 CLTYPE 云类型，china2 矢量掩膜后写入场景目录
    （output_dir/NC_H09_<时次>/）。幂等：产物已存在 → status="exists" 跳过。
    量纲：tbb 直接取 xarray 解码的 K 值（旧脚本 ×0.01+273.15 二次换算不复刻），
    并过物理 K 护栏（防复演）。
    bands：请求的通道号（1-6 可见光做 cos(SOZ) 归一化；7-16 热红外取 K）。
    """
    for p in (r21_path, clp_path, shp_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"文件不存在：{p}")
    slot = _slot_from_filename(r21_path)
    if not slot:
        raise ValueError(f"无法从文件名解析时次：{r21_path}")
    scene_dir = os.path.join(output_dir, f"NC_H09_{slot}")
    os.makedirs(scene_dir, exist_ok=True)

    band_results = []
    with xr.open_dataset(r21_path, decode_times=False) as ds:
        transform = _grid_transform(ds)
        # SOZ 仅可见光通道需要；全盘 float64 数组 ~288MB，按需加载
        cos_soz = None
        if any(b in VISIBLE_BANDS for b in bands):
            soz = ds["SOZ"].values.squeeze() if "SOZ" in ds else None
            if soz is not None:
                with np.errstate(invalid="ignore", divide="ignore"):
                    cos_soz = np.cos(np.deg2rad(soz))
                    cos_soz = np.where(cos_soz == 0, np.nan, cos_soz)

        for band in bands:
            out_path = os.path.join(scene_dir, f"H09_B{band:02d}.tif")
            if os.path.isfile(out_path):
                band_results.append({"band": band, "status": "exists",
                                     "output": out_path})
                continue
            try:
                var_name = (f"albedo_0{band}" if band in VISIBLE_BANDS
                            else f"tbb_{band:02d}")
                if var_name not in ds:
                    band_results.append({"band": band, "status": "error",
                                         "error": f"变量 {var_name} 不存在"})
                    continue
                data = ds[var_name].values.squeeze()
                if band in VISIBLE_BANDS:
                    if cos_soz is not None and data.shape == cos_soz.shape:
                        data = data / cos_soz
                    else:
                        band_results.append({"band": band, "status": "error",
                                             "error": "cos(SOZ) 缺失或尺寸不一致"})
                        continue
                else:
                    _check_kelvin_guardrail(band, data)
                _write_tif(out_path, data, transform)
                _mask_with_china2(out_path, shp_path)
                vmin, vmax, vratio = _file_value_stats(out_path)
                band_results.append({
                    "band": band, "var": var_name, "status": "success",
                    "output": out_path, "value_min": vmin,
                    "value_max": vmax, "valid_ratio": vratio,
                })
            except KelvinGuardrailError:
                raise  # 系统性解码问题：响亮失败，不进逐波段容错
            except Exception as e:  # 单波段失败不炸批次（镜像 lst_invert_date 模式）
                band_results.append({"band": band, "status": "error",
                                     "error": str(e)})

    # ---- CLTYPE 云类型（5km 网格，独立 transform）----
    cltype_path = os.path.join(scene_dir, "H09_CLTYPE.tif")
    if os.path.isfile(cltype_path):
        cltype_result = {"status": "exists", "output": cltype_path}
    else:
        try:
            with xr.open_dataset(clp_path, decode_times=False) as ds:
                if "CLTYPE" not in ds:
                    raise ValueError("云产品中不存在 CLTYPE 变量")
                cltype = ds["CLTYPE"].values.squeeze().astype(np.float32)
                transform = _grid_transform(ds)
            cltype[cltype == 255] = np.nan  # 255=Fill → NaN（旧口径）
            _write_tif(cltype_path, cltype, transform)
            _mask_with_china2(cltype_path, shp_path)
            vmin, vmax, vratio = _file_value_stats(cltype_path)
            cltype_result = {"status": "success", "output": cltype_path,
                             "value_min": vmin, "value_max": vmax,
                             "valid_ratio": vratio}
        except Exception as e:
            cltype_result = {"status": "error", "error": str(e)}

    n_ok = sum(1 for r in band_results if r["status"] in ("success", "exists"))
    return _json_safe({
        "slot": slot, "scene_dir": scene_dir,
        "total_bands": len(band_results), "success": n_ok,
        "band_results": band_results, "cltype": cltype_result,
        "note": ("tbb 直接取 xarray 解码 K 值（旧脚本 ×0.01+273.15 二次换算 bug "
                 "不复刻）；网格变换保留旧脚本 /len 怪癖以保证与历史产品逐像元对齐"),
    })


# ------------------------------------------------------------
# 工具 3：云掩膜 + 小斑块清理
# ------------------------------------------------------------
def apply_fire_cloud_mask(scene_dir):
    """
    基于 CLTYPE 云类型对 B07/B14 做云掩膜 + 小斑块清理（对应 fire.py
    process_cloud_masking）。云条件：CLTYPE>0 或 NaN → 云。
    CLTYPE 先 bilinear 重采样到 B07 网格（保真要求，勿改 nearest）。
    幂等：4 个产物（masked×2 + cleaned×2）齐全 → 全部 exists 跳过。
    """
    cltype_path = os.path.join(scene_dir, "H09_CLTYPE.tif")
    band_files = {
        "B07": (os.path.join(scene_dir, "H09_B07.tif"),
                os.path.join(scene_dir, "H09_B07_cloud_masked.tif"),
                os.path.join(scene_dir, "H09_B07_cloud_masked_cleaned.tif")),
        "B14": (os.path.join(scene_dir, "H09_B14.tif"),
                os.path.join(scene_dir, "H09_B14_cloud_masked.tif"),
                os.path.join(scene_dir, "H09_B14_cloud_masked_cleaned.tif")),
    }
    # 只检查 3 个输入（masked/cleaned 是输出，首跑时尚不存在）
    for p in (cltype_path, band_files["B07"][0], band_files["B14"][0]):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"缺少输入文件：{p}（先运行 himawari_extract_bands）")
    all_exist = all(os.path.isfile(t[1]) and os.path.isfile(t[2])
                    for t in band_files.values())
    if all_exist:
        return _json_safe({
            "scene_dir": scene_dir, "status": "exists",
            "bands": {tag: {"status": "exists", "masked": t[1], "cleaned": t[2]}
                      for tag, t in band_files.items()},
            "note": "云掩膜产物已存在（幂等跳过）",
        })

    with rasterio.open(band_files["B07"][0]) as ref_src:
        ref_profile = ref_src.profile
        ref_transform = ref_src.transform
        ref_shape = ref_src.shape
        ref_crs = ref_src.crs
    resampled_cltype = _resample_cltype_to_ref(cltype_path, ref_transform,
                                               ref_shape, ref_crs)
    cloud_condition = (resampled_cltype > 0) | np.isnan(resampled_cltype)
    cloud_pixels = int(cloud_condition.sum())
    cloud_ratio = round(float(cloud_pixels) / cloud_condition.size, 4)

    band_info = {}
    for tag, (band_path, masked_path, final_path) in band_files.items():
        if os.path.isfile(masked_path) and os.path.isfile(final_path):
            band_info[tag] = {"status": "exists", "masked": masked_path,
                              "cleaned": final_path}
            continue
        try:
            with rasterio.open(band_path) as src:
                band_data = src.read(1).astype(np.float32)
                orig_nodata = src.nodata
            masked_data = np.where(cloud_condition, np.nan, band_data)
            if orig_nodata is not None:  # NaN==NaN 恒 False，逐字保留旧口径（fire.py 389-391）
                orig_nodata_mask = (band_data == orig_nodata)
                masked_data = np.where(orig_nodata_mask, orig_nodata, masked_data)
            output_profile = ref_profile.copy()
            output_profile.update({
                "driver": "GTiff", "dtype": "float32",
                "nodata": orig_nodata if orig_nodata is not None else np.nan,
            })
            with rasterio.open(masked_path, "w", **output_profile) as dst:
                dst.write(masked_data.astype(output_profile["dtype"]), 1)
            _clean_small_objects(masked_path, final_path)
            _, _, valid_ratio = _file_value_stats(final_path)
            band_info[tag] = {"status": "success", "masked": masked_path,
                              "cleaned": final_path,
                              "valid_ratio_after": valid_ratio}
        except Exception as e:
            band_info[tag] = {"status": "error", "error": str(e)}

    return _json_safe({
        "scene_dir": scene_dir, "status": "success",
        "cloud_mask": {"cloud_pixels": cloud_pixels, "cloud_ratio": cloud_ratio},
        "bands": band_info,
        "note": "云条件 CLTYPE>0 或 NaN（0=晴空）；bilinear 重采样为旧口径保真要求；"
                "去除 <12 像元离散小斑块",
    })


# ------------------------------------------------------------
# 工具 4：上下文火点识别
# ------------------------------------------------------------
def generate_fire_mask(scene_dir, landuse_path):
    """
    上下文火点识别（对应 fire.py generate_fire_mask）。四条件（显式括号 =
    旧代码 & 优先于 | 的语义，fire.py 594 行）：
        c1: b07-b14 > diff_mean
        c2: b07 > b07_mean + 2.8*diff_std
        c3: b07-b14 > diff_mean + 2.5*diff_std
        c4: b14 > b14_mean + 2*b14_std
    火点 = ((c1&c2)|c3|c4) & 林地(landuse 20-25) & valid_mask
    输出 uint8 0/1（nodata=0）。幂等：H09_fire_point.tif 存在 → exists。
    """
    if not os.path.isfile(landuse_path):
        raise FileNotFoundError(f"土地利用文件不存在：{landuse_path}")
    b07_path = os.path.join(scene_dir, "H09_B07_cloud_masked_cleaned.tif")
    b14_path = os.path.join(scene_dir, "H09_B14_cloud_masked_cleaned.tif")
    fire_path = os.path.join(scene_dir, "H09_fire_point.tif")
    for p in (b07_path, b14_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"缺少云掩膜产物：{p}（先运行 apply_fire_cloud_mask）")
    if os.path.isfile(fire_path):
        return _json_safe({
            "scene_dir": scene_dir, "fire_tif": fire_path, "status": "exists",
            "fire_pixels": _count_fire_pixels(fire_path),
            "note": "火点产品已存在（幂等跳过）；conditions/stats 需重跑才可得",
        })

    with rasterio.open(b07_path) as src:
        b07 = src.read(1).astype(np.float32)
        meta = src.meta.copy()
        transform = src.transform
        crs = src.crs
        b07_bounds = src.bounds
        height, width = src.height, src.width
        nodata = meta["nodata"]
    with rasterio.open(b14_path) as src:
        b14 = src.read(1).astype(np.float32)

    landuse_arr = _landuse_to_scene_grid(landuse_path, scene_dir, crs,
                                         transform, b07_bounds, height, width)

    # ---- 统计与四条件（逐字复刻 fire.py 539-594）----
    # 注意：nodata=NaN 时 (b07 != nodata) 恒真（IEEE NaN≠NaN）——旧脚本原样保留，
    # NaN 像元靠后续比较运算自然落 0，此表达式为兼容性空操作（fire.py 539）
    valid_mask = (b07 != nodata) & (b14 != nodata)
    b07_clean = np.where(valid_mask, b07, np.nan)
    b14_clean = np.where(valid_mask, b14, np.nan)

    b07_mean = np.nanmean(b07_clean)
    b07_std = np.nanstd(b07_clean)
    b14_mean = np.nanmean(b14_clean)
    b14_std = np.nanstd(b14_clean)
    b_diff = b07_clean - b14_clean
    diff_mean = np.nanmean(b_diff)
    diff_std = np.nanstd(b_diff)

    c1 = b_diff > diff_mean
    c2 = b07_clean > (b07_mean + 2.8 * diff_std)
    c3 = b_diff > (diff_mean + 2.5 * diff_std)
    c4 = b14_clean > (b14_mean + 2 * b14_std)
    landuse_condition = (landuse_arr >= LANDUSE_FOREST[0]) & (landuse_arr < LANDUSE_FOREST[1])
    fire = ((c1 & c2) | c3 | c4) & landuse_condition & valid_mask
    fire_mask = np.where(fire, 1, 0).astype(np.uint8)
    fire_mask = np.where(valid_mask, fire_mask, 0)  # 旧口径（恒真下为空操作）

    meta.update({"driver": "GTiff", "dtype": "uint8", "count": 1,
                 "nodata": 0, "transform": transform})
    with rasterio.open(fire_path, "w", **meta) as dst:
        dst.write(fire_mask, 1)

    fire_pixels = int(fire_mask.sum())
    return _json_safe({
        "scene_dir": scene_dir, "fire_tif": fire_path, "status": "success",
        "fire_pixels": fire_pixels,
        "fire_ratio": round(float(fire_pixels) / fire_mask.size, 6),
        "conditions": {
            "c1": int(c1.sum()), "c2": int(c2.sum()), "c3": int(c3.sum()),
            "c4": int(c4.sum()), "landuse": int(landuse_condition.sum()),
            "valid": int(valid_mask.sum()),
        },
        "stats": {
            "b07_mean": round(float(b07_mean), 3),
            "b07_std": round(float(b07_std), 3),
            "b14_mean": round(float(b14_mean), 3),
            "b14_std": round(float(b14_std), 3),
            "diff_mean": round(float(diff_mean), 3),
            "diff_std": round(float(diff_std), 3),
        },
        "note": ("四条件仿射不变：量纲修正不改变检测结果（新旧掩膜差异仅来自 "
                 "float32 统计的阈值边界效应，见 verify_fire_tools.py V-F4）。"
                 "阈值口径为人工对照 NASA 火点监测网站调参所得（旧文档所称遗传算法"
                 "未在代码中实现），精度未经独立验证，火点计数存在误检/漏检可能"),
    })


# ------------------------------------------------------------
# 工具 5：火点统计
# ------------------------------------------------------------
def calculate_fire_stats(fire_tif, region_name=""):
    """
    火点统计（新工具，对齐 calculate_lst_stats 风格）：火点像素数/占比、
    逐行纬度 cos 精确面积、bbox、连通簇数与 top-5 簇面积。
    像元面积 = (|gt.a|·111.32km·cos(lat_row))·(|gt.e|·111.32km)，按逐行纬度
    计算（裁剪区纬度跨度大，均值纬度近似误差可达 ~5%）。
    """
    if not os.path.isfile(fire_tif):
        raise FileNotFoundError(f"火点产品不存在：{fire_tif}")
    with rasterio.open(fire_tif) as src:
        mask_arr = src.read(1)
        gt = src.transform
    fire = (mask_arr == 1)
    total = int(mask_arr.size)
    fire_pixels = int(fire.sum())
    fire_ratio = fire_pixels / total

    rows = np.arange(mask_arr.shape[0])
    lat_row = gt.f + (rows + 0.5) * gt.e  # gt.e < 0（北半球）
    pixel_area_km2 = ((abs(gt.a) * 111.32 * np.cos(np.deg2rad(lat_row)))
                      * (abs(gt.e) * 111.32))
    per_pixel_area = np.broadcast_to(pixel_area_km2[:, None].astype(np.float32),
                                     mask_arr.shape)
    fire_area_km2 = float(per_pixel_area[fire].sum()) if fire_pixels else 0.0
    mean_area = fire_area_km2 / fire_pixels if fire_pixels else None

    bbox = None
    center = None
    if fire_pixels:
        rs, cs = np.nonzero(fire)
        rmin, rmax = int(rs.min()), int(rs.max())
        cmin, cmax = int(cs.min()), int(cs.max())
        lon_min = float(gt.c + cmin * gt.a)
        lon_max = float(gt.c + (cmax + 1) * gt.a)
        lat_max = float(gt.f + rmin * gt.e)
        lat_min = float(gt.f + (rmax + 1) * gt.e)
        bbox = {"pixels": [rmin, rmax, cmin, cmax],
                "geo": [round(lon_min, 4), round(lat_min, 4),
                        round(lon_max, 4), round(lat_max, 4)]}
        center = [round((lon_min + lon_max) / 2, 4),
                  round((lat_min + lat_max) / 2, 4)]

    labeled, n_clusters = scipy_label(fire)
    top5 = []
    if n_clusters:
        sizes = np.bincount(labeled.ravel())[1:]
        order = np.argsort(sizes)[::-1][:5]
        for i, idx in enumerate(order):
            area = float(per_pixel_area[labeled == idx + 1].sum())
            top5.append({"rank": i + 1, "size_pixels": int(sizes[idx]),
                         "area_km2": round(area, 3)})

    return _json_safe({
        "region": region_name, "fire_tif": fire_tif,
        "fire_pixels": fire_pixels,
        "fire_ratio": round(fire_ratio, 6),
        "fire_area_km2": round(fire_area_km2, 3),
        "pixel_area_km2_mean": round(mean_area, 4) if mean_area is not None else None,
        "bbox": bbox, "center": center,
        "clusters": int(n_clusters),
        "top5_clusters": top5,
        "note": ("像元面积按逐行纬度 cos 计算（111.32km/° 近似）；"
                 "0=非火点或无效（云/非林地），nodata=0"),
    })
