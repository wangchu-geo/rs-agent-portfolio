# -*- coding: utf-8 -*-
"""
PM10(2.5) 大气污染物反演工具链

旧链：E:\\YYR\\PM10(2.5)\\test（13 个脚本，本模块逐字复刻其语义并封装为 Agent 工具）。
预报数据为模拟数据（本机无 WRF 预报 nc，PM10_TIQU.py 不做工具化），
因此本链评估口径 = 跑通反演流程证明反演能力，精度数字不做产品化声明。

旧脚本 ↔ 工具映射：
    csv_tiqu.py + csv整理.py  →  pm_prepare_stations   （站点筛选 + 逐时 CSV 转置）
    Himawari8_AOD.py          →  pm_extract_aod         （AOD nc→tif）
    ERA5_Process.py           →  pm_extract_era5        （ERA5 nc→7 气象 tif）
    PM10_TIQU.py              →  不做（WRF 预报提取，本机无 nc，用现成模拟预报）
    tif裁剪.py / -999_to_nan.py → 不做（PD/DEM/LU 裁剪已完成直接复用）
    PM10_RF_1KM_Landuse.py    →  pm_align_features / pm_train_rf / pm_predict_rf
    PM10_tianbu_Landuse.py    →  pm_correct_observation （观测校正填补）
    PM10_sanwei.py            →  pm_vertical_profile    （垂直廓线三维推算）
    PM2.5 三镜像               →  同上工具，pollutant 参数化（仅路径/目标列/命名不同）

已实证旧品缺陷（correct 模式默认修复，legacy 模式逐字复现用于验证）：
    1. RF 预测列序 bug（PM10_RF_1KM_Landuse.py:354-361）：训练特征按
       ['DEM','PD','AOD','LU']+sorted(气象) 排列，预测却用 os.listdir 序+DEM 首位
       → 11 列中 5 列错位（PD↔blh 互换、LU→lai_hv→lai_lv→landuse→PD 链移）；
       实测旧 ALIGNED 目录 listdir 序 ≠ sorted 序。
    2. AOD 双倍缩放 bug（Himawari8_AOD.py:107,119）：netCDF4 读取时已自动应用
       scale_factor=0.0002（读值 0.023~1.777 即真实 AOD），脚本又乘一次 0.0002
       → 特征量级 ~1e-4（旧 station_predictions.csv 中 AOD=5.07e-5 佐证）；
       且 missing_value -32768 未置 NaN（旧 Cropped tif 值域 -32768~0.00025 佐证）。
    3. sanwei 命名交叉 bug（PM10_sanwei.py:795 / PM25_sanwei.py:795）：PM10 脚本
       输出 "PM2.5_*_result.tif"、PM2.5 脚本输出 "PM10_*_result.tif"（互相写错名）。

数据口径（继承用户三原则）：
    - 单景跑通即推进：主验证 = PM10 20250210_12 一景；PM2.5 镜像参数化覆盖不单独验证；
    - 样本/预报为模拟数据 → R² 只做机械对照，评估=流程证明非精度；
    - 旧目录（test/、PM10（PM25）/）只读；新产物写 E:/YYR/PM10(2.5)/work/。
"""

import json
import os
import re
import sys
import glob
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import rasterio
import netCDF4 as nc
import xarray as xr
from osgeo import gdal, osr
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from rasterio.transform import from_bounds, from_origin
from scipy import ndimage
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_dilation
from scipy.interpolate import griddata, Rbf, interp1d, RegularGridInterpolator
from scipy.optimize import curve_fit, OptimizeWarning
from pyproj import Transformer
from pykrige.ok import OrdinaryKriging
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ndvi_agent"))
from pipeline_tools import _json_safe  # noqa: E402

# 静态特征文件名发现模式（对应旧脚本硬编码的 DEM_FILE/PD_FILE/AOD_FILE/LU_FILE）
_PAT_DEM = "Elevation"
_PAT_PD = "population_density"
_PAT_AOD = "AOT_Merged_Cropped"
_PAT_LU = "landuse"

# PM10/PM2.5 值域护栏（旧品实测 ~47~122，护栏口径为物理常识 + 提示）
_PM_HARD = (0.0, 500.0)
_PM_SOFT = (20.0, 200.0)
_AOD_HARD = (0.0, 3.0)


# ---------------------------------------------------------------------------
# 通用辅助
# ---------------------------------------------------------------------------

def _out(payload, **extra):
    """把结果 dict 转成 JSON 字符串（Agent Tool 返回约定）。"""
    payload.update(extra)
    return json.dumps(_json_safe(payload), ensure_ascii=False, allow_nan=False)


def _tif_attrs(path):
    """只读地读一个 tif 的关键属性（盘查/验证用）。"""
    with rasterio.open(path) as src:
        a = src.read(1)
        return {
            "shape": list(src.shape), "dtype": src.dtypes[0], "crs": str(src.crs),
            "nodata": (None if src.nodata is None else float(src.nodata)),
            "bounds": [round(float(b), 4) for b in src.bounds],
            "min": float(np.nanmin(a)) if np.any(~np.isnan(a)) else None,
            "max": float(np.nanmax(a)) if np.any(~np.isnan(a)) else None,
            "nan_pct": round(100 * float(np.isnan(a).mean()), 1),
        }


def _find_static_files(aligned_dir):
    """在已对齐目录里按文件名模式找 4 个静态特征（复刻旧脚本硬编码映射）。"""
    tifs = [f for f in os.listdir(aligned_dir) if f.endswith(".tif")]
    found = {}
    for pat in (_PAT_DEM, _PAT_PD, _PAT_AOD, _PAT_LU):
        hits = [f for f in tifs if pat in f]
        if len(hits) != 1:
            raise ValueError(f"按模式 '{pat}' 应恰好匹配 1 个文件，实际 {len(hits)}: {hits}")
        found[pat] = hits[0]
    return found


def _extract_station_features(station_csv, aligned_dir, target_col):
    """复刻 PM10_RF_1KM_Landuse.py:221-268 的站点特征提取（训练/预测/评估共用）。"""
    stations = pd.read_csv(station_csv, encoding="utf-8")
    static = _find_static_files(aligned_dir)
    feature_names = ["DEM", "PD", "AOD", "LU"]
    for feature_name, file_name in zip(
            feature_names,
            [static[_PAT_DEM], static[_PAT_PD], static[_PAT_AOD], static[_PAT_LU]]):
        values = []
        raster_path = os.path.join(aligned_dir, file_name)
        for _, row in stations.iterrows():
            values.append(_extract_raster_value(raster_path, row["经度"], row["纬度"]))
        stations[feature_name] = values

    weather_files = sorted([f for f in os.listdir(aligned_dir)
                            if f not in list(static.values()) and f.endswith(".tif")])
    for wfile in weather_files:
        feature_name = os.path.splitext(wfile)[0]
        values = []
        raster_path = os.path.join(aligned_dir, wfile)
        for _, row in stations.iterrows():
            values.append(_extract_raster_value(raster_path, row["经度"], row["纬度"]))
        stations[feature_name] = values
        feature_names.append(feature_name)

    stations_clean = stations.dropna(subset=feature_names + [target_col])
    return stations, stations_clean, feature_names


def _extract_raster_value(raster_path, lon, lat):
    """复刻 PM10_RF_1KM_Landuse.py:136-149。"""
    try:
        with rasterio.open(raster_path) as src:
            row, col = src.index(lon, lat)
            if 0 <= row < src.height and 0 <= col < src.width:
                value = src.read(1, window=Window(col, row, 1, 1))[0, 0]
                return value if value != src.nodata else np.nan
    except Exception as e:
        print(f"提取 {raster_path} 值时出错: {e}")
    return np.nan


# ===========================================================================
# 工具 1：pm_inspect_data —— 阶段状态盘查
# ===========================================================================

def pm_inspect_data(base_dir, date="20250210", hour="12"):
    """
    盘查 PM10(2.5) 反演链各阶段数据状态（只读）。

    :param base_dir: 旧链根目录，如 E:/YYR/PM10(2.5)/test
    :param date: 日期 YYYYMMDD（旧链只跑通 20250210 一景）
    :param hour: 时次（"12"）
    """
    hour_dir = f"{date}_{hour}"
    rep = {}

    def st(name, path):
        if path is None:
            rep[name] = "missing"
        elif os.path.isdir(path):
            rep[name] = "dir"
        elif os.path.isfile(path):
            rep[name] = "file"
        else:
            rep[name] = "missing"

    st("station_csvs", os.path.join(base_dir, "station", date))
    st("station_list", os.path.join(base_dir, "station", "站点列表.csv"))
    st("aod_dir", os.path.join(base_dir, "AOD"))
    st("aod_tiqu", os.path.join(base_dir, "AOD", "tiqu"))
    st("era5_nc", os.path.join(base_dir, "ERA5", "39b938147c1544637d0a9c28f24937c0"))
    st("era5_output", os.path.join(base_dir, "ERA5", "ERA5_output", f"{date}_{hour}00"))
    st("dem_tif", os.path.join(base_dir, "DEM-1KM", "FH_Elevation_1km.tif"))
    st("pd_tif", os.path.join(base_dir, "PD", "population_density_clip.tif"))
    st("lu_tif", os.path.join(base_dir, "Landuse", "landuse_clip.tif"))
    st("aligned_dir", os.path.join(base_dir, "ALIGNED_DIR_dem_2km", hour_dir))
    st("old_rf_result", os.path.join(base_dir, "pm10", hour_dir,
                                     f"{date}_{hour}00_PM10_2KM.tif"))
    st("old_tianbu_result", os.path.join(base_dir, "pm10", hour_dir,
                                         f"{date}_{hour}00_校正结果_2KM_tianbu_result_smoothed.tif"))
    st("old_sanwei_dir", os.path.join(base_dir, "pm10", hour_dir, "SANWEI"))
    st("yubao_dir", os.path.join(base_dir, "pm10", "yubao"))
    st("yubao_jjj_dir", os.path.join(base_dir, "pm10", "yubao_jjj"))
    st("old_product_level0", "E:/YYR/PM10(2.5)/PM10（PM25）/PM10_level0_result.tif")

    details = {}
    if rep["station_csvs"] == "dir":
        details["station_csv_count"] = len(glob.glob(
            os.path.join(base_dir, "station", date, "*.csv")))
    if rep["old_rf_result"] == "file":
        details["old_rf_result"] = _tif_attrs(
            os.path.join(base_dir, "pm10", hour_dir, f"{date}_{hour}00_PM10_2KM.tif"))
    if rep["old_tianbu_result"] == "file":
        details["old_tianbu_result"] = _tif_attrs(os.path.join(
            base_dir, "pm10", hour_dir, f"{date}_{hour}00_校正结果_2KM_tianbu_result_smoothed.tif"))
    if rep["old_product_level0"] == "file":
        details["old_product_level0"] = _tif_attrs(
            "E:/YYR/PM10(2.5)/PM10（PM25）/PM10_level0_result.tif")
    if rep["aligned_dir"] == "dir":
        details["aligned_tif_count"] = len(glob.glob(
            os.path.join(base_dir, "ALIGNED_DIR_dem_2km", hour_dir, "*.tif")))
        details["aligned_listdir_order"] = [
            f for f in os.listdir(os.path.join(base_dir, "ALIGNED_DIR_dem_2km", hour_dir))
            if f.endswith(".tif")]
    if rep["yubao_dir"] == "dir":
        details["yubao_files"] = sorted(os.listdir(os.path.join(base_dir, "pm10", "yubao")))
    if rep["yubao_jjj_dir"] == "dir":
        details["yubao_jjj_files"] = sorted(os.listdir(os.path.join(base_dir, "pm10", "yubao_jjj")))
    if rep["old_sanwei_dir"] == "dir":
        details["old_sanwei_files"] = sorted(os.listdir(
            os.path.join(base_dir, "pm10", hour_dir, "SANWEI")))

    return _out({"status": rep, "details": details})


# ===========================================================================
# 工具 2：pm_extract_aod —— AOD nc → GeoTIFF（Himawari8_AOD.py）
# ===========================================================================

def pm_extract_aod(nc_path, out_dir, crop_extent=None, mode="correct"):
    """
    Himawari-8 L3 AOD NetCDF → GeoTIFF（复刻 Himawari8_AOD.py）。

    :param nc_path: H09_*.nc 输入（变量 AOT_Merged/longitude/latitude）
    :param out_dir: 输出目录（H08_<北京时>_AOT_Merged_{Full,Cropped}.tif）
    :param crop_extent: (min_lon, min_lat, max_lon, max_lat) 或 None
    :param mode: "correct"=真实 AOD（netCDF4 已自动应用 scale_factor，不再二次缩放），
                 missing_value→NaN；"legacy"=逐字复刻旧品（双倍缩放 + 保留 -32768）
    """
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.basename(nc_path)
    m = re.search(r"H0(\d{1})_(\d{8})_(\d{4})_", filename)
    if m:
        utc_time = datetime.strptime(m.group(2) + m.group(3), "%Y%m%d%H%M")
    else:
        utc_time = datetime.utcnow()
    beijing_str = (utc_time + timedelta(hours=8)).strftime("%Y%m%d_%H%M")

    full_path = os.path.join(out_dir, f"H08_{beijing_str}_AOT_Merged_Full.tif")
    cropped_path = (os.path.join(out_dir, f"H08_{beijing_str}_AOT_Merged_Cropped.tif")
                    if crop_extent else None)
    outputs = [p for p in (full_path, cropped_path) if p]
    if all(os.path.exists(p) for p in outputs):
        return _out({"status": "exists", "files": outputs, "mode": mode,
                     "note": "全部输出已存在，短路跳过"})

    with nc.Dataset(nc_path, "r") as ds:
        for var in ("AOT_Merged", "longitude", "latitude"):
            if var not in ds.variables:
                raise ValueError(f"文件缺少必需变量: {var}")
        aod_var = ds.variables["AOT_Merged"]
        fill_value = aod_var.missing_value
        scale_factor = (aod_var.scale_factor
                        if hasattr(aod_var, "scale_factor") else 0.0002)
        add_offset = (aod_var.add_offset if hasattr(aod_var, "add_offset") else 0.0)
        raw = aod_var[:]  # netCDF4 已自动缩放（真实 AOD）
        lon = np.array(ds.variables["longitude"][:])
        lat = np.array(ds.variables["latitude"][:])

    if mode == "legacy":
        # 复刻旧品：对已缩放的读值再乘一次 scale_factor（双倍缩放 bug）
        underlying = np.ma.getdata(raw)
        aod_data = np.where(underlying != fill_value,
                            underlying * scale_factor + add_offset, fill_value)
    else:
        aod_data = np.ma.filled(raw, np.nan).astype(np.float64)

    if lat[0] < lat[-1]:
        aod_data = np.flipud(aod_data)
        lat = np.flip(lat)

    rows, cols = aod_data.shape
    lon_res = (lon[-1] - lon[0]) / (len(lon) - 1)
    lat_res = (lat[-1] - lat[0]) / (len(lat) - 1)
    geotransform = (lon[0], lon_res, 0, lat[0], 0, lat_res)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    metadata = {
        "scale_factor": str(scale_factor), "add_offset": str(add_offset),
        "source": "Himawari-8 L3 L3ARP Hourly AOD",
        "missing_value": str(fill_value), "original_file": filename,
        "data_time_utc": utc_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "data_time_beijing": (utc_time + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S CST"),
        "processing_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pm_tool_mode": mode,
    }

    driver = gdal.GetDriverByName("GTiff")
    full_ds = driver.Create(full_path, cols, rows, 1, gdal.GDT_Float32,
                            options=["COMPRESS=LZW", "PREDICTOR=2", "TILED=YES"])
    if full_ds is None:
        raise RuntimeError("无法创建输出文件")
    full_ds.SetGeoTransform(geotransform)
    full_ds.SetProjection(srs.ExportToWkt())
    full_band = full_ds.GetRasterBand(1)
    full_band.WriteArray(np.asarray(aod_data, dtype=np.float32))
    full_band.SetNoDataValue(float(fill_value) if mode == "legacy" else float("nan"))
    full_band.SetDescription("Aerosol Optical Thickness at 500nm")
    full_ds.SetMetadata(metadata)
    full_ds.FlushCache()
    full_ds = None

    if crop_extent:
        cropped_ds = gdal.Warp(
            cropped_path, full_path, format="GTiff",
            outputBounds=list(crop_extent),
            dstNodata=float(fill_value) if mode == "legacy" else float("nan"),
            outputType=gdal.GDT_Float32,
            creationOptions=["COMPRESS=LZW", "PREDICTOR=2", "TILED=YES"])
        if cropped_ds is None:
            raise RuntimeError("裁剪操作失败")
        cropped_ds.SetMetadata(metadata)
        cropped_ds.GetRasterBand(1).SetDescription(
            "Aerosol Optical Thickness at 500nm (Cropped)")
        cropped_ds.FlushCache()
        cropped_ds = None

    info = {"mode": mode, "files": outputs}
    with rasterio.open(full_path) as src:
        a = src.read(1)
        info["full_range"] = [float(np.nanmin(a)), float(np.nanmax(a))]
        info["full_nan_pct"] = round(100 * float(np.isnan(a).mean()), 1)
    return _out({"status": "ok"}, **info)


# ===========================================================================
# 工具 3：pm_extract_era5 —— ERA5 nc → 气象 GeoTIFF（ERA5_Process.py）
# ===========================================================================

def pm_extract_era5(nc_dir, out_dir):
    """
    ERA5 多 nc 合并 → 按北京时次输出各变量 GeoTIFF（复刻 ERA5_Process.py）。

    :param nc_dir: 含多个 .nc 的目录（每文件一变量，valid_time 坐标）
    :param out_dir: 输出根目录 → <out_dir>/<YYYYMMDD_HHMM>/<变量>.tif（0.25° 网格）
    """
    os.makedirs(out_dir, exist_ok=True)
    # 幂等短路：已有时次子目录则跳过
    existing_dirs = [d for d in os.listdir(out_dir)
                     if os.path.isdir(os.path.join(out_dir, d))]
    if existing_dirs:
        return _out({"status": "exists", "dirs": existing_dirs,
                     "note": "输出目录已有数据，短路跳过"})

    nc_files = [os.path.join(nc_dir, f) for f in os.listdir(nc_dir) if f.endswith(".nc")]
    datasets = []
    for file in nc_files:
        ds = xr.open_dataset(file)
        lon_coord = ("longitude" if "longitude" in ds.coords
                     else "lon" if "lon" in ds.coords else None)
        if lon_coord:
            lon = ds[lon_coord].values
            if lon.min() < 0 or lon.max() > 360:
                ds = ds.assign_coords(**{lon_coord: np.mod(lon, 360)})
        datasets.append(ds)
    merged = xr.merge(datasets)

    lon_coord = ("longitude" if "longitude" in merged.coords
                 else "lon" if "lon" in merged.coords else None)
    if lon_coord and not np.all(np.diff(merged[lon_coord].values) > 0):
        merged = merged.sortby(lon_coord)

    if "valid_time" not in merged.coords:
        raise ValueError("数据集缺少'valid_time'坐标")
    times = merged.valid_time.values
    datetimes = ([pd.Timestamp(t).to_pydatetime() for t in times]
                 if isinstance(times[0], np.datetime64) else list(times))

    made = []
    for idx, dt in enumerate(datetimes):
        beijing = dt + timedelta(hours=8)
        time_dir = os.path.join(out_dir, beijing.strftime("%Y%m%d_%H%M"))
        os.makedirs(time_dir, exist_ok=True)
        time_slice = merged.isel(valid_time=idx)
        for var_name in merged.data_vars:
            var_data = time_slice[var_name]
            if len(var_data.dims) != 2:
                continue
            lon_dim = ("longitude" if "longitude" in var_data.dims
                       else "lon" if "lon" in var_data.dims else None)
            lat_dim = ("latitude" if "latitude" in var_data.dims
                       else "lat" if "lat" in var_data.dims else None)
            if not lon_dim or not lat_dim:
                continue
            out_path = os.path.join(time_dir, f"{var_name}.tif")
            _save_era5_tif(var_data, out_path, lon_dim, lat_dim)
            made.append(out_path)
    return _out({"status": "ok", "time_steps": len(datetimes),
                 "files": [os.path.relpath(p, out_dir) for p in made]})


def _save_era5_tif(data_array, output_path, lon_dim, lat_dim):
    """复刻 ERA5_Process.save_as_geotiff_with_gdal（0.25° 常数分辨率语义）。"""
    data = data_array.values
    lats = np.asarray(data_array[lat_dim].values)
    lons = np.asarray(data_array[lon_dim].values)
    if not np.all(np.diff(lons) > 0):
        sort_idx = np.argsort(lons)
        lons = lons[sort_idx]
        data = data[:, sort_idx]
    if not np.all(np.diff(lats) < 0):
        sort_idx = np.argsort(lats)[::-1]
        lats = lats[sort_idx]
        data = data[sort_idx, :]

    transform = [lons.min(), 0.25, 0, lats.max(), 0, -0.25]
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(output_path, len(lons), len(lats), 1, gdal.GDT_Float32)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    out_ds.SetGeoTransform(transform)
    out_ds.SetProjection(srs.ExportToWkt())
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(np.asarray(data, dtype=np.float32))
    if "_FillValue" in data_array.attrs:
        out_band.SetNoDataValue(float(data_array.attrs["_FillValue"]))
    out_band.FlushCache()
    out_ds = None


# ===========================================================================
# 工具 4：pm_prepare_stations —— 站点筛选 + 逐时 CSV（csv_tiqu.py + csv整理.py）
# ===========================================================================

def pm_prepare_stations(raw_data_dir, site_list_csv, out_dir,
                        lon_range=(113, 120, 36, 43), date=None):
    """
    站点列表按经纬度范围筛选 + china_sites 逐时数据转置（复刻 csv_tiqu.py + csv整理.py）。

    :param raw_data_dir: china_sites_{date}.csv 所在目录（递归查找）
    :param site_list_csv: 站点列表 CSV（列: 监测点编码…经度/纬度…）
    :param out_dir: 输出目录 → <out_dir>/站点列表.csv + <out_dir>/<date>_<hour>.csv ×24
    :param lon_range: (min_lon, max_lon, min_lat, max_lat)
    :param date: YYYYMMDD；None 则取 raw_data_dir 下第一个 china_sites 文件
    """
    os.makedirs(out_dir, exist_ok=True)
    lon_min, lon_max, lat_min, lat_max = lon_range
    filtered_list = os.path.join(out_dir, "站点列表.csv")
    if os.path.exists(filtered_list):
        hour_files = sorted(f for f in os.listdir(out_dir)
                            if f != "站点列表.csv" and f.endswith(".csv"))
        return _out({"status": "exists", "site_list": os.path.basename(filtered_list),
                     "hour_files": hour_files, "note": "站点 CSV 已存在，短路跳过"})

    # --- 复刻 csv_tiqu.filter_stations ---
    df = pd.read_csv(site_list_csv)
    df["经度"] = pd.to_numeric(df["经度"], errors="coerce")
    df["纬度"] = pd.to_numeric(df["纬度"], errors="coerce")
    df_filtered = df[(df["经度"] >= lon_min) & (df["经度"] <= lon_max) &
                     (df["纬度"] >= lat_min) & (df["纬度"] <= lat_max)]
    df_filtered.to_csv(filtered_list, index=False, encoding="utf-8-sig")

    # --- 复刻 csv整理.py ---
    pat = (f"**/china_sites_{date}.csv" if date else "**/china_sites_*.csv")
    hits = sorted(glob.glob(os.path.join(raw_data_dir, pat), recursive=True))
    if not hits:
        raise ValueError(f"未找到 china_sites CSV: {os.path.join(raw_data_dir, pat)}")
    data_path = hits[0]

    try:
        sites_df = pd.read_csv(filtered_list, encoding="utf-8")
    except UnicodeDecodeError:
        sites_df = pd.read_csv(filtered_list, encoding="gbk")
    data_df = pd.read_csv(data_path)

    made = []
    for (d, h), group in data_df.groupby(["date", "hour"]):
        hour_data = group.drop(columns=["date", "hour"]).set_index("type")
        transposed_data = hour_data.T.reset_index()
        transposed_data.rename(columns={"index": "监测点编码"}, inplace=True)
        merged_df = pd.merge(sites_df, transposed_data, on="监测点编码", how="left")
        hour_file = os.path.join(out_dir, f"{int(d)}_{int(h)}.csv")
        merged_df.to_csv(hour_file, index=False)
        made.append(os.path.basename(hour_file))

    return _out({"status": "ok", "site_list": os.path.basename(filtered_list),
                 "n_stations": int(len(df_filtered)),
                 "hour_files": sorted(made), "source": data_path})


# ===========================================================================
# 工具 5：pm_align_features —— 11 特征对齐到 DEM-1KM 网格
# ===========================================================================

def pm_align_features(dem_tif, pd_tif, lu_tif, aod_tif, meteo_dir, out_dir):
    """
    所有特征栅格对齐到 DEM 参考系（复刻 PM10_RF_1KM_Landuse.align_all_rasters_to_dem）。

    :param dem_tif: DEM-1KM 基准栅格（FH_Elevation_1km.tif，网格=产品网格）
    :param pd_tif: 人口密度；:param lu_tif: 土地利用（最近邻）；:param aod_tif: AOD（双线性）
    :param meteo_dir: ERA5_output/<时次> 目录（7 个气象 tif，双线性）
    :param out_dir: 对齐输出目录（<date>_<hour>/），内容不对齐时清空重建
    """
    NODATA = np.nan
    os.makedirs(out_dir, exist_ok=True)
    if _check_alignment(out_dir, os.path.basename(dem_tif)):
        tifs = sorted(f for f in os.listdir(out_dir) if f.endswith(".tif"))
        return _out({"status": "exists", "aligned_files": tifs,
                     "note": "对齐目录已通过一致性检查，短路跳过"})

    with rasterio.open(dem_tif) as ref:
        ref_transform = ref.transform
        ref_crs = ref.crs
        ref_width = ref.width
        ref_height = ref.height
        ref_profile = ref.profile.copy()
        ref_profile.update({"dtype": "float32", "nodata": NODATA, "compress": "LZW"})

    aligned_dem = os.path.join(out_dir, os.path.basename(dem_tif))
    with rasterio.open(dem_tif) as src:
        data = src.read(1)
    with rasterio.open(aligned_dem, mode="w", **ref_profile) as dst:
        dst.write(data.astype(np.float32), 1)

    feature_files = {"PD": pd_tif, "AOD": aod_tif, "LU": lu_tif}
    for name, file_path in feature_files.items():
        aligned_path = os.path.join(out_dir, os.path.basename(file_path))
        resampling = Resampling.nearest if name == "LU" else Resampling.bilinear
        data = np.empty(shape=(ref_height, ref_width), dtype=np.float32)
        data.fill(NODATA)
        with rasterio.open(file_path) as src:
            reproject(source=rasterio.band(src, 1), destination=data,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=ref_transform, dst_crs=ref_crs,
                      resampling=resampling, dst_nodata=NODATA)
        with rasterio.open(aligned_path, mode="w", **ref_profile) as dst:
            dst.write(data, 1)

    weather_files = [f for f in os.listdir(meteo_dir) if f.endswith(".tif")]
    for wfile in weather_files:
        wpath = os.path.join(meteo_dir, wfile)
        aligned_path = os.path.join(out_dir, wfile)
        data = np.empty(shape=(ref_height, ref_width), dtype=np.float32)
        data.fill(NODATA)
        with rasterio.open(wpath) as src:
            reproject(source=rasterio.band(src, 1), destination=data,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=ref_transform, dst_crs=ref_crs,
                      resampling=Resampling.bilinear, dst_nodata=NODATA)
        with rasterio.open(aligned_path, mode="w", **ref_profile) as dst:
            dst.write(data, 1)

    tifs = sorted(f for f in os.listdir(out_dir) if f.endswith(".tif"))
    return _out({"status": "ok", "aligned_files": tifs,
                 "ref_shape": [ref_width, ref_height]})


def _check_alignment(directory, dem_name):
    """复刻 PM10_RF_1KM_Landuse.check_alignment。"""
    files = [f for f in os.listdir(directory) if f.endswith(".tif")]
    ref_path = os.path.join(directory, dem_name)
    if not files or not os.path.exists(ref_path):
        return False
    with rasterio.open(ref_path) as ref:
        ref_transform = ref.transform
        ref_shape = (ref.width, ref.height)
        ref_crs = ref.crs
    for f in files:
        if f == dem_name:
            continue
        with rasterio.open(os.path.join(directory, f)) as src:
            if (src.transform != ref_transform or
                    (src.width, src.height) != ref_shape or src.crs != ref_crs):
                return False
    return True


# ===========================================================================
# 工具 6：pm_train_rf —— 随机森林训练（复刻 PM10_RF_1KM_Landuse 训练段）
# ===========================================================================

def pm_train_rf(station_csv, aligned_dir, out_dir, pollutant="PM10"):
    """
    站点特征提取 + GridSearchCV + 五折 CV + 全量训练（训练语义逐字复刻旧品）。

    :param station_csv: 站点逐时 CSV（station/<date>/<date>_<hour>.csv）
    :param aligned_dir: 对齐特征目录（11 tif）
    :param out_dir: 输出目录 → model.pkl/scaler_X.pkl/scaler_y.pkl/metrics.json
    :param pollutant: "PM10" 或 "PM2.5"（PM2.5 为镜像链，仅目标列不同）
    """
    os.makedirs(out_dir, exist_ok=True)
    target_col = pollutant
    output_model = os.path.join(out_dir, "model.pkl")
    scaler_x_path = os.path.join(out_dir, "scaler_X.pkl")
    scaler_y_path = os.path.join(out_dir, "scaler_y.pkl")
    metrics_path = os.path.join(out_dir, "metrics.json")
    if os.path.exists(output_model) and os.path.exists(metrics_path):
        return _out({"status": "exists", "model": output_model,
                     "note": "模型与指标已存在，短路跳过"})

    stations, stations_clean, feature_names = _extract_station_features(
        station_csv, aligned_dir, target_col)
    n_valid = int(len(stations_clean))
    if n_valid == 0:
        raise ValueError(f"dropna 后有效样本为 0（{len(stations)} 站），无法训练")

    X = stations_clean[feature_names].values
    y = stations_clean[target_col].values

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.2, random_state=42)
    scaler_X.fit(X_train_raw)
    scaler_y.fit(y_train_raw.reshape(-1, 1))
    features_scaled = scaler_X.transform(X)
    target_scaled = scaler_y.transform(y.reshape(-1, 1)).ravel()

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_features": ["sqrt", "log2"],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid,
                               cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
    grid_search.fit(X_train_raw, y_train_raw)
    best_params = grid_search.best_params_
    best_mse = float(-grid_search.best_score_)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_list, r2_list = [], []
    for train_index, test_index in kf.split(features_scaled):
        X_tr, X_te = features_scaled[train_index], features_scaled[test_index]
        y_tr, y_te = target_scaled[train_index], target_scaled[test_index]
        model = RandomForestRegressor(**best_params, random_state=42)
        model.fit(X_tr, y_tr)
        pred_scaled = model.predict(X_te)
        predictions = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        y_te_orig = scaler_y.inverse_transform(y_te.reshape(-1, 1)).ravel()
        rmse_list.append(float(np.sqrt(mean_squared_error(y_te_orig, predictions))))
        r2_list.append(float(r2_score(y_te_orig, predictions)))

    final_model = RandomForestRegressor(**best_params, random_state=42)
    final_model.fit(features_scaled, target_scaled)

    joblib.dump(final_model, output_model)
    joblib.dump(scaler_X, scaler_x_path)
    joblib.dump(scaler_y, scaler_y_path)

    static = _find_static_files(aligned_dir)
    feature_files = {"DEM": static[_PAT_DEM], "PD": static[_PAT_PD],
                     "AOD": static[_PAT_AOD], "LU": static[_PAT_LU]}
    for fn in feature_names[4:]:
        feature_files[fn] = f"{fn}.tif"
    importance = {fn: float(imp) for fn, imp in
                  zip(feature_names, final_model.feature_importances_)}
    metrics = {
        "pollutant": pollutant,
        "n_stations": int(len(stations)),
        "n_valid_samples": n_valid,
        "feature_names": feature_names,
        "feature_files": feature_files,
        "best_params": best_params,
        "best_mse_unscaled": best_mse,
        "cv_rmse_list": rmse_list,
        "cv_r2_list": r2_list,
        "cv_mean_rmse": float(np.mean(rmse_list)),
        "cv_mean_r2": float(np.mean(r2_list)),
        "feature_importance": importance,
        "note": "样本为随机模拟数据：CV 指标只做流程证明与机械对照，不做精度声明",
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(metrics), f, ensure_ascii=False, indent=2)

    return _out({"status": "ok", "n_valid_samples": n_valid,
                 "feature_names": feature_names, "best_params": best_params,
                 "cv_mean_rmse": metrics["cv_mean_rmse"],
                 "cv_mean_r2": metrics["cv_mean_r2"], "model": output_model})


# ===========================================================================
# 工具 7：pm_predict_rf —— 全局反演（correct=训练列序 / legacy=旧品 listdir 列序）
# ===========================================================================

def pm_predict_rf(aligned_dir, model_dir, out_tif, station_csv=None,
                  mode="correct", pollutant="PM10", raster_order=None):
    """
    分块全局反演（复刻 PM10_RF_1KM_Landuse 预测段）。

    :param aligned_dir: 对齐特征目录
    :param model_dir: pm_train_rf 输出目录（model.pkl/scaler_X.pkl/scaler_y.pkl/metrics.json）
    :param out_tif: 反演结果 tif（float32/nodata=nan/LZW）
    :param station_csv: 站点 CSV（可选，用于输出 station_predictions.csv）
    :param mode: "correct"=按训练 feature_names 列序堆叠（修复列序 bug）；
                 "legacy"=os.listdir 序+DEM 首位（逐字复刻旧品 bug）
    :param pollutant: 与模型对应（"PM10"/"PM2.5"）
    :param raster_order: 仅 legacy 模式生效；显式给定文件名列表时按该序堆叠
                 （用于复刻旧 ALIGNED 目录的实测 listdir 序；None=用当前目录 os.listdir）
    """
    os.makedirs(os.path.dirname(out_tif), exist_ok=True)
    if os.path.exists(out_tif):
        return _out({"status": "exists", "out_tif": out_tif,
                     "note": "反演结果已存在，短路跳过"})
    metrics_path = os.path.join(model_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        raise ValueError(f"model_dir 缺少 metrics.json: {model_dir}")
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    feature_names = metrics["feature_names"]

    model = joblib.load(os.path.join(model_dir, "model.pkl"))
    scaler_X = joblib.load(os.path.join(model_dir, "scaler_X.pkl"))
    scaler_y = joblib.load(os.path.join(model_dir, "scaler_y.pkl"))

    if mode == "correct":
        fmap = metrics["feature_files"]
        ordered = [fmap[fn] for fn in feature_names]
        raster_files = [os.path.join(aligned_dir, f) for f in ordered]
    else:
        if raster_order is not None:
            names = list(raster_order)
        else:
            names = [f for f in os.listdir(aligned_dir) if f.endswith(".tif")]
        dem_name = metrics["feature_files"]["DEM"]
        if dem_name in names:
            names.remove(dem_name)
            names.insert(0, dem_name)
        raster_files = [os.path.join(aligned_dir, f) for f in names]
        if len(raster_files) != len(feature_names):
            raise ValueError(
                f"legacy 栅格数 {len(raster_files)} ≠ 特征数 {len(feature_names)}")

    for f in raster_files:
        if not os.path.exists(f):
            raise ValueError(f"对齐特征缺失: {f}")

    with rasterio.open(raster_files[0]) as ref:
        output_profile = ref.profile.copy()
    output_profile.update({"count": 1, "dtype": "float32",
                           "nodata": np.nan, "compress": "LZW"})

    raster_srcs = [rasterio.open(f) for f in raster_files]
    try:
        with rasterio.open(out_tif, mode="w", **output_profile) as dst:
            windows = [w for _, w in raster_srcs[0].block_windows()]
            for wi, window in enumerate(windows):
                height, width = window.height, window.width
                n_pixels = height * width
                features = np.full((n_pixels, len(feature_names)), np.nan,
                                   dtype=np.float32)
                for i, src in enumerate(raster_srcs):
                    data = src.read(1, window=window, masked=True)
                    features[:, i] = data.filled(np.nan).flatten()
                valid_mask = ~np.isnan(features).any(axis=1)
                if np.any(valid_mask):
                    vfs = scaler_X.transform(features[valid_mask])
                    pred_scaled = model.predict(vfs)
                    prediction = scaler_y.inverse_transform(
                        pred_scaled.reshape(-1, 1)).ravel()
                    result = np.full(n_pixels, output_profile["nodata"],
                                     dtype=np.float32)
                    result[valid_mask] = prediction
                    result_2d = result.reshape(height, width)
                else:
                    result_2d = np.full((height, width), output_profile["nodata"],
                                        dtype=np.float32)
                dst.write(result_2d, 1, window=window)
                if (wi + 1) % 50 == 0:
                    print(f"  分块预测进度: {wi + 1}/{len(windows)}")
    finally:
        for src in raster_srcs:
            src.close()

    # 站点反演值提取（复刻旧品 station_predictions 输出）
    station_out = None
    if station_csv and os.path.exists(station_csv):
        _, stations_clean, _ = _extract_station_features(
            station_csv, aligned_dir, pollutant)
        predictions = []
        with rasterio.open(out_tif) as src:
            for _, row in stations_clean.iterrows():
                lon, lat = row["经度"], row["纬度"]
                try:
                    row_idx, col_idx = src.index(lon, lat)
                    value = src.read(1, window=Window(col_idx, row_idx, 1, 1))[0, 0]
                    predictions.append(
                        np.nan if (value == src.nodata or np.isnan(value)) else value)
                except Exception:
                    predictions.append(np.nan)
        stations_clean["预测值"] = predictions
        valid_predictions = stations_clean.dropna(subset=["预测值"])
        result_df = valid_predictions.copy()
        result_df["绝对误差"] = np.abs(
            result_df[pollutant] - result_df["预测值"])
        station_out = os.path.join(os.path.dirname(out_tif), "station_predictions.csv")
        result_df.to_csv(station_out, index=False, encoding="utf-8-sig")

    with rasterio.open(out_tif) as src:
        a = src.read(1)
        info = {"shape": list(src.shape), "range": [float(np.nanmin(a)), float(np.nanmax(a))],
                "nan_pct": round(100 * float(np.isnan(a).mean()), 1)}
    return _out({"status": "ok", "mode": mode, "out_tif": out_tif,
                 "station_predictions": station_out, **info})


# ===========================================================================
# 工具 8：pm_correct_observation —— 观测校正填补（PM10_tianbu_Landuse.py）
# ===========================================================================

def pm_correct_observation(rf_tif, station_csv, lu_tif, forecast_tif, out_tif,
                           pollutant="PM10"):
    """
    监测站观测校正 + NaN 区域填补（逐字复刻 PM10_tianbu_Landuse.bias_correction_with_csv_and_shp）。

    产出：<out_tif> 及 9 个中间件（_forecast_smoothed/_LU_resampled/_preliminary/
    _ML_filled/_measured_surface/_distance_weights/_combined_filled/
    _mosaic_smoothed/_result_smoothed）。

    :param rf_tif: RF 反演结果（待校正基准栅格）
    :param station_csv: 站点逐时 CSV（含 PM10/PM2.5 实测列）
    :param lu_tif: 土地利用（重采样用）
    :param forecast_tif: 预报栅格（旧链用旧产品 PM10_level0_result.tif 当预报）
    :param out_tif: 校正结果输出（中间件由 replace(".tif", "_X.tif") 派生）
    """
    os.makedirs(os.path.dirname(out_tif), exist_ok=True)
    pm_col = "pm10" if pollutant == "PM10" else "pm2.5"
    expected = [out_tif.replace(".tif", "_result_smoothed.tif")]
    if os.path.exists(expected[0]):
        return _out({"status": "exists", "files": [out_tif] + expected,
                     "note": "校正终产物已存在，短路跳过"})

    with rasterio.open(rf_tif) as src:
        raster = src.read(1).astype(float)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata
        profile = src.profile
    if nodata is not None:
        raster[raster == nodata] = np.nan

    # 预报数据（重采样→平滑）
    if forecast_tif and os.path.exists(forecast_tif):
        try:
            forecast_raster = _resample_to_target(forecast_tif, profile)
            forecast_raster = forecast_raster.astype(float)
            with rasterio.open(forecast_tif) as forecast_src:
                forecast_nodata = forecast_src.nodata
                if forecast_nodata is not None:
                    forecast_raster[forecast_raster == forecast_nodata] = np.nan
            forecast_raster = _smooth_forecast_data(
                forecast_raster, sigma=1.5, max_iters=50)
            weights_output_path = out_tif.replace(".tif", "_forecast_smoothed.tif")
            profile.update(dtype=rasterio.float32, nodata=np.nan)
            with rasterio.open(weights_output_path, "w", **profile) as dst:
                dst.write(forecast_raster.astype(np.float32), 1)
        except Exception as e:
            print(f"预报数据处理失败: {e}")
            forecast_raster = raster.copy()
    else:
        forecast_raster = raster.copy()

    # 土地利用（重采样）
    try:
        lu_raster = _resample_to_target(lu_tif, profile)
        lu_raster = lu_raster.astype(float)
        with rasterio.open(lu_tif) as lu_src:
            lu_nodata = lu_src.nodata
            if lu_nodata is not None:
                lu_raster[lu_raster == lu_nodata] = np.nan
        lu_output_path = out_tif.replace(".tif", "_LU_resampled.tif")
        profile.update(dtype=rasterio.float32, nodata=np.nan)
        with rasterio.open(lu_output_path, "w", **profile) as dst:
            dst.write(lu_raster.astype(np.float32), 1)
    except Exception as e:
        print(f"土地利用数据处理失败: {e}")
        lu_raster = np.full_like(raster, np.nan)

    # 站点 CSV
    try:
        csv_data = pd.read_csv(station_csv, encoding="utf-8")
    except UnicodeDecodeError:
        csv_data = pd.read_csv(station_csv, encoding="gbk")
    csv_data = csv_data.rename(columns={"经度": "Lon", "纬度": "Lat", pollutant: pm_col})
    csv_data = csv_data.dropna(subset=["Lon", "Lat", pm_col]).copy()

    x_proj, y_proj = _project_points_to_crs(csv_data["Lon"].values,
                                            csv_data["Lat"].values, 4326, crs)
    all_x_proj = x_proj.copy()
    all_y_proj = y_proj.copy()
    all_measured = csv_data[pm_col].values.astype(np.float32)

    coords = [~transform * (x, y) for x, y in zip(x_proj, y_proj)]
    cols, rows = map(np.array, zip(*coords))
    valid_mask = (~np.isnan(cols)) & (~np.isnan(rows)) & \
                 (cols >= 0) & (cols < raster.shape[1]) & \
                 (rows >= 0) & (rows < raster.shape[0])
    if not np.any(valid_mask):
        raise ValueError("检查监测点坐标与栅格数据范围/CRS是否一致")
    csv_data = csv_data.loc[valid_mask].copy()
    x_proj = x_proj[valid_mask]
    y_proj = y_proj[valid_mask]
    cols = cols[valid_mask].astype(int)
    rows = rows[valid_mask].astype(int)

    valid_raster_mask = ~np.isnan(raster[rows, cols])
    if not np.any(valid_raster_mask):
        raise ValueError("没有有效的监测点用于校正（落在 NaN 区域）。")
    csv_data = csv_data[valid_raster_mask].copy()
    x_proj = x_proj[valid_raster_mask]
    y_proj = y_proj[valid_raster_mask]
    cols = cols[valid_raster_mask]
    rows = rows[valid_raster_mask]
    if len(csv_data) == 0:
        raise ValueError("没有有效的监测点用于校正")

    measured = csv_data[pm_col].values.astype(np.float32)
    predicted = raster[rows, cols].astype(np.float32)
    residuals = measured - predicted.astype(np.float32)

    # Kriging 残差
    nrows, ncols = raster.shape
    x_coords = np.arange(ncols) * transform.a + transform.c + transform.a / 2
    y_coords = np.arange(nrows) * transform.e + transform.f + transform.e / 2
    y_asc = y_coords
    flip_y_back = False
    if (len(y_coords) >= 2) and (y_coords[1] < y_coords[0]):
        y_asc = y_coords[::-1]
        flip_y_back = True
    OK = OrdinaryKriging(x_proj, y_proj, residuals, variogram_model="spherical",
                         verbose=False, enable_plotting=False)
    z, ss = OK.execute("grid", x_coords, y_asc)
    residual_grid = np.array(z, dtype=np.float32)
    if flip_y_back:
        residual_grid = residual_grid[::-1, :]

    corrected = raster.copy()
    valid_mask_raster = ~np.isnan(raster)
    corrected[valid_mask_raster] += residual_grid[valid_mask_raster]
    min_measured = max(float(np.nanmin(measured)), 1.0)
    corrected[valid_mask_raster & (corrected < min_measured)] = min_measured
    preliminary_output_path = out_tif.replace(".tif", "_preliminary.tif")
    profile.update(dtype=rasterio.float32, nodata=np.nan)
    with rasterio.open(preliminary_output_path, "w", **profile) as dst:
        dst.write(corrected.astype(np.float32), 1)

    # NaN 区域填补（XGBoost + 实测表面 + 距离权重）
    nan_mask = np.isnan(corrected)
    known_mask = ~np.isnan(corrected)
    if known_mask.any() and nan_mask.any():
        known_y, known_x = np.where(known_mask)
        known_lon, known_lat = rasterio.transform.xy(transform, known_y, known_x)
        known_vals = corrected[known_y, known_x]
        known_forecast = forecast_raster[known_y, known_x]
        known_lu = lu_raster[known_y, known_x]
        fill_y, fill_x = np.where(nan_mask)
        fill_lon, fill_lat = rasterio.transform.xy(transform, fill_y, fill_x)
        fill_forecast = forecast_raster[fill_y, fill_x]
        fill_lu = lu_raster[fill_y, fill_x]

        X_train = np.column_stack(
            [known_forecast, known_lon, known_lat, known_lu]).astype(np.float32)
        y_train = known_vals.astype(np.float32)
        X_pred = np.column_stack(
            [fill_forecast, fill_lon, fill_lat, fill_lu]).astype(np.float32)
        train_nan_mask = np.any(np.isnan(X_train), axis=1) | np.isnan(y_train)
        X_train = X_train[~train_nan_mask]
        y_train = y_train[~train_nan_mask]
        pred_nan_mask = np.any(np.isnan(X_pred), axis=1)
        X_pred_valid = X_pred[~pred_nan_mask]
        fill_y_valid = fill_y[~pred_nan_mask]
        fill_x_valid = fill_x[~pred_nan_mask]

        ml_filled = corrected.copy()
        if len(X_train) > 0 and len(X_pred_valid) > 0:
            model = XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6,
                                 subsample=0.8, colsample_bytree=0.8,
                                 random_state=42, n_jobs=-1, tree_method="hist")
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            train_r2 = r2_score(y_train, train_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            print(f"训练数据R²: {train_r2:.3f}, RMSE: {train_rmse:.3f}")
            fill_vals = model.predict(X_pred_valid).astype(np.float32)
            ml_filled[fill_y_valid, fill_x_valid] = fill_vals
            ml_output_path = out_tif.replace(".tif", "_ML_filled.tif")
            profile.update(dtype=rasterio.float32, nodata=np.nan)
            with rasterio.open(ml_output_path, "w", **profile) as dst:
                dst.write(ml_filled.astype(np.float32), 1)

        points_xy = np.column_stack([all_x_proj, all_y_proj])
        values_pm = all_measured.astype(np.float32)
        measured_surface = _create_measured_surface(
            points_xy, values_pm, corrected.shape, transform,
            variogram_model="spherical")
        measured_output_path = out_tif.replace(".tif", "_measured_surface.tif")
        profile.update(dtype=rasterio.float32, nodata=np.nan)
        with rasterio.open(measured_output_path, "w", **profile) as dst:
            dst.write(measured_surface.astype(np.float32), 1)

        distance_weights = _calculate_distance_weights(
            all_x_proj, all_y_proj, corrected.shape, transform,
            max_distance_deg=1, sigma=1)
        weights_output_path = out_tif.replace(".tif", "_distance_weights.tif")
        profile.update(dtype=rasterio.float32, nodata=np.nan)
        with rasterio.open(weights_output_path, "w", **profile) as dst:
            dst.write(distance_weights.astype(np.float32), 1)

        combined_filled = corrected.copy()
        combined_filled[nan_mask] = (
            distance_weights[nan_mask] * measured_surface[nan_mask] +
            (1 - distance_weights[nan_mask]) * ml_filled[nan_mask])
        combined_output_path = out_tif.replace(".tif", "_combined_filled.tif")
        profile.update(dtype=rasterio.float32, nodata=np.nan)
        with rasterio.open(combined_output_path, "w", **profile) as dst:
            dst.write(combined_filled.astype(np.float32), 1)

        corrected = _mosaic_with_cubic_convolution(
            corrected, combined_filled, known_mask, buffer_size=5)
        mosaic_output_path = out_tif.replace(".tif", "_mosaic_smoothed.tif")
        profile.update(dtype=rasterio.float32, nodata=np.nan)
        with rasterio.open(mosaic_output_path, "w", **profile) as dst:
            dst.write(corrected.astype(np.float32), 1)

    corrected = _smooth_forecast_data(corrected, sigma=0.5, max_iters=50)
    result_smoothed_output_path = out_tif.replace(".tif", "_result_smoothed.tif")
    profile.update(dtype=rasterio.float32, nodata=np.nan)
    with rasterio.open(result_smoothed_output_path, "w", **profile) as dst:
        dst.write(corrected.astype(np.float32), 1)
    profile.update(dtype=rasterio.float32, nodata=np.nan)
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(corrected.astype(np.float32), 1)

    intermediates = [out_tif.replace(".tif", s) for s in (
        "_forecast_smoothed", "_LU_resampled", "_preliminary", "_ML_filled",
        "_measured_surface", "_distance_weights", "_combined_filled",
        "_mosaic_smoothed", "_result_smoothed")]
    return _out({"status": "ok", "out_tif": out_tif, "intermediates": intermediates,
                 "n_stations_used": int(len(measured)),
                 "nan_filled": bool(known_mask.any() and nan_mask.any())})


def _project_points_to_crs(lon, lat, src_epsg, dst_crs):
    transformer = Transformer.from_crs(f"EPSG:{src_epsg}", dst_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.asarray(x), np.asarray(y)


def _resample_to_target(source_path, target_profile):
    with rasterio.open(source_path) as src:
        source_data = src.read(1)
        source_profile = src.profile.copy()
        if (src.transform == target_profile["transform"] and
                src.width == target_profile["width"] and
                src.height == target_profile["height"]):
            return source_data
        resampled_data = np.empty((target_profile["height"], target_profile["width"]),
                                  dtype=np.float32)
        reproject(source_data, resampled_data,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=target_profile["transform"],
                  dst_crs=target_profile["crs"], resampling=Resampling.bilinear)
        return resampled_data


def _smooth_forecast_data(forecast_data, sigma=1.5, max_iters=50):
    valid_mask = ~np.isnan(forecast_data)
    filled_data = forecast_data.copy()
    if np.any(np.isnan(filled_data)):
        nan_mask = np.isnan(filled_data)
        it = 0
        while np.any(nan_mask) and it < max_iters:
            it += 1
            prev_nan_count = np.sum(nan_mask)
            kernel = np.ones((3, 3))
            conv_result = ndimage.convolve(np.where(nan_mask, 0, filled_data),
                                           kernel, mode="constant", cval=0)
            conv_count = ndimage.convolve(np.where(nan_mask, 0, 1),
                                          kernel, mode="constant", cval=0)
            update_mask = nan_mask & (conv_count > 0)
            filled_data[update_mask] = conv_result[update_mask] / conv_count[update_mask]
            nan_mask = np.isnan(filled_data)
            if np.sum(nan_mask) == prev_nan_count:
                break
    if np.any(np.isnan(filled_data)):
        mask_valid = ~np.isnan(filled_data)
        dist, indices = distance_transform_edt(~mask_valid, return_indices=True)
        filled_data[np.isnan(filled_data)] = filled_data[
            indices[0][np.isnan(filled_data)], indices[1][np.isnan(filled_data)]]
    smoothed = gaussian_filter(filled_data, sigma=sigma)
    smoothed[~valid_mask] = np.nan
    return smoothed


def _calculate_distance_weights(lons, lats, grid_shape, transform, max_distance_deg, sigma):
    rows, ncols = grid_shape
    x_coords = np.arange(ncols) * transform.a + transform.c + transform.a / 2
    y_coords = np.arange(rows) * transform.e + transform.f + transform.e / 2
    xx, yy = np.meshgrid(x_coords, y_coords)
    grid_points = np.column_stack([xx.flatten(), yy.flatten()])
    station_points = np.column_stack([lons, lats])
    distances = np.sqrt(
        (grid_points[:, None, 0] - station_points[None, :, 0]) ** 2 +
        (grid_points[:, None, 1] - station_points[None, :, 1]) ** 2)
    weights_all = np.exp(-0.5 * (distances / max_distance_deg) ** 2)
    weights_sum = np.sum(weights_all, axis=1)
    weights_norm = weights_sum / np.max(weights_sum)
    weights = weights_norm.reshape(grid_shape)
    if sigma > 0:
        weights = gaussian_filter(weights, sigma=sigma)
    return weights


def _mosaic_with_cubic_convolution(original_data, filled_data, valid_mask, buffer_size):
    border_mask = binary_dilation(valid_mask, iterations=buffer_size) & ~valid_mask
    result = np.where(valid_mask, original_data, filled_data)
    if np.any(border_mask):
        border_y, border_x = np.where(border_mask)
        valid_y, valid_x = np.where(valid_mask)
        valid_points = np.column_stack([valid_x, valid_y])
        valid_values = original_data[valid_mask]
        if len(valid_points) > 3 and len(border_y) > 0:
            border_points = np.column_stack([border_x, border_y])
            try:
                rbf = Rbf(valid_points[:, 0], valid_points[:, 1], valid_values,
                          function="cubic")
                border_values = rbf(border_points[:, 0], border_points[:, 1])
                result[border_y, border_x] = border_values
            except Exception:
                border_values = griddata(valid_points, valid_values, border_points,
                                         method="linear")
                nan_mask = np.isnan(border_values)
                if np.any(nan_mask):
                    nearest_values = griddata(valid_points, valid_values,
                                              border_points[nan_mask], method="nearest")
                    border_values[nan_mask] = nearest_values
                result[border_y, border_x] = border_values
    return result


def _create_measured_surface(points, values, grid_shape, transform,
                             variogram_model="spherical"):
    rows, ncols = grid_shape
    x_coords = np.arange(ncols) * transform.a + transform.c + transform.a / 2
    y_coords = np.arange(rows) * transform.e + transform.f + transform.e / 2
    flip_y_back = False
    if (len(y_coords) > 1) and (y_coords[1] < y_coords[0]):
        y_coords = y_coords[::-1]
        flip_y_back = True
    OK = OrdinaryKriging(points[:, 0], points[:, 1], values,
                         variogram_model=variogram_model, verbose=False,
                         enable_plotting=False)
    z, ss = OK.execute("grid", x_coords, y_coords)
    measured_surface = np.array(z, dtype=np.float32)
    if flip_y_back:
        measured_surface = measured_surface[::-1]
    measured_surface[measured_surface < 0] = 0
    return measured_surface


# ===========================================================================
# 工具 9：pm_vertical_profile —— 垂直廓线三维推算（PM10_sanwei.py）
# ===========================================================================

class PM10VerticalProfileProcessor:
    """PM10 垂直剖面处理器（逐字复刻 PM10_sanwei.PM10VerticalProfileProcessor）。"""

    def __init__(self, level_heights=(4, 120, 250, 400)):
        self.level_heights = np.array(level_heights)
        self.profile_functions = None
        self.fitted_params = None
        self.geotransform = None
        self.projection = None

    @staticmethod
    def exponential_func(h, a, b, c):
        return a * np.exp(-b * h) + c

    @staticmethod
    def power_func(h, a, b, c):
        return a * (h + 1e-6) ** (-b) + c

    def read_tif_data(self, file_pattern):
        file_list = glob.glob(file_pattern)
        if not file_list:
            raise FileNotFoundError(f"未找到文件: {file_pattern}")
        if len(file_list) > 1:
            print(f"找到多个文件，使用第一个: {file_list[0]}")
        filename = file_list[0]
        dataset = gdal.Open(filename)
        if dataset is None:
            raise IOError(f"无法打开文件: {filename}")
        self.geotransform = dataset.GetGeoTransform()
        self.projection = dataset.GetProjection()
        band = dataset.GetRasterBand(1)
        data = band.ReadAsArray()
        cols = dataset.RasterXSize
        rows = dataset.RasterYSize
        lon_start = self.geotransform[0]
        lat_start = self.geotransform[3]
        lon_res = self.geotransform[1]
        lat_res = self.geotransform[5]
        lons = np.arange(lon_start + lon_res / 2, lon_start + cols * lon_res, lon_res)
        lats = np.arange(lat_start + lat_res / 2, lat_start + rows * lat_res, lat_res)
        dataset = None
        da = xr.DataArray(data, dims=["lat", "lon"],
                          coords={"lat": lats, "lon": lons},
                          attrs={"geotransform": self.geotransform,
                                 "projection": self.projection})
        return da

    def load_forecast_data(self, base_path):
        level_data = []
        for i in range(1, 5):
            file_pattern = os.path.join(base_path, f"*level{i}.tif")
            try:
                da = self.read_tif_data(file_pattern)
                level_data.append(da)
                print(f"成功加载第{i}层数据，形状: {da.shape}")
            except Exception as e:
                print(f"加载level{i}数据失败: {e}")
                nan_da = xr.full_like(level_data[0], np.nan)
                level_data.append(nan_da)
        pm10_4levels = xr.concat(level_data, dim="level")
        pm10_4levels["level"] = np.arange(1, 5)
        return pm10_4levels

    def load_corrected_data(self, corrected_tif):
        return self.read_tif_data(corrected_tif)

    def fit_vertical_profile(self, pm10_data, fit_func="power", save_parameters_path=None):
        print("开始拟合垂直剖面...")
        lat_coords = pm10_data.lat.values
        lon_coords = pm10_data.lon.values
        n_lat, n_lon = len(lat_coords), len(lon_coords)
        param_a = np.full((n_lat, n_lon), np.nan)
        param_b = np.full((n_lat, n_lon), np.nan)
        param_c = np.full((n_lat, n_lon), np.nan)
        r_squared = np.full((n_lat, n_lon), np.nan)
        total_pixels = n_lat * n_lon
        processed = 0
        successful_fits = 0
        failed_fits = 0
        warning_fails = 0

        prev_filters = warnings.filters
        warnings.filterwarnings(action="error", category=OptimizeWarning)
        warnings.filterwarnings(action="error", category=RuntimeWarning)
        try:
            for i in range(n_lat):
                for j in range(n_lon):
                    processed += 1
                    try:
                        concentrations = pm10_data.isel(lat=i, lon=j).values
                        if np.any(np.isnan(concentrations)) or np.any(concentrations <= 0):
                            failed_fits += 1
                            continue
                        if fit_func == "exponential":
                            fit_function = self.exponential_func
                            p0 = [concentrations[0] - concentrations[-1],
                                  0.1, concentrations[-1]]
                            bounds = ([1e-6, 0.001, 0], [500, np.inf, 300])
                        elif fit_func == "power":
                            fit_function = self.power_func
                            p0 = [concentrations[0], 0.5, concentrations[-1] * 0.5]
                            bounds = ([1e-6, 0.001, 0], [500, np.inf, 300])
                        else:
                            raise ValueError(f"不支持的拟合函数: {fit_func}")
                        interp_func = interp1d(self.level_heights, concentrations,
                                               kind="cubic", fill_value="extrapolate")
                        h_dense = np.linspace(self.level_heights.min(),
                                              self.level_heights.max(), num=150)
                        c_dense = interp_func(h_dense)
                        popt, pcov = curve_fit(fit_function, h_dense, c_dense,
                                               p0=p0, bounds=bounds, maxfev=5000)
                        if (not np.all(np.isfinite(popt)) or
                                not np.all(np.isfinite(np.diag(pcov)))):
                            failed_fits += 1
                            continue
                        if fit_func == "exponential":
                            max_h = self.level_heights.max()
                            if popt[1] * max_h > 100:
                                failed_fits += 1
                                continue
                        predicted = fit_function(self.level_heights, *popt)
                        if (np.any(~np.isfinite(predicted)) or
                                np.any(predicted <= 0) or
                                np.any(predicted > 10000) or
                                np.any(np.isinf(predicted))):
                            failed_fits += 1
                            continue
                        residuals = concentrations - predicted
                        ss_res = np.sum(residuals ** 2)
                        ss_tot = np.sum((c_dense - np.mean(c_dense)) ** 2)
                        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                        if r2 < 0.3:
                            failed_fits += 1
                            continue
                        param_a[i, j], param_b[i, j], param_c[i, j] = popt
                        r_squared[i, j] = r2
                        successful_fits += 1
                    except (RuntimeWarning, OptimizeWarning, RuntimeError):
                        failed_fits += 1
                        warning_fails += 1
                        continue
                    except Exception:
                        failed_fits += 1
                        continue
                    if processed % 1000 == 0:
                        print(f"已处理 {processed}/{total_pixels} 个像元，"
                              f"成功拟合 {successful_fits}，失败 {failed_fits}")
        finally:
            warnings.filters = prev_filters

        print(f"拟合统计: 总像元 {total_pixels}, 成功 {successful_fits} "
              f"({successful_fits / total_pixels * 100:.1f}%), "
              f"失败 {failed_fits}, 警告失败 {warning_fails}")

        if successful_fits > 0:
            self.fitted_params = {
                "a": param_a, "b": param_b, "c": param_c,
                "r_squared": r_squared, "fit_func": fit_func,
                "lat": lat_coords, "lon": lon_coords}
            if save_parameters_path:
                self.save_parameters_to_tiff(
                    param_a, param_b, param_c, r_squared,
                    pm10_data, fit_func, save_parameters_path)
        return self.fitted_params

    def save_parameters_to_tiff(self, param_a, param_b, param_c, r_squared,
                                pm10_data, fit_func, save_parameters_path):
        try:
            lat = pm10_data.lat.values
            lon = pm10_data.lon.values
            lat_res = abs(lat[1] - lat[0]) if len(lat) > 1 else 0.1
            lon_res = abs(lon[1] - lon[0]) if len(lon) > 1 else 0.1
            left = lon.min() - lon_res / 2
            right = lon.max() + lon_res / 2
            bottom = lat.min() - lat_res / 2
            top = lat.max() + lat_res / 2
            transform = from_bounds(left, bottom, right, top, len(lon), len(lat))
            multi_band_data = np.stack([param_a, param_b, param_c, r_squared])
            profile = {
                "driver": "GTiff", "height": len(lat), "width": len(lon),
                "count": 4, "dtype": np.float32, "crs": "EPSG:4326",
                "transform": transform, "compress": "lzw", "nodata": np.nan}
            with rasterio.open(save_parameters_path, mode="w", **profile) as dst:
                for i in range(4):
                    dst.write(multi_band_data[i].astype(np.float32), i + 1)
                dst.set_band_description(1, f"Parameter a ({fit_func} function)")
                dst.set_band_description(2, f"Parameter b ({fit_func} function)")
                dst.set_band_description(3, f"Parameter c ({fit_func} function)")
                dst.set_band_description(4, "R-squared")
            print(f"拟合参数已保存: {save_parameters_path}")
        except Exception as e:
            print(f"保存TIFF文件失败: {e}")

    def _gaussian_smooth(self, data, sigma=1.0):
        data_filled = np.nan_to_num(data, nan=0.0)
        mask = ~np.isnan(data)
        smoothed = gaussian_filter(data_filled, sigma=sigma)
        smoothed[~mask] = np.nan
        return smoothed

    def _smooth_parameters(self, param_a_fine, param_b_fine, param_c_fine,
                           method="gaussian", **kwargs):
        if method == "gaussian":
            sigma = kwargs.get("sigma", 1.0)
            param_a_smooth = self._gaussian_smooth(param_a_fine, sigma)
            param_b_smooth = self._gaussian_smooth(param_b_fine, sigma)
            if not np.all(np.isnan(param_c_fine)):
                param_c_smooth = self._gaussian_smooth(param_c_fine, sigma)
            else:
                param_c_smooth = param_c_fine
            return param_a_smooth, param_b_smooth, param_c_smooth
        return param_a_fine, param_b_fine, param_c_fine

    def calculate_other_levels(self, corrected_level0, target_heights=(120, 250, 400),
                               save_parameters_path=None):
        print("开始计算其他高度层的PM10浓度...")
        if self.fitted_params is None:
            raise ValueError("请先调用 fit_vertical_profile 方法拟合垂直剖面")
        target_lats = corrected_level0.lat.values
        target_lons = corrected_level0.lon.values
        orig_lats = self.fitted_params["lat"]
        orig_lons = self.fitted_params["lon"]
        param_a_fine = self._interpolate_to_target_grid(
            self.fitted_params["a"], orig_lats, orig_lons, target_lats, target_lons)
        param_b_fine = self._interpolate_to_target_grid(
            self.fitted_params["b"], orig_lats, orig_lons, target_lats, target_lons)
        param_c_fine = self._interpolate_to_target_grid(
            self.fitted_params["c"], orig_lats, orig_lons, target_lats, target_lons)
        param_a_fine, param_b_fine, param_c_fine = self._smooth_parameters(
            param_a_fine, param_b_fine, param_c_fine,
            method="gaussian", sigma=1.0)
        self._validate_interpolation_results(param_a_fine, param_b_fine, param_c_fine)
        if save_parameters_path:
            self._save_parameters_as_tiff(
                param_a_fine, param_b_fine, param_c_fine,
                corrected_level0, save_parameters_path)

        result_dict = {"level0": corrected_level0}
        fit_func_name = self.fitted_params["fit_func"]
        h0 = self.level_heights[0]
        for idx, target_h in enumerate(target_heights, 1):
            print(f"计算 {target_h}米 高度的浓度...")
            if fit_func_name == "exponential":
                original_h0_conc = param_a_fine * np.exp(-param_b_fine * h0) + param_c_fine
                valid_mask = (original_h0_conc > 1e-10) & (np.isfinite(original_h0_conc))
                scaling_factor = np.ones_like(original_h0_conc)
                scaling_factor[valid_mask] = (
                    corrected_level0.values[valid_mask] / original_h0_conc[valid_mask])
                scaling_factor = np.clip(scaling_factor, a_min=0.1, a_max=10)
                a_adjusted = param_a_fine * scaling_factor
                c_adjusted = param_c_fine * scaling_factor
                level_data = a_adjusted * np.exp(-param_b_fine * target_h) + c_adjusted
            elif fit_func_name == "power":
                original_h0_conc = param_a_fine * np.power(h0, -param_b_fine) + param_c_fine
                valid_mask = (original_h0_conc > 1e-10) & (np.isfinite(original_h0_conc))
                scaling_factor = np.ones_like(original_h0_conc)
                scaling_factor[valid_mask] = (
                    corrected_level0.values[valid_mask] / original_h0_conc[valid_mask])
                scaling_factor = np.clip(scaling_factor, a_min=0.1, a_max=10)
                a_adjusted = param_a_fine * scaling_factor
                c_adjusted = param_c_fine * scaling_factor
                level_data = a_adjusted * np.power(target_h, -param_b_fine) + c_adjusted
            else:
                raise ValueError(f"不支持的拟合函数: {fit_func_name}")
            level_da = xr.DataArray(
                level_data, dims=["lat", "lon"],
                coords={"lat": corrected_level0.lat.values,
                        "lon": corrected_level0.lon.values},
                attrs=corrected_level0.attrs)
            result_dict[f"level{idx}"] = level_da
        print("计算完成")
        return result_dict

    def _interpolate_to_target_grid(self, param, orig_lats, orig_lons,
                                    target_lats, target_lons):
        param_filled = self._fill_nan_smartly(param)
        target_lon_grid, target_lat_grid = np.meshgrid(target_lons, target_lats)
        target_points = np.column_stack(
            [target_lat_grid.ravel(), target_lon_grid.ravel()])
        try:
            interpolator = RegularGridInterpolator(
                (orig_lats, orig_lons), param_filled, method="linear",
                bounds_error=False, fill_value=np.nan)
            param_fine = interpolator(target_points)
            param_fine = param_fine.reshape(len(target_lats), len(target_lons))
        except Exception as e:
            print(f"RegularGridInterpolator失败: {e}，回退到最近邻插值")
            interpolator = RegularGridInterpolator(
                (orig_lats, orig_lons), param_filled, method="nearest",
                bounds_error=False, fill_value=np.nan)
            param_fine = interpolator(target_points)
            param_fine = param_fine.reshape(len(target_lats), len(target_lons))
        if np.isnan(param_fine).any():
            print("警告: 插值结果包含NaN，进行最终清理")
            param_fine = self._fill_nan_smartly(param_fine)
        assert param_fine.shape == (len(target_lats), len(target_lons)), \
            f"插值结果形状 {param_fine.shape} 与目标 {(len(target_lats), len(target_lons))} 不匹配"
        return param_fine

    @staticmethod
    def _fill_nan_smartly(data):
        if not np.isnan(data).any():
            return data
        mask = ~np.isnan(data)
        if not mask.any():
            return np.zeros_like(data)
        filled_data = data.copy()
        distances, indices = distance_transform_edt(
            ~mask, return_distances=True, return_indices=True)
        filled_data[~mask] = data[tuple(indices[:, ~mask])]
        if np.isnan(filled_data).any():
            global_mean = np.nanmean(data)
            if np.isnan(global_mean):
                global_mean = 0.0
            filled_data = np.where(np.isnan(filled_data), global_mean, filled_data)
        return filled_data

    def _validate_interpolation_results(self, param_a, param_b, param_c):
        print("验证插值结果...")
        nan_count_a = np.isnan(param_a).sum()
        nan_count_b = np.isnan(param_b).sum()
        nan_count_c = np.isnan(param_c).sum()
        print(f"参数A NaN数量: {nan_count_a}")
        print(f"参数B NaN数量: {nan_count_b}")
        print(f"参数C NaN数量: {nan_count_c}")
        if nan_count_a > 0 or nan_count_b > 0 or nan_count_c > 0:
            raise ValueError("插值结果包含NaN值，请检查插值过程")

    def _save_parameters_as_tiff(self, param_a, param_b, param_c,
                                 corrected_level0, save_path):
        print(f"保存参数到TIFF文件: {save_path}")
        lats = corrected_level0.lat.values
        lons = corrected_level0.lon.values
        if len(lats) > 1 and len(lons) > 1:
            lat_res = abs(lats[1] - lats[0])
            lon_res = abs(lons[1] - lons[0])
            left = lons.min()
            top = lats.max()
            transform = from_origin(left, top, lon_res, -lat_res)
        else:
            transform = from_origin(0, 0, 1, 1)
        multi_band_data = np.stack([param_a, param_b, param_c], axis=0)
        multi_band_data = multi_band_data.astype(np.float32)
        n_bands, height, width = multi_band_data.shape
        with rasterio.open(save_path, mode="w", driver="GTiff", height=height,
                           width=width, count=n_bands, dtype=np.float32,
                           crs="EPSG:4326", transform=transform,
                           compress="lzw") as dst:
            for band in range(n_bands):
                dst.write(multi_band_data[band], band + 1)
            dst.set_band_description(1, "Parameter A")
            dst.set_band_description(2, "Parameter B")
            dst.set_band_description(3, "Parameter C")
            dst.update_tags(
                fit_function=self.fitted_params["fit_func"],
                level0_height=f"{self.level_heights[0]}m",
                interpolation_method="Robust nearest-neighbor + bilinear")
        meta_path = save_path.replace(".tif", "_metadata.txt")
        with open(meta_path, "w") as f:
            f.write("参数TIFF文件元数据\n")
            f.write("=" * 40 + "\n")
            f.write(f"文件路径: {save_path}\n")
            f.write(f"拟合函数: {self.fitted_params['fit_func']}\n")
            f.write("波段1: 参数A\n")
            f.write("波段2: 参数B\n")
            f.write("波段3: 参数C\n")
            f.write(f"数据形状: {height} x {width}\n")
            f.write(f"坐标范围: 纬度 [{lats.min():.4f}, {lats.max():.4f}], "
                    f"经度 [{lons.min():.4f}, {lons.max():.4f}]\n")
            f.write(f"创建时间: {np.datetime64('now')}\n")
        print(f"元数据已保存到: {meta_path}")

    def save_to_tif(self, data_array, output_path):
        print(f"保存文件: {output_path}")
        data = data_array.values.astype(np.float32)
        geotransform = data_array.attrs.get("geotransform", self.geotransform)
        projection = data_array.attrs.get("projection", self.projection)
        rows, cols = data.shape
        driver = gdal.GetDriverByName("GTiff")
        out_dataset = driver.Create(output_path, cols, rows, 1, gdal.GDT_Float32)
        if out_dataset is None:
            raise IOError(f"无法创建文件: {output_path}")
        out_dataset.SetGeoTransform(geotransform)
        out_dataset.SetProjection(projection)
        out_band = out_dataset.GetRasterBand(1)
        out_band.WriteArray(data)
        out_band.SetNoDataValue(np.nan)
        out_band.FlushCache()
        out_dataset = None

    def save_all_levels(self, result_dict, output_dir, prefix):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建输出目录: {output_dir}")
        saved = []
        for level_name, data_array in result_dict.items():
            output_path = os.path.join(output_dir, f"{prefix}_{level_name}_result.tif")
            self.save_to_tif(data_array, output_path)
            print(f"已保存: {output_path}")
            saved.append(output_path)
        return saved

    def visualize_results(self, result_dict, save_png_path, title_pollutant):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n_levels = len(result_dict)
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
        axes = axes.flatten()
        vmin = min(da.min().item() for da in result_dict.values())
        vmax = max(da.max().item() for da in result_dict.values())
        for idx, (level_name, data_array) in enumerate(result_dict.items()):
            if idx >= 4:
                break
            im = axes[idx].imshow(data_array.values, cmap="RdYlGn_r",
                                  vmin=vmin, vmax=vmax,
                                  extent=[data_array.lon.min(), data_array.lon.max(),
                                          data_array.lat.min(), data_array.lat.max()])
            axes[idx].set_title(f"{title_pollutant} {level_name} (μg/m³)")
            axes[idx].set_xlabel("Longitude")
            axes[idx].set_ylabel("Latitude")
            cbar = plt.colorbar(im, ax=axes[idx])
            cbar.set_label(f"{title_pollutant} Concentration (μg/m³)")
        plt.tight_layout()
        plt.savefig(save_png_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        print(f"可视化已保存: {save_png_path}")


def pm_vertical_profile(forecast_dir, corrected_tif, out_dir,
                        level_heights=(40, 120, 250, 400), pollutant="PM10",
                        mode="correct", date_hour=None):
    """
    垂直廓线三维推算（复刻 PM10_sanwei.py main 流程，fit_func='exponential'）。

    :param forecast_dir: 4 层模拟预报目录（*level1~4.tif，如 pm10/yubao_jjj）
    :param corrected_tif: 校正后地面层（*ML_filled.tif）
    :param out_dir: 输出目录 → parameters.tif/parameters_fine.tif(+_metadata.txt)/
                    4 个 level tif/png
    :param level_heights: (h0, h1, h2, h3)，旧 main 用 (40,120,250,400)
    :param pollutant: "PM10"/"PM2.5"
    :param mode: "correct"=level 文件名用正确前缀（pollutant）；
                 "legacy"=复刻旧品命名交叉 bug（PM10→"PM2.5_" 前缀，PM2.5→"PM10_"）
    :param date_hour: "YYYYMMDD_HH"（仅用于 png 命名；None 不产 png）
    """
    os.makedirs(out_dir, exist_ok=True)
    if mode == "legacy":
        prefix = "PM2.5" if pollutant == "PM10" else "PM10"
        title_pollutant = prefix
        png_prefix = prefix.lower()
    else:
        prefix = pollutant
        title_pollutant = pollutant
        png_prefix = pollutant.lower()

    level_names = ["level0", "level1", "level2", "level3"]
    level_files = [os.path.join(out_dir, f"{prefix}_{n}_result.tif") for n in level_names]
    if all(os.path.exists(p) for p in level_files):
        return _out({"status": "exists", "files": level_files,
                     "note": "4 层结果已存在，短路跳过"})

    processor = PM10VerticalProfileProcessor(level_heights=level_heights)
    forecast_data = processor.load_forecast_data(forecast_dir)
    corrected_data = processor.load_corrected_data(corrected_tif)

    fitted = processor.fit_vertical_profile(
        forecast_data, fit_func="exponential",
        save_parameters_path=os.path.join(out_dir, "parameters.tif"))
    if fitted is None:
        raise RuntimeError("垂直廓线拟合全部失败（无成功像元）")

    results = processor.calculate_other_levels(
        corrected_data, target_heights=(120, 250, 400),
        save_parameters_path=os.path.join(out_dir, "parameters_fine.tif"))
    saved = processor.save_all_levels(results, out_dir, prefix)

    png_path = None
    if date_hour:
        png_path = os.path.join(out_dir, f"{png_prefix}_{date_hour}.png")
        processor.visualize_results(results, png_path, title_pollutant)

    with rasterio.open(os.path.join(out_dir, "parameters.tif")) as src:
        param_shape = list(src.shape)
    info = {"mode": mode, "prefix": prefix, "level_files": saved,
            "parameters_shape": param_shape, "png": png_path,
            "n_fit_success": int(np.sum(~np.isnan(fitted["a"]))),
            "note": "预报数据为模拟数据；结果仅论证流程，不做精度声明"}
    return _out({"status": "ok"}, **info)


# ===========================================================================
# 工具 10：pm_assess_model —— 模型质量自动评估（新工具）
# ===========================================================================

def pm_assess_model(model_dir, station_csv, aligned_dir, product_tif=None):
    """
    模型质量自动评估：重算 KFold CV 指标（确定性，应与训练记录一致）、特征重要性
    排序、站点回代散点指标（机械对照口径）。

    :param model_dir: pm_train_rf 输出目录
    :param station_csv: 站点逐时 CSV
    :param aligned_dir: 对齐特征目录
    :param product_tif: 反演产品（可选，做护栏统计）
    """
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    pollutant = metrics["pollutant"]
    feature_names = metrics["feature_names"]
    best_params = metrics["best_params"]

    model = joblib.load(os.path.join(model_dir, "model.pkl"))
    scaler_X = joblib.load(os.path.join(model_dir, "scaler_X.pkl"))
    scaler_y = joblib.load(os.path.join(model_dir, "scaler_y.pkl"))

    stations, stations_clean, feat_recheck = _extract_station_features(
        station_csv, aligned_dir, pollutant)
    X = stations_clean[feature_names].values
    y = stations_clean[pollutant].values
    features_scaled = scaler_X.transform(X)
    target_scaled = scaler_y.transform(y.reshape(-1, 1)).ravel()

    # 重跑 KFold CV（与训练同一段确定性代码）
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_list, r2_list = [], []
    for train_index, test_index in kf.split(features_scaled):
        m = RandomForestRegressor(**best_params, random_state=42)
        m.fit(features_scaled[train_index], target_scaled[train_index])
        pred = scaler_y.inverse_transform(
            m.predict(features_scaled[test_index]).reshape(-1, 1)).ravel()
        y_te = scaler_y.inverse_transform(
            target_scaled[test_index].reshape(-1, 1)).ravel()
        rmse_list.append(float(np.sqrt(mean_squared_error(y_te, pred))))
        r2_list.append(float(r2_score(y_te, pred)))

    cv_matches = (abs(np.mean(rmse_list) - metrics["cv_mean_rmse"]) < 1e-9 and
                  abs(np.mean(r2_list) - metrics["cv_mean_r2"]) < 1e-9)

    # 站点回代（样本内，机械对照）
    pred_station = scaler_y.inverse_transform(
        model.predict(features_scaled).reshape(-1, 1)).ravel()
    station_metrics = {
        "n": int(len(y)),
        "r2": float(r2_score(y, pred_station)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred_station))),
        "mae": float(np.mean(np.abs(y - pred_station))),
        "bias": float(np.mean(pred_station - y)),
    }

    importance = sorted(metrics["feature_importance"].items(),
                        key=lambda kv: -kv[1])
    importance_ranking = [{"feature": k, "importance": round(v, 6)}
                          for k, v in importance]

    result = {
        "status": "ok",
        "pollutant": pollutant,
        "n_valid_samples": metrics["n_valid_samples"],
        "feature_names_consistent": feature_names == feat_recheck,
        "cv_rmse_recomputed": round(float(np.mean(rmse_list)), 6),
        "cv_r2_recomputed": round(float(np.mean(r2_list)), 6),
        "cv_matches_train_record": bool(cv_matches),
        "cv_rmse_train_record": metrics["cv_mean_rmse"],
        "cv_r2_train_record": metrics["cv_mean_r2"],
        "importance_ranking": importance_ranking,
        "station_in_sample": station_metrics,
        "note": ("样本为随机模拟数据（无实测支撑）：站点回代 R² 属样本内过拟合，"
                 "仅作机械对照与 bug 修复方向佐证，不做精度声明"),
    }

    if product_tif and os.path.exists(product_tif):
        with rasterio.open(product_tif) as src:
            a = src.read(1)
        hard_ok = bool(np.nanmin(a) >= _PM_HARD[0] and np.nanmax(a) <= _PM_HARD[1])
        soft_ok = bool(np.nanmin(a) >= _PM_SOFT[0] and np.nanmax(a) <= _PM_SOFT[1])
        result["product"] = {
            "shape": list(src.shape),
            "range": [float(np.nanmin(a)), float(np.nanmax(a))],
            "nan_pct": round(100 * float(np.isnan(a).mean()), 1),
            "hard_guard_ok": hard_ok, "soft_guard_ok": soft_ok,
        }
    return _out(result)


if __name__ == "__main__":
    print(pm_inspect_data("E:/YYR/PM10(2.5)/test"))
