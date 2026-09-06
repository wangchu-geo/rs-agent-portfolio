# 遥感智能分析 Agent 工具链

把真实遥感处理管线封装为 **Agent 工具**，让大语言模型通过 Tool Use 自主编排
多时相反演、变化检测与联合分析，最终输出中文专业解读。

核心链路：**自然语言问题 → Agent 自主选择工具 → 真实遥感处理（SNAP / GDAL / 自研算法）→ 中文解读**

## 产品线

| 产品 | 目录 | 工具数 | 验证 | 能力 |
|---|---|---|---|---|
| NDVI + LST 双域联合 Agent | `ndvi_agent/` | 14 | 23+17+12 项断言 | SNAP 预处理→去云→镶嵌→裁剪→统计→变化检测；LST 反演 / ΔLST 对比 / LST-NDVI 联合分析（含跨年同季节对照） |
| 森林火点识别（Himawari-9 静止卫星） | `fire_agent/` | 5 | 86 项断言 | 10 分钟级观测；发现→提取→云掩膜→上下文火点识别→统计；修复旧脚本量纲 bug |
| 土壤相对湿度（S1+S2 双星协同 RF） | `soil_agent/` | 9 | 33+24 项断言 | S1/S2 特征→随机森林训练→反演→模型质量自动评估；修复特征列错位、无地理重投影、垃圾文件三缺陷 |
| PM10(2.5) 反演（多源特征） | `pm_agent/` | 10 | 34 项断言 | AOD/ERA5 提取→站点观测校正→垂直廓线三维推算；修复 RF 预测列序、AOD 双倍缩放、命名交叉三缺陷 |
| 水体总磷（S2+S3 双传感器） | `pwater_agent/` | 10 | 103 项断言 | 逐小时监测→时空匹配→SNAP 预处理链→6 波段组合指数→RF 反演→水体矢量裁剪；修复 6 缺陷 |

## 成果展示

| 产品 | 成果图 | 验证 | 旧品缺陷修复 |
|---|---|---|---|
| NDVI 多时相 | ![NDVI](portfolio/figures/doc_ndvi.png) | 23+29+31 项断言 | — |
| 地表温度 LST（Landsat TIRS，跨年同季节验证） | ![LST](portfolio/figures/doc_lst.png) | 23+17+12 项断言 | —（自研链，含跨年同季节对照） |
| 森林火点识别 | ![火点](portfolio/figures/doc_fire.png) | 86 项断言 | 量纲 bug |
| 土壤相对湿度 | ![土壤湿度](portfolio/figures/doc_soil_result.png) | 33+24 项断言 | 特征列错位、无地理重投影、垃圾文件 |
| PM10(2.5) 反演 | ![PM10](portfolio/figures/doc_pm10_example.png) ![PM2.5](portfolio/figures/doc_pm25_example.png) | 34 项断言 | RF 预测列序、AOD 双倍缩放、命名交叉 |
| 水体总磷 | ![总磷](portfolio/figures/doc_tp_example.png) | 103 项断言 | 掩膜坏值入模、阈值 0、÷0 语义、模型未落盘等 6 项 |

图注：成果图摘自本人产品手册原图（原产品链产物）；对比与联合分析图为工具链直接输出（未做后期修饰）。

## 多时相对比与联合分析成果

跨年同季节对照（同一研究区），对应 `compare_ndvi_dates`、
LST 对比与 LST-NDVI 联合分析工具的真实运行产物：

| 产品 | 对比/分析图 | 说明 |
|---|---|---|
| NDVI 变化对比 | ![NDVI 变化图](portfolio/figures/change_map_20250425_vs_20260510.png) ![NDVI 面积占比](portfolio/figures/area_proportion_20250425_vs_20260510.png) | 20250425 → 20260510：Δ=晚-早 分类（改善/稳定/退化/无效）与各类面积占比 |
| LST 变化对比 | ![LST 变化图](portfolio/figures/lst_change_map_20241231_vs_20251226.png) ![LST 面积占比](portfolio/figures/lst_area_proportion_20241231_vs_20251226.png) | 20241231 → 20251226：ΔLST 变化分类与各类面积占比 |
| LST-NDVI 联合分析 | ![联合 2025](portfolio/figures/lst_ndvi_scatter_20250430_20250425.png) ![联合 2026](portfolio/figures/lst_ndvi_scatter_20260425_20260510.png) | 近似同期 LST-NDVI 散点与相关性，两年对照看关系年际稳定性 |

## PM10(2.5) 反演验证结果

留一法站点验证：每次去掉一个监测站重新建模，在该站位置预测并与实测对比：

| 监测点编码 | 城市 | 监测值（μg/m³） | 验证模型结果¹（μg/m³） | 误差（μg/m³） | 相对误差（%） |
|---|---|---|---|---|---|
| 1104A | 沈阳 | 83.0 | 80.37 | +2.63 | 3.17 |
| 1317A | 郑州 | 109.0 | 102.58 | +6.42 | 5.89 |
| 1465A | 西安 | 98.0 | 105.20 | -7.20 | 7.35 |
| 2405A | 南阳 | 92.0 | 100.05 | -8.05 | 8.75 |
| 3005A | 徐州 | 54.0 | 66.15 | -12.15 | 22.52 |
| 3635A | 洛阳 | 138.0 | 118.01 | +19.99 | 14.48 |

¹ 验证模型结果为去掉该监测站重新建立模型在该监测站处得到的结果，用来评估模型准确度

## 反演流程图

| 产品 | 流程图 |
|---|---|
| 土壤相对湿度（Sentinel-1 雷达 + Sentinel-2 光学融合反演） | ![土壤湿度流程](portfolio/figures/doc_soil_flow.png) |
| 水体总磷（Sentinel-2 + Sentinel-3 随机森林反演） | ![总磷流程](portfolio/figures/doc_tp_flow.png) |
| PM10/2.5（二维反演 + 空值填补） | ![PM 流程](portfolio/figures/doc_pm_flow.png) |

## 工程方法

每个产品都以 `legacy` 模式逐字复现旧脚本（与旧产物 bit-exact 对照，max|Δ|=0），
再用 `correct` 模式修复实证缺陷并量化效果——**复刻可证明、修复可量化**。
验证脚本随代码发布（`verify_*.py`），每个产品的模块 docstring 是
"旧脚本 ↔ 工具映射 + 缺陷清单（含旧脚本行号）"的最快复习材料。

**数据口径（诚实工程）**：土壤湿度样本与 PM 预报数据为模拟数据，其评估口径 =
流程证明；站点实测与水体总磷为真实数据但样本有限，R² 仅作机械对照，精度数字
以各验证小节的统计口径为准。

## 目录结构

```
rs-agent-portfolio/
├── ndvi_agent/               # NDVI + LST 双域联合 Agent
│   ├── ndvi_tools.py         # NDVI 统计与覆盖分级
│   ├── pipeline_tools.py     # NDVI 多时相管线 7 工具
│   ├── ndvi_agent.py         # 8 工具 NDVI Agent
│   ├── lst_tools.py          # LST 反演/裁剪/统计/对比/联合分析 6 工具
│   ├── rs_agent.py           # 14 工具 NDVI+LST 合并 Agent
│   ├── ask_agent.py          # 问题文件驱动 NDVI Agent
│   ├── ask_rs_agent.py       # 问题文件驱动合并 Agent
│   ├── verify_lst_tools.py   # LST 工具链验证（V1.1-V1.10）
│   ├── verify_crossyear.py   # 跨年同季节验证（V3）
│   ├── run_ndvi_crossyear.py # 跨年 NDVI 链驱动 + 联合分析（V4）
│   └── smoke_one_tile.py     # 单景冒烟
├── fire_agent/               # 火点域
│   ├── fire_tools.py         # 火点反演链 5 工具
│   └── verify_fire_tools.py  # 验证（V-F1~V-F9，86 项断言）
├── soil_agent/               # 土壤湿度域
│   ├── soil_tools.py         # 土壤相对湿度反演链 9 工具
│   ├── verify_soil_tools.py  # 验证（V-S1~V-S10）
│   └── graphs/               # SNAP 图资产
├── pm_agent/                 # 大气污染物域
│   ├── pm_tools.py           # PM10(2.5) 反演链 10 工具
│   └── verify_pm_tools.py    # 验证（V-P1~V-P11）
├── pwater_agent/             # 水体磷域
│   ├── pw_tools.py           # 水体总磷反演链 10 工具
│   ├── verify_pw_tools.py    # 验证（V-PW1~V-PW14）
│   └── graphs/               # SNAP 图资产
└── portfolio/                # 成果图件
    └── figures/              # 成果图、对比分析与流程图
```

## 环境

- Python 3.9（GDAL 3.6.2 / numpy / rasterio / matplotlib / openai / python-dotenv /
  xarray / scikit-learn / geopandas）
- SNAP 13（`gpt.exe`，可直接读取 .SAFE.zip 压缩包）
- Windows 运行注意：需 `PYTHONUTF8=1`（中文路径与输出的编码）；建议使用
  conda 虚拟环境的 python 全路径运行，避免裸 `python` 命中商店占位程序

## 运行与验证

每个产品目录的 `verify_*.py` 为全量验证脚本（幂等：产物已存在时短路跳过）：

```bash
PYTHONUTF8=1 python ndvi_agent/verify_lst_tools.py
PYTHONUTF8=1 python ndvi_agent/verify_crossyear.py
PYTHONUTF8=1 python fire_agent/verify_fire_tools.py
PYTHONUTF8=1 python soil_agent/verify_soil_tools.py
PYTHONUTF8=1 python pm_agent/verify_pm_tools.py
PYTHONUTF8=1 python pwater_agent/verify_pw_tools.py
```

Agent 交互（把问题写进 UTF-8 文本文件，规避命令行中文编码问题）：

```bash
PYTHONUTF8=1 python ndvi_agent/ask_rs_agent.py question.txt
```

## 数据说明

影像与产物为本地数据资产，未随仓库发布；验证脚本依赖的数据路径在各脚本顶部
常量区声明，运行前需自备对应数据（哨兵2 .SAFE.zip、Landsat L2SP、Himawari-9 NC、
站点实测 CSV、SNAP 图资产等）。

## 已知局限

- LST 云掩膜为启发式（QA_PIXEL<2 或 >22000 置 NaN），可能残留薄云/漏检云影
- LST↔NDVI 联合分析存在物候差，相关关系只作近似同期解读，不推断因果
- 火点阈值为人工对照 NASA FIRMS 调参所得，精度未经独立验证，存在误检/漏检可能
- 土壤湿度样本为随机模拟数据：交叉验证 R²≈0 是随机目标的预期天花板而非模型缺陷，
  评估口径=跑通反演流程证明；样本点 R² 属样本内回代（高值=过拟合），仅作
  bug 修复方向的机械对照
- 水体总磷建模样本仅 45 个（3 站有效），测试 R²=0.3157 只做机械对照；
  训练特征为 GEE 点提取、反演输入为 SNAP 本地链重采样，两链口径差异为
  结构性风险（已如实注明）
- PM 预报数据为模拟数据；随机森林 1km / 垂直廓线 2km 分辨率较粗，
  算力支持可做大范围精细反演
- Agent 纪律：同一工具连续失败 3 次即停止并如实说明原因
