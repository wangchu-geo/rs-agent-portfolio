"""
NDVI + LST 双工具联合智能分析 Agent（14 个工具）

升级内容（相对 ndvi_agent.py 的 8 工具）：
    1. 合并 NDVI 域（8 工具，从 ndvi_agent 单源导入 schema，零漂移）
       与 LST 域（6 工具，来自 lst_tools.py，旧脚本 E:/YYR/LST/LST2.py 的改造）
    2. 模型根据用户问题自主判断走哪个域：NDVI 工作流 / LST 工作流 / 联合分析
    3. 联合分析：LST-NDVI 相关分析（analyze_lst_ndvi）+ 两期 ΔLST 对比（compare_lst_dates）

ndvi_agent.py 原样保留（回归基准，ask_agent.py 继续可用）。

运行方式：
    PYTHONUTF8=1 python rs_agent.py
    （修改下方 QUESTION 变量即可换需求；或用 ask_rs_agent.py <问题文件.txt>）
"""
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

# NDVI 域：schema 与函数映射从 ndvi_agent 单源导入（ndvi_agent 有 __main__ 保护，import 安全）
from ndvi_agent import tools as NDVI_TOOLS
from ndvi_agent import FUNCTION_MAP as NDVI_FUNCTION_MAP
from lst_tools import (
    analyze_lst_ndvi,
    calculate_lst_stats,
    compare_lst_dates,
    list_available_lst_scenes,
    lst_clip_date_to_aoi,
    lst_invert_date,
)

# .env 在上一级目录（E:\ai-learning\.env），无论从哪里运行都能找到
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ============================================================
# 第一步：LST 工具 schema（NDVI 8 个从 ndvi_agent 单源导入）
# ============================================================
LST_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_available_lst_scenes",
            "description": (
                "扫描本地目录，从 Landsat Collection 2 L2SP 影像 tar 文件名"
                "（如 LC09_L2SP_120036_20251226_20251227_02_T1.tar）解析获取日期"
                "（取第一个8位日期），按日期分组返回各日期影像路径列表。"
                "注意：本工具只识别 *.tar 文件，哨兵2 .SAFE.zip 请用 list_available_scenes。"
                "这是LST处理链的第一步。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "data_dir": {
                        "type": "string",
                        "description": "Landsat tar 影像存放目录，例如 E:/YYR/LST/multi_date"
                    }
                },
                "required": ["data_dir"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lst_invert_date",
            "description": (
                "对指定日期的一批 Landsat L2SP tar 执行：解压 → ST_B10 波段线性换算"
                "（K→℃）→ QA_PIXEL 启发式云掩膜（QA<2 或 QA>22000 置 NaN，"
                "与历史产品口径一致），输出浮点地表温度产品 LST_<日期>.tif（nodata=NaN）。"
                "一次调用处理该日期全部景，单景约2-4分钟（首次含解压），请耐心等待。"
                "已存在的产品自动跳过（幂等，可安全重跑）；force=true 时无视已存在产品、重新执行反演。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "scene_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "该日期全部景的 tar 路径列表，直接使用 list_available_lst_scenes 返回的 scenes_by_date[日期]"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "LST产品输出目录，如 E:/YYR/LST/multi_date/work/20251226/lst"
                    },
                    "extract_dir": {
                        "type": "string",
                        "description": "tar解压目录（可选），如 E:/YYR/LST/multi_date/work/20251226/extract；缺省为 output_dir 同级的 extract"
                    },
                    "force": {
                        "type": "boolean",
                        "description": "true=即使反演产品已存在也重新执行反演（用户明确选择重新跑时传true）；缺省false（幂等跳过）"
                    }
                },
                "required": ["scene_paths", "output_dir"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lst_clip_date_to_aoi",
            "description": (
                "用研究区矢量裁剪整景LST产品至研究区范围（自动处理坐标系不一致），"
                "背景置-9999，输出裁剪产品（供统计、对比与联合分析使用）。"
                "LST 为 30m 分辨率，与 NDVI 的裁剪工具是两套工具，不要混用。"
                "已存在的裁剪产品自动跳过（幂等）；force=true 时重新裁剪覆盖。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lst_tif": {
                        "type": "string",
                        "description": "lst_invert_date 输出的整景LST产品（LST_<日期>.tif）完整路径"
                    },
                    "shp_path": {
                        "type": "string",
                        "description": "研究区矢量文件，如 E:/YYR/NDVI/vector/1/1.shp（与NDVI共用同一矢量）"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "裁剪输出目录，如 E:/YYR/LST/multi_date/work/20251226/clip"
                    },
                    "force": {
                        "type": "boolean",
                        "description": "true=即使裁剪产品已存在也重新裁剪（用户明确选择重新跑时传true）；缺省false（幂等跳过）"
                    }
                },
                "required": ["lst_tif", "shp_path", "output_dir"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_lst_stats",
            "description": (
                "读取本地LST栅格（.tif，已完成反演与裁剪的产品），计算均值、最大值、"
                "最小值、标准差、有效像元占比（单位℃），以及各温度等级的面积占比："
                ">=30℃炎热 / 20-30℃暖热 / 10-20℃温和 / 0-10℃寒冷 / <0℃低温。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lst_path": {
                        "type": "string",
                        "description": "LST栅格文件的完整路径（整景产品或裁剪产品均可）"
                    },
                    "region_name": {
                        "type": "string",
                        "description": "研究区名称，例如 南昌市（可选）"
                    }
                },
                "required": ["lst_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_lst_dates",
            "description": (
                "两期LST裁剪产品对比（date1=较早，date2=较晚，Δ=晚−早，单位℃）："
                "将较早日期重采样到较晚日期的网格，逐像元求差并分类"
                "（Δ>=3升温，|Δ|<3稳定，Δ<=-3降温，任一日期无有效值为无效），"
                "输出差值GeoTIFF、分类GeoTIFF、变化分类图PNG与面积占比图PNG，"
                "并返回各类像元数、面积（公顷，30m像元=0.09公顷）与占比统计。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date1_path": {
                        "type": "string",
                        "description": "较早日期的裁剪LST结果路径（如 20251226 的 clip tif）"
                    },
                    "date2_path": {
                        "type": "string",
                        "description": "较晚日期的裁剪LST结果路径（如 20260425 的 clip tif）"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "对比结果输出目录，如 E:/YYR/LST/multi_date/compare"
                    },
                    "threshold": {
                        "type": "number",
                        "description": "升温/降温分类阈值（℃），默认3.0"
                    }
                },
                "required": ["date1_path", "date2_path", "output_dir"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_lst_ndvi",
            "description": (
                "LST-NDVI 联合相关分析：把NDVI 10m产品用平均聚合重采样到LST 30m的"
                "精确网格（3×3窗口，nodata不参与均值），统一无效像元与物理范围后，"
                "计算有效像元对的Pearson相关系数与线性回归（x=NDVI，y=LST℃），"
                "输出密度散点图PNG（hexbin+回归线）与30m对齐NDVI副产品。"
                "输入均为已完成反演/裁剪的本地产品路径。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lst_path": {
                        "type": "string",
                        "description": "LST栅格文件路径（裁剪产品或整景产品）"
                    },
                    "ndvi_path": {
                        "type": "string",
                        "description": "NDVI栅格文件路径（裁剪产品或整景产品，10m或任意分辨率均可）"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "联合分析输出目录，如 E:/YYR/LST/multi_date/joint"
                    }
                },
                "required": ["lst_path", "ndvi_path", "output_dir"]
            }
        }
    },
]

tools = NDVI_TOOLS + LST_TOOLS   # 14 个工具

FUNCTION_MAP = dict(NDVI_FUNCTION_MAP)
FUNCTION_MAP.update({
    "list_available_lst_scenes": list_available_lst_scenes,
    "lst_invert_date": lst_invert_date,
    "lst_clip_date_to_aoi": lst_clip_date_to_aoi,
    "calculate_lst_stats": calculate_lst_stats,
    "compare_lst_dates": compare_lst_dates,
    "analyze_lst_ndvi": analyze_lst_ndvi,
})

# ============================================================
# 第二步：合并版系统提示词 —— 遥感专业知识的落点
# ============================================================
SYSTEM_PROMPT = """你是一名拥有10年遥感数据分析经验的高级工程师，精通哨兵2号（Sentinel-2）与Landsat影像处理、NDVI与地表温度（LST）反演、植被与热环境时序分析。你的分析严谨客观，会主动识别数据质量问题并在结论中明确指出局限。

你有14个工具，分两个域，先判断用户问题属于哪个域：
A. NDVI域（8个）：list_available_scenes → snap_preprocess_date → convert_date_to_tif → apply_cloud_mask_date → mosaic_date → clip_date_to_aoi，统计 calculate_ndvi_stats，对比 compare_ndvi_dates。适用 .SAFE.zip 哨兵2影像。
B. LST域（6个）：list_available_lst_scenes → lst_invert_date → lst_clip_date_to_aoi，统计 calculate_lst_stats，对比 compare_lst_dates，联合分析 analyze_lst_ndvi。适用 *.tar Landsat影像。

NDVI工作流：
1. list_available_scenes 获取日期与影像；影像位于 E:/YYR/NDVI/multi_date/（.SAFE.zip，SNAP直接读取无需解压）
2. 每日期依次：snap_preprocess_date（XML流程）→ convert_date_to_tif → apply_cloud_mask_date → mosaic_date → clip_date_to_aoi；XML：E:/YYR/NDVI/NDVI_COLUD_MASK.xml；矢量：E:/YYR/NDVI/vector/1/1.shp
3. 两期裁剪品均已就绪：E:/YYR/NDVI/multi_date/work/20251228/clip/NDVI_20251228_clip.tif 与 E:/YYR/NDVI/multi_date/work/20260510/clip/NDVI_20260510_clip.tif（10m，nodata=-9999）。除非用户明确要求，严禁重跑NDVI处理链，直接复用

LST工作流：
1. list_available_lst_scenes 获取日期；影像位于 E:/YYR/LST/multi_date/（*.tar，需解压）；tar文件名中第一个8位日期是获取日期（如 LC09_L2SP_120036_20251226_20251227_02_T1.tar → 20251226）
2. 每日期：lst_invert_date（解压+反演+云掩膜）→ lst_clip_date_to_aoi（裁剪；研究区完全位于单景内，无需镶嵌）
3. 云掩膜为启发式（QA<2或>22000置NaN），与历史产品口径一致，可能残留薄云/漏检云影——解读时必须把该局限纳入考虑
4. 两期 LST 均裁剪完成后，用 compare_lst_dates 做 ΔLST 对比

联合分析：
- analyze_lst_ndvi 一次调用处理一对（LST, NDVI）产品。当前数据有两期各一景：LST 20251226 与 20260425，NDVI 20251228 与 20260510
- 配对约定：20251226(LST)↔20251228(NDVI) 差2天，近似同期；20260425(LST)↔20260510(NDVI) 差15天，存在物候差异，解读时必须声明这一局限
- 本任务只做相关分析与ΔLST对比，不做热岛等级制图

路径约定：
- LST工作树：E:/YYR/LST/multi_date/work/<YYYYMMDD>/{extract,lst,clip}/；LST对比：E:/YYR/LST/multi_date/compare/；联合分析：E:/YYR/LST/multi_date/joint/
- 旧目录只读，严禁读取或修改：E:/YYR/LST/result、E:/YYR/LST/data、E:/YYR/LST/temp、E:/YYR/NDVI/yuchuli、E:/YYR/NDVI/tif、E:/YYR/NDVI/quyun、E:/YYR/NDVI/result

关键阈值：
- LST温度分级：>=30℃炎热；20-30℃暖热；10-20℃温和；0-10℃寒冷；<0℃低温
- ΔLST（晚−早）：>=+3℃升温；|Δ|<3℃稳定；<=-3℃降温；任一日期无有效值（云/影/背景）为无效
- NDVI覆盖等级：<0.1裸地或建筑用地；0.1-0.3稀疏植被；0.3-0.6中等密度植被；>=0.6茂密植被
- ΔNDVI（晚−早）：>=+0.1改善；-0.1<Δ<0.1稳定；<=-0.1退化

解读纪律：
- 20260425→20260510 与 20251226→20251228 均为冬季→春季方向，地表温度升高属正常季节变化；若出现反常模式（如大范围降温），如实报告而不是强行圆场
- 跨年同季节 LST 对比（如 20241231 vs 20251226、20250430 vs 20260425）的 Δ 受两日瞬时天气差异影响：若 Δ 空间均匀（Δstd 小）且同季节均值差在 ±8℃ 内，解读为天气主导而非地表变化，不得宣称"变冷/变热趋势"；剥离天气需多景平均
- analyze_lst_ndvi 的 r<0 常见解释是植被蒸散降温（NDVI越高地表越凉）；相关性不构成因果结论；两期影像存在日期差时 r 只能作为近似同期关系解读
- 结论中优先引用工具返回的具体数字，简洁专业（不超过400字）

工具调用纪律：
- 处理类工具一次调用处理一个日期的全部景，单景耗时2-6分钟，请耐心等待返回结果，不要因等待而重复调用（工具幂等，重复调用会自动跳过）
- 工具返回error时，先对照场景发现工具的结果检查路径是否正确，修正后重试；同一工具连续失败3次则停止处理并向用户如实说明原因
- 回答中优先引用工具返回的具体数字，简洁专业

任务澄清规则（重要）：
- 当用户任务涉及 LST 产品（lst_invert_date、lst_clip_date_to_aoi、calculate_lst_stats、compare_lst_dates、analyze_lst_ndvi），且用户没有明确说明"复用现成产品"还是"从反演开始重新生产"时，第一轮不要调用任何工具，先反问一次让用户选择：
  "本地已有就绪的 LST 产品。请选择：A. 复用现成产品直接分析（快）；B. 从反演开始完整重新生产一遍（每景约 2-4 分钟）。"
- 用户明确说了"复用已就绪产品/直接复用/严禁重跑"或"从反演开始/重新跑一遍"时，不反问，直接执行
- 用户指定了具体文件路径时视为数据来源已明确，不反问
- 用户选择 B 时，调用 lst_invert_date 和 lst_clip_date_to_aoi 都要传 force=true，确实重新执行反演与裁剪"""


# ============================================================
# 第三步：Tool Use 完整循环（与 ndvi_agent 同构）
# ============================================================
def run_agent(user_question: str, verbose: bool = True, messages: list = None,
              return_history: bool = False):
    """执行一轮 Agent 对话。

    传入 messages 时沿用历史（多轮对话：Agent 反问后用户继续回复），并在原地更新；
    return_history=True 时返回 (answer, messages)，便于调用方继续下一轮。
    """
    if messages is None:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_question},
        ]
    else:
        messages.append({"role": "user", "content": user_question})

    round_num = 0
    while True:
        round_num += 1
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"第 {round_num} 轮请求：")
            print(f"{'=' * 60}")

        response = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=messages,
            tools=tools,
            extra_body={"thinking": {"type": "disabled"}}
        )
        msg = response.choices[0].message

        # 模型没有请求工具 → 说明它已经能直接回答了，循环结束
        if not msg.tool_calls:
            if verbose:
                print("模型未调用工具，直接给出回复")
            # 回复也记入历史（多轮对话时模型需要记得自己反问过什么）
            messages.append(msg)
            if return_history:
                return msg.content, messages
            return msg.content

        # 把模型的 tool_call 请求加入对话历史
        messages.append(msg)

        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            print(f"\n>> 模型请求调用工具：{func_name}")
            print(f">> 传入参数：{json.dumps(func_args, ensure_ascii=False, indent=2)}")

            # 真正执行函数（把工具返回的 dict 转成 json 字符串还给模型）
            if func_name not in FUNCTION_MAP:
                result = {"error": f"未知工具：{func_name}"}
            else:
                try:
                    result = FUNCTION_MAP[func_name](**func_args)
                except Exception as e:
                    # 报错信息也要还给模型，让它自己解释或调整参数重试
                    result = {"error": str(e)}

            if verbose:
                print(f">> 工具返回：{json.dumps(result, ensure_ascii=False, indent=2)[:800]}...")

            # 把工具执行结果加入对话历史
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, ensure_ascii=False)
            })


if __name__ == "__main__":
    # ============================================================
    # 测试入口：改这里的问题即可测试不同需求
    # ============================================================
    QUESTION = (
        "请对 E:/YYR/LST/multi_date 目录下的两期 Landsat 影像（20251226 与 20260425）"
        "执行地表温度反演、两期对比与 LST-NDVI 联合分析：\n"
        "1. 调用 list_available_lst_scenes 列出可用日期与影像\n"
        "2. 对两个日期分别执行 lst_invert_date（解压目录放 work/<日期>/extract，"
        "产品输出到 work/<日期>/lst）与 lst_clip_date_to_aoi（输出到 work/<日期>/clip，"
        "矢量 E:/YYR/NDVI/vector/1/1.shp），并对每个日期的裁剪结果调用 calculate_lst_stats\n"
        "3. 调用 compare_lst_dates 对比两期 LST（date1_path=20251226裁剪结果，"
        "date2_path=20260425裁剪结果，threshold=3.0），输出到 E:/YYR/LST/multi_date/compare\n"
        "4. 调用 analyze_lst_ndvi 两次做联合相关分析，输出到 E:/YYR/LST/multi_date/joint：\n"
        "   a) LST 20251226 裁剪结果 + NDVI 20251228 裁剪结果（E:/YYR/NDVI/multi_date/work/20251228/clip/NDVI_20251228_clip.tif）\n"
        "   b) LST 20260425 裁剪结果 + NDVI 20260510 裁剪结果（E:/YYR/NDVI/multi_date/work/20260510/clip/NDVI_20260510_clip.tif）\n"
        "   NDVI 裁剪产品已就绪，直接复用，严禁重跑NDVI处理链\n"
        "5. 给出中文专业解读（不超过400字）：两期地表温度水平与季节升温幅度、"
        "升温/稳定/降温面积构成、LST-NDVI 相关关系与可能机制，"
        "并声明两项数据局限（QA启发式漏云、4月对存在15天物候差）"
    )

    print("遥感智能分析 Agent（NDVI+LST 双工具）启动")
    print(f"用户需求：{QUESTION}\n")

    answer = run_agent(QUESTION)

    print(f"\n{'=' * 60}")
    print("Agent 最终回复：")
    print(f"{'=' * 60}")
    print(answer)
