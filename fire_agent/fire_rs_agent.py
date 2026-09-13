"""
fire_rs_agent.py —— 火点域接入 LLM 编排（5 工具）

把 fire_tools.py 的 5 个工具以 Function Calling schema 暴露给 DeepSeek，
模型根据用户自然语言需求自主调用：场景发现 → 波段提取 → 云掩膜 →
上下文火点识别 → 火点统计，并输出中文专业解读。

与其他域的关系：本文件只编排火点域；NDVI+LST 双域见 ndvi_agent/rs_agent.py。
如需多域合并，可仿 rs_agent.py 的 tools = NDVI_TOOLS + LST_TOOLS 模式拼合。

运行方式：
    PYTHONUTF8=1 python fire_rs_agent.py
    （修改下方 QUESTION 变量即可换需求；或用交互式入口 ask_fire_demo.py）
"""
import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from fire_tools import (
    apply_fire_cloud_mask,
    calculate_fire_stats,
    generate_fire_mask,
    himawari_extract_bands,
    list_available_himawari_scenes,
)

# .env 在上一级目录（E:\ai-learning\.env），无论从哪里运行都能找到
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ============================================================
# 第一步：火点工具 schema
# ============================================================
FIRE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_available_himawari_scenes",
            "description": (
                "扫描本地目录，从 Himawari-9 NC 文件名（NC_H09_<日期8位>_<UTC时次4位>_"
                "<产品>_FLDK.<网格>_<网格>.nc）解析时次（YYYYMMDD_HHMM），按槽位配对 "
                "R21 辐射产品与 L2CLP010 云产品，返回各时次的 r21_path/clp_path。"
                "这是火点处理链的第一步。只识别 NC_H09_*.nc，解析失败的进 unparsed，不抛异常。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "data_dir": {
                        "type": "string",
                        "description": "Himawari NC 影像存放目录，例如 E:/YYR/fire/Himawari"
                    }
                },
                "required": ["data_dir"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "himawari_extract_bands",
            "description": (
                "提取指定时次的 R21 波段与 L2CLP010 CLTYPE 云类型，用 china2 全国矢量"
                "掩膜后写入场景目录 output_dir/NC_H09_<时次>/。"
                "量纲：tbb 直接取 xarray 解码的开尔文 K 值并过物理护栏。"
                "bands：请求的通道号（1-6 可见光做 cos(SOZ) 归一化；7-16 热红外取 K），"
                "火点链固定用 (7,14)。"
                "幂等：产物已存在 → status='exists' 跳过，可安全重跑。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "r21_path": {
                        "type": "string",
                        "description": "R21 辐射产品 NC 路径，直接使用 list_available_himawari_scenes 返回的 r21_path"
                    },
                    "clp_path": {
                        "type": "string",
                        "description": "L2CLP010 云产品 NC 路径，直接使用 list_available_himawari_scenes 返回的 clp_path"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "场景目录的父目录，如 E:/YYR/fire/work（场景目录自动命名为 NC_H09_<时次>）"
                    },
                    "shp_path": {
                        "type": "string",
                        "description": "掩膜矢量，如 E:/YYR/fire/data/china2/china2.shp"
                    },
                    "bands": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "要提取的通道号列表，火点链固定 [7,14]（缺省即 [7,14]）"
                    }
                },
                "required": ["r21_path", "clp_path", "output_dir", "shp_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_fire_cloud_mask",
            "description": (
                "对场景目录中的 B07/B14 做云掩膜与小斑块清理。云条件：CLTYPE>0 或 NaN → 云；"
                "CLTYPE 先 bilinear 重采样到 B07 网格（保真口径，勿改 nearest）；"
                "掩膜后再去除 <12 像元的离散小斑块，输出 masked 与 cleaned 两级产物。"
                "幂等：4 个产物（masked×2 + cleaned×2）齐全 → 全部 exists 跳过。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "scene_dir": {
                        "type": "string",
                        "description": "himawari_extract_bands 输出的场景目录，如 E:/YYR/fire/work/NC_H09_20250415_0200"
                    }
                },
                "required": ["scene_dir"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_fire_mask",
            "description": (
                "上下文火点识别：四条件（b07/b14 为云掩膜清理后的开尔文亮温，diff=b07-b14 场景统计）："
                "c1: b07-b14>diff_mean; c2: b07>b07_mean+2.8*diff_std; "
                "c3: b07-b14>diff_mean+2.5*diff_std; c4: b14>b14_mean+2*b14_std；"
                "火点 = ((c1&c2)|c3|c4) & 林地(landuse 20-25) & valid_mask，"
                "输出 uint8 0/1 掩膜（nodata=0）。"
                "注意：阈值口径为人工对照 NASA FIRMS 火点监测网站调参所得，"
                "精度未经独立验证，火点计数存在误检/漏检可能。"
                "幂等：H09_fire_point.tif 存在 → exists。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "scene_dir": {
                        "type": "string",
                        "description": "完成云掩膜的场景目录，如 E:/YYR/fire/work/NC_H09_20250415_0200"
                    },
                    "landuse_path": {
                        "type": "string",
                        "description": "土地利用栅格，如 E:/YYR/fire/data/landuse.tif（林地类条件 landuse 20-25）"
                    }
                },
                "required": ["scene_dir", "landuse_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_fire_stats",
            "description": (
                "火点统计（对齐 calculate_lst_stats 风格）：火点像素数/占比、"
                "逐行纬度 cos 精确面积（像元面积 = |gt.a|·111.32km·cos(lat_row) × "
                "|gt.e|·111.32km，逐行计算）、bbox、连通簇数与 top-5 簇面积。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fire_tif": {
                        "type": "string",
                        "description": "generate_fire_mask 输出的火点掩膜，如 E:/YYR/fire/work/NC_H09_20250415_0200/H09_fire_point.tif"
                    },
                    "region_name": {
                        "type": "string",
                        "description": "区域名称，例如 全国（china2 掩膜范围），可选"
                    }
                },
                "required": ["fire_tif"]
            }
        }
    },
]

tools = FIRE_TOOLS

FUNCTION_MAP = {
    "list_available_himawari_scenes": list_available_himawari_scenes,
    "himawari_extract_bands": himawari_extract_bands,
    "apply_fire_cloud_mask": apply_fire_cloud_mask,
    "generate_fire_mask": generate_fire_mask,
    "calculate_fire_stats": calculate_fire_stats,
}

# ============================================================
# 第二步：火点域系统提示词 —— 遥感专业知识的落点
# ============================================================
SYSTEM_PROMPT = """你是一名拥有10年遥感数据分析经验的高级工程师，精通 Himawari-9 静止气象卫星热红外火点识别、云掩膜与森林火灾监测。你的分析严谨客观，会主动识别数据质量问题并在结论中明确指出局限。

你有5个工具（火点处理链）：
1. list_available_himawari_scenes 场景发现（第一步）
2. himawari_extract_bands 波段与云类型提取
3. apply_fire_cloud_mask 云掩膜与小斑块清理
4. generate_fire_mask 上下文火点识别
5. calculate_fire_stats 火点统计

数据与路径约定：
- 源数据：E:/YYR/fire/Himawari/，当前有 20250415 三个时次（0200/0300/0400 UTC）的 R21+L2CLP010 配对 NC
- 工作树：E:/YYR/fire/work/NC_H09_<YYYYMMDD_HHMM>/（提取/掩膜/火点产物全部落这里）
- 矢量：E:/YYR/fire/data/china2/china2.shp；土地利用：E:/YYR/fire/data/landuse.tif
- 旧目录只读，严禁读取或修改：E:/YYR/fire/result、E:/YYR/fire/脚本

火点工作流（每个时次）：
1. list_available_himawari_scenes(data_dir=E:/YYR/fire/Himawari) 获取各时次 r21_path/clp_path
2. himawari_extract_bands(r21_path, clp_path, output_dir=E:/YYR/fire/work, shp_path=E:/YYR/fire/data/china2/china2.shp, bands=(7,14))
3. apply_fire_cloud_mask(scene_dir=E:/YYR/fire/work/NC_H09_<时次>)
4. generate_fire_mask(scene_dir=同上, landuse_path=E:/YYR/fire/data/landuse.tif)
5. calculate_fire_stats(fire_tif=<scene_dir>/H09_fire_point.tif)

幂等与复用：
- 全部工具幂等：产物已存在自动跳过（exists）。三个时次的产物当前均已就绪，除非用户明确要求重新计算，直接复用；工具没有 force 参数，任何时候都不要建议或执行删除产物
- 严禁改动 E:/YYR/fire/work 下已有产物文件

解读纪律（诚实口径，必须遵守）：
- 火点四条件阈值是人工对照 NASA FIRMS 火点监测网站调参所得（旧文档所称"遗传算法"未在代码中实现），精度未经独立验证，火点计数存在误检/漏检可能——解读必须声明这一点
- 检测仅限林地类（landuse 20-25），非林地火点不在检测范围内
- 云掩膜口径 CLTYPE>0 或 NaN 置云：云下火点不可见，各时次火点数为晴空条件下的估计
- Himawari 热红外为 2km 级分辨率、10 分钟级重访：适合监测火点时空演变趋势，不适合像元级精确计数；单时次火点数与 NASA FIRMS（VIIRS/MODIS 400m~1km）不可直接比绝对值，趋势与相对变化更有意义
- tbb 为解码开尔文亮温（K），物理护栏已校验值域
- 三个时次 0200/0300/0400 UTC（即北京 10/11/12 时）为同日上午的逐时次演变：火点数的时次变化只反映监测时段内的火情动态，不得外推为整日趋势
- 结论中优先引用工具返回的具体数字，简洁专业（不超过400字）

工具调用纪律：
- 处理类工具单时次提取约数分钟，请耐心等待返回结果，不要因等待而重复调用（工具幂等，重复调用会自动跳过）
- 工具返回error时，先对照场景发现工具的结果检查路径是否正确，修正后重试；同一工具连续失败3次则停止处理并向用户如实说明原因
- 回答中优先引用工具返回的具体数字，简洁专业"""


# ============================================================
# 第三步：Tool Use 完整循环（与 ndvi_agent/rs_agent 同构）
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
        "请对 E:/YYR/fire/Himawari 下 20250415 三个时次的 Himawari-9 数据执行森林火点识别与统计：\n"
        "1. 调用 list_available_himawari_scenes 列出可用时次与影像配对\n"
        "2. 对每个时次依次执行 himawari_extract_bands（输出到 E:/YYR/fire/work，"
        "矢量 E:/YYR/fire/data/china2/china2.shp，bands=(7,14)）、apply_fire_cloud_mask、"
        "generate_fire_mask（landuse=E:/YYR/fire/data/landuse.tif）\n"
        "3. 对每个时次的 H09_fire_point.tif 调用 calculate_fire_stats\n"
        "4. 给出中文专业解读（不超过400字）：三个时次的火点数量与面积变化、空间分布特征，"
        "并声明检测口径局限（人工对照 FIRMS 调参、精度未经独立验证、云下不可见、仅林地）"
    )

    print("遥感智能分析 Agent（Himawari 火点）启动")
    print(f"用户需求：{QUESTION}\n")

    answer = run_agent(QUESTION)

    print(f"\n{'=' * 60}")
    print("Agent 最终回复：")
    print(f"{'=' * 60}")
    print(answer)
