"""
用问题文件驱动 Agent（避免每次临时改 ndvi_agent.py 的 QUESTION，
也规避命令行直接传中文在 Windows 下的编码坑）。

运行方式：
    PYTHONUTF8=1 python ask_agent.py question.txt
    （question.txt 为 UTF-8 文本，第一行起即完整问题内容）
"""
import sys

from ndvi_agent import run_agent

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法：python ask_agent.py <问题文件.txt>")
        sys.exit(1)
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        question = f.read().strip()
    if not question:
        print("问题文件为空")
        sys.exit(1)

    print("NDVI 智能分析 Agent 启动")
    print(f"用户需求：{question}\n")

    answer = run_agent(question)

    print(f"\n{'=' * 60}")
    print("Agent 最终回复：")
    print(f"{'=' * 60}")
    print(answer)
