"""集中配置。所有可变项都从 .env 读，代码里不出现 Key。

为什么单独一个文件：
    同一个参数（比如模型名、超时时间）一旦散落在三个文件里，
    改一处忘一处，就会出现「评测用 A 模型、线上跑 B 模型」这种查半天的问题。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent

# override=True 是必须的：默认不会覆盖已存在的系统环境变量，
# 如果 Windows 里残留一个旧的 key，它会静默压过 .env，报 401 却看不出原因。
load_dotenv(ROOT / ".env", override=True)

# ---- 模型 ----
API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
MODEL = os.getenv("MODEL_NAME", "deepseek-chat").strip()

# ---- 调用参数 ----
TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))   # 制度问答要可复现，默认 0
TIMEOUT = float(os.getenv("TIMEOUT_SECONDS", "60"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))

# ---- 数据 ----
CORPUS_DIR = ROOT / "corpus"

# ---- 检索开关 ----
# M1（第 1 阶段）先把全部条款塞进提示词，用来跑通最短链路；
# M2 接入向量检索后改成 "rag"，两条路用同一套评测集对比。
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "full").strip()   # full | rag
TOP_K = int(os.getenv("TOP_K", "8"))


def check() -> None:
    """启动自检：缺东西立刻报清楚，别等调用 API 时才炸。"""
    if not API_KEY:
        raise SystemExit(
            "❌ 未找到 DEEPSEEK_API_KEY\n"
            f"   请复制 {ROOT / '.env.example'} 为 .env，填入 Key"
        )
    if not CORPUS_DIR.exists() or not any(CORPUS_DIR.glob("*.md")):
        raise SystemExit(f"❌ 制度文档目录为空：{CORPUS_DIR}")


def describe() -> str:
    return (f"模型 {MODEL}｜接口 {BASE_URL}｜检索模式 {RETRIEVAL_MODE}"
            f"｜文档目录 {CORPUS_DIR}")
