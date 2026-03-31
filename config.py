"""项目配置管理"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


class TengriConfig:
    """讯蒙科技 Tengri API 配置"""
    API_KEY = os.getenv("TENGRI_API_KEY")
    BASE_URL = os.getenv("TENGRI_BASE_URL", "https://platform.heimori.cn/v1")
    MODEL = os.getenv("TENGRI_MODEL", "tengri-2-pro")
    EMBEDDING_MODEL = os.getenv("TENGRI_EMBEDDING_MODEL", "tengri-embedding-v1")
    TIMEOUT = 7200


class RAGConfig:
    """RAG 检索参数"""
    TOP_K = int(os.getenv("RAG_TOP_K", "5"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


class ConcurrencyConfig:
    """并发控制参数 -- 上游平台支持 ~2000 RPM

    评估阶段：JUDGE_MAX_WORKERS 控制同时评审的样本数，每个样本内部
    有 6 次 LLM 调用（2 Prompt × 3 runs），60 并发 × 6 = 360 峰值
    请求远低于 2000 RPM 上限。
    """
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
    EMBEDDING_MAX_WORKERS = int(os.getenv("EMBEDDING_MAX_WORKERS", "6"))
    EVIDENCE_MAX_WORKERS = int(os.getenv("EVIDENCE_MAX_WORKERS", "6"))
    LLM_MAX_WORKERS = int(os.getenv("LLM_MAX_WORKERS", "8"))
    JUDGE_MAX_WORKERS = int(os.getenv("JUDGE_MAX_WORKERS", "60"))
    REPORT_MAX_WORKERS = int(os.getenv("REPORT_MAX_WORKERS", "61"))


INDUSTRIES = {
    "制造业": [
        ("002594", "比亚迪"), ("000333", "美的集团"), ("000651", "格力电器"),
        ("600690", "海尔智家"), ("600031", "三一重工"), ("000157", "中联重科"),
        ("000338", "潍柴动力"), ("600019", "宝钢股份"), ("600585", "海螺水泥"),
        ("600660", "福耀玻璃"),
    ],
    "消费行业": [
        ("600519", "贵州茅台"), ("000858", "五粮液"), ("600887", "伊利股份"),
        ("603288", "海天味业"), ("002304", "洋河股份"), ("600600", "青岛啤酒"),
        ("000895", "双汇发展"), ("601888", "中国中免"), ("002507", "涪陵榨菜"),
        ("000568", "泸州老窖"),
    ],
    "医药行业": [
        ("600276", "恒瑞医药"), ("603259", "药明康德"), ("600436", "片仔癀"),
        ("000538", "云南白药"), ("000963", "华东医药"), ("600196", "复星医药"),
        ("600085", "同仁堂"), ("000661", "长春高新"), ("600763", "通策医疗"),
        ("600079", "人福医药"),
    ],
}

YEARS = list(range(2020, 2025))

CASE_STUDY_COMPANIES = [
    ("002594", "比亚迪", "制造业"),
    ("600031", "三一重工", "制造业"),
    ("600519", "贵州茅台", "消费行业"),
]
