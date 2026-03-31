# MAFS: Multi-Agent Financial Statement Analysis Framework

基于财务模型的多智能体 LLM 年报"计算--解读"一体化分析框架。

本项目为山东管理学院本科毕业论文《金融科技视角下基于财务模型的多智能体LLM年报"计算--解读"一体化框架研究》的实验实现。

## 系统架构

框架由五个专职 Agent 通过 LangGraph 状态图协作完成端到端年报分析：

```
年报 PDF
  │
  ▼
[文档解析 Agent] ─── PDF 结构化提取 + 向量索引构建
  │
  ▼
[指标计算 Agent] ─── 杜邦分析 / Z-Score / 偿债·盈利·营运指标
  │
  ▼
[证据对齐 Agent] ─── RAG 语义检索 + LLM 相关性过滤
  │
  ▼
[解读生成 Agent] ─── 结构化财务分析报告生成
  │
  ▼
[一致性自检 Agent] ─── 规则检查 + LLM 语义审查（未通过则回退重试）
```

## 环境要求

- Python >= 3.10
- 讯蒙科技 Tengri API 访问凭证

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API 密钥
cp .env.example .env
# 编辑 .env，填入 TENGRI_API_KEY

# 3. 运行实验
python run_experiment.py smoke_test   # 联调测试
python run_experiment.py main         # 主实验（30家公司 x 5年）
python run_experiment.py baselines    # 基线对比（规则/单一LLM/通用RAG）
python run_experiment.py evaluate     # 评估（含 LLM-as-Judge 双评）
python run_experiment.py robustness   # 稳健性检验
python run_experiment.py all          # 全部流程
```

## 目录结构

```
.
├── agents/                  # 五个专职 Agent 实现
│   ├── orchestrator.py      # LangGraph 编排器（两阶段并行 + 断点续传）
│   ├── document_parser.py   # 文档解析（LLM 辅助页面识别 + 正则提取 + 会计恒等式校验）
│   ├── indicator_calculator.py  # 指标计算（杜邦 / Z-Score / 偿债·盈利·营运）
│   ├── evidence_aligner.py  # 证据对齐（FAISS 检索 + LLM 相关性验证）
│   ├── interpretation_generator.py  # 解读生成
│   └── consistency_checker.py   # 一致性自检（规则 + LLM 语义审查）
├── baselines/               # 三类基线方法
│   ├── rule_based.py        # 基线1: 纯规则 + 公式计算
│   ├── single_llm.py        # 基线2: 单一 LLM 直接分析
│   └── general_rag.py       # 基线3: 通用 RAG（LangChain + FAISS）
├── evaluation/              # 评估模块
│   └── metrics.py           # 五维评价指标 + LLM-as-Judge 双评 + Cohen's Kappa
├── utils/                   # 工具库
│   ├── financial_formulas.py    # 财务指标计算公式
│   ├── llm_client.py        # LLM API 封装（线程安全 + 指数退避重试）
│   ├── pdf_parser.py        # PDF 解析（pdfplumber）
│   └── vector_store.py      # FAISS 向量存储（批量嵌入 + 并发检索）
├── data/                    # 年报 PDF 数据集（175 份）
│   ├── 制造业/              # 10 家公司 x 5 年
│   ├── 消费行业/            # 10 家公司 x 5 年
│   ├── 医药行业/            # 10 家公司 x 5 年
│   └── 科创板_稳健性/       # 5 家公司 x 5 年（稳健性检验）
├── config.py                # 项目配置（API / RAG 参数 / 样本定义）
├── models.py                # 统一输出数据模型
├── run_experiment.py        # 实验主入口（CLI）
├── requirements.txt         # Python 依赖
└── .env.example             # 环境变量模板
```

## 评价指标

| 指标 | 说明 |
|------|------|
| 计算正确率 | 与确定性公式计算基准的一致程度 |
| 公式验证通过率 | 杜邦分解自洽性等数学校验 |
| 证据对齐率 | 可追溯至年报原文的指标占比 |
| 口径一致性 | 同一公司跨年指标计算口径一致程度 |
| 解读质量 | LLM-as-Judge 双评（CPA + CFA 视角，各 3 次） |
| Cohen's Kappa | 两位评审间的一致性系数 |

## 数据集

实验数据来源于巨潮资讯网公开披露的 A 股上市公司年度报告（2020--2024 年），覆盖制造业、消费行业、医药行业三大行业共 30 家公司，以及科创板 5 家公司用于稳健性检验。

## 引用

如使用本项目代码或数据，请引用：

> 李奕铭. 金融科技视角下基于财务模型的多智能体LLM年报"计算--解读"一体化框架研究[D]. 济南: 山东管理学院, 2025.

## License

[MIT](LICENSE)
