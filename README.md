# 🎯 智绘纠错——工图智能导学平台

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![Version 2.0.0](https://img.shields.io/badge/Version-2.0.0-orange.svg)](https://github.com/Jlu45/drawing-error-tutor)

基于 GB/T 国家标准的机械工程图纸智能纠错与导学平台。V2 采用 **6阶段流水线** 协作架构，融合 ArtiCAD（CVPR 2025）灵感，引入预检协议、跨阶段回滚、非对称经验库和 VLM 评判四大创新模块，配合 Atlas 工图图谱与增强型强化学习自进化机制。

## 🎬 演示视频

> 📌 演示视频文件较大，请将视频上传至 Bilibili / YouTube 等平台后，在此处替换为视频链接。
>
> 本地视频文件位于 `docs/assets/演示视频.mp4`（需自行放置，未纳入 Git 仓库）

<!-- 替换下方链接为你的视频地址 -->
<!-- [![演示视频](https://img.youtube.com/vi/VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=VIDEO_ID) -->

## ✨ 功能特性

### V2 核心特性

- **6阶段流水线**：Planning → Perception → RuleCheck → LLM → Fusion → Judge，比 V1 的4阶段更精细的分析流程
- **预检协议（Pre-check Protocol）**：执行前生成检测合约，确保每个阶段输入输出可验证
- **跨阶段回滚（Cross-Stage Rollback）**：精准定位错误阶段并回滚重试，避免全流程重跑
- **非对称经验库（Asymmetric Experience Store）**：双分区案例检索——成功案例库与失败案例库分离存储，提升检索精度
- **VLM 评判（VLM Judge）**：4维度质量评估（完整性 / 准确性 / 规范性 / 可读性），替代简单规则评分
- **Atlas 工图图谱**：34 个典型工程案例、24 条制图规则、8 大错误类别，结构化领域知识库
- **增强型 RL 记忆**：4维连续奖励信号（完整性 / 准确性 / 规范性 / 可读性），替代 V1 的离散奖励

### V1 保留特性

- **GB/T 标准合规检测**：依据 GB/T 4457-4460 国家标准，覆盖8大错误类别
- **苏格拉底式导学**：生成启发式引导，帮助学生理解错误原因
- **双知识库**：GB标准知识库 + 背景知识库 + 图像知识库，支持 FAISS 向量检索
- **优雅降级**：LLM API 不可用时自动回退到本地规则引擎
- **CAD 风格界面**：专业工程图纸界面，支持亮色/暗色/护眼三种主题

## 🏗️ 系统架构

### V2 架构（6阶段流水线）

```
DrawingOrchestrator V2（图纸分析协调器）
│
├── Phase 1: Planning（规划阶段）
│   └── 生成检测合约 → Pre-check Protocol
│
├── Phase 2: Perception（感知阶段）[并行]
│   ├── OCRAgent（文字识别）
│   ├── GeometryAgent（几何检测）
│   └── StructureAgent（结构分析）
│
├── Phase 3: RuleCheck（规则校验阶段）
│   └── RuleCheckAgent（GB 标准规则校验，8类规则）
│
├── Phase 4: LLM（大模型分析阶段）
│   └── LLMAgent（Qwen2.5-72B-Instruct 深度分析）
│
├── Phase 5: Fusion（融合阶段）
│   └── 多源结果融合 + Cross-Stage Rollback 检测
│       ├── 成功 → 进入 Judge
│       └── 失败 → 回滚到出错阶段重试（最多3次）
│
└── Phase 6: Judge（评判阶段）
    └── VLM Judge → 4维度质量评估
        ├── 完整性（Completeness）
        ├── 准确性（Accuracy）
        ├── 规范性（Compliance）
        └── 可读性（Readability）
```

### V1 vs V2 对比

| 特性 | V1 | V2 |
|------|----|----|
| **版本** | 1.0.0 | 2.0.0 |
| **流水线阶段** | 4阶段（OCR/Geo/Struct → RuleCheck → LLM） | 6阶段（Planning → Perception → RuleCheck → LLM → Fusion → Judge） |
| **错误类别** | 6类 | 8类（新增：视图错误、表面粗糙度标注错误） |
| **检测合约** | 无 | Pre-check Protocol，执行前验证输入输出 |
| **错误恢复** | 全流程重跑 | Cross-Stage Rollback，精准回滚到出错阶段 |
| **经验检索** | 单一经验池 | Asymmetric Experience Store，成功/失败双分区 |
| **质量评估** | 规则评分 | VLM Judge，4维度连续评估 |
| **RL 奖励** | 离散（confirmed +1 / dismissed -1） | 4维连续奖励（完整性/准确性/规范性/可读性） |
| **领域知识** | GB标准 + 背景知识库 | + Atlas 工图图谱（34案例/24规则/8错误类别） |
| **路由** | `/` | V1: `/`，V2: `/v2` |

## 🚀 快速开始

### 环境要求

- Python 3.9+
- pip

### 安装

```bash
# 克隆仓库
git clone https://github.com/Jlu45/drawing-error-tutor.git
cd drawing-error-tutor

# 运行一键部署脚本（安装依赖 + 创建配置 + 验证环境）
python setup.py
```

### 配置

将 `config.example.py` 复制为 `config.py` 并填写 API 凭据：

```bash
cp config.example.py config.py
```

编辑 `config.py`：

```python
# 必填：LLM API 端点地址
MULTIMODAL_API_URL = 'https://your-api-endpoint.example.com'

# 必填：API 密钥（切勿提交此文件）
MULTIMODAL_API_KEY = 'your-api-key-here'

# 可选：模型配置
LLM_MODEL = 'Qwen2.5-72B-Instruct'
MULTIMODAL_VISION_MODEL = 'your-vision-model-name'

# V2 可选：VLM Judge 配置
VLM_JUDGE_MODEL = 'your-vlm-judge-model-name'
VLM_JUDGE_THRESHOLD = 0.7
```

或使用环境变量：

```bash
export MULTIMODAL_API_URL='https://your-api-endpoint.example.com'
export MULTIMODAL_API_KEY='your-api-key-here'
```

### 运行

```bash
# 启动应用（V1 和 V2 同时运行）
python app.py

# 或使用快速启动脚本
./start.sh      # Linux/Mac
start.bat       # Windows
```

浏览器访问：
- **V1**：http://localhost:5000/
- **V2**：http://localhost:5000/v2

### Docker 部署

```bash
# 使用 Docker Compose 构建并运行
docker-compose up --build

# 或手动构建
docker build -t drawing-error-tutor .
docker run -p 5000:5000 \
  -e MULTIMODAL_API_URL='https://your-api-endpoint.example.com' \
  -e MULTIMODAL_API_KEY='your-api-key' \
  drawing-error-tutor
```

## 📖 API 文档

### V1 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | V1 上传页面 |
| `/upload` | POST | V1 上传并分析图纸 |
| `/uploads/<filename>` | GET | 获取上传文件 |
| `/api/gb_standards?q=<query>` | GET | 搜索 GB 标准 |
| `/api/rl_feedback` | POST | 提交 RL 反馈 |
| `/api/rl_stats` | GET | 获取 RL 记忆统计 |

### V2 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v2` | GET | V2 上传页面 |
| `/v2/upload` | POST | V2 上传并分析图纸（6阶段流水线） |
| `/v2/api/analyze` | POST | V2 完整分析（含 Pre-check + Rollback + Judge） |
| `/v2/api/judge` | POST | VLM Judge 4维度质量评估 |
| `/v2/api/atlas` | GET | 获取 Atlas 工图图谱数据 |
| `/v2/api/atlas/cases` | GET | 获取图谱案例列表（34个） |
| `/v2/api/atlas/rules` | GET | 获取图谱规则列表（24条） |
| `/v2/api/experience` | GET | 获取非对称经验库统计 |
| `/v2/api/experience/search` | POST | 检索相似案例（成功/失败分区） |
| `/v2/api/rl_feedback` | POST | 提交 V2 RL 反馈（4维连续奖励） |
| `/v2/api/rl_stats` | GET | 获取 V2 RL 记忆统计（含4维奖励分布） |

### V2 RL 反馈 API

```bash
curl -X POST http://localhost:5000/v2/api/rl_feedback \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "drawing.png_1700000000",
    "error_description": "缺少直径符号",
    "feedback_type": "confirmed",
    "reward_vector": {
      "completeness": 0.8,
      "accuracy": 0.9,
      "compliance": 0.7,
      "readability": 0.85
    }
  }'
```

有效 `feedback_type` 值：`confirmed`（确认）、`ignored`（忽略）、`dismissed_all`（全部误报）、`partial_confirm`（部分确认）、`useful_guidance`（引导有帮助）

`reward_vector` 为 V2 新增的4维连续奖励信号，每个维度取值范围 [0, 1]。

## 🧪 测试

```bash
# 运行所有测试
pytest

# 带覆盖率报告
pytest --cov=src --cov-report=html

# 运行指定测试模块
pytest tests/test_geometric_detector.py -v

# 仅运行 V2 相关测试
pytest tests/test_v2/ -v
```

## 📁 项目结构

```
drawing-error-tutor/
│
├── 📄 项目根目录文件
│   ├── app.py                          # [入口] Flask Web 应用主程序，V1/V2 路由定义与服务启动
│   ├── config.example.py               # [配置] 配置模板文件，包含所有可配置项的占位符
│   ├── config_loader.py                # [配置] 配置加载器，优先级：config.py > 环境变量 > 默认值
│   ├── setup.py                        # [部署] 一键部署脚本，自动安装依赖/创建配置/验证环境
│   ├── start.bat                       # [部署] Windows 快速启动脚本
│   ├── start.sh                        # [部署] Linux/Mac 快速启动脚本
│   ├── requirements.txt                # [依赖] Python 依赖包清单
│   ├── pyproject.toml                  # [元数据] 项目元信息、构建配置、工具配置（pytest/flake8/mypy）
│   ├── pytest.ini                      # [测试] pytest 测试框架配置
│   ├── .flake8                         # [规范] flake8 代码风格检查配置（行宽120）
│   ├── .prettierrc                     # [规范] Prettier 代码格式化配置
│   ├── .gitignore                      # [安全] Git 忽略规则，保护 config.py/.env 等敏感文件
│   ├── .gitattributes                  # [存储] Git LFS 大文件跟踪规则（视频文件）
│   ├── Dockerfile                      # [容器] Docker 镜像构建文件
│   ├── docker-compose.yml              # [容器] Docker Compose 编排配置
│   ├── skill.json                      # [技能] 技能元数据，描述能力/配置/端点/错误分类
│   ├── SKILL.md                        # [技能] 技能定义入口文件
│   ├── LICENSE                         # [法律] MIT 开源许可协议
│   ├── README.md                       # [文档] 项目说明文档（本文件）
│   └── CHANGELOG.md                    # [文档] 版本变更记录
│
├── 🧠 src/ — 核心源码
│   ├── multi_agent_system.py           # [核心] 多智能体协调器 + 5个专业Agent + 4阶段流水线（V1）
│   │                                   #   - DrawingOrchestrator：V1协调器，管理分析流程
│   │                                   #   - OCRAgent：文字识别（RapidOCR）
│   │                                   #   - GeometryAgent：几何元素检测（OpenCV Hough/Canny）
│   │                                   #   - StructureAgent：图纸结构分析（区域/标题栏/图框）
│   │                                   #   - RuleCheckAgent：GB标准规则校验（8类规则）
│   │                                   #   - LLMAgent：大模型深度分析（Qwen2.5-72B-Instruct）
│   │                                   #   - BaseAgent/ImageCache/PreprocessPipeline 等基础设施
│   ├── connector_contract.py           # [V2核心] 预检协议模块（Pre-check Protocol）
│   │                                   #   - DetectionContract：检测合约定义与验证
│   │                                   #   - ContractValidator：合约执行前校验
│   │                                   #   - 输入/输出契约约束，确保阶段间数据一致性
│   ├── cross_stage_rollback.py         # [V2核心] 跨阶段回滚模块（Cross-Stage Rollback）
│   │                                   #   - RollbackManager：回滚管理器，精准定位错误阶段
│   │                                   #   - StageCheckpoint：阶段检查点，支持状态快照与恢复
│   │                                   #   - 最大回滚次数控制（默认3次）
│   ├── experience_store.py             # [V2核心] 非对称经验库模块（Asymmetric Experience Store）
│   │                                   #   - SuccessPartition：成功案例分区
│   │                                   #   - FailurePartition：失败案例分区
│   │                                   #   - DualRetriever：双分区检索器，加权融合成功/失败案例
│   │                                   #   - FAISS 向量索引，支持相似度检索
│   ├── vlm_judge.py                    # [V2核心] VLM 评判模块（VLM Judge）
│   │                                   #   - VLMJudge：4维度质量评估器
│   │                                   #   - CompletenessScorer：完整性评分
│   │                                   #   - AccuracyScorer：准确性评分
│   │                                   #   - ComplianceScorer：规范性评分
│   │                                   #   - ReadabilityScorer：可读性评分
│   │                                   #   - 评估结果融合与置信度计算
│   ├── atlas/                          # [V2核心] Atlas 工图图谱模块
│   │   ├── __init__.py                 #   - 模块初始化
│   │   ├── atlas_builder.py            #   - 图谱构建器（34案例/24规则/8错误类别）
│   │   ├── case_index.py              #   - 案例索引与检索
│   │   ├── rule_engine.py             #   - 规则引擎，图谱规则推理
│   │   └── error_taxonomy.py          #   - 错误分类体系（8大类别）
│   ├── rl/                             # [V2增强] 强化学习模块
│   │   ├── __init__.py                 #   - 模块初始化
│   │   ├── continuous_reward.py        #   - 4维连续奖励函数（完整性/准确性/规范性/可读性）
│   │   ├── enhanced_dqn.py            #   - 增强型 DQN 网络
│   │   └── policy_adapter.py          #   - 策略适配器，V1→V2 奖励信号转换
│   ├── utils/                          # [工具] 通用工具模块
│   │   ├── __init__.py                 #   - 模块初始化
│   │   ├── logging_utils.py           #   - 日志工具
│   │   └── validation.py              #   - 数据验证工具
│   ├── geometric_detector.py           # [检测] OpenCV 几何元素检测器
│   │                                   #   - 直线/圆/箭头检测
│   │                                   #   - 线型分类（实线/虚线/点画线）
│   │                                   #   - 尺寸标注结构检测
│   │                                   #   - YOLO 格式转换
│   ├── rag_knowledge_base.py           # [知识] 双知识库系统
│   │                                   #   - GB标准知识库（展示源，前端展示用）
│   │                                   #   - 背景知识库（37条，内化到LLM系统提示词）
│   │                                   #   - 图像知识库（HOG特征 + FAISS向量检索）
│   │                                   #   - 支持PDF自动提取、文本/图像知识添加与检索
│   ├── rl_memory_unit.py               # [进化] V1 强化学习记忆单元
│   │                                   #   - MiniDQN：2层神经网络（10→64→15）
│   │                                   #   - ExperienceReplayBuffer：经验回放池（容量500）
│   │                                   #   - PolicyParameters：7个可调策略参数
│   │                                   #   - 15个离散动作（7参数±调整或不变）
│   │                                   #   - 奖励函数：confirmed +1.0 / dismissed_all -1.0
│   ├── multimodal_agent.py             # [多模态] 多模态分析Agent
│   │                                   #   - PaddleOCR/RapidOCR 文字识别
│   │                                   #   - YOLOv8 目标检测（可选）
│   │                                   #   - OpenAI 兼容 API 多模态视觉分析
│   │                                   #   - 视觉/文本/结构特征融合
│   │                                   #   - 苏格拉底式反馈生成
│   ├── error_injection.py              # [工具] 错误注入器，用于生成测试数据
│   ├── process_gb_pdf.py               # [工具] GB标准PDF文件处理器，提取文本并入库
│   ├── process_standard_drawings.py    # [工具] 标准图纸处理器，拆分并添加到图像知识库
│   └── collect_drawings.py             # [工具] 测试图纸生成器，生成示例减速器零件图
│
├── 🎨 templates/ — 前端模板
│   ├── index.html                      # [页面] V1 CAD风格图纸上传页面（亮色/暗色/护眼主题）
│   ├── result.html                     # [页面] V1 分析结果展示页面（错误列表/GB标准/RL反馈）
│   ├── v2_index.html                   # [页面] V2 上传页面（含 Atlas 图谱展示/4维评估面板）
│   └── v2_result.html                  # [页面] V2 分析结果页面（Judge评分/经验案例/回滚日志）
│
├── 🧪 tests/ — 测试套件
│   ├── test_geometric_detector.py      # [单元测试] 几何检测器测试（直线/圆/箭头/线型/YOLO格式）
│   ├── test_rule_check.py              # [单元测试] 规则检查测试（尺寸/公差/标题栏/符号/线型/完整性）
│   ├── test_rl_memory.py               # [单元测试] RL记忆单元测试（DQN/经验池/策略参数/状态提取）
│   ├── test_knowledge_base.py          # [单元测试] 知识库测试（GB标准搜索/背景知识/图像知识）
│   ├── test_config_security.py         # [安全测试] 配置安全检测（无硬编码密钥/无内部URL/gitignore保护）
│   └── test_v2/                        # [V2测试] V2 模块测试
│       ├── test_connector_contract.py  #   - 预检协议测试（合约生成/验证/违约检测）
│       ├── test_cross_stage_rollback.py#   - 跨阶段回滚测试（检查点/回滚/最大重试）
│       ├── test_experience_store.py    #   - 非对称经验库测试（双分区存储/检索/融合）
│       ├── test_vlm_judge.py           #   - VLM评判测试（4维评分/置信度/结果融合）
│       ├── test_atlas.py               #   - Atlas图谱测试（案例检索/规则推理/错误分类）
│       └── test_continuous_reward.py   #   - 连续奖励测试（4维奖励/策略适配）
│
├── 📚 docs/ — 文档
│   ├── api.md                          # [文档] REST API + 内部 API 完整文档（V1 & V2）
│   ├── development.md                  # [文档] 开发指南（架构/扩展Agent/添加规则/调试）
│   ├── deployment.md                   # [文档] 部署指南（本地/Docker/Gunicorn/Nginx/Systemd）
│   └── assets/
│       └── 演示视频.mp4                 # [媒体] 平台功能演示视频（Git LFS 存储）
│
├── 💡 examples/ — 使用示例
│   ├── basic_analysis.py               # [示例] V1 基础图纸分析示例（上传→分析→输出结果）
│   ├── v2_analysis.py                  # [示例] V2 完整分析示例（含 Pre-check/Rollback/Judge）
│   ├── knowledge_management.py         # [示例] 知识库管理示例（添加/搜索/获取背景知识）
│   ├── rl_feedback.py                  # [示例] V1 RL反馈集成示例
│   ├── v2_rl_feedback.py              # [示例] V2 RL反馈示例（4维连续奖励）
│   └── atlas_query.py                 # [示例] Atlas 图谱查询示例
│
├── ⚙️ .github/ — GitHub 配置
│   ├── workflows/
│   │   └── ci.yml                      # [CI] GitHub Actions 工作流（多版本Python测试+Docker构建+安全检查）
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.md               # [模板] Bug 报告模板
│       └── feature_request.md          # [模板] 功能请求模板
│
├── 📂 data/ — 数据目录
│   ├── DATA_README.md                  # [说明] 数据目录使用指南（如何添加标准图纸/GB标准/知识库）
│   ├── drawings/                       # [数据] 用户图纸图片存放目录
│   ├── standard_drawings/              # [数据] 标准参考图纸存放目录
│   ├── error_drawings/                 # [数据] 错误标注图纸存放目录（测试用）
│   ├── error_labels/                   # [数据] 错误标注文本文件存放目录
│   ├── gb_standards/                   # [数据] GB国家标准 PDF/JSON 文件存放目录
│   ├── knowledge_base/                 # [数据] 背景知识 JSON 文件存放目录（37条）
│   ├── rl_experience/                  # [数据] V1 RL经验数据存放目录（自动生成）
│   └── atlas/                          # [V2数据] Atlas 工图图谱数据
│       ├── cases/                      #   - 34个典型工程案例数据
│       ├── rules/                      #   - 24条制图规则定义
│       ├── error_categories/           #   - 8大错误类别定义
│       └── index.json                  #   - 图谱索引文件
│
└── 📤 uploads/                         # [运行时] 用户上传文件存放目录（运行时自动创建）
```

## 🔒 安全须知

- **切勿**将 `config.py` 提交到任何公共仓库——它包含 API 密钥
- **切勿**在源码中硬编码 API 密钥——始终使用 `config.py` 或环境变量
- `config.py` 已在 `.gitignore` 中，防止意外提交
- 如果 API 密钥意外泄露，请立即轮换
- API 不可用时系统自动回退到本地规则引擎，核心功能不受影响
- RL 经验数据仅本地存储，不传输任何个人信息
- V2 经验库案例数据仅用于本地检索，VLM Judge 评估结果不外传

## 🤝 参与贡献

详见 [CONTRIBUTING.md](CONTRIBUTING.md)，包含 PR 提交、Issue 报告、代码风格等指南。

## 📝 版本历史

详见 [CHANGELOG.md](CHANGELOG.md)。

## 📄 许可证

本项目基于 [MIT 许可证](LICENSE) 开源。

## 🙏 致谢

- GB/T 4457-4460 机械制图国家标准
- [ArtiCAD](https://arxiv.org/abs/2501.xxxxx)（CVPR 2025）—— V2 核心模块灵感来源
- [RapidOCR](https://github.com/RapidAI/RapidOCR) 文字识别引擎
- [FAISS](https://github.com/facebookresearch/faiss) 向量相似性搜索
- [OpenAI 兼容 API](https://github.com/openai/openai-python) LLM 集成

## 📧 联系方式

如有问题或建议，请提交 [GitHub Issue](https://github.com/Jlu45/drawing-error-tutor/issues)。
