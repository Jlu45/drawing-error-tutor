# 🎯 智绘纠错——工图智能导学平台

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)

基于 GB/T 国家标准的机械工程图纸智能纠错与导学平台。采用5智能体协作架构、双知识库、苏格拉底式引导教学法和强化学习自进化机制。

## 🎬 演示视频

> 📌 演示视频文件较大，请将视频上传至 Bilibili / YouTube 等平台后，在此处替换为视频链接。
>
> 本地视频文件位于 `docs/assets/演示视频.mp4`（需自行放置，未纳入 Git 仓库）

<!-- 替换下方链接为你的视频地址 -->
<!-- [![演示视频](https://img.youtube.com/vi/VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=VIDEO_ID) -->

## ✨ 功能特性

- **多智能体协作分析**：5个专业智能体（OCR、几何、结构、规则检查、大模型）通过4阶段流水线协同工作
- **GB/T 标准合规检测**：依据 GB/T 4457-4460 国家标准，覆盖6大错误类别
- **苏格拉底式导学**：生成启发式引导，帮助学生理解错误原因
- **强化学习自进化**：基于 MiniDQN 的强化学习，根据用户反馈自适应调整分析策略
- **双知识库**：GB标准知识库 + 背景知识库 + 图像知识库，支持 FAISS 向量检索
- **优雅降级**：LLM API 不可用时自动回退到本地规则引擎
- **CAD 风格界面**：专业工程图纸界面，支持亮色/暗色/护眼三种主题

## 🏗️ 系统架构

```
DrawingOrchestrator（图纸分析协调器）
├── Phase 1（并行）：OCRAgent / GeometryAgent / StructureAgent
├── Phase 2（条件）：OCR 增强（RL 自适应阈值）
├── Phase 3：RuleCheckAgent（GB 标准规则校验）
└── Phase 4：LLMAgent（Qwen2.5-72B-Instruct 深度分析）
```

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
```

或使用环境变量：

```bash
export MULTIMODAL_API_URL='https://your-api-endpoint.example.com'
export MULTIMODAL_API_KEY='your-api-key-here'
```

### 运行

```bash
# 启动应用
python app.py

# 或使用快速启动脚本
./start.sh      # Linux/Mac
start.bat       # Windows
```

浏览器访问 http://localhost:5000

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

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 上传页面 |
| `/upload` | POST | 上传并分析图纸 |
| `/uploads/<filename>` | GET | 获取上传文件 |
| `/api/gb_standards?q=<query>` | GET | 搜索 GB 标准 |
| `/api/rl_feedback` | POST | 提交 RL 反馈 |
| `/api/rl_stats` | GET | 获取 RL 记忆统计 |

### RL 反馈 API

```bash
curl -X POST http://localhost:5000/api/rl_feedback \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "drawing.png_1700000000",
    "error_description": "缺少直径符号",
    "feedback_type": "confirmed"
  }'
```

有效 `feedback_type` 值：`confirmed`（确认）、`ignored`（忽略）、`dismissed_all`（全部误报）、`partial_confirm`（部分确认）、`useful_guidance`（引导有帮助）

## 🧪 测试

```bash
# 运行所有测试
pytest

# 带覆盖率报告
pytest --cov=src --cov-report=html

# 运行指定测试模块
pytest tests/test_geometric_detector.py -v
```

## 📁 项目结构

```
drawing-error-tutor/
│
├── 📄 项目根目录文件
│   ├── app.py                          # [入口] Flask Web 应用主程序，路由定义与服务启动
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
│   ├── SKILL.md                        # [技能] Trae IDE 技能定义入口文件
│   ├── LICENSE                         # [法律] MIT 开源许可协议
│   ├── README.md                       # [文档] 项目说明文档（本文件）
│   └── CHANGELOG.md                    # [文档] 版本变更记录
│
├── 🧠 src/ — 核心源码
│   ├── multi_agent_system.py           # [核心] 多智能体协调器 + 5个专业Agent + 4阶段流水线
│   │                                   #   - DrawingOrchestrator：协调器，管理分析流程
│   │                                   #   - OCRAgent：文字识别（RapidOCR）
│   │                                   #   - GeometryAgent：几何元素检测（OpenCV Hough/Canny）
│   │                                   #   - StructureAgent：图纸结构分析（区域/标题栏/图框）
│   │                                   #   - RuleCheckAgent：GB标准规则校验（6类规则）
│   │                                   #   - LLMAgent：大模型深度分析（Qwen2.5-72B-Instruct）
│   │                                   #   - BaseAgent/ImageCache/PreprocessPipeline 等基础设施
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
│   ├── rl_memory_unit.py               # [进化] 强化学习记忆单元
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
│   ├── index.html                      # [页面] CAD风格图纸上传页面（亮色/暗色/护眼主题）
│   └── result.html                     # [页面] 分析结果展示页面（错误列表/GB标准/RL反馈）
│
├── 🧪 tests/ — 测试套件
│   ├── test_geometric_detector.py      # [单元测试] 几何检测器测试（直线/圆/箭头/线型/YOLO格式）
│   ├── test_rule_check.py              # [单元测试] 规则检查测试（尺寸/公差/标题栏/符号/线型/完整性）
│   ├── test_rl_memory.py               # [单元测试] RL记忆单元测试（DQN/经验池/策略参数/状态提取）
│   ├── test_knowledge_base.py          # [单元测试] 知识库测试（GB标准搜索/背景知识/图像知识）
│   └── test_config_security.py         # [安全测试] 配置安全检测（无硬编码密钥/无内部URL/gitignore保护）
│
├── 📚 docs/ — 文档
│   ├── api.md                          # [文档] REST API + 内部 API 完整文档
│   ├── development.md                  # [文档] 开发指南（架构/扩展Agent/添加规则/调试）
│   ├── deployment.md                   # [文档] 部署指南（本地/Docker/Gunicorn/Nginx/Systemd）
│   └── assets/
│       └── 演示视频.mp4                 # [媒体] 平台功能演示视频（Git LFS 存储）
│
├── 💡 examples/ — 使用示例
│   ├── basic_analysis.py               # [示例] 基础图纸分析示例（上传→分析→输出结果）
│   ├── knowledge_management.py         # [示例] 知识库管理示例（添加/搜索/获取背景知识）
│   └── rl_feedback.py                  # [示例] RL反馈集成示例（查看策略参数/反馈类型说明）
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
│   └── rl_experience/                  # [数据] RL经验数据存放目录（自动生成）
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

## 🤝 参与贡献

详见 [CONTRIBUTING.md](CONTRIBUTING.md)，包含 PR 提交、Issue 报告、代码风格等指南。

## 📝 版本历史

详见 [CHANGELOG.md](CHANGELOG.md)。

## 📄 许可证

本项目基于 [MIT 许可证](LICENSE) 开源。

## 🙏 致谢

- GB/T 4457-4460 机械制图国家标准
- [RapidOCR](https://github.com/RapidAI/RapidOCR) 文字识别引擎
- [FAISS](https://github.com/facebookresearch/faiss) 向量相似性搜索
- [OpenAI 兼容 API](https://github.com/openai/openai-python) LLM 集成

## 📧 联系方式

如有问题或建议，请提交 [GitHub Issue](https://github.com/Jlu45/drawing-error-tutor/issues)。
