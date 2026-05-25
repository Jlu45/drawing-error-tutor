# 智绘纠错V2 - 工程图纸智能纠错系统

## 系统简介

智绘纠错V2是一个基于多智能体协同的工图智能纠错系统，支持6阶段流水线、预检机制、跨阶段回滚、非对称经验存储、VLM质量评审等核心功能。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API密钥

编辑 `智绘纠错V2/config.py` 文件，填入你的API密钥：

```python
MULTIMODAL_API_KEY = 'your-api-key-here'
```

或使用环境变量：

```bash
# Windows
set MULTIMODAL_API_KEY=your-api-key-here

# Linux/Mac
export MULTIMODAL_API_KEY=your-api-key-here
```

### 3. 启动服务

```bash
python app_v2.py
```

服务将运行在 http://127.0.0.1:5001

## 项目结构

```
v2/
├── app_v2.py              # 主入口文件
├── requirements.txt       # 依赖列表
├── templates/             # HTML模板
│   ├── index_v2.html     # 首页
│   └── result_v2.html    # 结果页
├── data/                  # 数据目录
│   ├── drawings/         # 上传图纸
│   ├── error_drawings/   # 错误图纸
│   ├── error_labels/     # 错误标签
│   ├── gb_standards/     # GB标准
│   ├── knowledge_base/   # 知识库
│   ├── standard_drawings/# 标准图纸
│   └── rl_experience/    # RL经验
├── uploads/              # 上传文件临时目录
└── 智绘纠错V2/           # 核心模块
    ├── config.py         # 配置文件
    ├── orchestrator_v2.py # 协调器
    ├── agents/           # 智能体模块
    ├── atlas/            # 图册能力包
    ├── data/atlas/       # 图册数据
    ├── rl/               # RL模块
    ├── tools/            # 工具脚本
    └── utils/            # 工具函数
```

## 配置说明

### API配置

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| MULTIMODAL_API_URL | API地址 | https://dashscope.aliyuncs.com/compatible-mode/v1 |
| MULTIMODAL_API_KEY | API密钥 | 空（必须配置） |
| LLM_MODEL | LLM模型 | deepseek-v4-flash |
| VLM_MODEL | VLM模型 | qwen3-vl-plus |

### 功能开关

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| VLM_JUDGE_ENABLED | 启用VLM质量评审 | True |
| ENABLE_ATLAS_PACK | 启用图册能力包 | True |

## 使用说明

1. 打开浏览器访问 http://127.0.0.1:5001
2. 点击或拖拽上传工程图纸（PNG/JPG/JPEG/GIF/BMP）
3. 可选择启用VLM质量评审（需要API支持）
4. 点击"开始智能纠错分析"
5. 查看分析结果：
   - 纠错详情：检测到的所有问题
   - AI分析：AI结构化分析结果
   - 知识库：相关GB标准和学习资源
   - 系统面板：流水线耗时和统计信息

## 注意事项

- 首次使用需要配置有效的API密钥
- 建议使用分辨率不低于1920×1080的图纸
- VLM质量评审需要额外的API调用，会产生费用
- 经验库数据会自动保存在 `data/rl_experience/` 目录

## 技术栈

- Python 3.8+
- Flask
- OpenCV
- NumPy
- OpenAI API (兼容DashScope等)

## 许可证

仅供学习和研究使用
