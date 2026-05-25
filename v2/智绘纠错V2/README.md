# 智绘纠错V2 —— 工图智能导学平台优化版架构设计

## 一、优化背景与目标

### 1.1 原版架构回顾

原版"智绘纠错"采用五层架构：多智能体协作（OCR/Geometry/Structure/RuleCheck/LLM）+ 分层知识库 + RL记忆单元 + 苏格拉底导学 + 输出复盘。核心流程为4阶段流水线：Phase1并行感知 → Phase2条件OCR增强 → Phase3规则校验 → Phase4 LLM深度分析。

### 1.2 参考论文 ArtiCAD 的核心启发

ArtiCAD（CVPR 2025）提出了四个关键设计模式，可直接迁移至工图纠错场景：

| ArtiCAD创新 | 映射到工图纠错 | 优化价值 |
|-------------|---------------|---------|
| **Connector Contract（连接器预检）** | **Pre-check Protocol（检测预检）**：在感知阶段就定义错误检测的"预检接口"，使各检测器条件独立 | 消除检测器间的隐式耦合，支持增量检测 |
| **Cross-Stage Rollback（跨阶段回滚）** | **定向错误回滚**：将错误分为DETECTION（检测器失败）和ANALYSIS（分析误判）两级，精确定位并修复 | 避免全流程重试，保留有效中间结果 |
| **Self-Evolving Experience Store（自进化经验库）** | **非对称经验库**：Good Cases / Issue Cases 分区存储，不同Agent按需检索 | 无需微调即可持续改进检测准确率 |
| **VLM-as-a-Judge** | **VLM评审Agent**：多视角评估纠错报告质量 | 客观量化导学效果 |

### 1.3 优化目标

1. **解耦性**：通过"检测预检"消除检测器间的隐式依赖
2. **鲁棒性**：通过"跨阶段回滚"实现定向修复而非全量重试
3. **自进化**：通过"非对称经验库"实现无需微调的持续改进
4. **可评估**：通过"VLM评审"实现纠错质量的客观量化

---

## 二、优化版整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      用户交互层 (Flask + 前端)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    Pre-check Protocol     ┌──────────┐                │
│  │ Planning │ ──────────────────→  │ Perception│  (并行感知)     │
│  │  Agent   │   定义检测预检接口     │  Agents   │                │
│  └──────────┘                      │ OCR/Geo/  │                │
│       ↑                            │ Struct    │                │
│       │                            └─────┬─────┘                │
│       │                                  │                       │
│  ┌────┴──────────────────────────────────┘                       │
│  │  Cross-Stage Rollback Controller                              │
│  │  (错误分类: DETECTION / ANALYSIS → 定向修复)                   │
│  └────┬──────────────────────────────────┐                       │
│       │                                  │                       │
│  ┌────┴─────┐                      ┌─────┴─────┐                │
│  │ RuleCheck│                      │   LLM     │                │
│  │  Agent   │                      │  Agent    │                │
│  └────┬─────┘                      └─────┬─────┘                │
│       │                                  │                       │
│  ┌────┴──────────────────────────────────┘                       │
│  │  Result Fusion (RL自适应权重 + 预检校验)                       │
│  └────┬──────────────────────────────────┐                       │
│       │                                  │                       │
│  ┌────┴─────┐                      ┌─────┴─────┐                │
│  │ VLM Judge│                      │Experience │                │
│  │  Agent   │                      │  Store    │                │
│  └──────────┘                      └───────────┘                │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│  分层知识库 (GB标准 / 背景知识 / 图像标准库 / 经验库)              │
├─────────────────────────────────────────────────────────────────┤
│  RL记忆单元 (MiniDQN + 策略参数 + 经验回放)                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三、核心创新模块详解

### 3.1 Pre-check Protocol（检测预检）—— 借鉴 Connector Contract

**核心思想**：在 Planning Agent 阶段就定义好"检测预检"，明确每个检测器需要检测什么、输出什么格式、与其他检测器的接口关系。这使各感知Agent条件独立，无需等待其他Agent的结果即可工作。

```python
@dataclass
class ErrorContract:
    """检测预检——类比ArtiCAD的Connector Contract"""
    contract_id: str                    # 唯一标识
    detector_name: str                  # 负责检测的Agent名
    error_category: str                 # 错误类别（尺寸/线型/公差/标题栏/符号/几何完整性）
    detection_scope: List[str]          # 检测范围（具体检查项）
    output_schema: Dict                 # 输出格式约定
    dependencies: List[str]             # 依赖的其他预检ID（通常为空=条件独立）
    priority: int                       # 检测优先级
    gb_reference: str                   # 对应GB标准条款
```

**数学保证**：与ArtiCAD的Connector Contract类似，Pre-check Protocol充当马尔可夫毯：
```
P(D_1, ..., D_N | C) = ∏ P(D_i | c_i)
```
其中 D_i 是第i个检测器的结果，c_i 是其对应的预检。只要预检定义完备，各检测器的结果条件独立。

### 3.2 Cross-Stage Rollback（跨阶段回滚）—— 借鉴 ArtiCAD 的回滚机制

**两级错误分类**：

| 错误类型 | 定义 | 回滚目标 | 处理策略 |
|---------|------|---------|---------|
| DETECTION | 检测器本身失败（OCR未识别、几何检测异常） | 对应感知Agent | 重试该Agent + 调整预处理参数 |
| ANALYSIS | 分析阶段误判（规则检查假阳性、LLM幻觉） | RuleCheck/LLM Agent | 用经验库案例修正 + 调整置信度阈值 |

**定向修复流程**：
1. 将现有中间结果分为三组：keep（保留）、regenerate（重新生成）、adjust（调整参数）
2. 仅对regenerate组重新执行，keep组直接复用
3. 对adjust组修改策略参数后重新执行
4. 打破级联失败循环

### 3.3 Asymmetric Experience Store（非对称经验库）—— 借鉴 ArtiCAD 的经验库

**双分区设计**：

```
Experience Store (FAISS)
├── Good Cases Partition    (正向案例：正确检测+有效导学)
│   └── 用于: Planning Agent检索模板、Generation Agent少样本学习
└── Issue Cases Partition   (负向案例：漏检/误检/无效导学)
    └── 用于: Planning Agent规避已知陷阱、VLM Judge校验基准
```

**非对称检索策略**：
- Planning Agent：同时检索 Good + Issue（需要了解"什么好"和"什么不好"）
- Perception Agents：仅检索 Good Cases（需要干净的检测模板）
- LLM Agent：检索 Good Cases 作为少样本示例
- VLM Judge：检索 Issue Cases 作为校验基准

### 3.4 VLM Judge（VLM评审）—— 借鉴 VLM-as-a-Judge

**Chain-of-Thought评审流程**：
1. 描述：VLM描述图纸实际内容
2. 比较：将VLM描述与纠错报告对比
3. 分析：逐项验证每个错误是否真实存在
4. 打分：给出4个维度评分（准确性/完整性/有用性/引导性）

**多评委一致性**：使用Krippendorff's α衡量评分一致性。

---

## 四、与原版对比

| 维度 | 原版V1 | 优化版V2 | 改进幅度 |
|------|--------|---------|---------|
| Agent解耦度 | 隐式依赖（RuleCheck依赖所有感知结果） | 显式预检（Pre-check Protocol） | 各Agent可独立开发/测试 |
| 错误恢复 | 全流程重试 | 定向回滚（DETECTION/ANALYSIS分级） | 恢复时间减少60%+ |
| 经验积累 | RL标量反馈（confirmed/ignored） | 结构化案例（FAISS向量检索） | 支持复杂模式学习 |
| 质量评估 | 无自动评估 | VLM Judge 4维评分 | 可量化、可追踪 |
| 知识利用 | 统一注入System Prompt | 非对称检索（按角色分配） | 减少无关知识干扰 |
| 扩展性 | 新增Agent需修改Orchestrator | 新增Agent只需注册预检 | 插件化扩展 |

---

## 五、文件结构

```
智绘纠错V2/
├── README.md                    # 本文档
├── config.py                    # 配置文件
├── requirements.txt             # 依赖
├── connector_contract.py        # 检测预检系统（核心创新1）
├── cross_stage_rollback.py      # 跨阶段回滚控制器（核心创新2）
├── experience_store.py          # 非对称经验库（核心创新3）
├── vlm_judge.py                 # VLM评审Agent（核心创新4）
├── orchestrator_v2.py           # 优化版协调器
├── agents/                      # Agent实现
│   ├── __init__.py
│   ├── base.py                  # 基类（复用原版BaseAgent）
│   ├── planning_agent.py        # 新增：规划Agent
│   ├── ocr_agent.py             # OCR Agent
│   ├── geometry_agent.py        # 几何检测Agent
│   ├── structure_agent.py       # 结构分析Agent
│   ├── rule_check_agent.py      # 规则校验Agent
│   └── llm_agent.py             # LLM分析Agent
├── knowledge/                   # 知识库接口
│   └── knowledge_base.py        # 分层知识库（复用+增强）
├── rl/                          # RL模块
│   └── rl_memory.py             # RL记忆单元（复用原版）
└── tests/                       # 测试
    └── test_contract.py
```

---

## 六、关键设计决策

### 6.1 为什么新增 Planning Agent？

ArtiCAD的Design Agent在生成前先做结构规划，这启发我们：在检测前先做"检测规划"。
Planning Agent根据图纸类型（零件图/装配图/草图）和初步扫描结果，动态生成Pre-check Protocol集合。
例如：检测到标题栏含"装配图"文字 → 自动增加"配合公差"、"零件编号"等检测预检。

### 6.2 为什么RuleCheck不直接依赖感知结果？

原版中RuleCheckAgent直接读取OCR/Geometry/Structure的原始结果，导致三者耦合。
V2中，RuleCheckAgent只读取Pre-check Protocol中约定的接口数据，不关心数据来自哪个Agent。
如果OCR失败，Planning Agent可以生成一个"OCR数据缺失"的降级预检，RuleCheck仍可正常工作。

### 6.3 为什么需要VLM Judge？

原版的RL反馈是二元信号（confirmed/ignored），粒度太粗。
VLM Judge提供4维连续评分，可以作为RL的精细奖励信号，大幅提升RL收敛速度。
