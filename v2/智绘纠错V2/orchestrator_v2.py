"""
优化版协调器 (Orchestrator V2)
================================
整合四大核心创新模块的协调器。

与原版DrawingOrchestrator的区别:
1. 新增 Planning Phase（预检机制）
2. 新增 Cross-Stage Rollback（定向修复）
3. 新增 Experience Store（非对称经验检索）
4. 新增 VLM Judge Phase（质量评审）
5. RL奖励信号从二元升级为4维连续评分
"""

import os
import time
import json
import logging
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from connector_contract import (
    ContractRegistry, ContractFactory, ErrorContract,
    ContractStatus, ErrorCategory
)
from cross_stage_rollback import RollbackController, RollbackAction
from experience_store import (
    AsymmetricExperienceStore, ExperienceCase, CaseType,
    AgentRole
)
from vlm_judge import VLMJudge, JudgeResult

import config

logger = logging.getLogger("OrchestratorV2")


class OrchestratorV2:
    """
    优化版协调器

    执行流程（6阶段）：
    Phase 0: Planning（预检生成 + 经验检索）
    Phase 1: Perception（并行感知，按预检执行）
    Phase 2: Rule Checking（规则校验，依赖Phase 1）
    Phase 3: LLM Analysis（深度分析，依赖Phase 2）
    Phase 4: Result Fusion（结果融合 + 预检校验）
    Phase 5: VLM Judge（质量评审，可选）
    """

    def __init__(self, api_url: str, api_key: str,
                 llm_model: str = "Qwen2.5-72B-Instruct",
                 vlm_model: str = "",
                 experience_dir: str = ""):
        self.api_url = api_url
        self.api_key = api_key
        self.llm_model = llm_model
        self.vlm_model = vlm_model

        # 核心模块
        self.rollback = RollbackController()
        self.experience = AsymmetricExperienceStore(persist_dir=experience_dir)

        # VLM Judge（可选）
        self.vlm_judge = None
        if vlm_model and api_url and api_key:
            self.vlm_judge = VLMJudge(api_url, api_key, vlm_model)

        # Agent注册表（延迟初始化）
        self._agents = {}
        self._initialized = False

        # RL记忆单元
        self._rl_memory = None

        # Atlas 图册能力包
        self._atlas_registry = None
        self._atlas_contract_extender = None
        self._atlas_feature_adapter = None
        self._atlas_rule_plugin = None
        self._atlas_context_retriever = None
        self._atlas_fusion = None
        self._atlas_vlm_fewshot = None
        self._atlas_enabled = getattr(config, 'ENABLE_ATLAS_PACK', False)

        if self._atlas_enabled:
            try:
                from atlas import (
                    AtlasRegistry, AtlasContractExtender,
                    AtlasFeatureAdapter, AtlasRulePlugin,
                )
                cases_path = getattr(config, 'ATLAS_CASES_PATH', '')
                rules_path = getattr(config, 'ATLAS_RULES_PATH', '')
                self._atlas_registry = AtlasRegistry(cases_path, rules_path)
                self._atlas_contract_extender = AtlasContractExtender(self._atlas_registry)
                self._atlas_feature_adapter = AtlasFeatureAdapter()
                self._atlas_rule_plugin = AtlasRulePlugin(self._atlas_registry)
                logger.info(f"[OrchestratorV2] Atlas 图册能力包已加载: "
                            f"{len(self._atlas_registry.cases)}条案例, "
                            f"{len(self._atlas_registry.rules)}条规则")
            except Exception as e:
                logger.warning(f"[OrchestratorV2] Atlas 图册能力包加载失败: {e}")
                self._atlas_enabled = False

        # 统计
        self._stats = {
            'total_analyses': 0,
            'total_rollbacks': 0,
            'avg_quality_score': 0.0,
            'phase_timings': {}
        }

        logger.info("[OrchestratorV2] 初始化完成")

    def _init_agents(self):
        """延迟初始化所有Agent"""
        if self._initialized:
            return

        try:
            # 导入Agent模块
            from agents.ocr_agent import OCRAgent
            from agents.geometry_agent import GeometryAgent
            from agents.structure_agent import StructureAgent
            from agents.rule_check_agent import RuleCheckAgent
            from agents.llm_agent import LLMAgent

            self._agents = {
                'ocr': OCRAgent(),
                'geometry': GeometryAgent(),
                'structure': StructureAgent(),
                'rule_check': RuleCheckAgent(),
                'llm': LLMAgent(self.api_url, self.api_key, self.llm_model)
            }

            # 预初始化所有Agent（加载模型等）
            for name, agent in self._agents.items():
                agent.initialize()
                logger.info(f"[OrchestratorV2] {name} Agent: "
                           f"{'OK' if agent._initialized else 'FAIL'}")

            # 尝试初始化RL记忆单元
            try:
                from rl.rl_memory import RLMemoryUnit
                self._rl_memory = RLMemoryUnit(state_dim=10)
                logger.info("[OrchestratorV2] RL记忆单元已加载")
            except ImportError:
                logger.info("[OrchestratorV2] RL记忆单元未加载，使用默认参数")

            self._initialized = True
            logger.info("[OrchestratorV2] 所有Agent初始化完成")

        except ImportError as e:
            logger.error(f"[OrchestratorV2] Agent导入失败: {e}")
            raise

    def analyze(self, image_path: str, background_knowledge: str = "",
                enable_judge: bool = False) -> Dict:
        """
        执行完整的图纸分析流水线

        Args:
            image_path: 图纸图像路径
            background_knowledge: 背景知识文本
            enable_judge: 是否启用VLM评审

        Returns:
            完整的分析结果字典
        """
        self._init_agents()
        total_start = time.time()

        logger.info(f"[OrchestratorV2] ===== 开始分析: {image_path} =====")

        # ===== Phase 0: Planning（预检生成） =====
        phase0_start = time.time()
        registry = self._phase0_planning(image_path, background_knowledge)
        phase0_time = (time.time() - phase0_start) * 1000

        # ===== Phase 1: Perception（并行感知） =====
        phase1_start = time.time()
        perception_results = self._phase1_perception(image_path, registry)
        phase1_time = (time.time() - phase1_start) * 1000

        # ===== Phase 2: Rule Checking =====
        phase2_start = time.time()
        rule_result = self._phase2_rule_check(perception_results, registry)
        phase2_time = (time.time() - phase2_start) * 1000

        # Phase 2完成后，标记规则检查相关预检
        if rule_result and getattr(rule_result, 'success', False):
            self._mark_contracts_completed(registry, {'rule_check': rule_result}, phase='rule_check')

        # ===== Phase 3: LLM Analysis =====
        phase3_start = time.time()
        llm_result = self._phase3_llm_analysis(perception_results, rule_result,
                                                 background_knowledge, registry)
        phase3_time = (time.time() - phase3_start) * 1000

        # Phase 3完成后，标记LLM分析预检
        if llm_result and getattr(llm_result, 'success', False):
            self._mark_contracts_completed(registry, {'llm': llm_result}, phase='llm_analysis')

        # ===== Phase 4: Result Fusion =====
        phase4_start = time.time()
        final_result = self._phase4_fusion(perception_results, rule_result,
                                            llm_result, registry)
        phase4_time = (time.time() - phase4_start) * 1000

        # ===== Phase 5: VLM Judge（可选） =====
        judge_result = None
        if enable_judge and self.vlm_judge:
            phase5_start = time.time()
            judge_result = self._phase5_judge(image_path, final_result)
            phase5_time = (time.time() - phase5_start) * 1000
        else:
            phase5_time = 0

        # 总时间
        total_time = (time.time() - total_start) * 1000

        # 构建完整结果
        result = {
            **final_result,
            'metrics': {
                'total_time_ms': round(total_time, 1),
                'phase_timings': {
                    'planning': round(phase0_time, 1),
                    'perception': round(phase1_time, 1),
                    'rule_check': round(phase2_time, 1),
                    'llm_analysis': round(phase3_time, 1),
                    'fusion': round(phase4_time, 1),
                    'vlm_judge': round(phase5_time, 1)
                },
                'contract_stats': registry.stats,
                'contract_list': [
                    {
                        'category': c.error_category.value,
                        'detector': c.detector_name,
                        'status': c.status.value,
                        'priority': c.priority,
                        'confidence': c.confidence,
                        'execution_time_ms': c.execution_time_ms
                    }
                    for c in registry.all_contracts
                ],
                'rollback_stats': self.rollback.stats,
                'experience_stats': self.experience.stats,
                'rl_stats': self._rl_memory.get_stats() if self._rl_memory else None,
                'atlas_stats': {
                    'enabled': self._atlas_enabled,
                    'total_cases': len(self._atlas_registry.cases) if self._atlas_registry else 0,
                    'total_rules': len(self._atlas_registry.rules) if self._atlas_registry else 0,
                    'active_rules': len(self._atlas_registry.get_active_rules()) if self._atlas_registry else 0,
                } if self._atlas_enabled else {'enabled': False},
            }
        }

        if judge_result:
            result['judge_result'] = judge_result.to_dict()

        # 更新统计
        self._stats['total_analyses'] += 1
        self._stats['total_rollbacks'] = self.rollback.stats['total_rollbacks']

        logger.info(f"[OrchestratorV2] ===== 分析完成: {total_time:.0f}ms =====")
        logger.info(f"[OrchestratorV2] 预检: {registry.stats['total']}个, "
                    f"回滚: {self.rollback.stats['total_rollbacks']}次, "
                    f"经验库: {self.experience.stats['total_cases']}条")

        # 重置回滚控制器
        self.rollback.reset()

        return result

    def _phase0_planning(self, image_path: str,
                          background_knowledge: str) -> ContractRegistry:
        """
        Phase 0: 规划阶段

        1. 快速扫描图纸（可选）
        2. 检索相关经验
        3. 生成Error Contract集合
        """
        logger.info("[Phase 0] Planning: 生成错误检测预检单")

        # 检索相关经验（Planning角色：Good + Issue）
        experience_context = self.experience.get_context_for_agent(
            AgentRole.PLANNING, query="工程图纸纠错检测规划", top_k=3
        )
        if experience_context:
            logger.info(f"[Phase 0] 检索到{self.experience.stats['total_cases']}条经验")

        # 生成预检单（默认为零件图，后续可根据OCR结果动态调整）
        registry = ContractFactory.create_contracts(
            drawing_type="part",
            initial_scan=None  # 可在此处传入快速扫描结果
        )

        # Atlas 图册契约扩展
        if self._atlas_enabled and self._atlas_contract_extender:
            try:
                rule_mode = getattr(config, 'ATLAS_RULE_MODE', 'safe')
                teacher_profile = 'default'
                if rule_mode == 'strict':
                    teacher_profile = 'strict'
                elif rule_mode == 'lenient':
                    teacher_profile = 'lenient'
                contracts = registry.all_contracts
                self._atlas_contract_extender.extend(
                    contracts,
                    drawing_type="part",
                    teacher_profile=teacher_profile,
                )
                atlas_count = sum(
                    len(c.metadata.get('atlas_subchecks', []))
                    for c in contracts
                    if hasattr(c, 'metadata') and c.metadata
                )
                logger.info(f"[Phase 0] Atlas 图册扩展: {atlas_count} 条子检查注入预检")
            except Exception as e:
                logger.warning(f"[Phase 0] Atlas 契约扩展失败: {e}")

        logger.info(f"[Phase 0] 生成 {registry.stats['total']} 个预检项")
        return registry

    def _phase1_perception(self, image_path: str,
                            registry: ContractRegistry) -> Dict[str, Any]:
        """
        Phase 1: 并行感知阶段

        按预检执行各感知Agent，支持跨阶段回滚。
        """
        logger.info("[Phase 1] Perception: 并行感知")

        results = {}
        perception_agents = ['ocr', 'geometry', 'structure']

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            for agent_name in perception_agents:
                agent = self._agents.get(agent_name)
                if agent:
                    futures[executor.submit(agent.analyze, image_path)] = agent_name

            for future in as_completed(futures):
                agent_name = futures[future]
                try:
                    result = future.result(timeout=60)
                    results[agent_name] = result
                    self.rollback.save_result(agent_name, result.data if result.success else {})

                    if not result.success:
                        # 触发回滚决策
                        decision = self.rollback.decide_rollback(
                            agent_name, Exception(result.errors[0] if result.errors else "Unknown")
                        )
                        if decision.action == RollbackAction.ADJUST_PARAMS:
                            # 使用调整后的参数重试
                            retry_result = agent.analyze(image_path, **decision.adjusted_params)
                            if retry_result.success:
                                results[agent_name] = retry_result
                                self.rollback.save_result(agent_name, retry_result.data)
                                logger.info(f"[Phase 1] {agent_name} 回滚重试成功")
                            else:
                                logger.warning(f"[Phase 1] {agent_name} 回滚重试失败")

                    logger.info(f"[Phase 1] {agent_name}: "
                                f"{'OK' if result.success else 'FAIL'} "
                                f"({result.execution_time_ms:.0f}ms)")
                except Exception as e:
                    logger.error(f"[Phase 1] {agent_name}: ERROR - {e}")
                    results[agent_name] = type('AgentResult', (), {
                        'success': False, 'data': {}, 'errors': [str(e)],
                        'execution_time_ms': 0, 'confidence': 0.0
                    })()

        # Phase 1完成后，标记感知相关预检为completed
        self._mark_contracts_completed(registry, results, phase='perception')

        return results

    def _phase2_rule_check(self, perception_results: Dict,
                            registry: ContractRegistry) -> Any:
        """
        Phase 2: 规则校验阶段

        依赖Phase 1的感知结果，按预检执行规则校验。
        """
        logger.info("[Phase 2] Rule Check: 规则校验")

        agent = self._agents.get('rule_check')
        if not agent:
            return type('AgentResult', (), {
                'success': False, 'data': {}, 'errors': ['RuleCheck Agent未初始化'],
                'execution_time_ms': 0, 'confidence': 0.0
            })()

        try:
            result = agent.analyze(
                "",
                ocr_result=perception_results.get('ocr'),
                geometry_result=perception_results.get('geometry'),
                structure_result=perception_results.get('structure')
            )
            self.rollback.save_result('rule_check', result.data if result.success else {})
            logger.info(f"[Phase 2] RuleCheck: {result.data.get('total_errors', 0)} errors "
                        f"({result.execution_time_ms:.0f}ms)")

            # Atlas 图册规则插件
            if self._atlas_enabled and self._atlas_rule_plugin and self._atlas_feature_adapter:
                try:
                    ocr_res = perception_results.get('ocr')
                    geo_res = perception_results.get('geometry')
                    struct_res = perception_results.get('structure')

                    ocr_data = ocr_res.data if ocr_res and getattr(ocr_res, 'success', False) else {}
                    geo_data = geo_res.data if geo_res and getattr(geo_res, 'success', False) else {}
                    struct_data = struct_res.data if struct_res and getattr(struct_res, 'success', False) else {}

                    img_h = struct_data.get('image_size', {}).get('height', 0)
                    img_w = struct_data.get('image_size', {}).get('width', 0)
                    image_shape = (img_h, img_w) if img_h > 0 and img_w > 0 else (0, 0)

                    atlas_features = self._atlas_feature_adapter.build(
                        ocr_data, geo_data, struct_data, image_shape
                    )

                    contracts = registry.all_contracts if registry else []
                    atlas_issues = self._atlas_rule_plugin.check(atlas_features, contracts)

                    if atlas_issues:
                        _cat_cn = {
                            'DIMENSION_ERROR':'尺寸标注', 'LINE_TYPE_ERROR':'线型错误', 'TOLERANCE_ERROR':'公差配合',
                            'TITLE_BLOCK_ERROR':'标题栏', 'SYMBOL_ERROR':'符号标注',
                            'GEOMETRY_INCOMPLETE_ERROR':'几何完整性', 'STRUCTURE_ERROR':'图纸结构',
                            'SURFACE_ERROR':'表面粗糙度', 'ROUGHNESS_ERROR':'表面粗糙度',
                            'WELD_ERROR':'焊接符号', 'SHEET_ERROR':'图幅规范',
                            'VIEW_ERROR':'视图标注', 'GENERAL_ERROR':'其他',
                        }
                        existing_errors = result.data.get('errors', [])
                        for issue in atlas_issues:
                            _raw_cat = issue.get('error_category', '')
                            _evidence = issue.get('evidence', {})
                            _marker = self._extract_marker_pos(_evidence, img_w, img_h)
                            existing_errors.append({
                                'type': _cat_cn.get(_raw_cat, _raw_cat),
                                'description': issue.get('title', ''),
                                'suggestion': issue.get('suggestion', ''),
                                'severity': {'confirmed_error': '高', 'suspected_issue': '中', 'suggestion': '低'}.get(issue.get('level', ''), '中'),
                                'source': 'atlas_rule',
                                'gb_reference': '',
                                'atlas_rule_id': issue.get('atlas_rule_id', ''),
                                'source_case_id': issue.get('source_case_id', ''),
                                'level': issue.get('level', 'suspected_issue'),
                                'confidence': issue.get('confidence', 0.0),
                                'evidence': _evidence,
                                'teaching_hint': issue.get('teaching_hint', ''),
                                'marker_x': _marker[0] if _marker else None,
                                'marker_y': _marker[1] if _marker else None,
                            })
                        result.data['errors'] = existing_errors
                        result.data['total_errors'] = len(existing_errors)
                        result.data['atlas_issues'] = atlas_issues
                        logger.info(f"[Phase 2] Atlas 图册规则: {len(atlas_issues)} 条新问题")
                except Exception as e:
                    logger.warning(f"[Phase 2] Atlas 图册规则插件执行失败: {e}")

            return result
        except Exception as e:
            logger.error(f"[Phase 2] RuleCheck ERROR: {e}")
            return type('AgentResult', (), {
                'success': False, 'data': {}, 'errors': [str(e)],
                'execution_time_ms': 0, 'confidence': 0.0
            })()

    def _extract_marker_pos(self, evidence: Dict, img_w: int, img_h: int):
        if not evidence or img_w <= 0 or img_h <= 0:
            return None
        try:
            if 'circle' in evidence:
                c = evidence['circle']
                cx, cy = c.get('center', (0, 0))
                return (float(cx) / img_w, float(cy) / img_h)
            if 'line' in evidence:
                l = evidence['line']
                s = l.get('start', (0, 0))
                e = l.get('end', (0, 0))
                mx = (float(s[0]) + float(e[0])) / 2
                my = (float(s[1]) + float(e[1])) / 2
                return (mx / img_w, my / img_h)
            if 'text_bbox' in evidence:
                bb = evidence['text_bbox']
                if len(bb) >= 4:
                    mx = (float(bb[0]) + float(bb[2])) / 2
                    my = (float(bb[1]) + float(bb[3])) / 2
                    return (mx / img_w, my / img_h)
            if 'contour_bbox' in evidence:
                bb = evidence['contour_bbox']
                if len(bb) >= 4:
                    mx = (float(bb[0]) + float(bb[2])) / 2
                    my = (float(bb[1]) + float(bb[3])) / 2
                    return (mx / img_w, my / img_h)
            if 'slot' in evidence:
                slot = evidence['slot']
                c1 = slot.get('circle1', {}).get('center', (0, 0))
                c2 = slot.get('circle2', {}).get('center', (0, 0))
                mx = (float(c1[0]) + float(c2[0])) / 2
                my = (float(c1[1]) + float(c2[1])) / 2
                return (mx / img_w, my / img_h)
        except Exception:
            pass
        return None

    def _mark_contracts_completed(self, registry: ContractRegistry,
                                   results: Dict, phase: str = ''):
        """
        根据Agent执行结果，标记对应预检为completed

        Args:
            registry: 预检注册表
            results: 当前阶段的Agent结果字典 {agent_name: AgentResult}
            phase: 当前阶段名称 ('perception' / 'rule_check' / 'llm_analysis')
        """
        # Agent到预检类别的映射
        agent_to_categories = {
            'ocr': ['标题栏', '尺寸标注'],
            'geometry': ['线型', '几何完整性'],
            'structure': ['图纸结构'],
            'rule_check': ['尺寸标注', '公差', '符号', '几何完整性'],
            'llm': ['其他']
        }

        for agent_name, result in results.items():
            if not result.success:
                continue

            categories = agent_to_categories.get(agent_name, [])
            if not categories:
                continue

            # 找到该Agent对应的预检并标记完成
            for contract in registry.all_contracts:
                if (contract.status in [ContractStatus.PENDING, ContractStatus.IN_PROGRESS]
                    and contract.detector_name == agent_name
                    and contract.error_category.value in categories):
                    contract.mark_completed(
                        result=result.data if hasattr(result, 'data') else {},
                        confidence=result.confidence,
                        exec_time=result.execution_time_ms
                    )
                    logger.debug(f"[预检] {contract.contract_id} ({contract.error_category.value}) "
                                f"→ completed by {agent_name}")

        logger.info(f"[预检更新] Phase={phase}: "
                   f"completed={len(registry.get_completed())}, "
                   f"pending={len(registry.get_pending())}, "
                   f"total={registry.stats['total']}")

    def _phase3_llm_analysis(self, perception_results: Dict, rule_result: Any,
                              background_knowledge: str,
                              registry: ContractRegistry) -> Any:
        """
        Phase 3: LLM深度分析阶段

        依赖Phase 1和Phase 2的结果，注入经验上下文。
        """
        logger.info("[Phase 3] LLM Analysis: 深度分析")

        agent = self._agents.get('llm')
        if not agent:
            return type('AgentResult', (), {
                'success': False, 'data': {}, 'errors': ['LLM Agent未初始化'],
                'execution_time_ms': 0, 'confidence': 0.0
            })()

        try:
            experience_context = self.experience.get_context_for_agent(
                AgentRole.ANALYSIS, query="工程图纸纠错分析", top_k=2
            )

            # Atlas 图册上下文检索
            atlas_context = ""
            if self._atlas_enabled and self._atlas_registry:
                try:
                    from atlas.atlas_context_retriever import AtlasContextRetriever
                    if self._atlas_context_retriever is None:
                        self._atlas_context_retriever = AtlasContextRetriever(self._atlas_registry.cases)
                    rule_errors = rule_result.data.get('errors', []) if rule_result and rule_result.success else []
                    atlas_cases = self._atlas_context_retriever.retrieve(
                        rule_errors, top_k=getattr(config, 'ATLAS_MAX_CONTEXT_CASES', 3)
                    )
                    if atlas_cases:
                        atlas_parts = []
                        for case in atlas_cases:
                            atlas_parts.append(
                                f"案例：{case.get('figure_no', '')} {case.get('case_name', '')}\n"
                                f"典型错误：{case.get('source_text', '')}\n"
                                f"教学提示：{case.get('teaching_hint', '')}"
                            )
                        atlas_context = "\n\n【图册参考案例】\n" + "\n\n".join(atlas_parts) + (
                            "\n\n注意：1. 图册案例只能作为参考；"
                            "2. 不得脱离当前图纸证据直接照搬；"
                            "3. AtlasRulePlugin输出suspected_issue时，报告中必须写'疑似'或'建议核对'。"
                        )
                        logger.info(f"[Phase 3] Atlas 图册上下文: {len(atlas_cases)} 条案例注入")
                except Exception as e:
                    logger.warning(f"[Phase 3] Atlas 上下文检索失败: {e}")

            combined_context = (experience_context or "") + atlas_context

            result = agent.analyze(
                "",
                ocr_result=perception_results.get('ocr'),
                geometry_result=perception_results.get('geometry'),
                structure_result=perception_results.get('structure'),
                rule_result=rule_result,
                background_knowledge=background_knowledge,
                experience_context=combined_context
            )

            if not result.success:
                logger.warning(f"[Phase 3] LLM Agent失败: {result.errors}, 启动降级方案")
                decision = self.rollback.decide_rollback('llm', Exception("LLM调用失败"))
                if decision.action == RollbackAction.USE_FALLBACK:
                    logger.info("[Phase 3] LLM降级到本地规则引擎")
                    result = self._generate_local_analysis(perception_results, rule_result)
                else:
                    logger.warning(f"[Phase 3] 回滚决策为{decision.action.value}，仍使用本地降级")
                    result = self._generate_local_analysis(perception_results, rule_result)

            self.rollback.save_result('llm', result.data if result.success else {})
            logger.info(f"[Phase 3] LLM: {'OK' if result.success else 'DEGRADED'} "
                        f"({result.execution_time_ms:.0f}ms) "
                        f"raw_response长度={len(result.data.get('raw_response', '')) if result.success else 0}")
            return result

        except Exception as e:
            logger.error(f"[Phase 3] LLM ERROR: {e}")
            fallback = self._generate_local_analysis(perception_results, rule_result)
            logger.info(f"[Phase 3] 异常降级: raw_response长度={len(fallback.data.get('raw_response', ''))}")
            return fallback

    def _phase4_fusion(self, perception_results: Dict, rule_result: Any,
                        llm_result: Any, registry: ContractRegistry) -> Dict:
        """
        Phase 4: 结果融合阶段

        融合规则检查和LLM分析的结果，执行预检校验。
        """
        logger.info("[Phase 4] Fusion: 结果融合")

        # 获取RL策略参数
        rl_params = None
        if self._rl_memory:
            rl_params = self._rl_memory.get_policy_params()

        # 提取错误列表
        rule_errors = rule_result.data.get('errors', []) if rule_result.success else []
        llm_errors = []
        llm_summary = ""
        llm_score = None
        llm_learning_points = []

        if llm_result.success:
            try:
                raw = llm_result.data.get('raw_response', '')
                start_idx = raw.find('{')
                end_idx = raw.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    llm_data = json.loads(raw[start_idx:end_idx])
                    llm_errors = llm_data.get('errors', [])
                    llm_summary = llm_data.get('summary', '')
                    llm_score = llm_data.get('overall_score', None)
                    llm_learning_points = llm_data.get('learning_points', [])
            except Exception:
                pass

        # 去重合并
        all_errors = list(rule_errors)
        existing_descs = {e.get('description', '') for e in all_errors}
        for le in llm_errors:
            desc = le.get('description', '')
            if desc and desc not in existing_descs:
                all_errors.append({
                    'type': le.get('type', 'LLM检测'),
                    'description': desc,
                    'suggestion': le.get('suggestion', ''),
                    'severity': le.get('severity', '中'),
                    'source': 'llm_analysis',
                    'gb_reference': le.get('gb_reference', '')
                })
                existing_descs.add(desc)

        # Atlas 图册结果融合
        if self._atlas_enabled:
            try:
                from atlas.atlas_fusion import AtlasEvidenceFusion
                if self._atlas_fusion is None:
                    self._atlas_fusion = AtlasEvidenceFusion()
                atlas_errors = [e for e in all_errors if e.get('source') == 'atlas_rule']
                base_errors = [e for e in all_errors if e.get('source') != 'atlas_rule']
                llm_only_errors = [e for e in all_errors if e.get('source') == 'llm_analysis']
                merged = self._atlas_fusion.merge(base_errors, atlas_errors, llm_only_errors)
                all_errors = merged
                logger.info(f"[Phase 4] Atlas 融合: base={len(base_errors)}, "
                            f"atlas={len(atlas_errors)}, llm={len(llm_only_errors)}, "
                            f"merged={len(merged)}")
            except Exception as e:
                logger.warning(f"[Phase 4] Atlas 融合失败: {e}")

        # 预检校验：检查是否所有预检都已完成
        contract_validation = self._validate_contracts(registry, all_errors)

        # 计算评分
        if rl_params:
            severity_weights = {
                '高': rl_params.severity_weight_high,
                '中': rl_params.severity_weight_medium,
                '低': rl_params.severity_weight_low
            }
            weighted = sum(severity_weights.get(e.get('severity', '中'),
                            rl_params.severity_weight_medium) for e in all_errors)
            base_score = max(0, 100 - weighted * rl_params.score_penalty_per_weight)
            fusion_ratio = rl_params.llm_score_fusion_ratio
            overall_score = int(base_score * (1 - fusion_ratio) +
                                llm_score * fusion_ratio) if llm_score is not None else base_score
        else:
            base_score = max(0, 100 - len(all_errors) * 8)
            overall_score = int(base_score * 0.5 + llm_score * 0.5) if llm_score is not None else base_score

        # 生成反馈
        feedback = llm_learning_points if llm_learning_points else [
            f'关于"{e.get("description", "")}"——请思考如何修正这个问题。'
            for e in all_errors[:5]
        ]
        if not feedback:
            feedback.append('请仔细检查图纸细节，确保符合GB/T制图标准。')

        # 错误分类统计
        error_categories = {}
        for e in all_errors:
            cat = e.get('type', '其他')
            error_categories[cat] = error_categories.get(cat, 0) + 1

        # OCR和几何检测结果
        ocr_texts = []
        if perception_results.get('ocr') and perception_results['ocr'].success:
            ocr_texts = perception_results['ocr'].data.get('texts', [])

        detection_items = []
        if perception_results.get('geometry') and perception_results['geometry'].success:
            geo = perception_results['geometry'].data
            for l in geo.get('lines', [])[:10]:
                detection_items.append({
                    'class': '直线', 'confidence': 1.0,
                    'bbox': [l['start'][0], l['start'][1], l['end'][0], l['end'][1]]
                })
            for c in geo.get('circles', []):
                detection_items.append({
                    'class': '圆', 'confidence': 1.0,
                    'bbox': [c['center'][0]-c['radius'], c['center'][1]-c['radius'],
                             c['center'][0]+c['radius'], c['center'][1]+c['radius']]
                })

        llm_raw_response = ''
        llm_model_name = 'local_rule_engine'
        if llm_result and getattr(llm_result, 'success', False):
            llm_data_dict = getattr(llm_result, 'data', {})
            if llm_data_dict:
                llm_raw_response = llm_data_dict.get('raw_response', '')
                llm_model_name = llm_data_dict.get('model', 'local_rule_engine')
            logger.info(f"[Phase 4] LLM结果: success=True, raw_response长度={len(llm_raw_response)}, model={llm_model_name}")
        else:
            logger.warning(f"[Phase 4] LLM结果: success=False, 将使用空raw_response")

        return {
            'ocr_results': ocr_texts,
            'detection_results': detection_items,
            'errors': all_errors,
            'feedback': feedback,
            'contract_validation': contract_validation,
            'geo_result': perception_results.get('geometry').data if perception_results.get('geometry') and perception_results['geometry'].success else None,
            'structure_result': perception_results.get('structure').data if perception_results.get('structure') and perception_results['structure'].success else None,
            'api_result': {
                'raw_response': llm_raw_response,
                'model': llm_model_name,
            },
            'report': {
                'total_errors': len(all_errors),
                'error_categories': error_categories,
                'overall_score': overall_score,
                'summary': llm_summary or f"共检测到{len(all_errors)}个问题需要关注"
            }
        }

    def _phase5_judge(self, image_path: str, final_result: Dict) -> JudgeResult:
        """
        Phase 5: VLM评审阶段

        使用VLM对纠错报告进行质量评审。
        """
        logger.info("[Phase 5] VLM Judge: 质量评审")
        return self.vlm_judge.judge(image_path, final_result)

    def _validate_contracts(self, registry: ContractRegistry,
                             errors: List[Dict]) -> Dict:
        """
        预检校验：检查是否所有预检的检测需求都被满足
        """
        validation = {
            'total_contracts': registry.stats['total'],
            'completed': len(registry.get_completed()),
            'failed': len(registry.get_failed()),
            'coverage': {}  # 各错误类别的覆盖情况
        }

        # 检查每个错误类别是否有对应的预检
        error_types_in_result = set(e.get('type', '') for e in errors)
        for contract in registry.all_contracts:
            cat = contract.error_category.value
            if cat not in validation['coverage']:
                validation['coverage'][cat] = {
                    'contract_exists': True,
                    'errors_found': cat in error_types_in_result,
                    'detector': contract.detector_name
                }

        return validation

    def _generate_local_analysis(self, perception_results: Dict, rule_result: Any) -> Any:
        """本地降级分析（当LLM不可用时）"""
        ocr_texts = []
        if perception_results.get('ocr') and perception_results['ocr'].success:
            ocr_texts = perception_results['ocr'].data.get('texts', [])

        rule_errors = rule_result.data.get('errors', []) if rule_result.success else []

        drawing_type = "工程图纸"
        if any('减速' in t.get('text', '') or '轴' in t.get('text', '') for t in ocr_texts):
            drawing_type = "减速器相关图纸"
        elif any('装配' in t.get('text', '') for t in ocr_texts):
            drawing_type = "装配图"

        content_parts = [t.get('text', '') for t in ocr_texts[:10]]
        content_summary = "、".join(content_parts) if content_parts else "未识别到文字内容"

        local_errors = []
        for e in rule_errors:
            local_errors.append({
                'type': e.get('type', ''),
                'description': e.get('description', ''),
                'suggestion': e.get('suggestion', ''),
                'severity': e.get('severity', '中'),
                'gb_reference': e.get('gb_reference', '')
            })

        total = len(local_errors)
        high = sum(1 for e in local_errors if e.get('severity') == '高')
        if total == 0:
            summary = "图纸整体符合机械制图基本规范。"
        elif high > 0:
            summary = f"图纸存在{high}个高严重度问题需优先修正，共{total}个问题。"
        else:
            summary = f"图纸存在{total}个中低严重度问题，建议按规范修正。"

        local_json = json.dumps({
            'drawing_type': drawing_type,
            'content_summary': content_summary,
            'errors': local_errors,
            'overall_score': max(0, 100 - total * 8),
            'summary': summary,
            'learning_points': [
                '请仔细检查图纸细节，确保所有标注符合GB/T制图标准'
            ]
        }, ensure_ascii=False)

        return type('AgentResult', (), {
            'success': True,
            'data': {'raw_response': local_json, 'model': 'local_rule_engine', 'usage': None},
            'errors': [],
            'execution_time_ms': 0,
            'confidence': 0.5
        })()

    def register_feedback(self, session_id: str, feedback_type: str,
                          drawing_type: str = "", error_category: str = "",
                          detection_result: str = "", correction: str = "",
                          accuracy: float = 0.0, helpfulness: float = 0.0):
        """
        注册用户反馈，创建经验案例

        Args:
            feedback_type: confirmed / ignored / dismissed_all / partial_confirm / useful_guidance
        """
        self.experience.create_case_from_feedback(
            session_id=session_id,
            drawing_type=drawing_type,
            error_category=error_category,
            detection_result=detection_result,
            correction=correction,
            feedback_type=feedback_type,
            accuracy=accuracy,
            helpfulness=helpfulness
        )

        # 如果有VLM Judge结果，同时更新RL奖励
        if self._rl_memory:
            rl_reward = {
                'confirmed': 1.0,
                'useful_guidance': 0.5,
                'partial_confirm': 0.3,
                'ignored': -0.5,
                'dismissed_all': -1.0
            }.get(feedback_type, 0.0)
            logger.info(f"[OrchestratorV2] RL反馈: {feedback_type} → reward={rl_reward}")

    @property
    def stats(self) -> Dict:
        return self._stats
