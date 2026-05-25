"""
Atlas 图册案例/规则统一加载器
==============================
从 atlas_cases.jsonl 和 atlas_rules.yaml 加载数据，
构建多维度索引以支持高效查询。
"""

import json
import logging
import os
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger("AtlasRegistry")

V2_CATEGORY_MAP = {
    "尺寸标注": "DIMENSION_ERROR",
    "线型": "LINE_TYPE_ERROR",
    "公差": "TOLERANCE_ERROR",
    "标题栏": "TITLE_BLOCK_ERROR",
    "符号": "SYMBOL_ERROR",
    "几何完整性": "GEOMETRY_INCOMPLETE_ERROR",
    "图纸结构": "STRUCTURE_ERROR",
    "表面粗糙度": "SURFACE_ERROR",
    "焊接符号": "WELD_ERROR",
    "图幅规范": "SHEET_ERROR",
    "视图标注": "VIEW_ERROR",
    "其他": "GENERAL_ERROR",
}


class AtlasRegistry:
    """图册案例/规则统一加载与索引"""

    def __init__(self, cases_path: str, rules_path: str):
        self.cases: List[Dict] = []
        self.rules: List[Dict] = []
        self._case_index: Dict[str, Dict] = {}
        self._rule_index: Dict[str, Dict] = {}
        self._category_rules: Dict[str, List[Dict]] = {}
        self._load_cases(cases_path)
        self._load_rules(rules_path)
        logger.info(
            f"[AtlasRegistry] 加载完成: {len(self.cases)} 条案例, "
            f"{len(self.rules)} 条规则, "
            f"{len(self._category_rules)} 个错误类别"
        )

    def _load_cases(self, path: str):
        try:
            if not os.path.exists(path):
                logger.warning(f"[AtlasRegistry] 案例文件不存在: {path}")
                return
            with open(path, "r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        case = json.loads(line)
                    except json.JSONDecodeError:
                        try:
                            fixed_line = self._fix_json_inner_quotes(line)
                            case = json.loads(fixed_line)
                        except (json.JSONDecodeError, Exception) as e2:
                            logger.warning(
                                f"[AtlasRegistry] 案例文件第{line_no}行解析失败: {e2}"
                            )
                            continue
                    self.cases.append(case)
                    cid = case.get("case_id", "")
                    if cid:
                        self._case_index[cid] = case
        except Exception as e:
            logger.error(f"[AtlasRegistry] 加载案例文件异常: {e}")

    @staticmethod
    def _fix_json_inner_quotes(line: str) -> str:
        result = []
        in_string = False
        i = 0
        while i < len(line):
            c = line[i]
            if c == '\\' and in_string and i + 1 < len(line):
                result.append(c)
                result.append(line[i + 1])
                i += 2
                continue
            if c == '"':
                if not in_string:
                    in_string = True
                    result.append(c)
                else:
                    ahead = line[i + 1:i + 3] if i + 1 < len(line) else ""
                    if ahead and ahead[0] in (',', '}', ']'):
                        in_string = False
                        result.append(c)
                    elif ahead and ahead[0] == ':':
                        in_string = False
                        result.append(c)
                    elif i + 1 == len(line):
                        in_string = False
                        result.append(c)
                    else:
                        result.append("'")
            else:
                result.append(c)
            i += 1
        return "".join(result)

    def _load_rules(self, path: str):
        try:
            if not os.path.exists(path):
                logger.warning(f"[AtlasRegistry] 规则文件不存在: {path}")
                return
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
            data = yaml.safe_load(self._preprocess_yaml(raw))
            if not isinstance(data, list):
                logger.warning("[AtlasRegistry] 规则文件格式异常，期望列表")
                return
            for rule in data:
                if not isinstance(rule, dict):
                    continue
                self.rules.append(rule)
                rid = rule.get("rule_id", "")
                if rid:
                    self._rule_index[rid] = rule
                cat = rule.get("v2_error_category", "")
                if cat:
                    self._category_rules.setdefault(cat, []).append(rule)
        except Exception as e:
            logger.error(f"[AtlasRegistry] 加载规则文件异常: {e}")

    @staticmethod
    def _preprocess_yaml(raw: str) -> str:
        raw = raw.replace("\u201c", "\u300c").replace("\u201d", "\u300d")
        raw = raw.replace("\u2018", "\u300e").replace("\u2019", "\u300f")
        fixed_lines = []
        for line in raw.split("\n"):
            stripped = line.lstrip()
            colon_idx = line.find(":")
            if colon_idx >= 0:
                after_colon = line[colon_idx + 1:]
                value_part = after_colon.strip()
                if (value_part.startswith('"') and value_part.endswith('"')
                        and len(value_part) > 2):
                    inner = value_part[1:-1]
                    if '"' in inner:
                        inner = inner.replace('"', "'")
                        indent_part = line[:colon_idx + 1] + after_colon[:len(after_colon) - len(after_colon.lstrip())]
                        line = indent_part + '"' + inner + '"'
            fixed_lines.append(line)
        return "\n".join(fixed_lines)

    def get_case(self, case_id: str) -> Optional[Dict]:
        return self._case_index.get(case_id)

    def get_rule(self, rule_id: str) -> Optional[Dict]:
        return self._rule_index.get(rule_id)

    def get_rules_by_v2_category(
        self, category: str, enabled_only: bool = True, qa_filter: bool = True
    ) -> List[Dict]:
        rules = self._category_rules.get(category, [])
        if enabled_only:
            rules = [r for r in rules if r.get("enabled", False)]
        if qa_filter:
            valid_statuses = {"reviewed", "published"}
            rules = [
                r
                for r in rules
                if self._get_case_qa_status(r.get("source_case_id", ""))
                in valid_statuses
            ]
        return rules

    def _get_case_qa_status(self, case_id: str) -> str:
        case = self._case_index.get(case_id)
        if case:
            return case.get("qa_status", "")
        return ""

    def get_cases_by_rule(self, rule_id: str) -> List[Dict]:
        rule = self._rule_index.get(rule_id)
        if not rule:
            return []
        source_case_id = rule.get("source_case_id", "")
        result = []
        for case in self.cases:
            if rule_id in case.get("rule_ids", []):
                result.append(case)
            elif source_case_id and case.get("case_id") == source_case_id:
                result.append(case)
        return result

    def get_active_rules(self) -> List[Dict]:
        valid_statuses = {"reviewed", "published"}
        result = []
        for rule in self.rules:
            if not rule.get("enabled", False):
                continue
            source_case_id = rule.get("source_case_id", "")
            qa_status = self._get_case_qa_status(source_case_id)
            if qa_status in valid_statuses:
                result.append(rule)
        return result

    def search_cases(self, keywords: List[str], top_k: int = 5) -> List[Dict]:
        if not keywords:
            return []
        scored = []
        kw_lower = [k.lower() for k in keywords]
        for case in self.cases:
            score = 0.0
            case_keywords = [k.lower() for k in case.get("keywords", [])]
            case_name = case.get("case_name", "").lower()
            source_text = case.get("source_text", "").lower()
            for kw in kw_lower:
                for ck in case_keywords:
                    if kw in ck or ck in kw:
                        score += 2.0
                        break
                else:
                    if kw in case_name:
                        score += 1.0
                    elif kw in source_text:
                        score += 0.5
            if score > 0:
                scored.append((score, case))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    @staticmethod
    def v2_category_from_error_category(error_category_value: str) -> str:
        return V2_CATEGORY_MAP.get(error_category_value, "GENERAL_ERROR")
