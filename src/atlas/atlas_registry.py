import json
import logging
import os
from typing import Dict, List, Optional

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
    def __init__(self, cases_path: str = "", rules_path: str = ""):
        self.cases_path = cases_path
        self.rules_path = rules_path
        self.cases: List[Dict] = []
        self.rules: List[Dict] = []
        self._case_index: Dict[str, Dict] = {}
        self._rule_index: Dict[str, Dict] = {}
        self._category_rules: Dict[str, List[Dict]] = {}
        self._load_cases(cases_path)
        self._load_rules(rules_path)
        logger.info(
            f"[AtlasRegistry] Loaded: {len(self.cases)} cases, "
            f"{len(self.rules)} rules, "
            f"{len(self._category_rules)} error categories"
        )

    def _load_cases(self, path: str):
        try:
            if not os.path.exists(path):
                logger.warning(f"[AtlasRegistry] Cases file not found: {path}")
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
                                f"[AtlasRegistry] Case line {line_no} parse failed: {e2}"
                            )
                            continue
                    self.cases.append(case)
                    cid = case.get("case_id", "")
                    if cid:
                        self._case_index[cid] = case
        except Exception as e:
            logger.error(f"[AtlasRegistry] Load cases error: {e}")

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
                logger.warning(f"[AtlasRegistry] Rules file not found: {path}")
                return
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
            try:
                import yaml
                data = yaml.safe_load(self._preprocess_yaml(raw))
            except ImportError:
                logger.warning("[AtlasRegistry] PyYAML not installed, skipping rules")
                return
            if not isinstance(data, list):
                logger.warning("[AtlasRegistry] Rules format unexpected, expected list")
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
            logger.error(f"[AtlasRegistry] Load rules error: {e}")

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

    def get_all_cases(self) -> List[Dict]:
        return self.cases

    def get_all_rules(self) -> List[Dict]:
        return self.rules

    def get_active_rules(self) -> List[Dict]:
        return [r for r in self.rules if r.get("enabled", True)]

    def get_rules_by_v2_category(self, category: str,
                                  enabled_only: bool = False,
                                  qa_filter: Optional[List[str]] = None) -> List[Dict]:
        rules = self._category_rules.get(category, [])
        if enabled_only:
            rules = [r for r in rules if r.get("enabled", True)]
        if qa_filter:
            rules = [r for r in rules if r.get("qa_status", "") in qa_filter]
        return rules

    def search_cases(self, query: str, top_k: int = 5) -> List[Dict]:
        query_lower = query.lower()
        scored = []
        for case in self.cases:
            score = 0
            for field in ["case_name", "source_text", "teaching_hint", "suggestion"]:
                val = case.get(field, "")
                if query_lower in val.lower():
                    score += 1
            for kw in case.get("keywords", []):
                if query_lower in kw.lower():
                    score += 2
            if score > 0:
                scored.append((score, case))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    def get_stats(self) -> Dict:
        categories = {}
        for case in self.cases:
            cat = case.get("v2_error_category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
        return {
            "total_cases": len(self.cases),
            "total_rules": len(self.rules),
            "active_rules": len(self.get_active_rules()),
            "case_categories": categories,
            "rule_categories": list(self._category_rules.keys()),
        }
