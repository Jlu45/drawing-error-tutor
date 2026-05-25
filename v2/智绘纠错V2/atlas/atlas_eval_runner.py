"""
Atlas 图册规则自动评测器
========================
对 OrchestratorV2 的输出进行结构化评测，
衡量图册规则在真实错误图片上的召回率、规则命中率、关键词覆盖率和误报率。

评测维度：
1. category_hit   — 错误类别是否命中
2. rule_hit       — 图册规则 ID 是否命中
3. suggestion_hit — 修正建议/关键词是否命中
4. no_wrong_cat   — 不应出现的类别是否被误报
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional

logger = logging.getLogger("AtlasEvalRunner")


class AtlasEvalRunner:

    def __init__(self, atlas_registry, eval_cases_path: str = ""):
        self.registry = atlas_registry
        self.eval_cases: List[Dict] = []
        if eval_cases_path:
            self._load_eval_cases(eval_cases_path)

    def _load_eval_cases(self, path: str):
        if not os.path.exists(path):
            logger.warning(f"[AtlasEvalRunner] 评测案例文件不存在: {path}")
            return
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    case = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"[AtlasEvalRunner] 第{line_no}行解析失败: {e}"
                    )
                    continue
                self.eval_cases.append(case)
        logger.info(
            f"[AtlasEvalRunner] 加载评测案例: {len(self.eval_cases)} 条"
        )

    def run_eval(self, orchestrator, test_images_dir: str = "") -> Dict:
        results = {
            "total_cases": len(self.eval_cases),
            "enabled_cases": 0,
            "category_recall": {},
            "rule_hit_rate": 0.0,
            "false_positive_per_case": 0.0,
            "confirmed_precision": 0.0,
            "atlas_case_display_rate": 0.0,
            "details": [],
        }

        if not self.eval_cases:
            logger.warning("[AtlasEvalRunner] 无评测案例，跳过评测")
            return results

        category_stats: Dict[str, Dict] = {}
        total_rule_hits = 0
        total_fp_count = 0
        total_confirmed = 0
        total_atlas_display = 0
        enabled_count = 0

        for eval_case in self.eval_cases:
            eval_id = eval_case.get("eval_id", "UNKNOWN")
            input_image = eval_case.get("input_image", "")
            expected = eval_case.get("expected", {})
            negative = eval_case.get("negative", {})
            score_weight = eval_case.get("score_weight", {})

            if not input_image:
                logger.warning(f"[AtlasEvalRunner] {eval_id}: 无 input_image，跳过")
                continue

            image_path = input_image
            if test_images_dir and not os.path.isabs(input_image):
                image_path = os.path.join(test_images_dir, input_image)

            if not os.path.exists(image_path):
                logger.warning(
                    f"[AtlasEvalRunner] {eval_id}: 图片不存在 {image_path}，跳过"
                )
                continue

            enabled_count += 1

            try:
                t0 = time.time()
                analysis = orchestrator.analyze(image_path)
                elapsed_ms = (time.time() - t0) * 1000
            except Exception as e:
                logger.error(f"[AtlasEvalRunner] {eval_id}: 分析异常 {e}")
                detail = self._make_detail(
                    eval_case, [], False, False, False, True, elapsed_ms=0
                )
                results["details"].append(detail)
                continue

            errors = analysis.get("errors", [])

            cat_hit = self._check_category_hit(errors, expected)
            rule_hit = self._check_rule_hit(errors, expected)
            kw_hit = self._check_keywords(errors, expected)
            neg_violation = self._check_negative(errors, negative)

            expected_cat = expected.get("error_category", "")
            if expected_cat not in category_stats:
                category_stats[expected_cat] = {
                    "total": 0,
                    "category_hit": 0,
                    "rule_hit": 0,
                    "keyword_hit": 0,
                }
            category_stats[expected_cat]["total"] += 1
            if cat_hit:
                category_stats[expected_cat]["category_hit"] += 1
            if rule_hit:
                category_stats[expected_cat]["rule_hit"] += 1
            if kw_hit:
                category_stats[expected_cat]["keyword_hit"] += 1

            if rule_hit:
                total_rule_hits += 1

            fp_count = 0
            if neg_violation:
                fp_count = len(neg_violation)
            total_fp_count += fp_count

            confirmed_errors = [
                e for e in errors
                if e.get("level") == "confirmed_error"
                or e.get("severity") == "高"
            ]
            if confirmed_errors:
                total_confirmed += 1

            atlas_errors = [
                e for e in errors if e.get("source") == "atlas_rule"
            ]
            if atlas_errors:
                total_atlas_display += 1

            w = score_weight or {
                "category_hit": 0.3,
                "rule_hit": 0.3,
                "suggestion_hit": 0.2,
                "no_wrong_category": 0.2,
            }
            w_cat = w.get("category_hit", 0.3)
            w_rule = w.get("rule_hit", 0.3)
            w_kw = w.get("suggestion_hit", 0.2)
            w_neg = w.get("no_wrong_category", 0.2)
            case_score = (
                w_cat * (1.0 if cat_hit else 0.0)
                + w_rule * (1.0 if rule_hit else 0.0)
                + w_kw * (1.0 if kw_hit else 0.0)
                + w_neg * (0.0 if neg_violation else 1.0)
            )

            detail = self._make_detail(
                eval_case,
                errors,
                cat_hit,
                rule_hit,
                kw_hit,
                neg_violation,
                case_score=round(case_score, 3),
                elapsed_ms=round(elapsed_ms, 1),
                fp_count=fp_count,
            )
            results["details"].append(detail)

        results["enabled_cases"] = enabled_count

        for cat, stats in category_stats.items():
            total = max(stats["total"], 1)
            results["category_recall"][cat] = {
                "total": stats["total"],
                "category_hit": stats["category_hit"],
                "category_recall": round(
                    stats["category_hit"] / total, 3
                ),
                "rule_hit": stats["rule_hit"],
                "rule_hit_rate": round(stats["rule_hit"] / total, 3),
                "keyword_hit": stats["keyword_hit"],
                "keyword_hit_rate": round(stats["keyword_hit"] / total, 3),
            }

        results["rule_hit_rate"] = round(
            total_rule_hits / max(enabled_count, 1), 3
        )
        results["false_positive_per_case"] = round(
            total_fp_count / max(enabled_count, 1), 3
        )
        results["confirmed_precision"] = round(
            total_confirmed / max(enabled_count, 1), 3
        )
        results["atlas_case_display_rate"] = round(
            total_atlas_display / max(enabled_count, 1), 3
        )

        return results

    def _check_category_hit(self, errors: List[Dict], expected: Dict) -> bool:
        expected_cat = expected.get("error_category", "")
        if not expected_cat:
            return False
        for err in errors:
            err_type = err.get("type", "")
            if err_type == expected_cat:
                return True
        return False

    def _check_rule_hit(self, errors: List[Dict], expected: Dict) -> bool:
        expected_rule = expected.get("atlas_rule_id", "")
        if not expected_rule:
            return False
        for err in errors:
            rule_id = err.get("atlas_rule_id", "")
            if rule_id == expected_rule:
                return True
        return False

    def _check_keywords(self, errors: List[Dict], expected: Dict) -> bool:
        keywords = expected.get("keywords", [])
        if not keywords:
            return True
        matched = 0
        for kw in keywords:
            for err in errors:
                desc = err.get("description", "")
                suggestion = err.get("suggestion", "")
                title = err.get("title", "")
                combined = f"{desc} {suggestion} {title}"
                if kw in combined:
                    matched += 1
                    break
        if not keywords:
            return True
        return matched >= max(1, len(keywords) // 2)

    def _check_negative(
        self, errors: List[Dict], negative: Dict
    ) -> List[str]:
        must_not = negative.get("must_not_categories", [])
        if not must_not:
            return []
        violations = []
        for err in errors:
            err_type = err.get("type", "")
            if err_type in must_not:
                violations.append(err_type)
        return list(set(violations))

    def _make_detail(
        self,
        eval_case: Dict,
        errors: List[Dict],
        cat_hit: bool,
        rule_hit: bool,
        kw_hit: bool,
        neg_violation,
        analysis_error: bool = False,
        case_score: float = 0.0,
        elapsed_ms: float = 0.0,
        fp_count: int = 0,
    ) -> Dict:
        return {
            "eval_id": eval_case.get("eval_id", ""),
            "input_image": eval_case.get("input_image", ""),
            "expected_category": eval_case.get("expected", {}).get(
                "error_category", ""
            ),
            "expected_rule": eval_case.get("expected", {}).get(
                "atlas_rule_id", ""
            ),
            "category_hit": cat_hit,
            "rule_hit": rule_hit,
            "keyword_hit": kw_hit,
            "negative_violation": neg_violation if neg_violation else [],
            "false_positive_count": fp_count,
            "case_score": case_score,
            "detected_error_count": len(errors),
            "analysis_error": analysis_error,
            "elapsed_ms": elapsed_ms,
        }

    def format_report(self, results: Dict) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("  AtlasPack-V2 Eval Report")
        lines.append("=" * 60)
        lines.append("")

        lines.append(f"  Total Cases:       {results['total_cases']}")
        lines.append(f"  Enabled Cases:     {results['enabled_cases']}")
        lines.append(f"  Rule Hit Rate:     {results['rule_hit_rate']:.1%}")
        lines.append(
            f"  Confirmed Prec:    {results['confirmed_precision']:.1%}"
        )
        lines.append(
            f"  FP Per Case:       {results['false_positive_per_case']:.3f}"
        )
        lines.append(
            f"  Atlas Display:     {results['atlas_case_display_rate']:.1%}"
        )
        lines.append("")

        lines.append("-" * 60)
        lines.append("  Category Breakdown")
        lines.append("-" * 60)
        cat_recall = results.get("category_recall", {})
        for cat, stats in sorted(cat_recall.items()):
            lines.append(
                f"  {cat:<24s}  "
                f"recall={stats['category_recall']:.1%}  "
                f"rule={stats['rule_hit_rate']:.1%}  "
                f"kw={stats['keyword_hit_rate']:.1%}  "
                f"({stats['category_hit']}/{stats['total']})"
            )
        lines.append("")

        lines.append("-" * 60)
        lines.append("  Per-Case Details")
        lines.append("-" * 60)
        for d in results.get("details", []):
            eid = d["eval_id"]
            cat_mark = "V" if d["category_hit"] else "X"
            rule_mark = "V" if d["rule_hit"] else "X"
            kw_mark = "V" if d["keyword_hit"] else "X"
            neg_mark = (
                "!" if d.get("negative_violation") else "-"
            )
            lines.append(
                f"  {eid:<40s}  "
                f"cat={cat_mark} rule={rule_mark} "
                f"kw={kw_mark} neg={neg_mark}  "
                f"score={d['case_score']:.3f}  "
                f"errs={d['detected_error_count']}  "
                f"fp={d['false_positive_count']}"
            )
            if d.get("negative_violation"):
                lines.append(
                    f"    -> 误报类别: {d['negative_violation']}"
                )
            if d.get("analysis_error"):
                lines.append(f"    -> 分析异常")

        lines.append("")
        lines.append("=" * 60)

        avg_score = 0.0
        details = results.get("details", [])
        if details:
            avg_score = sum(d["case_score"] for d in details) / len(details)
        lines.append(f"  Avg Case Score:    {avg_score:.3f}")
        lines.append("=" * 60)

        return "\n".join(lines)
