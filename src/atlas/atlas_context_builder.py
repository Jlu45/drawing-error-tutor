import logging
from typing import Dict, List, Optional

from .atlas_registry import AtlasRegistry
from .atlas_retriever import AtlasRetriever
from .atlas_rule_pack import AtlasRulePack
from .atlas_matcher import AtlasMatcher
from .atlas_case_schema import AtlasCase

logger = logging.getLogger("AtlasContextBuilder")

try:
    from config_loader import (
        ATLAS_MAX_CONTEXT_CASES as _MAX_CASES,
        ATLAS_SHOW_REFERENCE_IN_UI as _SHOW_REF
    )
    MAX_CONTEXT_CASES = _MAX_CASES
    SHOW_REFERENCE = _SHOW_REF
except ImportError:
    MAX_CONTEXT_CASES = 3
    SHOW_REFERENCE = True


class AtlasContextBuilder:
    def __init__(self, registry: AtlasRegistry, retriever: Optional[AtlasRetriever] = None,
                 rule_pack: Optional[AtlasRulePack] = None,
                 matcher: Optional[AtlasMatcher] = None):
        self.registry = registry
        self.retriever = retriever or AtlasRetriever(registry)
        self.rule_pack = rule_pack or AtlasRulePack(registry)
        self.matcher = matcher or AtlasMatcher(registry)
        self.max_cases = MAX_CONTEXT_CASES
        self.show_reference = SHOW_REFERENCE

    def build_context(self, analysis_result: Dict) -> Dict:
        similar_cases = self.retriever.retrieve_similar(analysis_result, top_k=self.max_cases)

        errors = analysis_result.get('errors', [])
        geo = analysis_result.get('geo_result', {})
        context = {
            'line_count': len(geo.get('lines', [])) if geo else 0,
            'circle_count': len(geo.get('circles', [])) if geo else 0,
            'arrow_count': len(geo.get('arrows', [])) if geo else 0
        }

        rule_enhanced_errors = self.rule_pack.apply_rules(errors, context)

        matched = self.matcher.match_errors(errors, context)

        atlas_context = {
            'similar_cases': self._format_cases(similar_cases),
            'rule_enhanced_errors': rule_enhanced_errors,
            'matched_patterns': self._format_matches(matched),
            'reference_suggestions': self._collect_suggestions(similar_cases, matched)
        }

        if self.show_reference:
            atlas_context['reference_cases'] = [
                {
                    'case_id': case.case_id,
                    'category': case.category,
                    'error_type': case.error_type,
                    'corrections': case.corrections,
                    'gb_references': case.gb_references
                }
                for case, score in similar_cases
            ]

        return atlas_context

    def build_llm_context(self, analysis_result: Dict) -> str:
        atlas_ctx = self.build_context(analysis_result)

        parts = []

        if atlas_ctx.get('similar_cases'):
            parts.append("【历史相似案例】")
            for i, case_info in enumerate(atlas_ctx['similar_cases'][:3], 1):
                parts.append(f"  案例{i}: {case_info['category']}/{case_info['error_type']} "
                              f"(相似度:{case_info['score']:.2f})")
                for corr in case_info.get('corrections', [])[:2]:
                    parts.append(f"    修正: {corr.get('description', '')}")

        if atlas_ctx.get('reference_suggestions'):
            parts.append("\n【Atlas参考建议】")
            for sug in atlas_ctx['reference_suggestions'][:5]:
                parts.append(f"  - {sug}")

        if atlas_ctx.get('matched_patterns'):
            parts.append("\n【匹配的规则模式】")
            for match in atlas_ctx['matched_patterns'][:3]:
                best = match.get('best_match')
                if best:
                    parts.append(f"  - [{best.get('type', '')}] {best.get('category', '')}: "
                                  f"{best.get('error_type', '')} "
                                  f"(置信度:{best.get('confidence', best.get('score', 0)):.2f})")

        return "\n".join(parts) if parts else ""

    def _format_cases(self, cases: list) -> List[Dict]:
        formatted = []
        for case, score in cases:
            formatted.append({
                'case_id': case.case_id,
                'category': case.category,
                'error_type': case.error_type,
                'description': case.description,
                'score': round(score, 3),
                'corrections': case.corrections,
                'gb_references': case.gb_references,
                'confidence': case.confidence
            })
        return formatted

    def _format_matches(self, matched: List[Dict]) -> List[Dict]:
        formatted = []
        for m in matched:
            entry = {
                'error_description': m['error'].get('description', ''),
                'error_type': m['error'].get('type', ''),
                'best_match': m.get('best_match')
            }
            formatted.append(entry)
        return formatted

    def _collect_suggestions(self, similar_cases: list,
                              matched: List[Dict]) -> List[str]:
        suggestions = []

        for case, score in similar_cases:
            for corr in case.corrections:
                sug = corr.get('description', '')
                if sug and sug not in suggestions:
                    suggestions.append(sug)

        for m in matched:
            best = m.get('best_match')
            if best and best.get('action', {}).get('suggestion'):
                sug = best['action']['suggestion']
                if sug not in suggestions:
                    suggestions.append(sug)

        return suggestions[:10]
