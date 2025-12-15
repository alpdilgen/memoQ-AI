"""
Prompt builder with structured batch formatting
"""

import json
from typing import List, Dict, Any, Optional


class PromptBuilder:
    """Build structured prompts with TM context and terminology"""
    
    def __init__(self, template_path: str = None, custom_template: str = None):
        if custom_template:
            self.template = custom_template
        elif template_path:
            with open(template_path, 'r', encoding='utf-8') as f:
                self.template = f.read()
        else:
            self.template = self._default_template()
    
    def _default_template(self) -> str:
        """Default prompt template"""
        return """You are a senior {source_lang}-to-{target_lang} translator with deep domain knowledge.

Translate the provided segments with accuracy, fluency, and brand consistency.

RULES:
- Preserve the exact structure and all tags ({{1}}, {{2}}, etc.)
- Do not add or remove spaces, line breaks, or formatting
- Maintain consistency across all segments
- Use the examples below as reference for style
- Apply terminology consistently from the provided list
- Avoid using forbidden terms

Return ONLY the translations in the specified format.
"""
    
    def build_prompt(self, source_lang: str, target_lang: str, segments: List, 
                    tm_context: Dict = None, tb_context: Dict = None,
                    chat_history: List = None, reference_context: str = None,
                    dnt_terms: List = None) -> str:
        """Build structured batch prompt"""
        
        if tm_context is None:
            tm_context = {}
        if tb_context is None:
            tb_context = {}
        if chat_history is None:
            chat_history = []
        if dnt_terms is None:
            dnt_terms = []
        
        # Build sections
        output = []
        
        # Header
        output.append("=" * 80)
        output.append(f"BATCH: {len(segments)} SEGMENTS")
        output.append(f"Source: {source_lang} | Target: {target_lang}")
        output.append("=" * 80)
        output.append("")
        
        # Segments to translate
        output.append("SEGMENTS TO TRANSLATE:")
        output.append("-" * 80)
        for seg in segments:
            output.append(f"[{seg.id}] {seg.source}")
        output.append("")
        
        # TM Examples (if any matches)
        if tm_context:
            output.append("EXAMPLES FROM TRANSLATION MEMORY:")
            output.append("-" * 80)
            for seg in segments:
                if seg.id in tm_context:
                    matches = tm_context[seg.id]
                    for match in matches[:1]:  # Show top match only
                        target = match.get('TargetSegment', match.get('target', ''))
                        match_rate = match.get('MatchRate', match.get('match_rate', 0))
                        output.append(f"({match_rate}% match)")
                        output.append(f"{seg.source}")
                        output.append(f"{target}")
                        output.append("")
        
        # Terminology
        terminology = self._extract_terminology(tm_context, tb_context)
        if terminology:
            output.append("TERMINOLOGY:")
            output.append("-" * 80)
            for source_term, target_term in sorted(terminology.items()):
                output.append(f"{source_term} = {target_term}")
            output.append("")
        
        # Forbidden terms
        if dnt_terms:
            output.append("DO NOT TRANSLATE:")
            output.append("-" * 80)
            for term in dnt_terms:
                output.append(f"• {term}")
            output.append("")
        
        # Output format
        output.append("OUTPUT FORMAT:")
        output.append("-" * 80)
        output.append("[ID] Translated text")
        output.append("")
        
        # JSON payload
        output.append("JSON PAYLOAD:")
        output.append("-" * 80)
        payload = {f"segment{seg.id}": seg.source for seg in segments}
        output.append(json.dumps(payload, indent=2, ensure_ascii=False))
        output.append("")
        output.append("=" * 80)
        
        return "\n".join(output)
    
    def _extract_terminology(self, tm_context: Dict, tb_context: Dict) -> Dict[str, str]:
        """Extract terminology from TM and TB matches"""
        terminology = {}
        
        # From TB
        for seg_id, matches in tb_context.items():
            if isinstance(matches, list):
                for match in matches:
                    if isinstance(match, dict):
                        source = match.get('source', match.get('Source', ''))
                        target = match.get('target', match.get('Target', ''))
                        if source and target:
                            terminology[source] = target
        
        return terminology
