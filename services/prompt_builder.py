from typing import List, Dict, Optional
from models.entities import TranslationSegment, TMMatch, TermMatch
import os

class PromptBuilder:
    """
    Builds prompts for translation with support for:
    - Custom prompt templates
    - Chat history (previous translations as context)
    - TM and TB context
    """
    
    def __init__(self, template_path: Optional[str] = None, custom_template: Optional[str] = None):
        """
        Initialize PromptBuilder.
        
        Args:
            template_path: Path to template file (used if custom_template not provided)
            custom_template: Direct template string (takes priority over template_path)
        """
        self.template = ""
        
        if custom_template:
            self.template = custom_template
        elif template_path and os.path.exists(template_path):
            with open(template_path, 'r', encoding='utf-8') as f:
                self.template = f.read()
        else:
            self.template = self._get_default_template()
    
    def set_custom_template(self, template_content: str):
        """Set a custom template from string content"""
        self.template = template_content
    
    def build_prompt(self, 
                     source_lang: str, 
                     target_lang: str, 
                     segments: List[TranslationSegment],
                     tm_context: Dict[str, List[TMMatch]], 
                     tb_context: Dict[str, List[TermMatch]],
                     chat_history: Optional[List[Dict[str, str]]] = None,
                     reference_context: Optional[str] = None,
                     dnt_terms: Optional[List[str]] = None) -> str:
        """
        Build the translation prompt.
        
        Args:
            source_lang: Source language name
            target_lang: Target language name
            segments: List of segments to translate
            tm_context: TM matches for each segment
            tb_context: TB matches for each segment
            chat_history: List of previous translations [{source, target}, ...]
            reference_context: Target language style reference text
            dnt_terms: List of terms that should NOT be translated
        
        Returns:
            Complete prompt string
        """
        
        # 0. Format Reference Context (target language style samples)
        reference_text = ""
        if reference_context:
            reference_text = f"STYLE REFERENCE (use these {target_lang} samples as style/tone guide):\n"
            reference_text += reference_context + "\n\n"
        
        # 0.5 Format DNT (Do Not Translate) list
        dnt_text = ""
        if dnt_terms and len(dnt_terms) > 0:
            dnt_text = "DO NOT TRANSLATE (keep these terms in original form):\n"
            for term in dnt_terms[:50]:  # Limit to 50 terms to save tokens
                dnt_text += f"- {term}\n"
            dnt_text += "\n"
        
        # 1. Format Chat History (previous translations for consistency)
        history_text = ""
        if chat_history and len(chat_history) > 0:
            history_text = "PREVIOUS TRANSLATIONS (for consistency):\n"
            for item in chat_history:
                history_text += f"{item['source']} -> {item['target']}\n"
            history_text += "\n"
        
        # 2. Format TM Context
        examples_text = ""
        unique_tm = {}
        
        for seg in segments:
            if seg.id in tm_context:
                for match in tm_context[seg.id]:
                    key = f"{match.source_text}->{match.target_text}"
                    unique_tm[key] = match
        
        if unique_tm:
            examples_text = "TRANSLATION MEMORY CONTEXT:\n"
            sorted_tm = sorted(unique_tm.values(), key=lambda x: x.similarity, reverse=True)
            for m in sorted_tm[:15]:
                examples_text += f"{m.source_text} -> {m.target_text} [{m.match_type} {m.similarity:.0f}%]\n"
        else:
            examples_text = "No Translation Memory matches available."
        
        # 3. Format TB Context
        terms_text = ""
        unique_tb = {}
        
        for seg in segments:
            if seg.id in tb_context:
                for match in tb_context[seg.id]:
                    unique_tb[match.source] = match.target
        
        if unique_tb:
            terms_text = "REQUIRED TERMINOLOGY:\n"
            for src, tgt in unique_tb.items():
                terms_text += f"- {src} = {tgt}\n"
        else:
            terms_text = "No specific terminology constraints."
        
        # 4. Format Segments
        seg_text = ""
        for seg in segments:
            seg_text += f"[{seg.id}] {seg.source}\n"
        
        # 5. Build prompt from template
        prompt = self.template
        
        # Replace placeholders
        prompt = prompt.replace("%SOURCELANG%", source_lang)
        prompt = prompt.replace("%TARGETLANG%", target_lang)
        prompt = prompt.replace("%EXAMPLES%", examples_text)
        prompt = prompt.replace("%TERMS%", terms_text)
        
        # Handle FORBIDDENTERMS placeholder
        if "%FORBIDDENTERMS%" in prompt:
            prompt = prompt.replace("%FORBIDDENTERMS%", dnt_text if dnt_text else "")
        elif dnt_text:
            # No placeholder in template, insert DNT after terms
            prompt = prompt.replace(terms_text, terms_text + "\n" + dnt_text)
        
        # Handle reference context - insert at the beginning after language instructions
        if reference_text:
            # Find a good insertion point - after first paragraph or after language info
            if "%EXAMPLES%" in self.template:
                # Insert before examples
                prompt = prompt.replace(examples_text, reference_text + examples_text)
            else:
                # Insert after the first instruction block
                first_newline = prompt.find('\n\n')
                if first_newline > 0:
                    prompt = prompt[:first_newline] + "\n\n" + reference_text + prompt[first_newline:]
        
        # Handle chat history - insert before examples if template has %EXAMPLES%
        if history_text:
            if "%EXAMPLES%" in self.template:
                # History was already placed, now add it before EXAMPLES content
                current_examples = reference_text + examples_text if reference_text else examples_text
                prompt = prompt.replace(current_examples, history_text + current_examples)
            else:
                # Add history at the beginning of context section
                prompt = prompt.replace(examples_text, history_text + examples_text)
        
        # Handle %SEGMENTS% - check if template already has the header
        if "%SEGMENTS%" in prompt:
            # Check if "SEGMENTS TO TRANSLATE" already exists before %SEGMENTS%
            if "SEGMENTS TO TRANSLATE:" in self.template and "%SEGMENTS%" in self.template:
                # Template has header, just replace placeholder with segments
                prompt = prompt.replace("%SEGMENTS%", seg_text)
            else:
                # Template doesn't have header, add it
                segments_block = "SEGMENTS TO TRANSLATE:\n" + seg_text
                prompt = prompt.replace("%SEGMENTS%", segments_block)
        else:
            # No placeholder, append segments to end
            prompt = prompt.rstrip() + "\n\nSEGMENTS TO TRANSLATE:\n" + seg_text
        
        # Add output format instruction if not present
        if "OUTPUT FORMAT:" not in prompt and "[ID]" not in prompt:
            prompt += "\nOUTPUT FORMAT:\n[ID] Translated text\n"
        
        return prompt
    
    def _get_default_template(self):
        return """You are a professional translator translating from %SOURCELANG% to %TARGETLANG%.
Your task is to translate the segments provided ensuring high accuracy and natural flow.

%EXAMPLES%

%TERMS%

INSTRUCTIONS:
1. Preserve all tags (e.g., {{1}}, {{2}}) exactly as they appear in source.
2. Use the provided Terminology and TM context where applicable.
3. Output format must be strictly: [ID] Translated Text

%SEGMENTS%
"""
