from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

# NOTE: Do NOT import TranslationSegment or anything from 'models.entities' here.
# This file DEFINES those classes.

@dataclass
class TMMatch:
    """Translation Memory match"""
    source_text: str
    target_text: str
    similarity: float  # 0-100
    match_type: str    # "EXACT", "SEMANTIC", "FUZZY"
    origin: str = "TM"
    
    def __repr__(self):
        return f"{self.source_text} -> {self.target_text} ({self.similarity:.1f}% {self.match_type})"

@dataclass
class TermMatch:
    """Terminology match"""
    source: str
    target: str
    
    def __repr__(self):
        return f"{self.source} = {self.target}"

@dataclass
class TranslationSegment:
    """XLIFF segment representation"""
    id: str
    source: str
    target: str = ""
    status: str = "new"
    tags_source: List[str] = field(default_factory=list)
    # Stores mapping of {{1}} -> XML Element for reconstruction
    tag_map: Dict[str, Any] = field(default_factory=dict) 
    
@dataclass
class QAError:
    """QA Error representation"""
    code: int
    segment_id: str
    description: str
    severity: str = "warning"
    original_target: str = ""
    fixed_target: str = ""
    status: str = "detected" # detected, fixed, manual_check

@dataclass
class TranslationResult:
    """Result from AI translation"""
    segment_id: str
    original_source: str
    translated_target: str
    tm_matches: List[TMMatch]
    term_matches: List[TermMatch]
    ai_model_used: str
    tokens_used: int