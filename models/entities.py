"""
Data models and entity classes for translation system
Supports local TMX, CSV termbases, and memoQ Server integration
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class TranslationSegment:
    """Represents a single XLIFF translation segment"""
    id: str
    source: str
    target: str = ""
    tag_map: Optional[Dict[str, str]] = None
    
    def __post_init__(self):
        """Validate segment"""
        if not self.id:
            raise ValueError("Segment ID cannot be empty")
        if not self.source:
            raise ValueError("Source text cannot be empty")


@dataclass
class TMMatch:
    """
    Universal Translation Memory Match object
    Works with local TMX files, memoQ Server, or any TM source
    """
    source_text: str
    target_text: str
    similarity: int  # 0-100 percentage
    match_type: str = "FUZZY"  # "EXACT" (100%), "FUZZY" (<100%), "CONTEXT"
    
    # Optional metadata
    source_file: Optional[str] = None
    project: Optional[str] = None
    domain: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Clean and validate match data"""
        # Strip whitespace
        self.source_text = self.source_text.strip() if self.source_text else ""
        self.target_text = self.target_text.strip() if self.target_text else ""
        
        # Validate similarity
        if not isinstance(self.similarity, int) or self.similarity < 0 or self.similarity > 100:
            raise ValueError(f"Invalid similarity: {self.similarity}, must be 0-100")
        
        # Validate match type
        valid_types = ["EXACT", "FUZZY", "CONTEXT"]
        if self.match_type not in valid_types:
            raise ValueError(f"Invalid match_type: {self.match_type}, must be one of {valid_types}")
        
        # Auto-set match type based on similarity if needed
        if self.similarity == 100 and self.match_type != "EXACT":
            self.match_type = "EXACT"
    
    def is_valid(self) -> bool:
        """Check if match has valid source and target"""
        return bool(self.source_text and self.target_text)
    
    def __repr__(self):
        src = self.source_text[:40] if self.source_text else "empty"
        tgt = self.target_text[:40] if self.target_text else "empty"
        return f"TMMatch('{src}...' → '{tgt}...' [{self.match_type} {self.similarity}%])"
    
    def __hash__(self):
        return hash((self.source_text, self.target_text, self.similarity))
    
    def __eq__(self, other):
        if not isinstance(other, TMMatch):
            return False
        return (self.source_text == other.source_text and
                self.target_text == other.target_text and
                self.similarity == other.similarity)


@dataclass
class TermMatch:
    """Represents a termbase (terminology) match"""
    source: str
    target: str
    context: Optional[str] = None
    part_of_speech: Optional[str] = None
    field: Optional[str] = None
    definition: Optional[str] = None
    
    # Metadata
    source_language: Optional[str] = None
    target_language: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Clean and validate term data"""
        self.source = self.source.strip() if self.source else ""
        self.target = self.target.strip() if self.target else ""
    
    def is_valid(self) -> bool:
        """Check if term has valid source and target"""
        return bool(self.source and self.target)
    
    def __repr__(self):
        return f"TermMatch({self.source} = {self.target})"
    
    def __hash__(self):
        return hash((self.source, self.target))
    
    def __eq__(self, other):
        if not isinstance(other, TermMatch):
            return False
        return self.source == other.source and self.target == other.target


@dataclass
class BatchResult:
    """Result of a translation batch"""
    batch_num: int
    segments: List[TranslationSegment]
    translations: Dict[str, str]  # {segment_id: translated_text}
    tokens_used: int = 0
    processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)
    
    def success_count(self) -> int:
        """Number of successfully translated segments"""
        return len(self.translations)
    
    def error_count(self) -> int:
        """Number of failed translations"""
        return len(self.errors)
    
    def __repr__(self):
        return f"BatchResult(batch={self.batch_num}, translations={self.success_count()}, errors={self.error_count()})"


@dataclass
class TranslationMetadata:
    """Metadata for a translation job"""
    source_language: str
    target_language: str
    source_file_name: str
    total_segments: int
    
    # Processing stats
    bypass_segments: int = 0  # From TM matches
    llm_segments: int = 0     # Via LLM
    total_cost: float = 0.0
    total_tokens: int = 0
    processing_time: float = 0.0
    
    # Context used
    tm_used: bool = False
    tb_used: bool = False
    dnt_used: bool = False
    memoq_used: bool = False
    reference_used: bool = False
    
    def __post_init__(self):
        """Validate metadata"""
        if self.total_segments != (self.bypass_segments + self.llm_segments):
            logger.warning(
                f"Segment count mismatch: total={self.total_segments}, "
                f"bypass={self.bypass_segments}, llm={self.llm_segments}"
            )


@dataclass
class TranslationConfig:
    """Configuration for translation processing"""
    # Language codes
    source_lang: str
    target_lang: str
    
    # Match thresholds
    acceptance_threshold: int = 95  # Minimum % to bypass LLM
    match_threshold: int = 70       # Minimum % to use in context
    
    # Batch settings
    batch_size: int = 20
    chat_history_length: int = 5
    
    # Context settings
    include_tm: bool = True
    include_tb: bool = True
    include_dnt: bool = True
    include_reference: bool = True
    
    # LLM settings
    model: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 4000
    
    def __post_init__(self):
        """Validate configuration"""
        if not 0 <= self.acceptance_threshold <= 100:
            raise ValueError(f"acceptance_threshold must be 0-100, got {self.acceptance_threshold}")
        if not 0 <= self.match_threshold <= 100:
            raise ValueError(f"match_threshold must be 0-100, got {self.match_threshold}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be ≥1, got {self.batch_size}")


@dataclass
class SegmentAnalysis:
    """Analysis of a single segment"""
    segment_id: str
    source_text: str
    
    # Match info
    local_tm_match: Optional[TMMatch] = None
    local_tm_matches: List[TMMatch] = field(default_factory=list)
    
    memoq_tm_match: Optional[TMMatch] = None
    memoq_tm_matches: List[TMMatch] = field(default_factory=list)
    
    # Context
    tb_matches: List[TermMatch] = field(default_factory=list)
    dnt_terms_found: List[str] = field(default_factory=list)
    
    # Decision
    should_bypass: bool = False
    bypass_reason: Optional[str] = None
    match_percentage: int = 0
    
    def get_best_match(self) -> Optional[TMMatch]:
        """Get highest match available"""
        all_matches = self.local_tm_matches + self.memoq_tm_matches
        if all_matches:
            return max(all_matches, key=lambda m: m.similarity)
        return None
    
    def __repr__(self):
        action = "BYPASS" if self.should_bypass else "LLM"
        return f"SegmentAnalysis({self.segment_id}, {action}, {self.match_percentage}%)"
