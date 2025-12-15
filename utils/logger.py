"""
Clean, readable translation log handler
"""

from datetime import datetime
from typing import Dict, List


class TransactionLogger:
    """Handles translation workflow logging with clean formatting"""
    
    def __init__(self):
        self.logs = []
        self.matched_segments = []
        self.unmatched_segments = []
        self.encoding_issues = []
        self.llm_segments = []
        self.start_time = datetime.now()
    
    def info(self, message: str):
        """Log info message, filtering out JSON dumps"""
        # Skip verbose JSON responses from memoQ
        if message.startswith("memoQ lookup for") and "{" in message:
            return
        
        # Log match score and bypass info
        if "memoQ match score" in message or "BYPASS" in message:
            self.logs.append(f"{self._timestamp()} | {message}")
        # Log config and general info
        elif any(x in message for x in ["Started translation", "Source:", "TM Acceptance", 
                                         "Chat History", "memoQ Server:", "Using", "Analysis complete"]):
            self.logs.append(f"{self._timestamp()} | {message}")
    
    def log_batch_start(self, batch_num: int, segments: list):
        """Log batch processing start"""
        self.logs.append(f"{self._timestamp()} | Batch {batch_num}: {len(segments)} segments to LLM")
    
    def log_llm_interaction(self, prompt: str, response: str):
        """Log LLM interaction"""
        self.logs.append(f"{self._timestamp()} | LLM processing...")
    
    def log_tm_matches(self, tm_context: Dict):
        """Track TM matches"""
        for seg_id in tm_context:
            self.matched_segments.append(seg_id)
    
    def log_tb_matches(self, tb_context: Dict):
        """Track TB matches"""
        pass
    
    def add_matched_segment(self, seg_id: str):
        """Record matched segment"""
        if seg_id not in self.matched_segments:
            self.matched_segments.append(seg_id)
    
    def add_unmatched_segment(self, seg_id: str, encoding_issue: bool = False):
        """Record unmatched segment"""
        if seg_id not in self.unmatched_segments:
            self.unmatched_segments.append(seg_id)
        if encoding_issue and seg_id not in self.encoding_issues:
            self.encoding_issues.append(seg_id)
    
    def add_llm_segment(self, seg_id: str):
        """Record segment sent to LLM"""
        if seg_id not in self.llm_segments:
            self.llm_segments.append(seg_id)
    
    def _timestamp(self) -> str:
        """Get current timestamp"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def get_content(self) -> str:
        """Return formatted log with summary"""
        output = []
        
        # Header
        output.append("=" * 80)
        output.append("TRANSLATION WORKFLOW LOG")
        output.append("=" * 80)
        output.append("")
        
        # Main logs
        for log in self.logs:
            output.append(log)
        
        output.append("")
        output.append("-" * 80)
        output.append("SUMMARY")
        output.append("-" * 80)
        output.append("")
        
        # Match results
        total = len(self.matched_segments) + len(self.unmatched_segments) + len(self.llm_segments)
        
        if self.matched_segments:
            output.append(f"✓ TM MATCHES: {len(self.matched_segments)} segments")
            segments_str = ", ".join(sorted(self.matched_segments, key=lambda x: int(x) if x.isdigit() else 0))
            output.append(f"  [{segments_str}]")
            output.append("")
        
        if self.encoding_issues:
            output.append(f"⚠ ENCODING ISSUES: {len(self.encoding_issues)} segments")
            segments_str = ", ".join(sorted(self.encoding_issues, key=lambda x: int(x) if x.isdigit() else 0))
            output.append(f"  [{segments_str}]")
            output.append(f"  Reason: & character encoded as &amp; in TM")
            output.append("")
        
        if self.llm_segments:
            output.append(f"→ LLM PROCESSED: {len(self.llm_segments)} segments")
            segments_str = ", ".join(sorted(self.llm_segments, key=lambda x: int(x) if x.isdigit() else 0))
            output.append(f"  [{segments_str}]")
            output.append("")
        
        # Statistics
        output.append("-" * 80)
        if total > 0:
            matched_pct = (len(self.matched_segments) / total * 100)
            output.append(f"Total Segments: {total}")
            output.append(f"TM Coverage: {len(self.matched_segments)} ({matched_pct:.1f}%)")
            output.append(f"LLM Required: {len(self.llm_segments)}")
            output.append(f"Encoding Issues: {len(self.encoding_issues)}")
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        output.append(f"Duration: {elapsed:.1f}s")
        output.append("=" * 80)
        
        return "\n".join(output)
