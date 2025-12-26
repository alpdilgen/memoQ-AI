"""
OPTIMIZED TM MATCHER - Production Ready
Implements 5 critical improvements:
1. Proper normalization
2. Exact match index (O(1) lookup)
3. Edit distance fuzzy matching
4. Proper threshold classification
5. Performance optimization
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# ═════════════════════════════════════════════════════════════════════════════════
# CHANGE #1: PROPER NORMALIZATION FUNCTION
# ═════════════════════════════════════════════════════════════════════════════════

def normalize(text: str) -> str:
    """
    Normalize text for TM matching
    
    Steps:
    1. Remove XML tags but keep content
    2. Remove placeholders ({}, {{1}}, etc.)
    3. Remove extra whitespace
    4. Convert to lowercase
    5. Keep punctuation and diacritics (important for Cyrillic)
    """
    # Remove XML tags: <ph>, <bpt>, <ept>, etc.
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove placeholders: {}, {{1}}, {{2}}, etc.
    text = re.sub(r'{.*?}', '', text)
    
    # Remove/normalize whitespace but preserve word boundaries
    text = ' '.join(text.split())
    
    # Strip leading/trailing
    text = text.strip()
    
    # Lowercase for case-insensitive matching
    text = text.lower()
    
    return text


# ═════════════════════════════════════════════════════════════════════════════════
# LEVENSHTEIN EDIT DISTANCE
# ═════════════════════════════════════════════════════════════════════════════════

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein distance between two strings
    Used by ALL CAT tools for fuzzy matching
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def edit_distance_similarity(s1: str, s2: str) -> float:
    """
    Convert Levenshtein distance to similarity score (0-1.0)
    Higher score = more similar
    """
    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    
    if max_len == 0:
        return 1.0
    
    similarity = 1 - (distance / max_len)
    return max(0.0, min(1.0, similarity))  # Clamp to 0-1


# ═════════════════════════════════════════════════════════════════════════════════
# CHANGE #4: THRESHOLD CLASSIFICATION
# ═════════════════════════════════════════════════════════════════════════════════

def classify_match_level(score: float) -> str:
    """
    Classify match score into CAT-standard categories
    
    Industry standard thresholds:
    - 1.0 (100%): Exact match
    - 0.95-0.99: 95%-99% fuzzy
    - 0.85-0.94: 85%-94% fuzzy
    - 0.75-0.84: 75%-84% fuzzy
    - 0.50-0.74: 50%-74% fuzzy
    - <0.50: No match
    """
    if score >= 1.0:
        return "100%"
    elif score >= 0.95:
        return "95%-99%"
    elif score >= 0.85:
        return "85%-94%"
    elif score >= 0.75:
        return "75%-84%"
    elif score >= 0.50:
        return "50%-74%"
    else:
        return "No match"


# ═════════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═════════════════════════════════════════════════════════════════════════════════

@dataclass
class TMMatch:
    """Result of a TM match query"""
    score: float                    # 0-1.0 similarity
    level: str                      # "100%", "95%-99%", etc.
    tm_source: str                  # Original TM source text
    tm_target: str                  # TM target translation
    match_type: str                 # "EXACT" or "FUZZY" or "NONE"


# ═════════════════════════════════════════════════════════════════════════════════
# CHANGE #2: OPTIMIZED TM MATCHER WITH EXACT MATCH INDEX
# ═════════════════════════════════════════════════════════════════════════════════

class OptimizedTMMatcher:
    """
    Production-ready TM matcher with:
    - O(1) exact match lookup via dictionary index
    - Edit distance fuzzy matching
    - Performance optimization (optional)
    """
    
    def __init__(self, tm_entries: List[Dict] = None):
        """
        Initialize matcher
        
        Args:
            tm_entries: List of dicts with 'source' and 'target' keys
        """
        self.tm_entries = tm_entries or []
        
        # CHANGE #2: Build exact match index for O(1) lookup
        self.exact_match_index = {}      # {normalized_source: target}
        self.normalized_entries = []     # For fuzzy matching
        
        if self.tm_entries:
            self._build_indices()
    
    def _build_indices(self):
        """Build exact match index and normalized entries for fuzzy matching"""
        for entry in self.tm_entries:
            source = entry.get('source', '')
            target = entry.get('target', '')
            
            # Normalize once
            normalized_source = normalize(source)
            
            # Store in exact match index
            if normalized_source not in self.exact_match_index:
                self.exact_match_index[normalized_source] = target
            
            # Store for fuzzy matching
            self.normalized_entries.append({
                'source_normalized': normalized_source,
                'source_original': source,
                'target': target
            })
    
    def load_tm(self, tm_entries: List[Dict]):
        """Load/reload TM entries and rebuild indices"""
        self.tm_entries = tm_entries
        self.exact_match_index.clear()
        self.normalized_entries.clear()
        self._build_indices()
    
    def match(self, source_text: str) -> TMMatch:
        """
        Match source segment against TM
        
        Strategy:
        1. Normalize source
        2. Try exact match (O(1) lookup)
        3. If no exact match, find best fuzzy match
        4. Classify and return
        """
        source_normalized = normalize(source_text)
        
        # CHANGE #2A: Fast exact match check (O(1))
        if source_normalized in self.exact_match_index:
            return TMMatch(
                score=1.0,
                level='100%',
                tm_source=source_normalized,
                tm_target=self.exact_match_index[source_normalized],
                match_type='EXACT'
            )
        
        # CHANGE #2B: Fuzzy match (O(n) but optimized)
        return self._fuzzy_match(source_normalized)
    
    def _fuzzy_match(self, source_normalized: str) -> TMMatch:
        """Find best fuzzy match using edit distance"""
        
        best_score = 0
        best_entry = None
        
        # CHANGE #5 (optional): Optimize by filtering similar-length entries
        source_len = len(source_normalized)
        
        for entry in self.normalized_entries:
            tm_source = entry['source_normalized']
            tm_len = len(tm_source)
            
            # Optional optimization: skip very different lengths
            # (disabled by default for accuracy, enable for speed)
            # if abs(tm_len - source_len) > 20:  # Skip if >20 chars different
            #     continue
            
            # Calculate similarity
            score = edit_distance_similarity(source_normalized, tm_source)
            
            if score > best_score:
                best_score = score
                best_entry = entry
        
        # No match found
        if best_entry is None or best_score < 0.50:
            return TMMatch(
                score=0,
                level='No match',
                tm_source='',
                tm_target='',
                match_type='NONE'
            )
        
        # Match found
        level = classify_match_level(best_score)
        return TMMatch(
            score=best_score,
            level=level,
            tm_source=best_entry['source_original'],
            tm_target=best_entry['target'],
            match_type='FUZZY'
        )
    
    def match_batch(self, segments: List[str]) -> List[TMMatch]:
        """Match multiple segments"""
        return [self.match(seg) for seg in segments]
    
    def get_statistics(self, segments: List[str]) -> Dict:
        """
        Analyze segments and return statistics
        
        Returns:
            {
                'total_segments': 39,
                'total_words': 182,
                'by_level': {
                    '100%': {'segments': 2, 'words': 4},
                    '95%-99%': {'segments': 18, 'words': 51},
                    ...
                }
            }
        """
        matches = self.match_batch(segments)
        
        stats = {
            'total_segments': len(segments),
            'total_words': 0,
            'by_level': {}
        }
        
        for segment, match in zip(segments, matches):
            # Count words in segment
            word_count = len(normalize(segment).split())
            stats['total_words'] += word_count
            
            # Group by level
            level = match.level
            if level not in stats['by_level']:
                stats['by_level'][level] = {'segments': 0, 'words': 0}
            
            stats['by_level'][level]['segments'] += 1
            stats['by_level'][level]['words'] += word_count
        
        return stats


# ═════════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════════

def words_in_segment(text: str) -> int:
    """Count words in a segment (after normalization)"""
    normalized = normalize(text)
    if not normalized:
        return 0
    return len(normalized.split())


def create_matcher_from_tmx(tmx_file_path: str) -> OptimizedTMMatcher:
    """
    Create matcher from TMX file
    
    Args:
        tmx_file_path: Path to TMX file
    
    Returns:
        OptimizedTMMatcher instance
    """
    import xml.etree.ElementTree as ET
    
    try:
        # Try UTF-16 first (memoQ default)
        with open(tmx_file_path, 'r', encoding='utf-16') as f:
            content = f.read()
    except:
        # Fall back to UTF-8
        with open(tmx_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    
    root = ET.fromstring(content)
    
    tm_entries = []
    for tu in root.findall('.//tu'):
        tuv_list = tu.findall('tuv')
        if len(tuv_list) >= 2:
            source_seg = tuv_list[0].find('seg')
            target_seg = tuv_list[1].find('seg')
            
            if source_seg is not None and target_seg is not None:
                source = ''.join(source_seg.itertext()).strip()
                target = ''.join(target_seg.itertext()).strip()
                
                if source:
                    tm_entries.append({'source': source, 'target': target})
    
    matcher = OptimizedTMMatcher(tm_entries)
    return matcher


# ═════════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Load TMX
    matcher = create_matcher_from_tmx('/mnt/user-data/uploads/bg-tr_palnomoshtni.tmx')
    
    # Test segments
    test_segments = [
        "превод от български на турски език",
        "пълномощно",
        "подписаният марин иванов стефанов",
        "адрес гр. стамбилийски, ул. райко даскалов 37",
    ]
    
    print("Individual matches:")
    for seg in test_segments:
        match = matcher.match(seg)
        print(f"  {seg[:40]}: {match.level} ({match.score*100:.1f}%)")
    
    print("\nStatistics:")
    stats = matcher.get_statistics(test_segments)
    print(f"  Total: {stats['total_segments']} segments, {stats['total_words']} words")
    for level, data in stats['by_level'].items():
        print(f"    {level}: {data['segments']} segs, {data['words']} words")
