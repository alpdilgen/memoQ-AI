from services.caching import CacheManager
from models.entities import TMMatch
from rapidfuzz import fuzz, process
from typing import List, Optional, Tuple
import time

class TMatcher:
    """
    Optimized TM Matcher with:
    - O(1) exact match via hash lookup
    - RapidFuzz for fast fuzzy matching (10-20x faster than difflib)
    - Disk caching for TM data
    - Configurable acceptance threshold for LLM bypass
    """
    
    def __init__(self, tmx_content: bytes, src_lang: str, tgt_lang: str, acceptance_threshold: float = 95.0):
        """
        Initialize TM matcher with parsed and cached TM data.
        
        Args:
            tmx_content: Raw TMX file bytes
            src_lang: Source language code (e.g., 'en', 'en-US')
            tgt_lang: Target language code (e.g., 'de', 'de-DE')
            acceptance_threshold: Minimum match % for direct TM usage (bypass LLM)
        """
        self.acceptance_threshold = acceptance_threshold
        
        start_time = time.time()
        
        # Load TM data (from cache if available)
        self.entries, self.exact_lookup, self.file_hash = CacheManager.load_tm_data(
            tmx_content, src_lang, tgt_lang
        )
        
        # Prepare source texts for fuzzy search
        self.sources = [entry['source'] for entry in self.entries]
        
        load_time = time.time() - start_time
        self.load_time = load_time
        self.tu_count = len(self.entries)
        
    def get_exact_match(self, query_text: str) -> Optional[TMMatch]:
        """
        O(1) exact match lookup using hash table.
        
        Args:
            query_text: Source text to find exact match for
            
        Returns:
            TMMatch if exact match found, None otherwise
        """
        normalized_query = query_text.strip().lower()
        
        if normalized_query in self.exact_lookup:
            # Find original source text for proper casing
            target = self.exact_lookup[normalized_query]
            
            # Find original source (for display purposes)
            original_source = query_text  # Default to query
            for entry in self.entries:
                if entry['source'].strip().lower() == normalized_query:
                    original_source = entry['source']
                    break
            
            return TMMatch(
                source_text=original_source,
                target_text=target,
                similarity=100.0,
                match_type="EXACT"
            )
        
        return None
    
    def get_fuzzy_matches(self, query_text: str, limit: int = 5, threshold: float = 65.0) -> List[TMMatch]:
        """
        Fast fuzzy matching using RapidFuzz.
        
        Args:
            query_text: Source text to find matches for
            limit: Maximum number of matches to return
            threshold: Minimum similarity score (0-100)
            
        Returns:
            List of TMMatch objects sorted by similarity (descending)
        """
        if not self.sources:
            return []
        
        # RapidFuzz process.extract returns list of (match, score, index)
        # Using fuzz.ratio for standard Levenshtein-based similarity
        results = process.extract(
            query_text,
            self.sources,
            scorer=fuzz.ratio,
            limit=limit,
            score_cutoff=threshold
        )
        
        matches = []
        for match_text, score, idx in results:
            matches.append(TMMatch(
                source_text=self.entries[idx]['source'],
                target_text=self.entries[idx]['target'],
                similarity=score,
                match_type="EXACT" if score >= 99.5 else "FUZZY"
            ))
        
        return matches
    
    def extract_matches(self, query_text: str, limit: int = 5, threshold: float = 65.0) -> Tuple[List[TMMatch], bool]:
        """
        Main matching method - combines exact and fuzzy matching.
        
        Args:
            query_text: Source text to find matches for
            limit: Maximum number of matches to return
            threshold: Minimum similarity score (0-100)
            
        Returns:
            Tuple of (matches, bypass_llm):
            - matches: List of TMMatch objects
            - bypass_llm: True if best match >= BYPASS_THRESHOLD (95%)
        """
        matches = []
        bypass_llm = False
        
        # Clean query for matching (remove tag placeholders)
        clean_query = query_text.replace('{{', '').replace('}}', '')
        clean_query = ''.join([c for c in clean_query if not c.isdigit() or c in clean_query.split()])
        clean_query = ' '.join(clean_query.split())  # Normalize whitespace
        
        # 1. Try exact match first (O(1))
        exact = self.get_exact_match(clean_query)
        if exact:
            matches.append(exact)
            bypass_llm = True  # 100% match = bypass LLM
            
            # Still get some fuzzy matches for context (optional)
            fuzzy = self.get_fuzzy_matches(clean_query, limit=limit-1, threshold=threshold)
            for m in fuzzy:
                if m.source_text != exact.source_text:  # Avoid duplicate
                    matches.append(m)
            
            return matches[:limit], bypass_llm
        
        # 2. Fuzzy match
        matches = self.get_fuzzy_matches(clean_query, limit=limit, threshold=threshold)
        
        # Check if best match meets acceptance threshold (for bypass)
        if matches and matches[0].similarity >= self.acceptance_threshold:
            bypass_llm = True
        
        return matches, bypass_llm
    
    def get_best_match(self, query_text: str, threshold: float = 65.0) -> Optional[TMMatch]:
        """
        Get single best match for a query.
        Convenience method for bypass decisions.
        
        Returns:
            Best TMMatch or None if no match above threshold
        """
        matches, _ = self.extract_matches(query_text, limit=1, threshold=threshold)
        return matches[0] if matches else None
    
    def should_bypass_llm(self, query_text: str, match_threshold: float = 65.0) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Determine if segment should bypass LLM translation.
        
        Args:
            query_text: Source text to check
            match_threshold: Minimum threshold for any match
        
        Returns:
            Tuple of (should_bypass, translation, match_score):
            - should_bypass: True if match >= acceptance_threshold
            - translation: The TM translation to use (if bypass), None otherwise
            - match_score: The match percentage
        """
        matches, bypass = self.extract_matches(query_text, limit=1, threshold=match_threshold)
        
        if matches:
            best_match = matches[0]
            if best_match.similarity >= self.acceptance_threshold:
                return True, best_match.target_text, best_match.similarity
            return False, None, best_match.similarity
        
        return False, None, None
