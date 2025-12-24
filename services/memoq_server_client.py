"""
memoQ Server API Client with complete TM and TB functionality
Handles authentication, lookups, and normalization of responses
"""

import requests
import logging
import re
from typing import List, Dict, Optional, Tuple, Any
from requests.auth import HTTPBasicAuth
from models.entities import TMMatch, TermMatch

logger = logging.getLogger(__name__)


# ===== NORMALIZATION FUNCTIONS =====

def normalize_memoq_tm_response(memoq_response: Any, segment_id: str = "batch", match_threshold: int = 70) -> List[TMMatch]:
    """
    Convert memoQ Server TM API response to standard TMMatch objects
    
    Args:
        memoq_response: Raw response from memoQ Server API
        segment_id: ID of the segment being matched (for logging)
        match_threshold: Minimum match % to include (0-100)
    
    Returns:
        List of TMMatch objects, sorted by similarity descending
    
    Handles:
        - Nested JSON structure from memoQ API
        - XML tags in segments (<seg>text</seg>)
        - Multiple matches per segment
        - Match rate scoring
    """
    matches = []
    
    try:
        # Handle different response structures
        if isinstance(memoq_response, dict):
            result_list = memoq_response.get('Result', [])
        elif isinstance(memoq_response, list):
            result_list = memoq_response
        else:
            logger.warning(f"Unexpected response type for {segment_id}: {type(memoq_response)}")
            return []
        
        if not result_list:
            return []
        
        # Extract TM hits from first result
        if isinstance(result_list[0], dict):
            tm_hits = result_list[0].get('TMHits', [])
        else:
            tm_hits = result_list
        
        if not tm_hits:
            return []
        
        # Process each TM hit
        for hit in tm_hits:
            if not isinstance(hit, dict):
                continue
            
            match_rate = hit.get('MatchRate', 0)
            
            # Skip if below threshold
            if match_rate < match_threshold:
                continue
            
            trans_unit = hit.get('TransUnit', {})
            
            if not trans_unit:
                continue
            
            # Extract source and target segments
            source_seg = trans_unit.get('SourceSegment', '')
            target_seg = trans_unit.get('TargetSegment', '')
            
            if not source_seg or not target_seg:
                continue
            
            # Clean XML tags: <seg>text</seg> → text
            source_text = re.sub(r'</?seg>', '', source_seg).strip()
            target_text = re.sub(r'</?seg>', '', target_seg).strip()
            
            # Skip empty matches
            if not source_text or not target_text:
                continue
            
            # Determine match type
            match_type = "EXACT" if match_rate == 100 else "FUZZY"
            
            # Extract optional metadata
            metadata = {}
            custom_metas = trans_unit.get('CustomMetas', [])
            for meta in custom_metas:
                if isinstance(meta, dict):
                    name = meta.get('Name', '')
                    value = meta.get('Value', '')
                    if name and value:
                        metadata[name] = value
            
            # Create standard TMMatch object
            try:
                match = TMMatch(
                    source_text=source_text,
                    target_text=target_text,
                    similarity=match_rate,
                    match_type=match_type,
                    project=metadata.get('Project'),
                    domain=metadata.get('Domain'),
                    metadata=metadata
                )
                
                matches.append(match)
                logger.debug(f"[{segment_id}] TM match: {match_rate}% - {source_text[:50]}")
            
            except ValueError as e:
                logger.warning(f"[{segment_id}] Invalid TMMatch: {e}")
                continue
    
    except Exception as e:
        logger.error(f"Error normalizing memoQ TM response for {segment_id}: {e}", exc_info=True)
        return []
    
    # Sort by similarity descending (best first)
    matches.sort(key=lambda x: x.similarity, reverse=True)
    
    # Limit to top 10 to save tokens
    return matches[:10]


def normalize_memoq_tb_response(memoq_response: Any, segment_id: str = "batch") -> List[TermMatch]:
    """
    Convert memoQ Server TB API response to standard TermMatch objects
    
    Args:
        memoq_response: Raw response from memoQ Server TB lookup
        segment_id: ID of the segment being matched (for logging)
    
    Returns:
        List of TermMatch objects
    
    Handles:
        - Different TB response structures
        - Term metadata (POS, definition, context)
    """
    terms = []
    
    try:
        # Handle different response structures
        if isinstance(memoq_response, dict):
            result_list = memoq_response.get('Result', memoq_response.get('TermResults', []))
        elif isinstance(memoq_response, list):
            result_list = memoq_response
        else:
            return []
        
        if not result_list:
            return []
        
        # Process TB results
        for result in result_list:
            if not isinstance(result, dict):
                continue
            
            # Handle various field names
            source_term = (
                result.get('SourceTerm') or 
                result.get('source_term') or 
                result.get('Source') or
                result.get('source', '')
            )
            
            target_term = (
                result.get('TargetTerm') or 
                result.get('target_term') or 
                result.get('Target') or
                result.get('target', '')
            )
            
            if not source_term or not target_term:
                continue
            
            source_term = source_term.strip()
            target_term = target_term.strip()
            
            # Create standard TermMatch object
            try:
                term = TermMatch(
                    source=source_term,
                    target=target_term,
                    context=result.get('context') or result.get('Context'),
                    part_of_speech=result.get('part_of_speech') or result.get('POS'),
                    field=result.get('field') or result.get('Field'),
                    definition=result.get('definition') or result.get('Definition')
                )
                
                terms.append(term)
                logger.debug(f"[{segment_id}] TB term: {source_term} = {target_term}")
            
            except ValueError as e:
                logger.warning(f"[{segment_id}] Invalid TermMatch: {e}")
                continue
    
    except Exception as e:
        logger.error(f"Error normalizing memoQ TB response for {segment_id}: {e}", exc_info=True)
        return []
    
    return terms


# ===== MEMOQ SERVER CLIENT =====

class MemoQServerClient:
    """
    Client for memoQ Server REST API v5
    
    Handles:
    - Authentication (Basic Auth or Token)
    - TM lookups with normalization
    - TB (termbase) lookups
    - Caching of TM/TB lists
    - Error handling and logging
    """
    
    def __init__(self, 
                 server_url: str, 
                 username: str, 
                 password: str = "",
                 use_token: bool = False,
                 timeout: int = 30,
                 verify_ssl: bool = True):
        """
        Initialize memoQ Server client
        
        Args:
            server_url: Base URL of memoQ Server (e.g., http://localhost:8080)
            username: Username or API token
            password: Password (empty if using token)
            use_token: If True, username is treated as API token
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        
        # Create session with authentication
        self.session = requests.Session()
        
        if use_token:
            self.session.headers['Authorization'] = f'Bearer {username}'
            logger.info("memoQ Client: Using token authentication")
        else:
            self.session.auth = HTTPBasicAuth(username, password)
            logger.info("memoQ Client: Using basic authentication")
        
        # Set common headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        # Caches
        self._tm_cache: Dict[str, List] = {}
        self._tb_cache: Dict[str, List] = {}
        
        logger.info(f"MemoQServerClient initialized: {self.server_url}")
    
    def _make_request(self, 
                      method: str, 
                      endpoint: str, 
                      params: Optional[Dict] = None, 
                      body: Optional[Dict] = None,
                      timeout: Optional[int] = None) -> Dict:
        """
        Make HTTP request to memoQ Server
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint (e.g., /tms)
            params: Query parameters
            body: JSON body for POST/PUT
            timeout: Request timeout
        
        Returns:
            Response JSON
        
        Raises:
            Exception: On network or API errors
        """
        url = f"{self.server_url}/api/v5{endpoint}"
        req_timeout = timeout or self.timeout
        
        try:
            logger.debug(f"{method} {url}")
            
            if method == "GET":
                response = self.session.get(
                    url, 
                    params=params, 
                    timeout=req_timeout,
                    verify=self.verify_ssl
                )
            
            elif method == "POST":
                response = self.session.post(
                    url, 
                    json=body, 
                    params=params, 
                    timeout=req_timeout,
                    verify=self.verify_ssl
                )
            
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            
            result = response.json()
            logger.debug(f"Response: {len(str(result))} chars")
            return result
        
        except requests.exceptions.Timeout:
            error_msg = f"Request timeout: {url}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error: {url} - {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP {response.status_code}: {url} - {response.text[:200]}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        except Exception as e:
            error_msg = f"Request failed: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
    
    def login(self) -> bool:
        """
        Login to memoQ Server (validate credentials)
        
        Returns:
            True if login successful, False otherwise
        """
        try:
            result = self._make_request("GET", "/status")
            logger.info(f"✓ memoQ Server login successful")
            return True
        
        except Exception as e:
            logger.error(f"✗ memoQ Server login failed: {e}")
            return False
    
    def test_connection(self) -> bool:
        """
        Test connection to memoQ Server
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            result = self._make_request("GET", "/status")
            logger.info(f"✓ memoQ Server connection successful")
            return True
        
        except Exception as e:
            logger.error(f"✗ memoQ Server connection failed: {e}")
            return False
    
    def list_tms(self, 
                 src_lang: Optional[str] = None, 
                 tgt_lang: Optional[str] = None,
                 force_refresh: bool = False) -> List[Dict]:
        """
        List all Translation Memories
        
        Args:
            src_lang: Filter by source language code (optional)
            tgt_lang: Filter by target language code (optional)
            force_refresh: Ignore cache and fetch fresh
        
        Returns:
            List of TM objects with metadata
        """
        # Check cache
        cache_key = f"tms_{src_lang}_{tgt_lang}"
        if not force_refresh and cache_key in self._tm_cache:
            logger.debug(f"Using cached TMs: {cache_key}")
            return self._tm_cache[cache_key]
        
        params = {}
        if src_lang:
            params['srcLang'] = src_lang
        if tgt_lang:
            params['tgtLang'] = tgt_lang
        
        try:
            result = self._make_request("GET", "/tms", params=params if params else None)
            
            # Cache result
            tm_list = result if isinstance(result, list) else result.get('TMs', [])
            self._tm_cache[cache_key] = tm_list
            
            logger.info(f"Listed {len(tm_list)} TMs (cache: {cache_key})")
            return tm_list
        
        except Exception as e:
            logger.error(f"Failed to list TMs: {e}")
            return []
    
    def list_termbases(self,
                       src_lang: Optional[str] = None,
                       tgt_lang: Optional[str] = None,
                       force_refresh: bool = False) -> List[Dict]:
        """
        List all Termbases
        
        Args:
            src_lang: Filter by source language
            tgt_lang: Filter by target language
            force_refresh: Ignore cache and fetch fresh
        
        Returns:
            List of termbase objects with metadata
        """
        # Check cache
        cache_key = f"tbs_{src_lang}_{tgt_lang}"
        if not force_refresh and cache_key in self._tb_cache:
            logger.debug(f"Using cached termbases: {cache_key}")
            return self._tb_cache[cache_key]
        
        params = {}
        if src_lang:
            params['srcLang'] = src_lang
        if tgt_lang:
            params['tgtLang'] = tgt_lang
        
        try:
            result = self._make_request("GET", "/termbases", params=params if params else None)
            
            # Cache result
            tb_list = result if isinstance(result, list) else result.get('TBs', [])
            self._tb_cache[cache_key] = tb_list
            
            logger.info(f"Listed {len(tb_list)} termbases (cache: {cache_key})")
            return tb_list
        
        except Exception as e:
            logger.error(f"Failed to list termbases: {e}")
            return []
    
    def list_tbs(self,
                 src_lang: Optional[str] = None,
                 tgt_lang: Optional[str] = None,
                 force_refresh: bool = False) -> List[Dict]:
        """
        Alias for list_termbases() for backward compatibility
        
        Args:
            src_lang: Filter by source language
            tgt_lang: Filter by target language
            force_refresh: Ignore cache and fetch fresh
        
        Returns:
            List of termbase objects with metadata
        """
        return self.list_termbases(src_lang, tgt_lang, force_refresh)
    
    def lookup_segments(self,
                        tm_guid: str,
                        segments: List[str],
                        match_threshold: int = 70) -> Dict[int, List[TMMatch]]:
        """
        Lookup segments in Translation Memory with normalization
        
        Args:
            tm_guid: Translation Memory GUID
            segments: List of source text segments
            match_threshold: Minimum match % to include (0-100)
        
        Returns:
            Dict: {segment_index: [TMMatch objects]}
                  Returns empty dict if no matches or error
        
        Handles:
            - Multiple segments per request
            - Normalization of response
            - Error recovery
        """
        if not segments:
            return {}
        
        # Clean segments: remove placeholder tags {{1}}, etc.
        clean_segments = []
        for seg in segments:
            # Remove XML-style tags
            clean_seg = re.sub(r'\{\{[\d]+\}\}', '', seg).strip()
            # Remove extra whitespace
            clean_seg = re.sub(r'\s+', ' ', clean_seg)
            if clean_seg:
                clean_segments.append(clean_seg)
        
        if not clean_segments:
            return {}
        
        try:
            logger.info(f"TM lookup: {len(clean_segments)} segments in {tm_guid}")
            
            # Prepare request body
            body = {
                "lookupSegmentBatch": clean_segments,
                "penaltyForContextMismatch": 1,
                "numberOfSearchResults": 5  # Return top 5 matches per segment
            }
            
            # Make API request
            results = self._make_request(
                "POST",
                f"/tms/{tm_guid}/concord",
                body=body
            )
            
            # Normalize responses for each segment
            normalized_results = {}
            
            if isinstance(results, dict) and 'Result' in results:
                # Single result with all hits
                normalized = normalize_memoq_tm_response(
                    results,
                    segment_id="batch",
                    match_threshold=match_threshold
                )
                if normalized:
                    for idx in range(len(clean_segments)):
                        normalized_results[idx] = normalized
            
            elif isinstance(results, list):
                # List of results per segment
                for idx, result in enumerate(results):
                    if idx < len(clean_segments):
                        normalized = normalize_memoq_tm_response(
                            result,
                            segment_id=str(idx),
                            match_threshold=match_threshold
                        )
                        if normalized:
                            normalized_results[idx] = normalized
            
            logger.info(f"TM lookup complete: {len(normalized_results)} segments with matches")
            return normalized_results
        
        except Exception as e:
            logger.error(f"TM lookup failed: {e}", exc_info=True)
            return {}
    
    def lookup_terms(self,
                     tb_guid: str,
                     term_candidates: List[str]) -> List[TermMatch]:
        """
        Lookup terms in Termbase
        
        Args:
            tb_guid: Termbase GUID
            term_candidates: List of terms to lookup
        
        Returns:
            List[TermMatch]: Matching terms with normalization applied
        
        Handles:
            - Multiple term lookups
            - Normalization of response
            - Error recovery
        """
        if not term_candidates:
            return []
        
        try:
            logger.info(f"TB lookup: {len(term_candidates)} terms in {tb_guid}")
            
            # Prepare request
            body = {
                "lookupItems": term_candidates,
                "numberOfSearchResults": 3  # Top 3 results per term
            }
            
            # Make API request
            results = self._make_request(
                "POST",
                f"/termbases/{tb_guid}/lookup",
                body=body
            )
            
            # Normalize TB response
            normalized = normalize_memoq_tb_response(results, segment_id="batch")
            
            # Remove duplicates while preserving order
            seen = set()
            unique_terms = []
            for term in normalized:
                key = (term.source, term.target)
                if key not in seen:
                    seen.add(key)
                    unique_terms.append(term)
            
            logger.info(f"TB lookup complete: {len(unique_terms)} unique terms found")
            return unique_terms
        
        except Exception as e:
            logger.error(f"TB lookup failed: {e}", exc_info=True)
            return []
    
    def clear_cache(self):
        """Clear all cached TM and TB lists"""
        self._tm_cache.clear()
        self._tb_cache.clear()
        logger.info("memoQ client cache cleared")
