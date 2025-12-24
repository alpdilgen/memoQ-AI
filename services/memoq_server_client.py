"""
memoQ Server REST API Client
Handles communication with memoQ Server for TM and TB operations
"""

import requests
import logging
import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


# ===== NORMALIZATION FUNCTIONS =====

def normalize_memoq_tm_response(memoq_response: Dict, segment_id: str = "batch", match_threshold: int = 70) -> List:
    """
    Convert memoQ TM API response to standard TMMatch objects
    
    Args:
        memoq_response: Raw response from memoQ Server API
        segment_id: ID of the segment being matched (for logging)
        match_threshold: Minimum match % to include (0-100)
    
    Returns:
        List of TMMatch-compatible dicts with proper fields
    """
    from models.entities import TMMatch
    
    matches = []
    
    try:
        result_list = memoq_response.get('Result', [])
        
        if not result_list:
            return []
        
        tm_hits = result_list[0].get('TMHits', []) if result_list else []
        
        if not tm_hits:
            return []
        
        for hit in tm_hits:
            match_rate = hit.get('MatchRate', 0)
            
            if match_rate < match_threshold:
                continue
            
            trans_unit = hit.get('TransUnit', {})
            
            if not trans_unit:
                continue
            
            source_seg = trans_unit.get('SourceSegment', '')
            target_seg = trans_unit.get('TargetSegment', '')
            
            if not source_seg or not target_seg:
                continue
            
            # Clean XML tags: <seg>text</seg> → text
            source_text = re.sub(r'</?seg>', '', source_seg).strip()
            target_text = re.sub(r'</?seg>', '', target_seg).strip()
            
            if not source_text or not target_text:
                continue
            
            match_type = "EXACT" if match_rate == 100 else "FUZZY"
            
            try:
                match = TMMatch(
                    source_text=source_text,
                    target_text=target_text,
                    similarity=match_rate,
                    match_type=match_type
                )
                matches.append(match)
                logger.debug(f"[{segment_id}] TM match: {match_rate}% - {source_text[:50]}")
            except Exception as e:
                logger.warning(f"[{segment_id}] Invalid TMMatch: {e}")
                continue
    
    except Exception as e:
        logger.error(f"Error normalizing memoQ TM response for {segment_id}: {e}")
        return []
    
    # Sort by similarity descending
    matches.sort(key=lambda x: x.similarity, reverse=True)
    return matches[:10]  # Limit to top 10


def normalize_memoq_tb_response(memoq_response: Dict, segment_id: str = "batch") -> List:
    """
    Convert memoQ TB API response to standard TermMatch objects
    
    Args:
        memoq_response: Raw response from memoQ Server TB lookup
        segment_id: ID of the segment being matched (for logging)
    
    Returns:
        List of TermMatch-compatible dicts
    """
    from models.entities import TermMatch
    
    terms = []
    
    try:
        result_list = memoq_response.get('Result', [])
        
        if not result_list:
            return []
        
        for result in result_list:
            if not isinstance(result, dict):
                continue
            
            tb_hits = result.get('TBHits', [])
            
            for hit in tb_hits:
                trans_unit = hit.get('TransUnit', {})
                
                if not trans_unit:
                    continue
                
                source_term = trans_unit.get('SourceTerm', '')
                target_term = trans_unit.get('TargetTerm', '')
                
                if not source_term or not target_term:
                    continue
                
                source_term = source_term.strip()
                target_term = target_term.strip()
                
                try:
                    term = TermMatch(
                        source=source_term,
                        target=target_term
                    )
                    terms.append(term)
                    logger.debug(f"[{segment_id}] TB term: {source_term} = {target_term}")
                except Exception as e:
                    logger.warning(f"[{segment_id}] Invalid TermMatch: {e}")
                    continue
    
    except Exception as e:
        logger.error(f"Error normalizing memoQ TB response for {segment_id}: {e}")
        return []
    
    return terms


class MemoQServerClient:
    """
    REST API client for memoQ Server
    Handles Authentication, TM, and TB operations
    """
    
    def __init__(
        self,
        server_url: str,
        username: str,
        password: str,
        verify_ssl: bool = False,
        timeout: int = 30
    ):
        """
        Initialize memoQ Server connection
        
        Args:
            server_url: Base URL (e.g., https://mirage.memoq.com:8091/adaturkey)
            username: memoQ username
            password: memoQ password
            verify_ssl: SSL certificate verification
            timeout: Request timeout
        """
        self.server_url = server_url.rstrip('/')
        self.username = username
        self.password = password
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.base_path = "/memoqserverhttpapi/v1"
        
        self.token = None
        self.token_expiry = None
        self.token_buffer = 300  # 5 min buffer
        
        self._tm_cache = {}
        self._tb_cache = {}
    
    def login(self) -> bool:
        """Authenticate with memoQ Server"""
        url = f"{self.server_url}{self.base_path}/auth/login"
        payload = {
            "UserName": self.username,
            "Password": self.password,
            "LoginMode": 0
        }
        
        try:
            response = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                verify=self.verify_ssl,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            self.token = data.get("AccessToken")
            self.token_expiry = datetime.now() + timedelta(minutes=55)
            
            logger.info(f"✓ Authenticated as {data.get('Name')}")
            return True
            
        except Exception as e:
            logger.error(f"Login failed: {e}")
            raise Exception(f"Authentication failed: {str(e)}")
    
    def _ensure_token(self) -> bool:
        """Ensure token is valid"""
        if self.token is None:
            return self.login()
        
        if datetime.now() > (self.token_expiry - timedelta(seconds=self.token_buffer)):
            logger.warning("Token expiring, refreshing...")
            return self.login()
        
        return True
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict:
        """Make REST API request"""
        if not self._ensure_token():
            raise Exception("Authentication failed")
        
        url = f"{self.server_url}{self.base_path}{endpoint}"
        
        request_params = {"authToken": self.token}
        if params:
            request_params.update(params)
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        try:
            if method == "GET":
                response = requests.get(
                    url,
                    params=request_params,
                    headers=headers,
                    verify=self.verify_ssl,
                    timeout=self.timeout
                )
            elif method == "POST":
                logger.debug(f"POST URL: {url}")
                logger.debug(f"POST Params: {request_params}")
                logger.debug(f"POST Data: {data}")
                response = requests.post(
                    url,
                    json=data,
                    params=request_params,
                    headers=headers,
                    verify=self.verify_ssl,
                    timeout=self.timeout
                )
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            logger.debug(f"Response Status: {response.status_code}")
            logger.debug(f"Response Text: {response.text}")
            
            response.raise_for_status()
            result = response.json()
            logger.debug(f"Response JSON: {result}")
            return result
            
        except requests.exceptions.HTTPError as e:
            try:
                error_data = response.json()
                error_code = error_data.get("ErrorCode", "Unknown")
                error_msg = error_data.get("Message", "")
                raise Exception(f"HTTP {response.status_code}: {error_code}: {error_msg}")
            except:
                raise Exception(f"HTTP {response.status_code}: {str(e)}")
        
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            raise Exception(f"Request failed: {str(e)}")
    
    # ==================== TRANSLATION MEMORY ====================
    
    def list_tms(
        self,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
        force_refresh: bool = False
    ) -> List[Dict]:
        """List all Translation Memories"""
        cache_key = f"tms_{src_lang}_{tgt_lang}"
        
        if not force_refresh and cache_key in self._tm_cache:
            return self._tm_cache[cache_key]
        
        endpoint = "/tms"
        params = {}
        
        if src_lang:
            params["srcLang"] = src_lang
        if tgt_lang:
            params["targetLang"] = tgt_lang
        
        result = self._make_request("GET", endpoint, params=params if params else None)
        self._tm_cache[cache_key] = result
        
        logger.info(f"Listed {len(result)} TMs")
        return result
    
    def lookup_segments(
        self,
        tm_guid: str,
        segments: List[str],
        match_threshold: int = 70
    ) -> Dict:
        """
        Lookup segments in Translation Memory
        
        Args:
            tm_guid: Translation Memory GUID
            segments: List of source segments to lookup
            match_threshold: Minimum match percentage (50-102)
        
        Returns:
            Dict with normalized TMMatch objects: {segment_index: [TMMatch objects]}
        """
        # Clean segments: remove XML tag placeholders {{1}}, {{2}}, etc.
        cleaned_segments = []
        for seg in segments:
            clean_text = seg.replace('{{', '').replace('}}', '')
            # Remove digits that were part of placeholders
            parts = clean_text.split()
            clean_text = ' '.join(p for p in parts if p.strip())
            cleaned_segments.append(clean_text.strip())
        
        # Build correct payload according to memoQ API v1 documentation
        # IMPORTANT: Segments must be wrapped in <seg> XML tags
        payload = {
            "Segments": [
                {"Segment": f"<seg>{seg}</seg>"}
                for seg in cleaned_segments
            ],
            "Options": {
                "MatchThreshold": match_threshold,
                "AdjustFuzzyMatches": False,
                "InlineTagStrictness": 2,
                "OnlyBest": False,
                "OnlyUnambiguous": False,
                "ShowFragmentHits": False,
                "ReverseLookup": False
            }
        }
        
        endpoint = f"/tms/{tm_guid}/lookupsegments"
        
        try:
            logger.debug(f"TM lookup payload: {payload}")
            result = self._make_request("POST", endpoint, data=payload)
            logger.info(f"TM lookup raw response: {result}")
            
            # Check if result has hits
            if result and isinstance(result, dict):
                result_list = result.get("Result", [])
                logger.info(f"TM lookup Result count: {len(result_list) if result_list else 0}")
                
                if result_list:
                    for i, segment_result in enumerate(result_list):
                        hits = segment_result.get("TMHits", [])
                        logger.info(f"Segment {i+1} TM hits: {len(hits)}")
                    
                    # Normalize the response to TMMatch objects
                    normalized_matches = normalize_memoq_tm_response(
                        result,
                        segment_id="batch",
                        match_threshold=match_threshold
                    )
                    
                    # Return in dict format for backward compatibility
                    if normalized_matches:
                        return {0: normalized_matches}
                    return {}
                else:
                    logger.warning(f"TM lookup returned Result but empty")
                    return {}
            else:
                logger.warning(f"TM lookup returned unexpected format: {result}")
                return {}
        except Exception as e:
            logger.error(f"TM lookup error: {e}", exc_info=True)
            return {}
    
    def concordance_search(
        self,
        tm_guid: str,
        search_terms: List[str],
        results_limit: int = 64
    ) -> Dict:
        """Concordance search in Translation Memory"""
        payload = {
            "SearchExpression": search_terms,
            "Options": {
                "ResultsLimit": results_limit,
                "Ascending": False,
                "Column": 3
            }
        }
        
        endpoint = f"/tms/{tm_guid}/concordance"
        return self._make_request("POST", endpoint, data=payload)
    
    # ==================== TERMBASE ====================
    
    def list_tbs(
        self,
        languages: Optional[List[str]] = None,
        force_refresh: bool = False
    ) -> List[Dict]:
        """List all Termbases"""
        cache_key = f"tbs_{'_'.join(languages or [])}"
        
        if not force_refresh and cache_key in self._tb_cache:
            return self._tb_cache[cache_key]
        
        endpoint = "/tbs"
        params = None
        
        if languages:
            params = {f"lang[{i}]": lang for i, lang in enumerate(languages)}
        
        result = self._make_request("GET", endpoint, params=params)
        self._tb_cache[cache_key] = result
        
        logger.info(f"Listed {len(result)} TBs")
        return result
    
    def lookup_terms(
        self,
        tb_guid: str,
        search_terms: List[str],
        src_lang: str = "eng",
        tgt_lang: Optional[str] = "tur"
    ) -> List:
        """
        Lookup terms in Termbase
        
        Args:
            tb_guid: Termbase GUID
            search_terms: List of terms to lookup
            src_lang: Source language code (default: "eng" for English)
            tgt_lang: Target language code (optional, default: "tur" for Turkish)
        
        Returns:
            List of normalized TermMatch objects
        """
        # Clean search terms: remove XML tag placeholders
        cleaned_terms = []
        for term in search_terms:
            clean_text = term.replace('{{', '').replace('}}', '')
            parts = clean_text.split()
            clean_text = ' '.join(p for p in parts if p.strip())
            cleaned_terms.append(clean_text.strip())
        
        # Build correct payload according to memoQ API v1 documentation
        # IMPORTANT: Segments must be wrapped in <seg> XML tags
        payload = {
            "SourceLanguage": src_lang,
            "Segments": [f"<seg>{term}</seg>" for term in cleaned_terms]
        }
        
        # Add target language if specified
        if tgt_lang:
            payload["TargetLanguage"] = tgt_lang
        
        endpoint = f"/tbs/{tb_guid}/lookupterms"
        
        try:
            logger.debug(f"TB lookup payload: {payload}")
            result = self._make_request("POST", endpoint, data=payload)
            logger.info(f"TB lookup raw response: {result}")
            
            # Check if result has hits
            if result and isinstance(result, dict):
                result_list = result.get("Result", [])
                logger.info(f"TB lookup Result count: {len(result_list) if result_list else 0}")
                
                if result_list:
                    for i, segment_result in enumerate(result_list):
                        hits = segment_result.get("TBHits", [])
                        logger.info(f"Segment {i+1} TB hits: {len(hits)}")
                    
                    # Normalize the response to TermMatch objects
                    normalized_terms = normalize_memoq_tb_response(
                        result,
                        segment_id="batch"
                    )
                    return normalized_terms
                else:
                    logger.warning(f"TB lookup returned Result but empty")
                    return []
            else:
                logger.warning(f"TB lookup returned unexpected format: {result}")
                return []
        except Exception as e:
            logger.error(f"TB lookup error: {e}", exc_info=True)
            return []
