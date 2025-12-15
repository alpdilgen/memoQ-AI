"""
memoQ Server REST API Client
Handles communication with memoQ Server for TM and TB operations
"""

import requests
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

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
            server_url: Base URL (e.g., https://mirage.memoq.com:9091/adaturkey)
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
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            try:
                error_data = response.json()
                error_code = error_data.get("ErrorCode", "Unknown")
                error_msg = error_data.get("Message", "")
                raise Exception(f"HTTP {response.status_code}: {error_code}: {error_msg}")
            except:
                raise Exception(f"HTTP {response.status_code}: {str(e)}")
        
        except Exception as e:
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
            API response with matches
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
        payload = {
            "Segments": [
                {"Segment": seg}
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
            result = self._make_request("POST", endpoint, data=payload)
            if result:
                logger.info(f"TM lookup successful")
                return result
            else:
                logger.warning(f"TM lookup returned empty result")
                return {}
        except Exception as e:
            logger.error(f"TM lookup error: {e}")
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
    ) -> Dict:
        """
        Lookup terms in Termbase
        
        Args:
            tb_guid: Termbase GUID
            search_terms: List of terms to lookup
            src_lang: Source language code (default: "eng" for English)
            tgt_lang: Target language code (optional, default: "tur" for Turkish)
        
        Returns:
            API response with matching terms
        """
        # Clean search terms: remove XML tag placeholders
        cleaned_terms = []
        for term in search_terms:
            clean_text = term.replace('{{', '').replace('}}', '')
            parts = clean_text.split()
            clean_text = ' '.join(p for p in parts if p.strip())
            cleaned_terms.append(clean_text.strip())
        
        # Build correct payload according to memoQ API v1 documentation
        payload = {
            "SourceLanguage": src_lang,
            "Segments": cleaned_terms
        }
        
        # Add target language if specified
        if tgt_lang:
            payload["TargetLanguage"] = tgt_lang
        
        endpoint = f"/tbs/{tb_guid}/lookupterms"
        
        try:
            result = self._make_request("POST", endpoint, data=payload)
            if result:
                logger.info(f"TB lookup successful")
                return result
            else:
                logger.warning(f"TB lookup returned empty result")
                return {}
        except Exception as e:
            logger.error(f"TB lookup error: {e}")
            return {}
