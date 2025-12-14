# services/memoq_server_client.py
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
            # Try to bubble up memoQ error payloads for easier debugging
            try:
                error_data = response.json()
                error_code = error_data.get("ErrorCode", "Unknown")
                error_msg = error_data.get("Message", "")
                raise Exception(f"{error_code}: {error_msg}")
            except Exception:
                # If response is not JSON, include text for context
                raise Exception(
                    f"HTTP {response.status_code}: {str(e)} | Response: {response.text}"
                )
        
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
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
    ) -> Dict:
        """Lookup segments in Translation Memory.

        Different memoQ server versions expect language codes either in the
        payload, in the query string, or both. To maximize compatibility we try
        multiple combinations and return as soon as one succeeds.
        """
        segment_objects = [
            {"Segment": seg}
            for seg in segments
        ]

        endpoint = f"/tms/{tm_guid}/lookupsegments"

        base_payload = {"Segments": segment_objects}
        payload_with_langs = dict(base_payload)
        if source_lang:
            payload_with_langs["SourceLangCode"] = source_lang
        if target_lang:
            payload_with_langs["TargetLangCode"] = target_lang

        params_with_langs = {}
        if source_lang:
            params_with_langs["srcLang"] = source_lang
        if target_lang:
            params_with_langs["targetLang"] = target_lang

        attempts: List[Tuple[Dict, Dict]] = []

        if params_with_langs:
            attempts.append((params_with_langs, payload_with_langs))
        attempts.append((params_with_langs or None, base_payload))
        attempts.append((None, payload_with_langs))

        last_error: Optional[Exception] = None

        for attempt_params, attempt_payload in attempts:
            try:
                return self._make_request(
                    "POST",
                    endpoint,
                    data=attempt_payload,
                    params=attempt_params,
                )
            except Exception as exc:  # pragma: no cover - network errors
                last_error = exc
                logger.warning(
                    "memoQ TM lookup failed with params=%s payload_keys=%s: %s",
                    attempt_params,
                    list(attempt_payload.keys()),
                    exc,
                )

        raise last_error if last_error else Exception("TM lookup failed")
    
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
        languages: Optional[List[str]] = None
    ) -> Dict:
        """Lookup terms in Termbase with compatibility fallbacks."""
        base_payload = {"SearchTerms": search_terms}
        payload_with_langs = dict(base_payload)

        params_with_langs = None
        if languages:
            params_with_langs = {f"lang[{i}]": lang for i, lang in enumerate(languages)}
            payload_with_langs["Languages"] = languages

        endpoint = f"/tbs/{tb_guid}/lookupterms"

        attempts: List[Tuple[Optional[Dict], Dict]] = []
        if params_with_langs:
            attempts.append((params_with_langs, payload_with_langs))
        attempts.append((params_with_langs, base_payload))
        attempts.append((None, payload_with_langs))

        last_error: Optional[Exception] = None

        for attempt_params, attempt_payload in attempts:
            try:
                return self._make_request(
                    "POST",
                    endpoint,
                    data=attempt_payload,
                    params=attempt_params,
                )
            except Exception as exc:  # pragma: no cover - network errors
                last_error = exc
                logger.warning(
                    "memoQ TB lookup failed with params=%s payload_keys=%s: %s",
                    attempt_params,
                    list(attempt_payload.keys()),
                    exc,
                )

        raise last_error if last_error else Exception("TB lookup failed")
