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
    
    def authenticate(self) -> bool:
        """Authenticate and get token"""
        try:
            url = f"{self.server_url}{self.base_path}/auth/login"
            payload = {
                "UserName": self.username,
                "Password": self.password
            }
            
            response = requests.post(
                url,
                json=payload,
                verify=self.verify_ssl,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            self.token = data.get('Token')
            expires_in = data.get('ExpiresIn', 3600)
            self.token_expiry = datetime.utcnow() + timedelta(seconds=expires_in)
            
            return self.token is not None
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            return False
    
    def _ensure_token(self) -> bool:
        """Ensure valid token, re-authenticate if needed"""
        if not self.token or (self.token_expiry and datetime.utcnow() > self.token_expiry - timedelta(seconds=self.token_buffer)):
            return self.authenticate()
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
            logger.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Request Error: {str(e)}")
            raise
        except ValueError as e:
            logger.error(f"JSON Decode Error: {str(e)}")
            raise
    
    def get_translation_memories(self) -> List[Dict]:
        """Get list of available Translation Memories"""
        try:
            endpoint = "/tms"
            result = self._make_request("GET", endpoint)
            if isinstance(result, dict) and 'Result' in result:
                return result['Result']
            return []
        except Exception as e:
            logger.error(f"Failed to get TMs: {str(e)}")
            return []
    
    def get_termbases(self) -> List[Dict]:
        """Get list of available Termbases"""
        try:
            endpoint = "/tbs"
            result = self._make_request("GET", endpoint)
            if isinstance(result, dict) and 'Result' in result:
                return result['Result']
            return []
        except Exception as e:
            logger.error(f"Failed to get TBs: {str(e)}")
            return []
    
    def lookup_segments(
        self,
        tm_guid: str,
        segments: List[str]
    ) -> Dict:
        """Lookup segments in Translation Memory"""
        segment_objects = [
            {"Segment": seg}
            for seg in segments
        ]
        
        payload = {"Segments": segment_objects}
        endpoint = f"/tms/{tm_guid}/lookupsegments"
        
        return self._make_request("POST", endpoint, data=payload)
    
    def concordance_search(
        self,
        tm_guid: str,
        search_terms: List[str],
        results_limit: int = 64
    ) -> Dict:
        """Concordance search in Translation Memory"""
        payload = {
            "SearchExpression": search_terms,
            "ResultsLimit": results_limit
        }
        endpoint = f"/tms/{tm_guid}/concordancesearch"
        return self._make_request("POST", endpoint, data=payload)
    
    def lookup_terms(
        self,
        tb_guid: str,
        terms: List[str]
    ) -> Dict:
        """Lookup terms in Termbase"""
        payload = {"Terms": terms}
        endpoint = f"/tbs/{tb_guid}/lookupterms"
        return self._make_request("POST", endpoint, data=payload)
    
    def insert_term(
        self,
        tb_guid: str,
        source_lang: str,
        target_lang: str,
        source_term: str,
        target_term: str,
        definition: str = ""
    ) -> bool:
        """Insert a term into Termbase"""
        payload = {
            "SourceLanguage": source_lang,
            "TargetLanguage": target_lang,
            "SourceTerm": source_term,
            "TargetTerm": target_term,
            "Definition": definition
        }
        try:
            endpoint = f"/tbs/{tb_guid}/insertterm"
            self._make_request("POST", endpoint, data=payload)
            return True
        except Exception as e:
            logger.error(f"Failed to insert term: {str(e)}")
            return False
    
    def close_connection(self):
        """Close connection and cleanup"""
        self.token = None
        self.token_expiry = None
