# services/memoq_ui.py
"""
Streamlit UI Components for memoQ Server Integration
Provides UI for connecting to memoQ Server and selecting TMs/TBs
"""

import streamlit as st
from typing import Dict, List, Tuple, Optional
from services.memoq_server_client import MemoQServerClient
import logging

logger = logging.getLogger(__name__)

class MemoQUI:
    """Streamlit UI for memoQ Server Integration"""
    
    @staticmethod
    def show_connection_settings() -> Optional[MemoQServerClient]:
        """
        Show memoQ Server connection settings in sidebar
        
        Returns:
            MemoQServerClient instance if connected, None otherwise
        """
        with st.sidebar:
            st.divider()
            st.subheader("🔗 memoQ Server Connection")
            
            # Initialize session state for memoQ settings
            if 'memoq_server_url' not in st.session_state:
                st.session_state.memoq_server_url = "https://mirage.memoq.com:8091/adaturkey"
            if 'memoq_username' not in st.session_state:
                st.session_state.memoq_username = "promotloc"
            if 'memoq_password' not in st.session_state:
                st.session_state.memoq_password = ""
            if 'memoq_verify_ssl' not in st.session_state:
                st.session_state.memoq_verify_ssl = False
            
            # Connection form
            with st.form("memoq_connection"):
                server_url = st.text_input(
                    "Server URL",
                    value=st.session_state.memoq_server_url,
                    help="memoQ Server base URL"
                )
                
                username = st.text_input(
                    "Username",
                    value=st.session_state.memoq_username,
                    help="memoQ server username"
                )
                
                password = st.text_input(
                    "Password",
                    type="password",
                    value=st.session_state.memoq_password,
                    help="memoQ server password"
                )
                
                verify_ssl = st.checkbox(
                    "Verify SSL",
                    value=st.session_state.memoq_verify_ssl,
                    help="Check for production, uncheck for self-signed certificates"
                )
                
                submitted = st.form_submit_button("🔐 Connect", use_container_width=True)
                
                if submitted:
                    st.session_state.memoq_server_url = server_url
                    st.session_state.memoq_username = username
                    st.session_state.memoq_password = password
                    st.session_state.memoq_verify_ssl = verify_ssl
                    st.session_state.memoq_connected = False
                    st.session_state.memoq_client = None
            
            # Connection status
            if 'memoq_connected' in st.session_state and st.session_state.memoq_connected:
                st.success("✓ Connected to memoQ Server")
                if st.button("🔌 Disconnect", use_container_width=True):
                    st.session_state.memoq_connected = False
                    st.session_state.memoq_client = None
                    st.rerun()
                
                return st.session_state.memoq_client
            
            elif 'memoq_client' in st.session_state and st.session_state.memoq_client:
                # Try to use existing connection
                try:
                    st.session_state.memoq_connected = True
                    st.success("✓ Connected to memoQ Server")
                    return st.session_state.memoq_client
                except:
                    st.session_state.memoq_connected = False
                    st.session_state.memoq_client = None
                    st.error("⚠️ Connection lost")
                    return None
            
            return None
    
    @staticmethod
    def show_memoq_data_loader(
        client: MemoQServerClient,
        src_lang: str,
        tgt_lang: str
    ) -> Tuple[List[str], List[str]]:
        """
        Show memoQ Data Loader UI with TM and TB selection
        
        Args:
            client: MemoQServerClient instance
            src_lang: Source language code
            tgt_lang: Target language code
        
        Returns:
            Tuple of (selected_tm_guids, selected_tb_guids)
        """
        # Initialize session state
        if 'memoq_tms_loaded' not in st.session_state:
            st.session_state.memoq_tms_loaded = False
            st.session_state.memoq_tbs_loaded = False
            st.session_state.memoq_tms_list = []
            st.session_state.memoq_tbs_list = []
            st.session_state.selected_tm_guids = []
            st.session_state.selected_tb_guids = []
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            load_button = st.button(
                "📥 Load memoQ Data",
                use_container_width=True,
                help="Load Translation Memories and Termbases from memoQ Server"
            )
        
        with col2:
            if st.session_state.memoq_tms_loaded:
                st.success(
                    f"✓ Loaded {len(st.session_state.memoq_tms_list)} TM(s) and "
                    f"{len(st.session_state.memoq_tbs_list)} TB(s)"
                )
        
        # Load data when button is pressed
        if load_button:
            with st.spinner("Loading memoQ data..."):
                try:
                    # Convert language codes for memoQ API
                    memoq_src = MemoQUI._get_memoq_lang_code(src_lang)
                    memoq_tgt = MemoQUI._get_memoq_lang_code(tgt_lang)
                    
                    # Load TMs
                    tms = client.list_tms(src_lang=memoq_src, tgt_lang=memoq_tgt)
                    st.session_state.memoq_tms_list = tms
                    
                    # Load TBs
                    tbs = client.list_tbs(languages=[memoq_src, memoq_tgt])
                    st.session_state.memoq_tbs_list = tbs
                    
                    st.session_state.memoq_tms_loaded = True
                    st.session_state.memoq_tbs_loaded = True
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Failed to load memoQ data: {str(e)}")
                    logger.error(f"memoQ data load error: {e}")
        
        # Display TMs and TBs
        if st.session_state.memoq_tms_loaded and st.session_state.memoq_tbs_loaded:
            st.divider()
            
            col1, col2 = st.columns(2)
            
            # ==================== TRANSLATION MEMORIES ====================
            with col1:
                st.subheader("📚 Translation Memories")
                
                if st.session_state.memoq_tms_list:
                    # Create columns for checkboxes and info
                    tm_selected = []
                    
                    for tm in st.session_state.memoq_tms_list:
                        tm_col1, tm_col2 = st.columns([0.5, 3.5])
                        
                        with tm_col1:
                            # Checkbox
                            is_selected = st.checkbox(
                                label="Select",
                                value=tm["TMGuid"] in st.session_state.selected_tm_guids,
                                key=f"tm_{tm['TMGuid']}"
                            )
                            if is_selected:
                                tm_selected.append(tm["TMGuid"])
                        
                        with tm_col2:
                            # TM Info
                            st.markdown(f"**{tm['FriendlyName']}**")
                            st.caption(
                                f"👥 {tm['SourceLangCode'].upper()} → {tm['TargetLangCode'].upper()} | "
                                f"📝 {tm.get('NumEntries', 0)} entries | "
                                f"📂 {tm.get('Domain', 'N/A')}"
                            )
                    
                    # Update selected list
                    st.session_state.selected_tm_guids = tm_selected
                else:
                    st.info("No Translation Memories found for this language pair")
            
            # ==================== TERMBASES ====================
            with col2:
                st.subheader("📖 Termbases")
                
                if st.session_state.memoq_tbs_list:
                    # Create columns for checkboxes and info
                    tb_selected = []
                    
                    for tb in st.session_state.memoq_tbs_list:
                        tb_col1, tb_col2 = st.columns([0.5, 3.5])
                        
                        with tb_col1:
                            # Checkbox
                            is_selected = st.checkbox(
                                label="Select",
                                value=tb["TBGuid"] in st.session_state.selected_tb_guids,
                                key=f"tb_{tb['TBGuid']}"
                            )
                            if is_selected:
                                tb_selected.append(tb["TBGuid"])
                        
                        with tb_col2:
                            # TB Info
                            st.markdown(f"**{tb['FriendlyName']}**")
                            st.caption(
                                f"🌐 {', '.join(tb.get('Languages', []))} | "
                                f"📝 {tb.get('NumEntries', 0)} terms | "
                                f"📂 {tb.get('Domain', 'N/A')}"
                            )
                    
                    # Update selected list
                    st.session_state.selected_tb_guids = tb_selected
                else:
                    st.info("No Termbases found for this language pair")
            
            # Display summary
            st.divider()
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Selected TMs", len(st.session_state.selected_tm_guids))
            
            with col2:
                st.metric("Selected TBs", len(st.session_state.selected_tb_guids))
            
            with col3:
                if st.session_state.selected_tm_guids or st.session_state.selected_tb_guids:
                    st.success(f"✓ Ready to use {len(st.session_state.selected_tm_guids) + len(st.session_state.selected_tb_guids)} resources")
        
        return st.session_state.selected_tm_guids, st.session_state.selected_tb_guids
    
    @staticmethod
    def _get_memoq_lang_code(lang_code: str) -> str:
        """
        Convert language code to memoQ format
        
        Args:
            lang_code: ISO language code (e.g., 'en', 'tr')
        
        Returns:
            memoQ language code (e.g., 'eng', 'tur')
        """
        # Map common language codes to memoQ 3-letter codes
        lang_map = {
            'en': 'eng',
            'tr': 'tur',
            'hu': 'hun',
            'de': 'ger',
            'fr': 'fra',
            'es': 'spa',
            'it': 'ita',
            'pt': 'por',
            'pl': 'pol',
            'ru': 'rus',
            'ja': 'jpn',
            'zh': 'zho',
            'ar': 'ara',
            'ko': 'kor',
            'nl': 'nld',
            'sv': 'swe',
            'no': 'nor',
            'da': 'dan',
            'fi': 'fin',
            'el': 'ell',
            'he': 'heb',
            'th': 'tha',
            'vi': 'vie',
            'bg': 'bul',
            'ro': 'ron',
            'cs': 'ces',
            'sk': 'slk',
            'uk': 'ukr',
            'et': 'est',
            'lv': 'lav',
            'lt': 'lit',
        }
        
        return lang_map.get(lang_code, lang_code)
    
    @staticmethod
    def show_tm_lookup_results(
        client: MemoQServerClient,
        tm_guids: List[str],
        source_text: str,
        min_match_rate: int = 75
    ) -> Optional[Dict]:
        """
        Show TM lookup results
        
        Args:
            client: MemoQServerClient instance
            tm_guids: List of selected TM GUIDs
            source_text: Source text to lookup
            min_match_rate: Minimum match rate to display
        
        Returns:
            Best match or None
        """
        if not tm_guids or not source_text:
            return None
        
        best_match = None
        best_rate = 0
        
        with st.spinner("🔍 Searching Translation Memories..."):
            for tm_guid in tm_guids:
                try:
                    results = client.lookup_segments(tm_guid, [source_text])
                    hits = results.get("Result", [{}])[0].get("TMHits", [])
                    
                    for hit in hits:
                        match_rate = hit.get("MatchRate", 0)
                        if match_rate >= min_match_rate and match_rate > best_rate:
                            best_match = hit
                            best_rate = match_rate
                
                except Exception as e:
                    logger.warning(f"TM lookup error for {tm_guid}: {e}")
        
        if best_match:
            with st.expander(f"✓ TM Match ({best_match['MatchRate']}%)", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Source (from TM):**")
                    source_seg = best_match["TransUnit"]["SourceSegment"]
                    st.code(source_seg.replace("<seg>", "").replace("</seg>", ""))
                
                with col2:
                    st.markdown("**Target (from TM):**")
                    target_seg = best_match["TransUnit"]["TargetSegment"]
                    st.code(target_seg.replace("<seg>", "").replace("</seg>", ""))
                
                # Show metadata
                st.divider()
                meta = best_match["TransUnit"]
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.caption(f"📂 Domain: {meta.get('Domain', 'N/A')}")
                with col2:
                    st.caption(f"👤 Creator: {meta.get('Creator', 'N/A')}")
                with col3:
                    st.caption(f"📅 Created: {meta.get('Created', 'N/A')[:10]}")
        
        return best_match
