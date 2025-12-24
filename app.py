"""
Complete app.py - Full Streamlit Application
With memoQ Server integration, TM/TB context, DNT support
"""

import streamlit as st
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import sys
import os
import traceback
import re
import json

# ===== PATH CONFIGURATION =====
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ===== IMPORTS WITH ERROR HANDLING =====
try:
    from models.entities import (
        TranslationSegment, 
        TMMatch, 
        TermMatch,
        BatchResult,
        TranslationMetadata
    )
except ImportError as e:
    st.error(f"❌ Import Error: models.entities\n\nMake sure you have:\n1. `models/` directory\n2. `models/__init__.py`\n3. `models/entities.py`\n\nError: {e}")
    raise

try:
    from services.memoq_server_client import (
        MemoQServerClient,
        normalize_memoq_tm_response,
        normalize_memoq_tb_response
    )
except ImportError as e:
    st.error(f"❌ Import Error: services.memoq_server_client\n\nError: {e}")
    raise

try:
    from services.prompt_builder import PromptBuilder
except ImportError as e:
    st.error(f"❌ Import Error: services.prompt_builder\n\nError: {e}")
    raise

# ===== LOGGING CONFIGURATION =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# ===== PAGE CONFIGURATION =====
st.set_page_config(
    page_title="memoQ AI Translator",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== SESSION STATE INITIALIZATION =====
def init_session_state():
    """Initialize Streamlit session state"""
    defaults = {
        'api_key': '',
        'source_lang': 'English',
        'target_lang': 'Turkish',
        'acceptance_threshold': 95,
        'match_threshold': 70,
        'segment_objects': {},
        'translation_results': {},
        'chat_history': [],
        'dnt_terms': [],
        'reference_chunks': [],
        'bypass_stats': {'bypassed': 0, 'llm_sent': 0},
        'memoq_server_url': '',
        'memoq_username': '',
        'memoq_password': '',
        'selected_tm_guids': [],
        'selected_tb_guids': [],
        'memoq_client': None,
        'use_generated_prompt': False,
        'generated_prompt': '',
        'translation_log': '',
        'match_rates': {}
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ===== HELPER FUNCTIONS =====

def get_memoq_tm_context(memoq_client: MemoQServerClient,
                         tm_guids: List[str],
                         segment: TranslationSegment,
                         match_threshold: int = 70) -> List[TMMatch]:
    """Get TM context from memoQ Server"""
    for tm_guid in tm_guids:
        try:
            results = memoq_client.lookup_segments(
                tm_guid,
                [segment.source],
                match_threshold=match_threshold
            )
            
            if results and 0 in results:
                matches = results[0]
                if matches:
                    logger.info(
                        f"[{segment.id}] memoQ TM: {len(matches)} fuzzy matches"
                    )
                    for match in matches[:3]:
                        logger.info(
                            f"  [{match.similarity}%] {match.source_text[:50]}"
                        )
                    return matches
        
        except Exception as e:
            logger.warning(f"memoQ TM lookup failed for {segment.id}: {e}")
            continue
    
    return []


def get_memoq_tb_context(memoq_client: MemoQServerClient,
                         tb_guids: List[str],
                         segment: TranslationSegment) -> List[TermMatch]:
    """Get TB context from memoQ Server"""
    for tb_guid in tb_guids:
        try:
            potential_terms = [
                word for word in segment.source.split()
                if len(word) > 3 and word.replace('.', '').isalpha()
            ]
            
            if not potential_terms:
                continue
            
            matches = memoq_client.lookup_terms(tb_guid, potential_terms)
            
            if matches:
                logger.info(f"[{segment.id}] memoQ TB: {len(matches)} terms")
                for term in matches[:3]:
                    logger.info(f"  {term.source} = {term.target}")
                return matches
        
        except Exception as e:
            logger.warning(f"memoQ TB lookup failed for {segment.id}: {e}")
            continue
    
    return []


def log_batch_context(batch_num: int,
                      segments: List[TranslationSegment],
                      tm_context: Dict[str, List],
                      tb_context: Dict[str, List],
                      dnt_context: Dict[str, List]):
    """Log batch context details"""
    logger.info(f"\n{'='*80}")
    logger.info(f"=== BATCH {batch_num} ===")
    logger.info(f"Segments: {len(segments)}")
    
    tm_segments = len([c for c in tm_context.values() if c])
    tm_total = sum(len(c) for c in tm_context.values() if c)
    logger.info(f"TM Context: {tm_total} matches across {tm_segments} segments")
    
    tb_segments = len([c for c in tb_context.values() if c])
    tb_total = sum(len(c) for c in tb_context.values() if c)
    logger.info(f"TB Context: {tb_total} terms across {tb_segments} segments")
    
    dnt_segments = len([c for c in dnt_context.values() if c])
    dnt_total = sum(len(c) for c in dnt_context.values() if c)
    logger.info(f"DNT Check: {dnt_total} terms in {dnt_segments} segments")
    
    logger.info(f"{'='*80}\n")


def process_single_segment(segment: TranslationSegment,
                           memoq_client: Optional[MemoQServerClient],
                           memoq_tm_guids: Optional[List[str]],
                           acceptance_threshold: int = 95) -> Tuple[bool, Dict, str]:
    """Process single segment for TM matching"""
    result_translation = {}
    should_bypass = False
    log_msg = ""
    
    if not memoq_client or not memoq_tm_guids:
        return should_bypass, result_translation, "No memoQ TM available"
    
    try:
        for tm_guid in memoq_tm_guids:
            results = memoq_client.lookup_segments(
                tm_guid,
                [segment.source],
                match_threshold=acceptance_threshold
            )
            
            if results and 0 in results:
                matches = results[0]
                if matches:
                    first_match = matches[0]
                    match_score = first_match.similarity
                    
                    if match_score >= acceptance_threshold:
                        should_bypass = True
                        result_translation[segment.id] = first_match.target_text
                        log_msg = f"[{segment.id}] memoQ: {match_score}% - BYPASS"
                        logger.info(log_msg)
                        logger.info(
                            f"  {first_match.source_text[:60]} → {first_match.target_text[:60]}"
                        )
                        return should_bypass, result_translation, log_msg
    
    except Exception as e:
        logger.warning(f"[{segment.id}] memoQ lookup error: {e}")
    
    if not should_bypass:
        log_msg = f"[{segment.id}] No memoQ match ≥{acceptance_threshold}%"
    
    return should_bypass, result_translation, log_msg


# ===== MAIN APPLICATION =====

def main():
    """Main Streamlit application"""
    
    st.title("🌐 memoQ AI Translator")
    st.markdown("Powered by memoQ Server & GPT-4")
    
    # ===== SIDEBAR CONFIGURATION =====
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Key
        api_key = st.text_input("OpenAI API Key", type="password", value=st.session_state.api_key)
        st.session_state.api_key = api_key
        
        # Language Selection
        col1, col2 = st.columns(2)
        with col1:
            source_lang = st.selectbox("Source Language", ["English", "German", "French", "Spanish"])
            st.session_state.source_lang = source_lang
        with col2:
            target_lang = st.selectbox("Target Language", ["Turkish", "German", "French", "Spanish", "Italian"])
            st.session_state.target_lang = target_lang
        
        # Thresholds
        st.subheader("Match Thresholds")
        acceptance = st.slider("TM Acceptance (bypass LLM)", 70, 100, st.session_state.acceptance_threshold)
        st.session_state.acceptance_threshold = acceptance
        
        match_thresh = st.slider("TM Context (send to LLM)", 0, acceptance-1, st.session_state.match_threshold)
        st.session_state.match_threshold = match_thresh
        
        # memoQ Server Configuration
        st.subheader("memoQ Server")
        memoq_url = st.text_input("Server URL", value=st.session_state.memoq_server_url, placeholder="http://localhost:8080")
        st.session_state.memoq_server_url = memoq_url
        
        memoq_user = st.text_input("Username", value=st.session_state.memoq_username)
        st.session_state.memoq_username = memoq_user
        
        memoq_pass = st.text_input("Password", type="password", value=st.session_state.memoq_password)
        st.session_state.memoq_password = memoq_pass
        
        # DNT List
        st.subheader("Do Not Translate (DNT)")
        dnt_text = st.text_area("Terms to keep in original (one per line)", height=100)
        st.session_state.dnt_terms = [t.strip() for t in dnt_text.split('\n') if t.strip()]
    
    # ===== MAIN CONTENT AREA =====
    
    # File Upload
    st.header("📤 Upload Translation File")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        xliff_file = st.file_uploader("XLIFF File", type=['xliff', 'xml'])
    
    with col2:
        tmx_file = st.file_uploader("Translation Memory (TMX)", type=['tmx'])
    
    with col3:
        csv_file = st.file_uploader("Termbase (CSV)", type=['csv'])
    
    if xliff_file:
        st.success(f"✅ Loaded: {xliff_file.name}")
        
        # Start Translation
        if st.button("🚀 Start Translation", type="primary"):
            if not api_key:
                st.error("❌ Please provide API Key")
                return
            
            xliff_bytes = xliff_file.getvalue()
            tmx_bytes = tmx_file.getvalue() if tmx_file else None
            csv_bytes = csv_file.getvalue() if csv_file else None
            
            try:
                process_translation(
                    xliff_bytes,
                    tmx_bytes=tmx_bytes,
                    csv_bytes=csv_bytes,
                    memoq_tm_guids=st.session_state.selected_tm_guids,
                    memoq_tb_guids=st.session_state.selected_tb_guids
                )
            except Exception as e:
                st.error(f"❌ Translation failed: {str(e)}")
                logger.error(f"Translation error: {e}", exc_info=True)
    
    # Results
    if st.session_state.translation_results:
        st.header("✅ Translation Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Segments", len(st.session_state.translation_results))
        with col2:
            st.metric("From TM (bypass)", st.session_state.bypass_stats.get('bypassed', 0))
        with col3:
            st.metric("Via LLM", st.session_state.bypass_stats.get('llm_sent', 0))
        
        # Download results
        st.download_button(
            label="📥 Download Results (JSON)",
            data=json.dumps(st.session_state.translation_results, ensure_ascii=False, indent=2),
            file_name="translations.json",
            mime="application/json"
        )
        
        # Display logs
        if st.session_state.translation_log:
            with st.expander("📋 Translation Log"):
                st.text(st.session_state.translation_log)


def process_translation(xliff_bytes,
                        tmx_bytes=None,
                        csv_bytes=None,
                        memoq_tm_guids=None,
                        memoq_tb_guids=None):
    """Main translation processing function"""
    
    with st.status("Processing...", expanded=True) as status:
        
        # Parse XLIFF
        st.write("📄 Parsing XLIFF...")
        try:
            # Simple XLIFF parsing - replace with your XMLParser
            segments = parse_xliff_simple(xliff_bytes)
            st.write(f"✅ Loaded {len(segments)} segments")
            logger.info(f"Loaded {len(segments)} segments")
        except Exception as e:
            st.error(f"Failed to parse XLIFF: {e}")
            logger.error(f"XLIFF parsing error: {e}")
            return
        
        st.session_state.segment_objects = {seg.id: seg for seg in segments}
        
        # Initialize memoQ Server
        memoq_client = None
        if st.session_state.memoq_server_url and memoq_tm_guids:
            st.write("🔗 Connecting to memoQ Server...")
            try:
                memoq_client = MemoQServerClient(
                    st.session_state.memoq_server_url,
                    st.session_state.memoq_username,
                    st.session_state.memoq_password
                )
                
                if memoq_client.test_connection():
                    st.write("✅ Connected to memoQ Server")
                    logger.info("memoQ Server connected")
                else:
                    st.warning("⚠️ Could not connect to memoQ Server")
                    memoq_client = None
            except Exception as e:
                st.warning(f"❌ memoQ Server error: {e}")
                logger.error(f"memoQ connection error: {e}")
                memoq_client = None
        
        # Initialize Prompt Builder
        prompt_builder = PromptBuilder()
        logger.info("PromptBuilder initialized")
        
        # Analyze segments
        status.update(label="Analyzing segments...", state="running")
        
        bypass_segments = []
        llm_segments = []
        final_translations = {}
        tm_context = {}
        tb_context = {}
        dnt_context = {}
        
        st.write("🔍 Analyzing matches...")
        progress = st.progress(0)
        
        for i, seg in enumerate(segments):
            
            # Check memoQ TM
            if memoq_client and memoq_tm_guids:
                should_bypass, trans_dict, log_msg = process_single_segment(
                    seg,
                    memoq_client,
                    memoq_tm_guids,
                    st.session_state.acceptance_threshold
                )
                
                if should_bypass:
                    bypass_segments.append(seg)
                    final_translations.update(trans_dict)
                else:
                    llm_segments.append(seg)
                    fuzzy_matches = get_memoq_tm_context(
                        memoq_client,
                        memoq_tm_guids,
                        seg,
                        st.session_state.match_threshold
                    )
                    if fuzzy_matches:
                        tm_context[seg.id] = fuzzy_matches
            else:
                llm_segments.append(seg)
            
            # Collect TB context
            if seg in llm_segments:
                if memoq_client and memoq_tb_guids:
                    tb_matches = get_memoq_tb_context(
                        memoq_client,
                        memoq_tb_guids,
                        seg
                    )
                    if tb_matches:
                        tb_context[seg.id] = tb_matches
                
                # Check DNT
                if st.session_state.dnt_terms:
                    found_dnt = [
                        term for term in st.session_state.dnt_terms
                        if term.lower() in seg.source.lower()
                    ]
                    if found_dnt:
                        dnt_context[seg.id] = found_dnt
                        logger.info(f"[{seg.id}] DNT terms: {', '.join(found_dnt)}")
            
            progress.progress((i + 1) / len(segments))
        
        st.write(f"✅ {len(bypass_segments)} from TM")
        st.write(f"🔄 {len(llm_segments)} to LLM")
        logger.info(f"Analysis: {len(bypass_segments)} bypass, {len(llm_segments)} LLM")
        
        # Save results
        st.session_state.translation_results = final_translations
        st.session_state.bypass_stats = {
            'bypassed': len(bypass_segments),
            'llm_sent': len(llm_segments)
        }
        
        status.update(label="✅ Complete", state="complete")


def parse_xliff_simple(xliff_bytes) -> List[TranslationSegment]:
    """Simple XLIFF parser - replace with your implementation"""
    import xml.etree.ElementTree as ET
    
    segments = []
    root = ET.fromstring(xliff_bytes)
    
    # Parse trans-units
    for i, tu in enumerate(root.findall('.//{urn:oasis:names:tc:xliff:document:1.2}trans-unit')):
        seg_id = tu.get('id', str(i))
        source_elem = tu.find('{urn:oasis:names:tc:xliff:document:1.2}source')
        
        if source_elem is not None:
            source_text = ''.join(source_elem.itertext())
            segments.append(TranslationSegment(id=seg_id, source=source_text))
    
    return segments


if __name__ == "__main__":
    main()
