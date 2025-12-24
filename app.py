"""
STREAMLIT APPLICATION - Updated key sections for memoQ Server integration

This file contains the UPDATED SECTIONS that need to be integrated into your existing app.py
Copy these sections and replace the corresponding sections in your app.py

Key areas updated:
1. Imports
2. Helper functions for memoQ context collection
3. Main process_translation function
4. Logging enhancements
"""

# ===== IMPORTS SECTION (ADD TO TOP OF APP.PY) =====
import streamlit as st
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import re

# Import updated models and services
from models.entities import (
    TranslationSegment, TMMatch, TermMatch, 
    BatchResult, TranslationMetadata, SegmentAnalysis
)
from services.memoq_server_client import (
    MemoQServerClient,
    normalize_memoq_tm_response,
    normalize_memoq_tb_response
)
from services.prompt_builder import PromptBuilder
# ... other existing imports ...

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===== HELPER FUNCTIONS (ADD TO APP.PY) =====

def get_memoq_tm_context(memoq_client: MemoQServerClient,
                         tm_guids: List[str],
                         segment: TranslationSegment,
                         match_threshold: int = 70) -> List[TMMatch]:
    """
    Get TM context from memoQ Server for a segment
    Returns normalized TMMatch objects
    
    Args:
        memoq_client: MemoQServerClient instance
        tm_guids: List of TM GUIDs to search
        segment: TranslationSegment object
        match_threshold: Minimum match % to include
    
    Returns:
        List of TMMatch objects sorted by similarity (best first)
    """
    for tm_guid in tm_guids:
        try:
            # Lookup returns already-normalized TMMatch objects
            results = memoq_client.lookup_segments(
                tm_guid,
                [segment.source],
                match_threshold=match_threshold
            )
            
            if results and 0 in results:
                matches = results[0]
                if matches:
                    logger.info(
                        f"[{segment.id}] memoQ TM: {len(matches)} fuzzy matches (threshold ≥{match_threshold}%)"
                    )
                    # Log top 3 matches
                    for match in matches[:3]:
                        logger.info(
                            f"  [{match.similarity}%] {match.source_text[:50]}... "
                            f"→ {match.target_text[:50]}..."
                        )
                    return matches
        
        except Exception as e:
            logger.warning(f"memoQ TM lookup failed for segment {segment.id}: {e}")
            continue
    
    logger.info(f"[{segment.id}] No memoQ TM matches found (threshold ≥{match_threshold}%)")
    return []


def get_memoq_tb_context(memoq_client: MemoQServerClient,
                         tb_guids: List[str],
                         segment: TranslationSegment) -> List[TermMatch]:
    """
    Get TB context from memoQ Server for a segment
    Returns normalized TermMatch objects
    
    Args:
        memoq_client: MemoQServerClient instance
        tb_guids: List of TB GUIDs to search
        segment: TranslationSegment object
    
    Returns:
        List of TermMatch objects
    """
    for tb_guid in tb_guids:
        try:
            # Extract potential terms from segment
            # Simple approach: words > 3 chars
            potential_terms = [
                word for word in segment.source.split()
                if len(word) > 3 and word.replace('.', '').isalpha()
            ]
            
            if not potential_terms:
                continue
            
            # Lookup returns already-normalized TermMatch objects
            matches = memoq_client.lookup_terms(tb_guid, potential_terms)
            
            if matches:
                logger.info(f"[{segment.id}] memoQ TB: {len(matches)} terms found")
                for term in matches[:3]:
                    logger.info(f"  {term.source} = {term.target}")
                return matches
        
        except Exception as e:
            logger.warning(f"memoQ TB lookup failed for segment {segment.id}: {e}")
            continue
    
    logger.info(f"[{segment.id}] No memoQ TB terms found")
    return []


def log_batch_context(batch_num: int,
                      segments: List[TranslationSegment],
                      tm_context: Dict[str, List],
                      tb_context: Dict[str, List],
                      dnt_context: Dict[str, List]):
    """
    Log detailed context information for a batch before sending to LLM
    
    Args:
        batch_num: Batch number
        segments: List of segments in batch
        tm_context: {segment_id: [TMMatch objects]}
        tb_context: {segment_id: [TermMatch objects]}
        dnt_context: {segment_id: [DNT terms]}
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"=== BATCH {batch_num} ===")
    logger.info(f"Segments in batch: {len(segments)}")
    
    # TM Summary
    tm_segments_with_context = len([c for c in tm_context.values() if c])
    tm_total_matches = sum(len(c) for c in tm_context.values() if c)
    logger.info(f"TM Context: {tm_total_matches} matches across {tm_segments_with_context} segments")
    
    # TB Summary
    tb_segments_with_context = len([c for c in tb_context.values() if c])
    tb_total_terms = sum(len(c) for c in tb_context.values() if c)
    logger.info(f"TB Context: {tb_total_terms} terms across {tb_segments_with_context} segments")
    
    # DNT Summary
    dnt_segments_with_terms = len([c for c in dnt_context.values() if c])
    dnt_total = sum(len(c) for c in dnt_context.values() if c)
    logger.info(f"DNT Check: {dnt_total} forbidden terms in {dnt_segments_with_terms} segments")
    
    logger.info(f"{'='*80}\n")


def log_prompt_details(batch_num: int, prompt: str):
    """Log prompt details"""
    tm_lines = len([l for l in prompt.split('\n') if '[FUZZY' in l or '[EXACT' in l])
    tb_lines = len([l for l in prompt.split('\n') if '=' in l and l.count('=') == 1 and '-' in l])
    
    logger.info(f">>> PROMPT SENT TO LLM (Batch {batch_num}) >>>")
    logger.info(f"Prompt size: {len(prompt)} characters")
    logger.info(f"TM context lines: {tm_lines}")
    logger.info(f"TB context lines: {tb_lines}")
    logger.info(">>> END PROMPT >>>")


def process_single_segment(segment: TranslationSegment,
                           memoq_client: Optional[MemoQServerClient],
                           memoq_tm_guids: Optional[List[str]],
                           memoq_tb_guids: Optional[List[str]],
                           acceptance_threshold: int = 95,
                           match_threshold: int = 70) -> Tuple[bool, Dict, str]:
    """
    Process a single segment for TM matching and context collection
    
    Args:
        segment: TranslationSegment to process
        memoq_client: memoQ Server client instance
        memoq_tm_guids: List of TM GUIDs
        memoq_tb_guids: List of TB GUIDs
        acceptance_threshold: % threshold to bypass LLM
        match_threshold: % threshold for context inclusion
    
    Returns:
        Tuple of (should_bypass, final_translations_dict, log_message)
    """
    result_translation = {}
    should_bypass = False
    log_msg = ""
    
    if not memoq_client or not memoq_tm_guids:
        return should_bypass, result_translation, "No memoQ TM available"
    
    # Check for 100% match in memoQ
    try:
        for tm_guid in memoq_tm_guids:
            # Lookup with high threshold to find near-perfect matches
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
                    
                    # If meets bypass threshold, use it
                    if match_score >= acceptance_threshold:
                        should_bypass = True
                        result_translation[segment.id] = first_match.target_text
                        log_msg = f"[{segment.id}] memoQ match: {match_score}% - BYPASS"
                        logger.info(f"{log_msg}")
                        logger.info(
                            f"  {first_match.source_text[:60]}... "
                            f"→ {first_match.target_text[:60]}..."
                        )
                        return should_bypass, result_translation, log_msg
                    
                    else:
                        # Below threshold, but log for reference
                        log_msg = f"[{segment.id}] memoQ match: {match_score}% (below {acceptance_threshold}%)"
                        logger.info(log_msg)
    
    except Exception as e:
        logger.warning(f"[{segment.id}] memoQ lookup error: {e}")
    
    if not should_bypass:
        log_msg = f"[{segment.id}] No memoQ match ≥{acceptance_threshold}% - needs LLM"
    
    return should_bypass, result_translation, log_msg


# ===== MAIN PROCESS_TRANSLATION FUNCTION (UPDATED SECTIONS) =====
# This shows the key updated sections of process_translation

def process_translation(xliff_file,
                        tmx_bytes=None,
                        csv_bytes=None,
                        custom_prompt_content=None,
                        memoq_tm_guids=None,
                        memoq_tb_guids=None):
    """
    Main translation processing function
    Handles TM matching, batch processing, and LLM translation
    
    UPDATED: Full memoQ Server integration with context collection
    
    Args:
        xliff_file: XLIFF file object
        tmx_bytes: Optional local TMX file bytes
        csv_bytes: Optional local CSV termbase bytes
        custom_prompt_content: Optional custom prompt template
        memoq_tm_guids: List of memoQ TM GUIDs
        memoq_tb_guids: List of memoQ TB GUIDs
    """
    
    if not st.session_state.api_key:
        st.error("❌ Please provide an API Key in settings.")
        return
    
    with st.status("Processing translation...", expanded=True) as status:
        
        # ===== PHASE 1: PARSE XLIFF =====
        st.write("📄 Parsing XLIFF file...")
        segments = []  # This should come from your XML parser
        # segments = XMLParser.parse_xliff(xliff_file.getvalue())
        
        st.write(f"✅ Loaded {len(segments)} segments")
        logger.info(f"Loaded {len(segments)} segments from XLIFF")
        
        # Initialize session state
        st.session_state.segment_objects = {seg.id: seg for seg in segments}
        st.session_state.chat_history = []
        
        # Create transaction logger
        logger.info(f"Started translation job for {len(segments)} segments.")
        logger.info(f"Source: {st.session_state.source_lang} | Target: {st.session_state.target_lang}")
        logger.info(f"TM Acceptance: ≥{st.session_state.acceptance_threshold}%")
        logger.info(f"TM Match (context): ≥{st.session_state.match_threshold}%")
        
        # ===== PHASE 2: INITIALIZE memoQ SERVER CLIENT =====
        memoq_client = None
        
        if memoq_tm_guids or memoq_tb_guids:
            st.write("🔗 Connecting to memoQ Server...")
            
            try:
                if st.session_state.get('memoq_client'):
                    memoq_client = st.session_state.memoq_client
                    
                    # Test connection
                    if memoq_client.test_connection():
                        st.write("✅ Connected to memoQ Server")
                        
                        if memoq_tm_guids:
                            st.write(f"   • {len(memoq_tm_guids)} Translation Memory(ies)")
                            logger.info(f"memoQ TMs selected: {len(memoq_tm_guids)}")
                        
                        if memoq_tb_guids:
                            st.write(f"   • {len(memoq_tb_guids)} Termbase(s)")
                            logger.info(f"memoQ TBs selected: {len(memoq_tb_guids)}")
                    
                    else:
                        st.warning("⚠️ Could not connect to memoQ Server")
                        logger.warning("memoQ Server connection failed")
                        memoq_client = None
            
            except Exception as e:
                st.warning(f"❌ memoQ Server error: {str(e)}")
                logger.error(f"memoQ Server error: {e}", exc_info=True)
                memoq_client = None
        
        # ===== PHASE 3: INITIALIZE PROMPT BUILDER =====
        if st.session_state.use_generated_prompt and st.session_state.generated_prompt:
            prompt_builder = PromptBuilder(custom_template=st.session_state.generated_prompt)
        elif custom_prompt_content:
            prompt_builder = PromptBuilder(custom_template=custom_prompt_content)
        else:
            prompt_builder = PromptBuilder()  # Uses default template
        
        logger.info("PromptBuilder initialized")
        
        # ===== PHASE 4: ANALYZE SEGMENTS & COLLECT CONTEXT =====
        status.update(label="Analyzing segments...", state="running")
        
        bypass_segments = []
        llm_segments = []
        final_translations = {}
        tm_context = {}
        tb_context = {}
        dnt_context = {}
        match_rates = {}
        
        st.write("🔍 Analyzing TM matches and collecting context...")
        analysis_progress = st.progress(0)
        
        for i, seg in enumerate(segments):
            
            # ===== Check memoQ TM for 100% match =====
            if memoq_client and memoq_tm_guids:
                should_bypass, trans_dict, log_msg = process_single_segment(
                    seg,
                    memoq_client,
                    memoq_tm_guids,
                    memoq_tb_guids,
                    acceptance_threshold=st.session_state.acceptance_threshold,
                    match_threshold=st.session_state.match_threshold
                )
                
                if should_bypass:
                    bypass_segments.append(seg)
                    final_translations.update(trans_dict)
                    # Extract match rate from memoQ response
                    match_rates[seg.id] = 100
                else:
                    llm_segments.append(seg)
                    
                    # Collect fuzzy matches for LLM context
                    fuzzy_matches = get_memoq_tm_context(
                        memoq_client,
                        memoq_tm_guids,
                        seg,
                        match_threshold=st.session_state.match_threshold
                    )
                    if fuzzy_matches:
                        tm_context[seg.id] = fuzzy_matches
            
            else:
                # No memoQ - all segments to LLM
                llm_segments.append(seg)
            
            # ===== Collect TB context for LLM segments =====
            if seg in llm_segments:
                
                # memoQ Termbase
                if memoq_client and memoq_tb_guids:
                    tb_matches = get_memoq_tb_context(
                        memoq_client,
                        memoq_tb_guids,
                        seg
                    )
                    if tb_matches:
                        tb_context[seg.id] = tb_matches
                
                # Check DNT list
                if st.session_state.dnt_terms:
                    found_dnt = []
                    for dnt_term in st.session_state.dnt_terms:
                        if dnt_term.lower() in seg.source.lower():
                            found_dnt.append(dnt_term)
                    
                    if found_dnt:
                        dnt_context[seg.id] = found_dnt
                        logger.info(f"[{seg.id}] DNT terms detected: {', '.join(found_dnt)}")
            
            analysis_progress.progress((i + 1) / len(segments))
        
        # Summary
        st.session_state.bypass_stats = {
            'bypassed': len(bypass_segments),
            'llm_sent': len(llm_segments)
        }
        
        st.write(f"✅ **{len(bypass_segments)}** segments from memoQ TM (≥{st.session_state.acceptance_threshold}% match)")
        st.write(f"🔄 **{len(llm_segments)}** segments need LLM translation")
        
        logger.info(
            f"Segment analysis complete: {len(bypass_segments)} bypass, "
            f"{len(llm_segments)} LLM, {len(tm_context)} with TM context"
        )
        
        # ===== PHASE 5: PROCESS LLM SEGMENTS IN BATCHES =====
        if llm_segments:
            status.update(label=f"Translating {len(llm_segments)} segments...", state="running")
            
            llm_progress = st.progress(0)
            batch_translations_history = []
            batch_size = 20  # Configure as needed
            
            for i in range(0, len(llm_segments), batch_size):
                batch = llm_segments[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(llm_segments) + batch_size - 1) // batch_size
                
                st.write(f"📤 **Batch {batch_num}/{total_batches}** ({len(batch)} segments)")
                
                # ===== Prepare batch context =====
                batch_tm = {seg.id: tm_context.get(seg.id, []) for seg in batch}
                batch_tb = {seg.id: tb_context.get(seg.id, []) for seg in batch}
                batch_dnt = {seg.id: dnt_context.get(seg.id, []) for seg in batch}
                
                # Log batch context
                log_batch_context(batch_num, batch, batch_tm, batch_tb, batch_dnt)
                
                # ===== BUILD PROMPT =====
                prompt = prompt_builder.build_prompt(
                    source_lang=st.session_state.source_lang,
                    target_lang=st.session_state.target_lang,
                    segments=batch,
                    tm_context=batch_tm,
                    tb_context=batch_tb,
                    chat_history=batch_translations_history[-50:],  # Last 50 translations
                    reference_context=None,  # Optional reference text
                    dnt_terms=st.session_state.dnt_terms
                )
                
                # Log prompt details
                log_prompt_details(batch_num, prompt)
                
                # ===== SEND TO LLM =====
                try:
                    # Your LLM translation call here
                    # response_text = translator.translate_batch(prompt)
                    
                    logger.info(f"Batch {batch_num} translation completed")
                
                except Exception as e:
                    st.error(f"❌ Batch {batch_num} failed: {str(e)}")
                    logger.error(f"Batch {batch_num} failed: {e}", exc_info=True)
                
                llm_progress.progress((i + len(batch)) / len(llm_segments))
        
        # ===== PHASE 6: SAVE RESULTS =====
        st.session_state.translation_results = final_translations
        st.session_state.match_rates = match_rates
        st.session_state.chat_history = batch_translations_history
        
        status.update(label="✅ Translation Complete!", state="complete")
        
        st.success(f"""
        **Translation Complete!**
        - ✅ {len(bypass_segments)} segments from memoQ TM (no LLM cost)
        - 🔄 {len(llm_segments)} segments via LLM
        - 📊 Total: {len(final_translations)} translations
        """)
        
        logger.info(
            f"Translation job complete: "
            f"{len(bypass_segments)} bypass + "
            f"{len(llm_segments)} LLM = "
            f"{len(final_translations)} total"
        )


# ===== END OF UPDATED SECTIONS =====
# Integrate these functions and sections into your existing app.py
