import streamlit as st
import pandas as pd
import time
from datetime import datetime
from services.tm_matcher import TMatcher
from services.tb_matcher import TBMatcher
from services.prompt_builder import PromptBuilder
from services.ai_translator import AITranslator
from services.caching import CacheManager
from services.doc_analyzer import DocumentAnalyzer, PromptGenerator
from services.embedding_matcher import EmbeddingMatcher, get_embedding_cost_estimate
from utils.xml_parser import XMLParser
from utils.logger import TransactionLogger
import config
from services.memoq_server_client import MemoQServerClient
from services.memoq_ui import MemoQUI
from analysis_screen import show_analysis_screen
# --- Setup ---
st.set_page_config(page_title=config.APP_NAME, layout="wide", page_icon="🌍")

# Session state initialization
if 'translation_results' not in st.session_state:
    st.session_state.translation_results = {}
if 'segment_objects' not in st.session_state:
    st.session_state.segment_objects = {}
if 'translation_log' not in st.session_state:
    st.session_state.translation_log = ""
if 'tm_info' not in st.session_state:
    st.session_state.tm_info = None
if 'bypass_stats' not in st.session_state:
    st.session_state.bypass_stats = {'bypassed': 0, 'llm_sent': 0}
if 'detected_languages' not in st.session_state:
    st.session_state.detected_languages = {'source': None, 'target': None}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
# Prompt Builder state
if 'generated_prompt' not in st.session_state:
    st.session_state.generated_prompt = None
if 'prompt_metadata' not in st.session_state:
    st.session_state.prompt_metadata = {}
if 'use_generated_prompt' not in st.session_state:
    st.session_state.use_generated_prompt = False
# Reference file state
if 'reference_chunks' not in st.session_state:
    st.session_state.reference_chunks = []
if 'embedding_matcher' not in st.session_state:
    st.session_state.embedding_matcher = None
if 'reference_embeddings_ready' not in st.session_state:
    st.session_state.reference_embeddings_ready = False
# DNT (Do Not Translate) list
if 'dnt_terms' not in st.session_state:
    st.session_state.dnt_terms = []

# memoQ Server state
if 'memoq_server_url' not in st.session_state:
    st.session_state.memoq_server_url = "https://mirage.memoq.com:9091/adaturkey"
if 'memoq_username' not in st.session_state:
    st.session_state.memoq_username = ""
if 'memoq_password' not in st.session_state:
    st.session_state.memoq_password = ""
if 'memoq_verify_ssl' not in st.session_state:
    st.session_state.memoq_verify_ssl = False
if 'memoq_connected' not in st.session_state:
    st.session_state.memoq_connected = False
if 'memoq_client' not in st.session_state:
    st.session_state.memoq_client = None
if 'selected_tm_guids' not in st.session_state:
    st.session_state.selected_tm_guids = []
if 'selected_tb_guids' not in st.session_state:
    st.session_state.selected_tb_guids = []
if 'memoq_tms_list' not in st.session_state:
    st.session_state.memoq_tms_list = []
if 'memoq_tbs_list' not in st.session_state:
    st.session_state.memoq_tbs_list = []

# Analysis screen state
if 'analysis_triggered' not in st.session_state:
    st.session_state.analysis_triggered = False
if 'ready_to_translate' not in st.session_state:
    st.session_state.ready_to_translate = False

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Language Settings
    st.subheader("🌐 Languages")
    
    detected_src = st.session_state.detected_languages.get('source')
    detected_tgt = st.session_state.detected_languages.get('target')
    
    # Convert detected ISO codes (en-gb) to memoQ codes (eng-GB)
    if detected_src:
        parts = detected_src.split('-')
        if len(parts) == 2:
            # en-gb -> eng-GB
            base_lang_map = {
                'en': 'eng', 'tr': 'tur', 'de': 'ger', 'fr': 'fra', 'es': 'spa',
                'it': 'ita', 'pt': 'por', 'pl': 'pol', 'ru': 'rus', 'ja': 'jpn',
                'zh': 'zho', 'ar': 'ara', 'ko': 'kor', 'nl': 'nld', 'sv': 'swe',
                'no': 'nor', 'da': 'dan', 'fi': 'fin', 'el': 'ell', 'he': 'heb',
                'th': 'tha', 'vi': 'vie', 'bg': 'bul', 'ro': 'ron', 'cs': 'ces',
                'sk': 'slk', 'uk': 'ukr', 'et': 'est', 'lv': 'lav', 'lt': 'lit'
            }
            base = base_lang_map.get(parts[0], parts[0])
            detected_src = f"{base}-{parts[1].upper()}"
        else:
            # Single code like 'en' -> 'eng'
            base_lang_map = {
                'en': 'eng', 'tr': 'tur', 'de': 'ger', 'fr': 'fra', 'es': 'spa',
                'it': 'ita', 'pt': 'por', 'pl': 'pol', 'ru': 'rus', 'ja': 'jpn',
                'zh': 'zho', 'ar': 'ara', 'ko': 'kor', 'nl': 'nld', 'sv': 'swe',
                'no': 'nor', 'da': 'dan', 'fi': 'fin', 'el': 'ell', 'he': 'heb',
                'th': 'tha', 'vi': 'vie', 'bg': 'bul', 'ro': 'ron', 'cs': 'ces',
                'sk': 'slk', 'uk': 'ukr', 'et': 'est', 'lv': 'lav', 'lt': 'lit'
            }
            detected_src = base_lang_map.get(detected_src, detected_src)
    
    if detected_tgt:
        parts = detected_tgt.split('-')
        if len(parts) == 2:
            base_lang_map = {
                'en': 'eng', 'tr': 'tur', 'de': 'ger', 'fr': 'fra', 'es': 'spa',
                'it': 'ita', 'pt': 'por', 'pl': 'pol', 'ru': 'rus', 'ja': 'jpn',
                'zh': 'zho', 'ar': 'ara', 'ko': 'kor', 'nl': 'nld', 'sv': 'swe',
                'no': 'nor', 'da': 'dan', 'fi': 'fin', 'el': 'ell', 'he': 'heb',
                'th': 'tha', 'vi': 'vie', 'bg': 'bul', 'ro': 'ron', 'cs': 'ces',
                'sk': 'slk', 'uk': 'ukr', 'et': 'est', 'lv': 'lav', 'lt': 'lit'
            }
            base = base_lang_map.get(parts[0], parts[0])
            detected_tgt = f"{base}-{parts[1].upper()}"
        else:
            base_lang_map = {
                'en': 'eng', 'tr': 'tur', 'de': 'ger', 'fr': 'fra', 'es': 'spa',
                'it': 'ita', 'pt': 'por', 'pl': 'pol', 'ru': 'rus', 'ja': 'jpn',
                'zh': 'zho', 'ar': 'ara', 'ko': 'kor', 'nl': 'nld', 'sv': 'swe',
                'no': 'nor', 'da': 'dan', 'fi': 'fin', 'el': 'ell', 'he': 'heb',
                'th': 'tha', 'vi': 'vie', 'bg': 'bul', 'ro': 'ron', 'cs': 'ces',
                'sk': 'slk', 'uk': 'ukr', 'et': 'est', 'lv': 'lav', 'lt': 'lit'
            }
            detected_tgt = base_lang_map.get(detected_tgt, detected_tgt)
    
    lang_keys = list(config.SUPPORTED_LANGUAGES.keys())
    
    src_default = lang_keys.index(detected_src) if detected_src in lang_keys else lang_keys.index('eng')
    tgt_default = lang_keys.index(detected_tgt) if detected_tgt in lang_keys else lang_keys.index('tur')
    
    src_code = st.selectbox(
        "Source Language", 
        lang_keys, 
        index=src_default,
        format_func=lambda x: f"{config.SUPPORTED_LANGUAGES[x]} ({x})" + (" ✓" if x == detected_src else "")
    )
    tgt_code = st.selectbox(
        "Target Language", 
        lang_keys, 
        index=tgt_default,
        format_func=lambda x: f"{config.SUPPORTED_LANGUAGES[x]} ({x})" + (" ✓" if x == detected_tgt else "")
    )
    
    if detected_src and detected_tgt:
        st.caption(f"🔍 Auto-detected: {detected_src} → {detected_tgt}")
    
    st.divider()
    
    # AI Settings
    st.subheader("🤖 AI Settings")
    api_key = st.text_input("API Key", type="password")
    model = st.selectbox("Model", config.OPENAI_MODELS)
    
    st.divider()
    
    # TM Settings
    st.subheader("📚 TM Settings")
    
    acceptance_threshold = st.slider(
        "TM Acceptance Threshold",
        min_value=70,
        max_value=100,
        value=config.DEFAULT_ACCEPTANCE_THRESHOLD,
        help="Matches ≥ this value bypass LLM (direct TM usage)"
    )
    
    match_threshold = st.slider(
        "TM Match Threshold",
        min_value=50,
        max_value=100,
        value=config.DEFAULT_MATCH_THRESHOLD,
        help="Matches ≥ this value are sent as context to LLM"
    )
    
    if acceptance_threshold <= match_threshold:
        st.warning("Acceptance should be higher than Match threshold")
    
    st.divider()
    
    # Chat History Settings
    st.subheader("💬 Chat History")
    chat_history_length = st.slider(
        "Previous batches to include",
        min_value=0,
        max_value=10,
        value=config.DEFAULT_CHAT_HISTORY,
        help="Number of previous translation batches to include for consistency"
    )
    
    st.divider()
    
    # Cache Management
    st.subheader("🗄️ TM Cache")
    cache_files = CacheManager.get_cache_info()
    if cache_files:
        st.caption(f"{len(cache_files)} cached TM(s)")
        if st.button("🗑️ Clear All Cache", type="secondary", use_container_width=True):
            count = CacheManager.clear_tm_cache()
            st.success(f"Cleared {count} cache file(s)")
            st.rerun()
    else:
        st.caption("No cached TMs")

# ==================== memoQ SERVER CONNECTION ====================
    st.divider()
    st.subheader("🔗 memoQ Server")
    
    with st.form("memoq_connection_form"):
        memoq_url = st.text_input(
            "Server URL",
            value=st.session_state.memoq_server_url,
            help="memoQ Server base URL",
            key="memoq_url_input"
        )
        
        memoq_user = st.text_input(
            "Username",
            value=st.session_state.memoq_username,
            key="memoq_user_input"
        )
        
        memoq_pass = st.text_input(
            "Password",
            type="password",
            value=st.session_state.memoq_password,
            key="memoq_pass_input"
        )
        
        memoq_ssl = st.checkbox(
            "Verify SSL",
            value=st.session_state.memoq_verify_ssl,
            help="Disable for self-signed certificates"
        )
        
        memoq_connect = st.form_submit_button("🔐 Connect", use_container_width=True)
    
    if memoq_connect:
        st.session_state.memoq_server_url = memoq_url
        st.session_state.memoq_username = memoq_user
        st.session_state.memoq_password = memoq_pass
        st.session_state.memoq_verify_ssl = memoq_ssl
        
        try:
            client = MemoQServerClient(
                server_url=memoq_url,
                username=memoq_user,
                password=memoq_pass,
                verify_ssl=memoq_ssl
            )
            client.login()
            st.session_state.memoq_client = client
            st.session_state.memoq_connected = True
            st.success("✓ Connected to memoQ Server")
        except Exception as e:
            st.error(f"Connection failed: {str(e)}")
            st.session_state.memoq_connected = False
            st.session_state.memoq_client = None
    
    if st.session_state.memoq_connected and st.session_state.memoq_client:
        st.success("✓ Connected to memoQ Server")
        if st.button("🔌 Disconnect", use_container_width=True):
            st.session_state.memoq_connected = False
            st.session_state.memoq_client = None
            st.rerun()
    
    # Show if using generated prompt
    if st.session_state.use_generated_prompt and st.session_state.generated_prompt:
        st.divider()
        st.success("✨ Using generated prompt")
        if st.button("❌ Clear Generated Prompt"):
            st.session_state.use_generated_prompt = False
            st.session_state.generated_prompt = None
            st.rerun()


# --- Helper Functions ---

def parse_reference_file(content: bytes, filename: str) -> list:
    """
    Parse reference file (target-only text) into chunks for style reference.
    Supports TXT, DOCX, PDF, HTML, RTF, and Excel formats.
    
    Returns list of text chunks (sentences/paragraphs).
    """
    chunks = []
    filename_lower = filename.lower()
    
    try:
        # === TXT ===
        if filename_lower.endswith('.txt'):
            text = None
            for encoding in ['utf-8', 'utf-8-sig', 'utf-16', 'latin-1', 'cp1252', 'iso-8859-9']:
                try:
                    text = content.decode(encoding)
                    break
                except:
                    continue
            
            if text:
                text = text.replace('\r\n', '\n').replace('\r', '\n')
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                
                import re
                for line in lines:
                    clean_line = re.sub(r'\d+$', '', line).strip()
                    if clean_line and len(clean_line) > 15:
                        chunks.append(clean_line)
        
        # === DOCX ===
        elif filename_lower.endswith('.docx'):
            from docx import Document
            import io
            doc = Document(io.BytesIO(content))
            for para in doc.paragraphs:
                text = para.text.strip()
                if text and len(text) > 15:
                    chunks.append(text)
        
        # === PDF ===
        elif filename_lower.endswith('.pdf'):
            import io
            try:
                import pdfplumber
                with pdfplumber.open(io.BytesIO(content)) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            lines = [line.strip() for line in text.split('\n') if line.strip()]
                            for line in lines:
                                if len(line) > 15:
                                    chunks.append(line)
            except ImportError:
                st.warning("PDF support requires pdfplumber: pip install pdfplumber")
        
        # === HTML ===
        elif filename_lower.endswith(('.html', '.htm')):
            try:
                from bs4 import BeautifulSoup
                text = None
                for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                    try:
                        text = content.decode(encoding)
                        break
                    except:
                        continue
                
                if text:
                    soup = BeautifulSoup(text, 'html.parser')
                    # Remove script and style elements
                    for element in soup(['script', 'style', 'head', 'meta', 'link']):
                        element.decompose()
                    
                    # Get text from paragraphs, divs, list items, etc.
                    for tag in soup.find_all(['p', 'div', 'li', 'td', 'th', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                        text = tag.get_text(strip=True)
                        if text and len(text) > 15:
                            chunks.append(text)
            except ImportError:
                st.warning("HTML support requires beautifulsoup4: pip install beautifulsoup4")
        
        # === RTF ===
        elif filename_lower.endswith('.rtf'):
            try:
                from striprtf.striprtf import rtf_to_text
                text = rtf_to_text(content.decode('latin-1', errors='ignore'))
                if text:
                    text = text.replace('\r\n', '\n').replace('\r', '\n')
                    lines = [line.strip() for line in text.split('\n') if line.strip()]
                    for line in lines:
                        if len(line) > 15:
                            chunks.append(line)
            except ImportError:
                st.warning("RTF support requires striprtf: pip install striprtf")
        
        # === Excel (XLSX, XLS) ===
        elif filename_lower.endswith(('.xlsx', '.xls')):
            import io
            try:
                df = pd.read_excel(io.BytesIO(content), header=None)
                # Iterate through all cells
                for col in df.columns:
                    for value in df[col]:
                        if pd.notna(value):
                            text = str(value).strip()
                            if text and len(text) > 15:
                                # Skip if it's just a number
                                try:
                                    float(text.replace(',', '.'))
                                    continue
                                except:
                                    chunks.append(text)
            except Exception as e:
                st.warning(f"Excel parsing error: {e}")
                    
    except Exception as e:
        st.warning(f"Error parsing reference file: {e}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_chunks = []
    for chunk in chunks:
        if chunk not in seen:
            seen.add(chunk)
            unique_chunks.append(chunk)
    
    return unique_chunks


def get_reference_samples(chunks: list, batch_num: int, samples_per_batch: int = 5, max_chars: int = 1500) -> str:
    """
    Get reference samples for a batch using rotating selection.
    
    Args:
        chunks: List of reference text chunks
        batch_num: Current batch number (for rotation)
        samples_per_batch: How many samples to include
        max_chars: Maximum total characters for all samples
        
    Returns:
        Formatted string of reference samples
    """
    if not chunks:
        return ""
    
    # Rotating selection - different chunks for each batch
    total_chunks = len(chunks)
    start_idx = (batch_num * samples_per_batch) % total_chunks
    
    selected = []
    total_len = 0
    
    for i in range(samples_per_batch):
        idx = (start_idx + i) % total_chunks
        chunk = chunks[idx]
        
        # Truncate long chunks
        if len(chunk) > 300:
            chunk = chunk[:300] + "..."
        
        if total_len + len(chunk) > max_chars:
            break
            
        selected.append(chunk)
        total_len += len(chunk)
    
    if not selected:
        return ""
    
    return "\n".join(f"• {s}" for s in selected)


def parse_dnt_file(content: bytes, filename: str) -> list:
    """
    Parse Do Not Translate / Forbidden Terms file.
    Supports TXT and CSV formats.
    
    Returns list of terms that should not be translated.
    """
    terms = []
    filename_lower = filename.lower()
    
    try:
        if filename_lower.endswith('.txt'):
            text = None
            for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                try:
                    text = content.decode(encoding)
                    break
                except:
                    continue
            
            if text:
                for line in text.split('\n'):
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith('#'):
                        terms.append(line)
        
        elif filename_lower.endswith('.csv'):
            text = None
            for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                try:
                    text = content.decode(encoding)
                    break
                except:
                    continue
            
            if text:
                for line in text.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Take first column
                        parts = line.split(',')
                        term = parts[0].strip().strip('"').strip("'")
                        # Skip header-like entries
                        if term.lower() not in ['term', 'forbidden', 'dnt', 'do not translate', 'source']:
                            if term:
                                terms.append(term)
    
    except Exception as e:
        st.warning(f"Error parsing DNT file: {e}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_terms = []
    for term in terms:
        if term not in seen:
            seen.add(term)
            unique_terms.append(term)
    
    return unique_terms


def apply_tm_to_segment(source_with_tags: str, tm_translation: str) -> str:
    """Apply TM translation while preserving source tags."""
    import re
    
    source_tags = re.findall(r'\{\{\d+\}\}', source_with_tags)
    
    if not source_tags:
        return tm_translation
    
    tm_tags = re.findall(r'\{\{\d+\}\}', tm_translation)
    
    if set(source_tags) == set(tm_tags):
        return tm_translation
    
    result = tm_translation
    
    leading_match = re.match(r'^(\{\{\d+\}\})+', source_with_tags)
    if leading_match:
        leading_tags = leading_match.group()
        if not result.startswith(leading_tags):
            result = leading_tags + result
    
    trailing_match = re.search(r'(\{\{\d+\}\})+$', source_with_tags)
    if trailing_match:
        trailing_tags = trailing_match.group()
        if not result.endswith(trailing_tags):
            result = result + trailing_tags
    
    return result


def get_chat_history_context(history: list, max_items: int) -> list:
    """Get recent translation history for context."""
    if not history or max_items <= 0:
        return []
    return history[-max_items:]


# --- Main Translation Logic ---

def process_translation(xliff_bytes, tmx_bytes, csv_bytes, custom_prompt_content=None, memoq_tm_guids=None, memoq_tb_guids=None):
    if not api_key:
        st.error("Please provide an API Key.")
        return

    with st.status("Processing...", expanded=True) as status:
        
        # 1. Parse XLIFF
        st.write("📄 Parsing XLIFF...")
        segments = XMLParser.parse_xliff(xliff_bytes)
        st.write(f"✅ Loaded {len(segments)} segments")
        
        st.session_state.segment_objects = {seg.id: seg for seg in segments}
        st.session_state.chat_history = []
        
        # Initialize Logger
        logger = TransactionLogger()
        logger.info(f"Started translation job for {len(segments)} segments.")
        logger.info(f"Source: {src_code} | Target: {tgt_code} | Model: {model}")
        logger.info(f"TM Acceptance: ≥{acceptance_threshold}% | TM Match: ≥{match_threshold}%")
        logger.info(f"Chat History Length: {chat_history_length}")
        
        if st.session_state.reference_chunks:
            logger.info(f"Reference file: {len(st.session_state.reference_chunks)} style samples loaded")
        
        if st.session_state.use_generated_prompt:
            logger.info("Using generated prompt from Prompt Builder")
        
        # 2. Initialize TM Matcher
        tm_matcher = None
        if tmx_bytes:
            st.write("🔄 Loading Translation Memory...")
            load_start = time.time()
            
            tm_matcher = TMatcher(
                tmx_bytes, 
                src_code, 
                tgt_code, 
                acceptance_threshold=acceptance_threshold
            )
            
            load_time = time.time() - load_start
            
            if load_time < 2:
                st.write(f"✅ TM Ready: {tm_matcher.tu_count:,} TUs (cached, {load_time:.1f}s)")
            else:
                st.write(f"✅ TM Indexed: {tm_matcher.tu_count:,} TUs ({load_time:.1f}s)")
            
            st.session_state.tm_info = {
                'tu_count': tm_matcher.tu_count,
                'load_time': load_time,
                'file_hash': tm_matcher.file_hash
            }
            logger.info(f"TM loaded: {tm_matcher.tu_count} TUs in {load_time:.2f}s")
        
        # 3. Initialize TB Matcher
        tb_matcher = None
        if csv_bytes:
            st.write("🔄 Loading Termbase...")
            tb_matcher = TBMatcher(csv_bytes)
            st.write(f"✅ Termbase Ready: {tb_matcher.term_count:,} terms")
            logger.info(f"Termbase loaded: {tb_matcher.term_count} terms (columns: {tb_matcher.src_col} → {tb_matcher.tgt_col})")
        
        # 3.5 Initialize memoQ Server client if TMs/TBs selected
        memoq_client = None
        if memoq_tm_guids or memoq_tb_guids:
            try:
                if st.session_state.get('memoq_client'):
                    memoq_client = st.session_state.memoq_client
                    st.write(f"🔗 Using memoQ Server TM/TB resources")
                    if memoq_tm_guids:
                        st.write(f"   • {len(memoq_tm_guids)} Translation Memory(ies)")
                    if memoq_tb_guids:
                        st.write(f"   • {len(memoq_tb_guids)} Termbase(s)")
                    logger.info(f"memoQ Server: {len(memoq_tm_guids)} TMs, {len(memoq_tb_guids)} TBs")
            except Exception as e:
                st.warning(f"Could not connect to memoQ Server: {str(e)}")
                logger.info(f"memoQ connection error: {e}")
        
        # 4. Initialize Prompt Builder
        # Priority: Generated prompt > Custom file > Default
        if st.session_state.use_generated_prompt and st.session_state.generated_prompt:
            prompt_builder = PromptBuilder(custom_template=st.session_state.generated_prompt)
            logger.info("Using generated prompt template from Prompt Builder.")
        elif custom_prompt_content:
            prompt_builder = PromptBuilder(custom_template=custom_prompt_content)
            logger.info("Using custom prompt template from file.")
        else:
            prompt_builder = PromptBuilder(template_path=config.PROMPT_TEMPLATE_PATH)
            logger.info("Using default prompt template.")
        
        translator = AITranslator("OpenAI", api_key, model)
        
        status.update(label="Analyzing segments...", state="running")
        
        # 5. Analyze segments
        bypass_segments = []
        llm_segments = []
        final_translations = {}
        tm_context = {}
        tb_context = {}
        
        st.write("🔍 Analyzing TM matches...")
        analysis_progress = st.progress(0)
        
        for i, seg in enumerate(segments):
            if tm_matcher:
                should_bypass, tm_translation, match_score = tm_matcher.should_bypass_llm(
                    seg.source, 
                    match_threshold=match_threshold
                )
                
                if should_bypass and tm_translation:
                    bypass_segments.append(seg)
                    final_translations[seg.id] = apply_tm_to_segment(seg.source, tm_translation)
                    logger.info(f"[{seg.id}] BYPASS ({match_score:.0f}% TM match)")
                else:
                    llm_segments.append(seg)
                    matches, _ = tm_matcher.extract_matches(seg.source, threshold=match_threshold)
                    if matches:
                        tm_context[seg.id] = matches
            # Check memoQ TMs if available
            elif memoq_client and memoq_tm_guids:
                try:
                    for tm_guid in memoq_tm_guids:
                        results = memoq_client.lookup_segments(tm_guid, [seg.source])
                        logger.info(f"memoQ lookup for {seg.id}: {results}")
                        
                        if results and isinstance(results, dict):
                            # Parse memoQ response structure
                            result_list = results.get('Result', [])
                            
                            if result_list and len(result_list) > 0:
                                tm_hits = result_list[0].get('TMHits', [])
                                
                                if tm_hits:
                                    # Get the first match
                                    hit = tm_hits[0]
                                    match_score = hit.get('MatchRate', 0)  # MatchRate not MatchScore
                                    trans_unit = hit.get('TransUnit', {})
                                    
                                    # Extract target from <seg> tags
                                    target_segment = trans_unit.get('TargetSegment', '')
                                    if target_segment:
                                        # Remove <seg></seg> tags
                                        target_text = target_segment.replace('<seg>', '').replace('</seg>', '')
                                    else:
                                        target_text = seg.source
                                    
                                    logger.info(f"[{seg.id}] memoQ match score: {match_score}%")
                                    
                                    if match_score >= acceptance_threshold:
                                        bypass_segments.append(seg)
                                        final_translations[seg.id] = target_text
                                        logger.info(f"[{seg.id}] BYPASS ({match_score}% memoQ TM match)")
                                        break
                                    elif match_score >= match_threshold:
                                        llm_segments.append(seg)
                                        tm_context[seg.id] = [{'MatchRate': match_score, 'TargetSegment': target_text}]
                                        logger.info(f"[{seg.id}] CONTEXT ({match_score}% memoQ fuzzy match)")
                                        break
                                else:
                                    llm_segments.append(seg)
                            else:
                                llm_segments.append(seg)
                        else:
                            llm_segments.append(seg)
                except Exception as e:
                    logger.info(f"memoQ TM lookup error for {seg.id}: {str(e)}")
                    llm_segments.append(seg)
            else:
                llm_segments.append(seg)
            
            if tb_matcher and seg in llm_segments:
                tb_matches = tb_matcher.extract_matches(seg.source)
                if tb_matches:
                    tb_context[seg.id] = tb_matches
            # Check memoQ TBs if available
            elif memoq_client and memoq_tb_guids and seg in llm_segments:
                try:
                    for tb_guid in memoq_tb_guids:
                        tb_results = memoq_client.lookup_terms(tb_guid, [seg.source])
                        if tb_results:
                            tb_context[seg.id] = tb_results
                            break
                except Exception as e:
                    logger.info(f"memoQ TB lookup error for {seg.id}: {e}")
            
            analysis_progress.progress((i + 1) / len(segments))
        
        st.session_state.bypass_stats = {
            'bypassed': len(bypass_segments),
            'llm_sent': len(llm_segments)
        }
        
        st.write(f"✅ **{len(bypass_segments)}** segments from TM (≥{acceptance_threshold}% match)")
        st.write(f"🔄 **{len(llm_segments)}** segments need LLM translation")
        
        logger.info(f"Analysis complete: {len(bypass_segments)} bypass, {len(llm_segments)} LLM")
        logger.log_tm_matches(tm_context)
        logger.log_tb_matches(tb_context)
        
        # 6. Process LLM segments
        if llm_segments:
            status.update(label=f"Translating {len(llm_segments)} segments...", state="running")
            
            llm_progress = st.progress(0)
            batch_translations_history = []
            
            for i in range(0, len(llm_segments), config.BATCH_SIZE):
                batch = llm_segments[i:i + config.BATCH_SIZE]
                batch_num = (i // config.BATCH_SIZE) + 1
                total_batches = (len(llm_segments) + config.BATCH_SIZE - 1) // config.BATCH_SIZE
                
                st.write(f"📤 Batch {batch_num}/{total_batches} ({len(batch)} segments)")
                
                logger.log_batch_start(batch_num, batch)
                
                batch_tm = {seg.id: tm_context.get(seg.id, []) for seg in batch}
                batch_tb = {seg.id: tb_context.get(seg.id, []) for seg in batch}
                
                history_context = get_chat_history_context(
                    batch_translations_history, 
                    chat_history_length * config.BATCH_SIZE
                )
                
                if history_context:
                    logger.info(f"Chat history: {len(history_context)} previous translations included")
                
                # Get reference samples for this batch
                reference_samples = ""
                
                # Use semantic matching if embedding matcher is ready
                if st.session_state.reference_embeddings_ready and st.session_state.embedding_matcher:
                    try:
                        # Get source texts from batch
                        source_texts = [seg.source for seg in batch]
                        
                        # Find semantically similar references for all segments in batch
                        matcher = st.session_state.embedding_matcher
                        matches_dict = matcher.find_similar_batch(
                            source_texts,
                            top_k=3,
                            min_similarity=0.35
                        )
                        
                        # Collect unique references
                        all_matches = []
                        seen_indices = set()
                        for seg_matches in matches_dict.values():
                            for m in seg_matches:
                                if m.index not in seen_indices:
                                    all_matches.append(m)
                                    seen_indices.add(m.index)
                        
                        # Sort by similarity and format
                        all_matches.sort(key=lambda x: x.similarity, reverse=True)
                        reference_samples = matcher.format_reference_context(all_matches[:8], max_chars=2000)
                        
                        if reference_samples:
                            logger.info(f"Semantic reference: {len(all_matches)} matches, {len(reference_samples)} chars")
                            
                    except Exception as e:
                        logger.info(f"Semantic reference error: {e}")
                        # Fallback to simple sampling
                        if st.session_state.reference_chunks:
                            reference_samples = get_reference_samples(
                                st.session_state.reference_chunks,
                                batch_num,
                                samples_per_batch=5,
                                max_chars=1500
                            )
                
                # Fallback: simple rotating samples (no embeddings)
                elif st.session_state.reference_chunks:
                    reference_samples = get_reference_samples(
                        st.session_state.reference_chunks,
                        batch_num,
                        samples_per_batch=5,
                        max_chars=1500
                    )
                    if reference_samples:
                        logger.info(f"Reference (rotating): {len(reference_samples)} chars of style samples")
                
                # Get DNT terms
                dnt_terms = st.session_state.dnt_terms if st.session_state.dnt_terms else None
                if dnt_terms:
                    logger.info(f"DNT list: {len(dnt_terms)} forbidden terms")
                
                prompt = prompt_builder.build_prompt(
                    config.SUPPORTED_LANGUAGES[src_code],
                    config.SUPPORTED_LANGUAGES[tgt_code],
                    batch, 
                    batch_tm, 
                    batch_tb,
                    chat_history=history_context,
                    reference_context=reference_samples,
                    dnt_terms=dnt_terms
                )
                
                try:
                    response_text, tokens = translator.translate_batch(prompt)
                    logger.log_llm_interaction(prompt, response_text)
                    
                    lines = response_text.strip().split('\n')
                    batch_results = []
                    
                    for line in lines:
                        if line.startswith('[') and ']' in line:
                            try:
                                seg_id = line[line.find('[')+1:line.find(']')]
                                trans_text = line[line.find(']')+1:].strip()
                                final_translations[seg_id] = trans_text
                                
                                seg_obj = st.session_state.segment_objects.get(seg_id)
                                if seg_obj:
                                    batch_results.append({
                                        'source': seg_obj.source,
                                        'target': trans_text
                                    })
                            except:
                                pass
                    
                    batch_translations_history.extend(batch_results)
                    
                except Exception as e:
                    err_msg = f"Batch {batch_num} failed: {e}"
                    st.error(err_msg)
                    logger.info(f"ERROR: {err_msg}")
                
                llm_progress.progress((i + len(batch)) / len(llm_segments))
        
        # 7. Save results
        st.session_state.translation_results = final_translations
        st.session_state.translation_log = logger.get_content()
        st.session_state.chat_history = batch_translations_history if llm_segments else []
        
        status.update(label="✅ Translation Complete!", state="complete")
        
        st.success(f"""
        **Translation Complete!**
        - {len(bypass_segments)} segments from TM (no API cost)
        - {len(llm_segments)} segments via LLM
        - {len(final_translations)} total translations
        """)


# --- UI Layout ---

st.title("🚀 Enhanced Translation Assistant")
st.markdown("AI-powered translation with TM, Termbase & Smart Prompt Builder")

tab1, tab2, tab3 = st.tabs(["📂 Workspace", "📊 Results", "✨ Prompt Builder"])

# === TAB 1: WORKSPACE ===
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        xliff_file = st.file_uploader(
            "📄 Upload Document (XLIFF)", 
            type=['xlf', 'xliff', 'mqxliff'],
            help="MemoQ XLIFF, Standard XLIFF"
        )
        
        if xliff_file:
            xliff_file.seek(0)
            detected_src, detected_tgt = XMLParser.detect_languages(xliff_file.getvalue())
            if detected_src and detected_tgt:
                st.session_state.detected_languages = {
                    'source': detected_src,
                    'target': detected_tgt
                }
                st.caption(f"🔍 Detected: {detected_src} → {detected_tgt}")
        
        # ==================== memoQ SERVER RESOURCES ====================
        st.markdown("---")
        st.markdown("##### 🔗 memoQ Server Resources")
        
        if st.session_state.memoq_connected and st.session_state.memoq_client:
            # Load TM/TB data
            selected_tms, selected_tbs = MemoQUI.show_memoq_data_loader(
                client=st.session_state.memoq_client,
                src_lang=src_code,
                tgt_lang=tgt_code
            )
            
            # Store selections
            st.session_state.selected_tm_guids = selected_tms
            st.session_state.selected_tb_guids = selected_tbs
            
            # Show status
            if selected_tms or selected_tbs:
                st.info(
                    f"✓ Using {len(selected_tms)} TM(s) and {len(selected_tbs)} TB(s) from memoQ Server"
                )
        else:
            st.warning("🔗 Not connected to memoQ Server. Configure connection in sidebar.")
        
        st.markdown("---")
        
        # ANALYSIS SCREEN - Show if file uploaded and ready
        if xliff_file and len(st.session_state.selected_tm_guids) > 0:
            if not st.session_state.get('ready_to_translate', False):
                show_analysis_screen(xliff_file, len(st.session_state.selected_tm_guids))
                st.markdown("---")
        
        # Reference file for style/tone with semantic matching
        st.markdown("---")
        st.markdown("##### 📑 Semantic Reference (Optional)")
        
        reference_file = st.file_uploader(
            "Reference File (Target Language Only)",
            type=['txt', 'docx', 'pdf', 'html', 'htm', 'rtf', 'xlsx', 'xls'],
            help="Previously translated text for style/terminology reference. Supports TXT, DOCX, PDF, HTML, RTF, Excel."
        )
        
        if reference_file:
            reference_file.seek(0)
            chunks = parse_reference_file(reference_file.getvalue(), reference_file.name)
            st.session_state.reference_chunks = chunks
            
            if chunks:
                # Show cost estimate
                cost_info = get_embedding_cost_estimate(len(chunks), 100)  # Estimate for 100 segments
                
                col_ref1, col_ref2 = st.columns(2)
                with col_ref1:
                    st.metric("Reference Samples", len(chunks))
                with col_ref2:
                    st.metric("Est. Embedding Cost", cost_info['total_cost_formatted'])
                
                # Button to create embeddings
                if not st.session_state.reference_embeddings_ready:
                    if api_key:
                        if st.button("🧠 Create Semantic Index", type="secondary", use_container_width=True):
                            with st.spinner("Creating embeddings... This may take a minute."):
                                try:
                                    matcher = EmbeddingMatcher(api_key)
                                    
                                    # Progress callback
                                    progress_bar = st.progress(0)
                                    def update_progress(current, total):
                                        progress_bar.progress(current / total if total > 0 else 0)
                                    
                                    count, was_cached = matcher.load_reference(chunks, update_progress)
                                    
                                    st.session_state.embedding_matcher = matcher
                                    st.session_state.reference_embeddings_ready = True
                                    
                                    if was_cached:
                                        st.success(f"✅ Loaded {count} cached embeddings")
                                    else:
                                        st.success(f"✅ Created {count} embeddings")
                                    st.rerun()
                                    
                                except Exception as e:
                                    st.error(f"Embedding error: {e}")
                    else:
                        st.warning("⚠️ API Key required for semantic matching")
                else:
                    st.success("✅ Semantic index ready")
                    if st.button("🔄 Reset Index"):
                        st.session_state.reference_embeddings_ready = False
                        st.session_state.embedding_matcher = None
                        st.rerun()
                
                with st.expander("Preview reference samples"):
                    for i, chunk in enumerate(chunks[:5]):
                        st.caption(f"{i+1}. {chunk[:100]}..." if len(chunk) > 100 else f"{i+1}. {chunk}")
                    if len(chunks) > 5:
                        st.caption(f"... and {len(chunks) - 5} more")
        
        st.markdown("---")
        
        # DNT (Do Not Translate) List
        dnt_file = st.file_uploader(
            "🚫 Do Not Translate List (TXT/CSV)",
            type=['txt', 'csv'],
            help="Terms that should remain in source language (brand names, product codes, etc.)"
        )
        
        if dnt_file:
            dnt_file.seek(0)
            terms = parse_dnt_file(dnt_file.getvalue(), dnt_file.name)
            st.session_state.dnt_terms = terms
            if terms:
                st.success(f"🚫 **{len(terms)}** forbidden terms loaded")
                with st.expander("Preview DNT terms"):
                    # Show first 20 terms
                    for term in terms[:20]:
                        st.caption(f"• {term}")
                    if len(terms) > 20:
                        st.caption(f"... and {len(terms) - 20} more")
        
        prompt_file = st.file_uploader(
            "📝 Custom Prompt Template (TXT)", 
            type=['txt'],
            help="Optional: Upload your own prompt template",
            disabled=st.session_state.use_generated_prompt
        )
        
        if st.session_state.use_generated_prompt:
            st.info("✨ Using prompt from Prompt Builder tab")
        
    with col2:
        st.info("""
        **How it works:**
        1. Segments ≥ Acceptance threshold → Direct TM
        2. Segments ≥ Match threshold → LLM with TM context
        3. Chat history provides consistency across batches
        4. Reference file provides style/tone guidance
        
        **Prompt Priority:**
        1. Generated prompt (from Prompt Builder)
        2. Custom file upload
        3. Default template
        """)
        
        if st.button("🚀 Start Translation", type="primary", use_container_width=True, disabled=not st.session_state.get('ready_to_translate', False)):
            if st.session_state.get('ready_to_translate', False):
                if xliff_file:
                    xliff_file.seek(0)
                    
                    custom_prompt = None
                    if prompt_file and not st.session_state.use_generated_prompt:
                        prompt_file.seek(0)
                        custom_prompt = prompt_file.read().decode('utf-8')
                    
                    process_translation(
                        xliff_file.getvalue(),
                        tmx_bytes=None,
                        csv_bytes=None,
                        custom_prompt_content=custom_prompt,
                        memoq_tm_guids=st.session_state.selected_tm_guids,
                        memoq_tb_guids=st.session_state.selected_tb_guids
                    )
                else:
                    st.error("XLIFF file is required.")
            else:
                st.error("⬆️ Complete analysis first")

# === TAB 2: RESULTS ===
with tab2:
    if st.session_state.translation_results:
        st.subheader("Translation Output")
        
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Total Segments", len(st.session_state.translation_results))
        with col_stat2:
            bypassed = st.session_state.bypass_stats.get('bypassed', 0)
            st.metric(f"From TM (≥{acceptance_threshold}%)", bypassed)
        with col_stat3:
            st.metric("Via LLM", st.session_state.bypass_stats.get('llm_sent', 0))
        
        st.divider()
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            if xliff_file:
                xliff_file.seek(0)
                final_xml = XMLParser.update_xliff(
                    xliff_file.getvalue(),
                    st.session_state.translation_results,
                    st.session_state.get('segment_objects', {})
                )
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = xliff_file.name.rsplit('.', 1)[0]
                extension = xliff_file.name.rsplit('.', 1)[1] if '.' in xliff_file.name else 'xliff'
                output_filename = f"{base_name}_translated_{timestamp}.{extension}"
                
                st.download_button(
                    "⬇️ Download Translated File",
                    final_xml,
                    file_name=output_filename,
                    mime="application/xml"
                )
                
        with col_res2:
            if st.session_state.translation_log:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    "📜 Download Detailed Log",
                    st.session_state.translation_log,
                    file_name=f"translation_log_{timestamp}.txt",
                    mime="text/plain"
                )
        
        st.divider()
        st.subheader("Preview")
        
        preview_data = []
        for seg_id, trans in st.session_state.translation_results.items():
            seg_obj = st.session_state.segment_objects.get(seg_id)
            source = seg_obj.source if seg_obj else "N/A"
            preview_data.append({
                'ID': seg_id,
                'Source': source[:50] + '...' if len(source) > 50 else source,
                'Translation': trans[:50] + '...' if len(trans) > 50 else trans
            })
        
        df = pd.DataFrame(preview_data)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No results yet. Run translation in Workspace tab.")

# === TAB 3: PROMPT BUILDER ===
with tab3:
    st.subheader("✨ Smart Prompt Builder")
    st.markdown("Generate optimized prompts from Analysis Reports and Style Guides")
    
    col_pb1, col_pb2 = st.columns([1, 1])
    
    with col_pb1:
        st.markdown("#### 📄 Upload Documents")
        
        analysis_file = st.file_uploader(
            "📊 Analysis Report (DOCX)",
            type=['docx'],
            help="AICONTEXT analysis report",
            key="analysis_docx"
        )
        
        style_file = st.file_uploader(
            "📋 Style Guide (DOCX)",
            type=['docx'],
            help="Translation style guide",
            key="style_docx"
        )
        
        dnt_file = st.file_uploader(
            "🚫 Do Not Translate / Forbidden Terms (TXT/CSV)",
            type=['txt', 'csv'],
            help="List of terms to avoid in translation. One term per line or CSV format.",
            key="dnt_file"
        )
        
        # Parse DNT file
        forbidden_terms = []
        if dnt_file:
            dnt_file.seek(0)
            dnt_content = dnt_file.getvalue().decode('utf-8', errors='ignore')
            
            if dnt_file.name.endswith('.csv'):
                # Parse CSV - take first column
                for line in dnt_content.strip().split('\n'):
                    if line.strip():
                        # Handle comma-separated
                        parts = line.split(',')
                        term = parts[0].strip().strip('"').strip("'")
                        if term and not term.lower().startswith(('term', 'forbidden', 'dnt', '#')):
                            forbidden_terms.append(term)
            else:
                # Parse TXT - one term per line
                for line in dnt_content.strip().split('\n'):
                    term = line.strip()
                    if term and not term.startswith('#'):
                        forbidden_terms.append(term)
            
            st.success(f"🚫 **{len(forbidden_terms)}** forbidden terms loaded")
            with st.expander("Preview forbidden terms"):
                st.write(", ".join(forbidden_terms[:20]) + ("..." if len(forbidden_terms) > 20 else ""))
        
        # Analyze uploaded files
        analysis_result = None
        style_result = None
        
        if analysis_file:
            analysis_file.seek(0)
            analysis_result = DocumentAnalyzer.analyze_file(
                analysis_file.getvalue(), 
                analysis_file.name
            )
            
            with st.expander("📊 Analysis Report Extracted Data", expanded=True):
                if analysis_result.domain:
                    st.success(f"**Domain:** {analysis_result.domain}")
                if analysis_result.domain_composition:
                    st.write("**Domain Composition:**")
                    for comp in analysis_result.domain_composition:
                        st.write(f"  • {comp}")
                if analysis_result.terminology_categories:
                    st.write(f"**Terminology:** {len(analysis_result.terminology_categories)} categories")
                if analysis_result.critical_numbers:
                    st.write(f"**Critical Numbers:** {len(analysis_result.critical_numbers)} items")
                    
        if style_file:
            style_file.seek(0)
            style_result = DocumentAnalyzer.analyze_file(
                style_file.getvalue(),
                style_file.name
            )
            
            with st.expander("📋 Style Guide Extracted Data", expanded=True):
                if style_result.style_rules:
                    st.write(f"**Style Rules:** {len(style_result.style_rules)} rules")
                if style_result.formatting_rules:
                    st.write(f"**Formatting Rules:** {len(style_result.formatting_rules)} rules")
                if style_result.gender_inclusivity:
                    st.write(f"**Gender/Inclusivity:** {len(style_result.gender_inclusivity)} rules")
                if style_result.do_not_translate:
                    st.write(f"**DNT Items:** {len(style_result.do_not_translate)} items")
        
        st.divider()
        
        # Generate button
        if st.button("🔮 Generate Prompt", type="primary", use_container_width=True, 
                     disabled=(not analysis_file and not style_file and not dnt_file)):
            
            prompt, metadata = PromptGenerator.generate(
                analysis=analysis_result,
                style_guide=style_result,
                source_lang=config.SUPPORTED_LANGUAGES[src_code],
                target_lang=config.SUPPORTED_LANGUAGES[tgt_code],
                forbidden_terms=forbidden_terms
            )
            
            st.session_state.generated_prompt = prompt
            st.session_state.prompt_metadata = metadata
            st.success("✅ Prompt generated!")
    
    with col_pb2:
        st.markdown("#### 📝 Generated Prompt")
        
        if st.session_state.generated_prompt:
            # Show metadata
            meta = st.session_state.prompt_metadata
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Style Rules", meta.get('style_rules_count', 0))
            with col_m2:
                st.metric("Term Categories", meta.get('terminology_categories', 0))
            with col_m3:
                st.metric("Format Rules", meta.get('formatting_rules_count', 0))
            with col_m4:
                st.metric("🚫 Forbidden", meta.get('forbidden_terms_count', 0))
            
            # Editable prompt
            edited_prompt = st.text_area(
                "Edit prompt (optional):",
                value=st.session_state.generated_prompt,
                height=400,
                key="prompt_editor"
            )
            
            # Update if edited
            if edited_prompt != st.session_state.generated_prompt:
                st.session_state.generated_prompt = edited_prompt
            
            st.divider()
            
            # Action buttons
            col_act1, col_act2, col_act3 = st.columns(3)
            
            with col_act1:
                if st.button("✅ Use This Prompt", type="primary", use_container_width=True):
                    st.session_state.use_generated_prompt = True
                    st.success("Prompt activated! Go to Workspace tab to start translation.")
                    
            with col_act2:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    "⬇️ Download",
                    st.session_state.generated_prompt,
                    file_name=f"cat_tool_prompt_{timestamp}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col_act3:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.session_state.generated_prompt = None
                    st.session_state.prompt_metadata = {}
                    st.session_state.use_generated_prompt = False
                    st.rerun()
        else:
            st.info("""
            **How to use:**
            1. Upload Analysis Report (AICONTEXT output) and/or Style Guide
            2. Click "Generate Prompt"
            3. Review and edit if needed
            4. Click "Use This Prompt" to activate
            5. Go to Workspace tab and start translation
            
            **Extracted elements:**
            - Domain & context from Analysis Report
            - Technical protocols (decimal, units, etc.)
            - Style rules from Style Guide
            - Formatting rules
            - Terminology categories
            """)
