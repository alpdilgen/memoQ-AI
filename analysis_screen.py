"""
ANALYSIS SCREEN - Pre-translation workflow
Allows user to analyze file and see cost estimate before translating

Workflow:
  1. Upload source file
  2. Upload TM file (if different from default)
  3. Click "Analyze"
  4. See analysis results in table
  5. See cost estimate
  6. Click "Translate" to proceed
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import json


def _get_processing_type(level):
    """Get processing type for match level"""
    if level == '100%':
        return "⚡ Bypass"
    elif level in ['95%-99%', '85%-94%', '75%-84%', '50%-74%']:
        return "🤖 Context"
    else:
        return "✍️ LLM"


def calculate_cost_estimate(analysis_results, model="gpt-4o"):
    """
    Calculate estimated cost based on analysis
    
    Args:
        analysis_results: Results from matcher.analyze_file()
        model: LLM model (default: gpt-4o)
    
    Returns:
        {
            'bypass_cost': 0,
            'context_cost': float,
            'llm_cost': float,
            'total_cost': float,
            'breakdown': {
                'bypass': int,
                'context': int,
                'llm_only': int
            }
        }
    """
    # Average tokens per segment based on translation log
    tokens_per_segment = 100  # Conservative estimate
    
    # Pricing (gpt-4o current rates)
    if model == "gpt-4o":
        input_price = 0.00025  # $0.25/1M tokens
        output_price = 0.001   # $1/1M tokens
    elif model == "gpt-4-turbo":
        input_price = 0.00001  # $0.01/1M tokens
        output_price = 0.00003 # $0.03/1M tokens
    else:
        input_price = 0.00025
        output_price = 0.001
    
    # Cost per segment for LLM translation
    cost_per_segment = (tokens_per_segment * input_price) + (tokens_per_segment * output_price)
    
    # Calculate breakdown
    bypass_segs = analysis_results['by_level'].get('100%', {}).get('segments', 0)
    context_segs = sum(
        analysis_results['by_level'].get(level, {}).get('segments', 0)
        for level in ['95%-99%', '85%-94%', '75%-84%', '50%-74%']
    )
    llm_segs = analysis_results['total_segments'] - bypass_segs - context_segs
    
    # Calculate costs
    bypass_cost = 0  # No LLM cost
    context_cost = context_segs * cost_per_segment * 0.5  # 50% discount (using TM context)
    llm_cost = llm_segs * cost_per_segment
    total_cost = bypass_cost + context_cost + llm_cost
    
    return {
        'bypass_cost': 0.0,
        'context_cost': context_cost,
        'llm_cost': llm_cost,
        'total_cost': total_cost,
        'breakdown': {
            'bypass': bypass_segs,
            'context': context_segs,
            'llm_only': llm_segs
        },
        'tokens': {
            'bypass': 0,
            'context': context_segs * tokens_per_segment,
            'llm_only': llm_segs * tokens_per_segment
        }
    }


def render_analysis_screen():
    """Render the pre-translation analysis screen"""
    
    st.markdown("## 📋 File Analysis & Cost Estimate")
    st.markdown("---")
    
    # Initialize session state for analysis
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'cost_estimate' not in st.session_state:
        st.session_state.cost_estimate = None
    if 'analyzed_file' not in st.session_state:
        st.session_state.analyzed_file = None
    
    # Step 1: File Upload
    st.markdown("### Step 1️⃣: Upload Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        source_file = st.file_uploader(
            "📄 Upload source file (SDLXLIFF, XLIFF, MQXLIFF)",
            type=['sdlxliff', 'xliff', 'xlf', 'mqxliff'],
            key="analysis_source_file"
        )
    
    with col2:
        tm_file = st.file_uploader(
            "💾 Upload Translation Memory (TMX) - Optional",
            type=['tmx'],
            key="analysis_tm_file",
            help="If not provided, will use configured default TM"
        )
    
    # Step 2: Analyze Button
    st.markdown("### Step 2️⃣: Analyze File")
    
    analyze_button = st.button(
        "🔍 Analyze File",
        type="primary",
        use_container_width=True,
        disabled=source_file is None
    )
    
    if analyze_button and source_file is not None:
        with st.spinner("📊 Analyzing file..."):
            try:
                from tm_matcher_single import TMatcher
                import tempfile
                import os
                
                # Save uploaded files temporarily
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Save source file
                    source_path = os.path.join(tmpdir, source_file.name)
                    with open(source_path, 'wb') as f:
                        f.write(source_file.getbuffer())
                    
                    # Load TM (use uploaded or default)
                    if tm_file is not None:
                        tm_path = os.path.join(tmpdir, tm_file.name)
                        with open(tm_path, 'wb') as f:
                            f.write(tm_file.getbuffer())
                    else:
                        # Use default TM from config
                        tm_path = st.session_state.get('default_tm_path', 'tm.tmx')
                    
                    # Run analysis
                    matcher = TMatcher(tm_path)
                    analysis_results = matcher.analyze_file(source_path)
                    
                    # Calculate cost
                    cost_estimate = calculate_cost_estimate(analysis_results)
                    
                    # Store results
                    st.session_state.analysis_done = True
                    st.session_state.analysis_results = analysis_results
                    st.session_state.cost_estimate = cost_estimate
                    st.session_state.analyzed_file = source_file.name
                    
                    st.success("✅ Analysis complete!")
                    st.rerun()
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.session_state.analysis_done = False
    
    # Step 3: Display Results (if analysis done)
    if st.session_state.analysis_done and st.session_state.analysis_results is not None:
        st.markdown("### Step 3️⃣: Analysis Results")
        
        analysis = st.session_state.analysis_results
        cost = st.session_state.cost_estimate
        
        # Summary boxes
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📊 Total Segments",
                analysis['total_segments'],
                help="Total segments in file"
            )
        
        with col2:
            st.metric(
                "📝 Total Words",
                analysis['total_words'],
                help="Total words to translate"
            )
        
        with col3:
            tm_coverage = (cost['breakdown']['bypass'] + cost['breakdown']['context']) / analysis['total_segments'] * 100
            st.metric(
                "🎯 TM Coverage",
                f"{tm_coverage:.1f}%",
                help="Percentage of segments with TM matches"
            )
        
        with col4:
            st.metric(
                "💰 Est. Cost",
                f"${cost['total_cost']:.2f}",
                help="Estimated translation cost"
            )
        
        st.markdown("---")
        
        # Detailed breakdown table
        st.markdown("#### Match Level Distribution")
        
        breakdown_data = []
        for level in ['100%', '95%-99%', '85%-94%', '75%-84%', '50%-74%', 'No match']:
            if level in analysis['by_level']:
                data = analysis['by_level'][level]
                pct = (data['words'] / analysis['total_words'] * 100) if analysis['total_words'] > 0 else 0
                breakdown_data.append({
                    'Match Level': level,
                    'Segments': data['segments'],
                    'Words': data['words'],
                    'Percentage': f"{pct:.1f}%",
                    'Processing': _get_processing_type(level)
                })
        
        df = pd.DataFrame(breakdown_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Cost breakdown
        st.markdown("#### 💰 Cost Breakdown")
        
        cost_data = [
            {
                'Category': 'Bypass (≥95% match)',
                'Segments': cost['breakdown']['bypass'],
                'Cost': f"${cost['bypass_cost']:.4f}",
                'Note': 'Use existing translation'
            },
            {
                'Category': 'Context (60-94% match)',
                'Segments': cost['breakdown']['context'],
                'Cost': f"${cost['context_cost']:.4f}",
                'Note': '50% discount with TM context'
            },
            {
                'Category': 'LLM Only (<60% match)',
                'Segments': cost['breakdown']['llm_only'],
                'Cost': f"${cost['llm_cost']:.4f}",
                'Note': 'Full cost'
            }
        ]
        
        cost_df = pd.DataFrame(cost_data)
        st.dataframe(cost_df, use_container_width=True, hide_index=True)
        
        # Total cost box
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Without TM", f"${cost['bypass_cost'] + cost['context_cost'] + cost['llm_cost']:.4f}")
        
        with col2:
            savings = (cost['context_cost'] + cost['llm_cost']) - (cost['context_cost']*0.5 + cost['llm_cost'])
            st.metric("Savings", f"${savings:.4f}")
        
        with col3:
            st.metric("Total Cost", f"${cost['total_cost']:.4f}", label_visibility="visible")
        
        st.markdown("---")
        
        # Information about processing
        with st.expander("ℹ️ Processing Details"):
            st.markdown("""
            **How segments are processed:**
            
            - **Bypass (≥95%)**: Exact or near-exact TM matches. Uses translation directly from memory. Zero AI cost.
            
            - **Context (60-94%)**: Fuzzy matches. AI translates but uses TM as reference for consistency. 50% cost.
            
            - **LLM Only (<60%)**: No meaningful TM match. Full AI translation at standard cost.
            
            **Cost Calculation:**
            - Model: GPT-4o
            - ~100 tokens per segment average
            - Input: $0.25/1M tokens
            - Output: $1/1M tokens
            - Total: ~$0.0015 per segment
            """)
        
        st.markdown("---")
        
        # Step 4: Proceed to Translation
        st.markdown("### Step 4️⃣: Ready to Translate?")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("✅ Proceed to Translation", type="primary", use_container_width=True):
                st.session_state.ready_to_translate = True
                st.session_state.pending_source_file = source_file
                st.rerun()
        
        with col2:
            if st.button("🔄 Clear & Analyze Another", use_container_width=True):
                st.session_state.analysis_done = False
                st.session_state.analysis_results = None
                st.session_state.cost_estimate = None
                st.rerun()
        
        with col3:
            st.markdown("")  # Spacer
    
    else:
        # No analysis yet - show placeholder
        if not st.session_state.analysis_done:
            st.info("👈 Upload a file and click **Analyze File** to see cost estimate and match distribution")


if __name__ == "__main__":
    render_analysis_screen()
