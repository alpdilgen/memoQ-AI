"""
XML Parser for XLIFF files
Handles parsing, updating, and memoQ metadata
"""

import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional
import re
from datetime import datetime
from dataclasses import dataclass


@dataclass
class Segment:
    """Represents an XLIFF segment"""
    id: str
    source: str
    target: str = ""


class XMLParser:
    """Parse and manipulate XLIFF files"""
    
    @staticmethod
    def parse_xliff(xliff_bytes: bytes) -> List[Segment]:
        """
        Parse XLIFF file and extract segments.
        
        Args:
            xliff_bytes: Raw bytes of XLIFF file
            
        Returns:
            List of Segment objects
        """
        segments = []
        
        try:
            root = ET.fromstring(xliff_bytes)
            
            # Define namespaces
            namespaces = {
                'xliff': 'urn:oasis:names:tc:xliff:document:1.2',
                'mq': 'MQXliff'
            }
            
            # Find all trans-units
            trans_units = root.findall('.//xliff:trans-unit', namespaces)
            
            if not trans_units:
                # Try without namespace
                trans_units = root.findall('.//trans-unit')
            
            for trans_unit in trans_units:
                seg_id = trans_unit.get('id')
                
                # Find source and target
                source_elem = trans_unit.find('xliff:source', namespaces)
                if source_elem is None:
                    source_elem = trans_unit.find('source')
                
                target_elem = trans_unit.find('xliff:target', namespaces)
                if target_elem is None:
                    target_elem = trans_unit.find('target')
                
                source_text = XMLParser._extract_text(source_elem) if source_elem is not None else ""
                target_text = XMLParser._extract_text(target_elem) if target_elem is not None else ""
                
                if seg_id and source_text:
                    segments.append(Segment(
                        id=seg_id,
                        source=source_text,
                        target=target_text
                    ))
        
        except Exception as e:
            raise Exception(f"Error parsing XLIFF: {str(e)}")
        
        return segments
    
    @staticmethod
    def _extract_text(element) -> str:
        """Extract text from element, including inline tags."""
        if element is None:
            return ""
        
        # Get direct text
        text = element.text or ""
        
        # Process children
        for child in element:
            # Handle inline tags like <bpt>, <ept>, <ph>, <it>
            if child.tag.endswith(('bpt', 'ept', 'ph', 'it')):
                # Add placeholder for tag
                tag_id = child.get('id', '')
                text += f"{{{{{tag_id}}}}}"
            
            # Add child text
            if child.text:
                text += child.text
            
            # Add tail text
            if child.tail:
                text += child.tail
        
        return text.strip()
    
    @staticmethod
    def detect_languages(xliff_bytes: bytes) -> Tuple[Optional[str], Optional[str]]:
        """
        Detect source and target languages from XLIFF.
        
        Returns:
            Tuple of (source_lang_code, target_lang_code)
        """
        try:
            root = ET.fromstring(xliff_bytes)
            
            # Try to find file element with source-language and target-language
            file_elem = root.find('.//file') or root.find('.//{urn:oasis:names:tc:xliff:document:1.2}file')
            
            if file_elem is not None:
                src_lang = file_elem.get('source-language')
                tgt_lang = file_elem.get('target-language')
                return src_lang, tgt_lang
        
        except Exception:
            pass
        
        return None, None
    
    @staticmethod
    def update_xliff(
        xliff_bytes: bytes,
        translations: Dict[str, str],
        segment_objects: Dict[str, Segment] = None,
        match_rates: Dict[str, int] = None
    ) -> bytes:
        """
        Update XLIFF with translations and match rates.
        
        Args:
            xliff_bytes: Original XLIFF bytes
            translations: Dict of segment_id -> translation_text
            segment_objects: Dict of segment_id -> Segment object (optional)
            match_rates: Dict of segment_id -> match_percentage (optional)
            
        Returns:
            Updated XLIFF as bytes
        """
        if match_rates is None:
            match_rates = {}
        
        if segment_objects is None:
            segment_objects = {}
        
        try:
            # Parse the XLIFF
            root = ET.fromstring(xliff_bytes)
            
            # Register namespaces to preserve them
            namespaces = {
                'xliff': 'urn:oasis:names:tc:xliff:document:1.2',
                'mq': 'MQXliff'
            }
            
            for prefix, uri in namespaces.items():
                ET.register_namespace(prefix, uri)
            
            # Define namespace URIs for searching
            xliff_ns = 'urn:oasis:names:tc:xliff:document:1.2'
            
            # Find all trans-units
            trans_units = root.findall(f'.//{{{xliff_ns}}}trans-unit')
            if not trans_units:
                trans_units = root.findall('.//trans-unit')
            
            for trans_unit in trans_units:
                seg_id = trans_unit.get('id')
                
                if seg_id in translations:
                    # Update target element
                    target_elem = trans_unit.find(f'{{{xliff_ns}}}target')
                    if target_elem is None:
                        target_elem = trans_unit.find('target')
                    
                    if target_elem is None:
                        # Create target element if it doesn't exist
                        source_elem = trans_unit.find(f'{{{xliff_ns}}}source')
                        if source_elem is None:
                            source_elem = trans_unit.find('source')
                        
                        if source_elem is not None:
                            source_index = list(trans_unit).index(source_elem)
                            target_elem = ET.Element('target')
                            trans_unit.insert(source_index + 1, target_elem)
                    
                    # Set the translation text
                    if target_elem is not None:
                        target_elem.text = translations[seg_id]
                        target_elem.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
            
            # Convert back to string for post-processing
            output_str = ET.tostring(root, encoding='unicode')
            
            # Now apply match rate metadata using regex (surgical approach)
            for seg_id, match_rate in match_rates.items():
                if match_rate == 100:
                    output_str = XMLParser._add_tm_match_metadata_to_segment(output_str, seg_id)
            
            # Convert to bytes
            return output_str.encode('utf-8')
        
        except Exception as e:
            raise Exception(f"Error updating XLIFF: {str(e)}")
    
    @staticmethod
    def _add_tm_match_metadata_to_segment(xml_str: str, seg_id: str) -> str:
        """
        Add memoQ TM match metadata to a specific segment (100% matches).
        
        Uses surgical regex approach to preserve XML structure.
        Modifies trans-unit opening tag to add/update memoQ attributes.
        
        Args:
            xml_str: XML string
            seg_id: Segment ID to update (must match id attribute)
            
        Returns:
            Updated XML string with mq:percent="100" and related attributes
        """
        timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Find trans-unit opening tag for this segment
        # Pattern: <trans-unit ... id="seg_id" ...>
        # Handles both quoted and unquoted id values
        pattern = rf'(<trans-unit\s+[^>]*?\sid=["\']?{re.escape(seg_id)}["\']?[^>]*?)>'
        
        def modify_opening_tag(match):
            """Modify the trans-unit opening tag to add match rate metadata"""
            opening_tag = match.group(1)
            
            # ===== 1. Change/add mq:status to ManuallyConfirmed =====
            if 'mq:status=' in opening_tag:
                opening_tag = re.sub(
                    r'mq:status="[^"]*"',
                    'mq:status="ManuallyConfirmed"',
                    opening_tag
                )
            else:
                opening_tag = opening_tag.rstrip() + ' mq:status="ManuallyConfirmed"'
            
            # ===== 2. Add/update mq:percent="100" =====
            if 'mq:percent=' in opening_tag:
                # Update existing
                opening_tag = re.sub(
                    r'mq:percent="[^"]*"',
                    'mq:percent="100"',
                    opening_tag
                )
            else:
                # Add new
                opening_tag = opening_tag.rstrip() + ' mq:percent="100"'
            
            # ===== 3. Add translator commit match rate =====
            if 'mq:translatorcommitmatchrate=' not in opening_tag:
                opening_tag = opening_tag.rstrip() + ' mq:translatorcommitmatchrate="100"'
            
            # ===== 4. Add translator commit username =====
            if 'mq:translatorcommitusername=' not in opening_tag:
                opening_tag = opening_tag.rstrip() + ' mq:translatorcommitusername="System"'
            
            # ===== 5. Add translator commit timestamp =====
            if 'mq:translatorcommittimestamp=' not in opening_tag:
                opening_tag = opening_tag.rstrip() + f' mq:translatorcommittimestamp="{timestamp}"'
            
            # ===== 6. Update last changed timestamp =====
            if 'mq:lastchangedtimestamp=' in opening_tag:
                opening_tag = re.sub(
                    r'mq:lastchangedtimestamp="[^"]*"',
                    f'mq:lastchangedtimestamp="{timestamp}"',
                    opening_tag
                )
            else:
                opening_tag = opening_tag.rstrip() + f' mq:lastchangedtimestamp="{timestamp}"'
            
            # ===== 7. Update last changing user =====
            if 'mq:lastchanginguser=' in opening_tag:
                opening_tag = re.sub(
                    r'mq:lastchanginguser="[^"]*"',
                    'mq:lastchanginguser="System"',
                    opening_tag
                )
            else:
                opening_tag = opening_tag.rstrip() + ' mq:lastchanginguser="System"'
            
            # Return modified tag with closing >
            return opening_tag + '>'
        
        # Apply regex substitution
        return re.sub(pattern, modify_opening_tag, xml_str)
