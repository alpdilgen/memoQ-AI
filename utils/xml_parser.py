import xml.etree.ElementTree as ET
from models.entities import TranslationSegment
from typing import List, Dict, Tuple, Optional
import re
import copy

class XMLParser:
    
    @staticmethod
    def detect_languages(content: bytes) -> Tuple[Optional[str], Optional[str]]:
        """
        Detect source and target languages from XLIFF file.
        
        Returns:
            Tuple of (source_lang, target_lang) or (None, None) if not found
        """
        try:
            # Try different encodings
            xml_str = None
            for enc in ['utf-8', 'utf-8-sig', 'utf-16', 'latin-1']:
                try:
                    xml_str = content.decode(enc)
                    break
                except:
                    continue
            
            if not xml_str:
                return None, None
            
            root = ET.fromstring(xml_str)
            
            # XLIFF stores languages in <file> element
            ns = {'x': 'urn:oasis:names:tc:xliff:document:1.2'}
            
            # Try with namespace
            file_elem = root.find('.//x:file', ns)
            if file_elem is None:
                # Try without namespace
                file_elem = root.find('.//file')
            
            if file_elem is None:
                # Try root if it's the file element itself
                file_elem = root
            
            source_lang = file_elem.get('source-language')
            target_lang = file_elem.get('target-language')
            
            # Keep full language codes with variants (en-gb, en-us, etc)
            # Convert to lowercase for consistency
            if source_lang:
                source_lang = source_lang.lower()
            if target_lang:
                target_lang = target_lang.lower()
            
            return source_lang, target_lang
            
        except Exception as e:
            print(f"Language detection error: {e}")
            return None, None
    
    @staticmethod
    def _extract_text_with_tags(element) -> Tuple[str, Dict[str, ET.Element]]:
        """
        Converts an XML element's mixed content into a string with placeholders.
        Returns: ("Hello {{1}}World{{2}}", {"{{1}}": Element_BPT, "{{2}}": Element_EPT})
        """
        tag_map = {}
        tag_counter = 1
        
        # 1. Start with the text of the parent element (before the first child tag)
        text_content = element.text if element.text else ""
        
        # 2. Iterate over children (inline tags like <bpt>, <ept>, <ph>)
        for child in list(element):
            placeholder = f"{{{{{tag_counter}}}}}"
            
            # Store a DEEP COPY of the tag to preserve attributes (id, type, etc.)
            # We strip the tail because the tail belongs to the text flow, not the tag itself
            tag_copy = copy.deepcopy(child)
            tag_copy.tail = None 
            tag_map[placeholder] = tag_copy
            
            # Append placeholder to text
            text_content += placeholder
            
            # Append the text that follows this tag (child.tail)
            if child.tail:
                text_content += child.tail
                
            tag_counter += 1
            
        return text_content, tag_map

    @staticmethod
    def _reconstruct_element(target_element: ET.Element, translated_text: str, tag_map: Dict[str, ET.Element]):
        """
        Reconstructs XML structure from a string with {{n}} placeholders.
        Populates the target_element with text and child nodes.
        """
        # Regex to split by placeholders (capturing group keeps the delimiters)
        # e.g. "Hello {{1}} World" -> ['Hello ', '{{1}}', ' World']
        parts = re.split(r'(\{\{\d+\}\})', translated_text)
        
        last_element = None
        
        for part in parts:
            if not part: continue
            
            if re.match(r'^\{\{\d+\}\}$', part):
                # It is a tag placeholder
                if part in tag_map:
                    # Clone the original tag
                    new_tag = copy.deepcopy(tag_map[part])
                    new_tag.tail = "" # Reset tail, we will fill it if text follows
                    target_element.append(new_tag)
                    last_element = new_tag
                else:
                    # Tag missing or hallucinated? Treat as text to be safe
                    if last_element is not None:
                        last_element.tail = (last_element.tail or "") + part
                    else:
                        target_element.text = (target_element.text or "") + part
            else:
                # It is text
                if last_element is not None:
                    # If we just added a tag, this text belongs to its tail
                    last_element.tail = (last_element.tail or "") + part
                else:
                    # If no tags yet, this is the main text of the element
                    target_element.text = (target_element.text or "") + part

    @staticmethod
    def parse_xliff(content: bytes) -> List[TranslationSegment]:
        try:
            ET.register_namespace('', "urn:oasis:names:tc:xliff:document:1.2")
            ET.register_namespace('mq', "MQXliff")
            
            tree = ET.ElementTree(ET.fromstring(content))
            root = tree.getroot()
            segments = []
            
            ns = {'x': 'urn:oasis:names:tc:xliff:document:1.2', 'mq': 'MQXliff'}
            
            for trans_unit in root.findall(".//x:trans-unit", ns):
                seg_id = trans_unit.get('id')
                source_node = trans_unit.find("x:source", ns)
                target_node = trans_unit.find("x:target", ns)
                
                if source_node is not None:
                    # USE NEW EXTRACTION LOGIC
                    source_text, tag_map = XMLParser._extract_text_with_tags(source_node)
                    
                    # Get existing target text (optional, usually empty for new translation)
                    target_text = ""
                    if target_node is not None:
                         target_text = "".join(target_node.itertext())

                    segments.append(TranslationSegment(
                        id=seg_id,
                        source=source_text,
                        target=target_text,
                        tag_map=tag_map # Store map for reconstruction
                    ))
            return segments
        except Exception as e:
            print(f"XLIFF Parsing Error: {e}")
            return []

    @staticmethod
    def update_xliff(
        original_content: bytes, 
        translations: Dict[str, str], 
        segments_map: Dict[str, TranslationSegment],
        match_rates: Dict[str, int] = None
    ) -> bytes:
        """
        Updates XLIFF with proper tag reconstruction and match rate metadata.
        
        Args:
            original_content: Original XLIFF bytes
            translations: Dict of segment ID -> translated text
            segments_map: Dict of segment ID -> TranslationSegment object
            match_rates: Dict of segment ID -> match percentage (optional)
        
        Returns:
            Updated XLIFF content as bytes
        """
        ET.register_namespace('', "urn:oasis:names:tc:xliff:document:1.2")
        ET.register_namespace('mq', "MQXliff")
        
        tree = ET.ElementTree(ET.fromstring(original_content))
        root = tree.getroot()
        
        ns = {'x': 'urn:oasis:names:tc:xliff:document:1.2', 'mq': 'MQXliff'}
        
        for trans_unit in root.findall(".//x:trans-unit", ns):
            seg_id = trans_unit.get('id')
            
            if seg_id in translations:
                target = trans_unit.find("x:target", ns)
                if target is None:
                    target = ET.SubElement(trans_unit, "{urn:oasis:names:tc:xliff:document:1.2}target")
                
                # Clear existing content
                target.text = None
                for child in list(target):
                    target.remove(child)
                
                # RECONSTRUCT CONTENT WITH TAGS
                trans_text = translations[seg_id]
                segment_obj = segments_map.get(seg_id)
                
                if segment_obj and segment_obj.tag_map:
                    XMLParser._reconstruct_element(target, trans_text, segment_obj.tag_map)
                else:
                    # Fallback if no tags mapped
                    target.text = trans_text
                
                # ===================== ADD MATCH RATE METADATA =====================
                if match_rates and seg_id in match_rates:
                    match_rate = match_rates[seg_id]
                    
                    # Add mq:percent attribute to trans-unit
                    trans_unit.set('{MQXliff}percent', str(match_rate))
                    
                    # Add mq:insertedmatch element with match details
                    inserted_match = ET.Element('{MQXliff}insertedmatch')
                    inserted_match.set('matchtype', '2')
                    inserted_match.set('source', 'TM')
                    inserted_match.set('matchrate', str(match_rate))
                    
                    # Create source element with original text
                    source_elem = ET.SubElement(inserted_match, '{urn:oasis:names:tc:xliff:document:1.2}source')
                    source_elem.set('xml:space', 'preserve')
                    source_elem.text = segment_obj.source if segment_obj else ""
                    
                    # Create target element with translated text
                    target_elem = ET.SubElement(inserted_match, '{urn:oasis:names:tc:xliff:document:1.2}target')
                    target_elem.set('xml:space', 'preserve')
                    target_elem.text = trans_text
                    
                    # Insert insertedmatch before closing trans-unit
                    trans_unit.append(inserted_match)
                
                # Update status
                for key in list(trans_unit.attrib.keys()):
                    if "status" in key:
                        trans_unit.attrib[key] = "Translated"

        # Final string generation and repair
        output_str = ET.tostring(root, encoding='unicode')
        
        # Restore MemoQ specific prefixes (Nuclear Fix)
        if 'xmlns:mq="MQXliff"' not in output_str:
            output_str = output_str.replace('<xliff ', '<xliff xmlns:mq="MQXliff" ')
            
        match = re.search(r'xmlns:(\w+)="MQXliff"', output_str)
        if match:
            wrong_prefix = match.group(1)
            if wrong_prefix != 'mq':
                output_str = output_str.replace(f'{wrong_prefix}:', 'mq:')
                output_str = output_str.replace(f'xmlns:{wrong_prefix}', 'xmlns:mq')

        final_output = f'<?xml version="1.0" encoding="UTF-8"?>\n{output_str}'
        return final_output.encode('utf-8')

    @staticmethod
    def parse_tmx(content: bytes) -> List[Dict]:
        """Standard TMX parsing (unchanged logic, just re-included for completeness)"""
        root = None
        encodings_to_try = ['utf-16', 'utf-8-sig', 'utf-8', 'latin-1']
        for enc in encodings_to_try:
            try:
                xml_str = content.decode(enc)
                xml_str = re.sub(r'<\?xml.*encoding=["\'].*["\'].*\?>', '', xml_str, count=1)
                root = ET.fromstring(xml_str)
                break
            except Exception:
                continue
        if root is None: return []

        entries = []
        for tu in root.findall('.//tu'):
            tuvs = tu.findall('.//tuv')
            if len(tuvs) >= 2:
                entry = {}
                for tuv in tuvs:
                    lang = tuv.get('{http://www.w3.org/XML/1998/namespace}lang')
                    seg = tuv.find('seg')
                    text = "".join(seg.itertext()) if seg is not None else ""
                    if lang: entry[lang.lower()] = text
                entries.append(entry)
        return entries
