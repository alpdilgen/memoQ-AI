import xml.etree.ElementTree as ET
from models.entities import TranslationSegment
from typing import List, Dict, Tuple, Optional
import re
import copy
from datetime import datetime

class XMLParser:
    
    @staticmethod
    def detect_languages(content: bytes) -> Tuple[Optional[str], Optional[str]]:
        """Detect source and target languages from XLIFF file."""
        try:
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
            ns = {'x': 'urn:oasis:names:tc:xliff:document:1.2'}
            
            file_elem = root.find('.//x:file', ns)
            if file_elem is None:
                file_elem = root.find('.//file')
            if file_elem is None:
                file_elem = root
            
            source_lang = file_elem.get('source-language')
            target_lang = file_elem.get('target-language')
            
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
        """Convert XML element's mixed content into string with placeholders."""
        tag_map = {}
        tag_counter = 1
        text_content = element.text if element.text else ""
        
        for child in list(element):
            placeholder = f"{{{{{tag_counter}}}}}"
            tag_copy = copy.deepcopy(child)
            tag_copy.tail = None 
            tag_map[placeholder] = tag_copy
            text_content += placeholder
            if child.tail:
                text_content += child.tail
            tag_counter += 1
            
        return text_content, tag_map

    @staticmethod
    def _reconstruct_element(target_element: ET.Element, translated_text: str, tag_map: Dict[str, ET.Element]):
        """Reconstruct XML structure from string with {{n}} placeholders."""
        parts = re.split(r'(\{\{\d+\}\})', translated_text)
        last_element = None
        
        for part in parts:
            if not part: continue
            
            if re.match(r'^\{\{\d+\}\}$', part):
                if part in tag_map:
                    new_tag = copy.deepcopy(tag_map[part])
                    new_tag.tail = ""
                    target_element.append(new_tag)
                    last_element = new_tag
                else:
                    if last_element is not None:
                        last_element.tail = (last_element.tail or "") + part
                    else:
                        target_element.text = (target_element.text or "") + part
            else:
                if last_element is not None:
                    last_element.tail = (last_element.tail or "") + part
                else:
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
                    source_text, tag_map = XMLParser._extract_text_with_tags(source_node)
                    target_text = ""
                    if target_node is not None:
                         target_text = "".join(target_node.itertext())

                    segments.append(TranslationSegment(
                        id=seg_id,
                        source=source_text,
                        target=target_text,
                        tag_map=tag_map
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
        Update XLIFF with translations and match rate metadata.
        
        Uses ElementTree for parsing, but works with RAW XML strings for output
        to preserve original structure while making surgical changes.
        """
        ET.register_namespace('', "urn:oasis:names:tc:xliff:document:1.2")
        ET.register_namespace('mq', "MQXliff")
        
        # Parse to update target content
        tree = ET.ElementTree(ET.fromstring(original_content))
        root = tree.getroot()
        
        ns = {'x': 'urn:oasis:names:tc:xliff:document:1.2', 'mq': 'MQXliff'}
        
        # Track which segments have 100% matches
        tm_100_segments = {}
        if match_rates:
            for seg_id, rate in match_rates.items():
                if rate == 100:
                    tm_100_segments[seg_id] = True
        
        # Update targets using ElementTree
        for trans_unit in root.findall(".//x:trans-unit", ns):
            seg_id = trans_unit.get('id')
            
            if seg_id in translations:
                target = trans_unit.find("x:target", ns)
                if target is None:
                    target = ET.SubElement(trans_unit, "{urn:oasis:names:tc:xliff:document:1.2}target")
                
                target.text = None
                for child in list(target):
                    target.remove(child)
                
                trans_text = translations[seg_id]
                segment_obj = segments_map.get(seg_id)
                
                if segment_obj and segment_obj.tag_map:
                    XMLParser._reconstruct_element(target, trans_text, segment_obj.tag_map)
                else:
                    target.text = trans_text

        # Convert to string
        output_str = ET.tostring(root, encoding='unicode')
        
        # Fix namespaces
        if 'xmlns:mq="MQXliff"' not in output_str:
            output_str = output_str.replace('<xliff ', '<xliff xmlns:mq="MQXliff" ')
            
        match = re.search(r'xmlns:(\w+)="MQXliff"', output_str)
        if match:
            wrong_prefix = match.group(1)
            if wrong_prefix != 'mq':
                output_str = output_str.replace(f'{wrong_prefix}:', 'mq:')
                output_str = output_str.replace(f'xmlns:{wrong_prefix}', 'xmlns:mq')
        
        # Now do SURGICAL MODIFICATIONS for 100% matches on the raw XML string
        # This preserves the original structure perfectly
        if tm_100_segments:
            for seg_id in tm_100_segments.keys():
                output_str = XMLParser._add_tm_match_metadata_to_segment(
                    output_str, seg_id
                )

        final_output = f'<?xml version="1.0" encoding="UTF-8"?>\n{output_str}'
        return final_output.encode('utf-8')

    @staticmethod
    def _add_tm_match_metadata_to_segment(xml_str: str, seg_id: str) -> str:
        """
        Add TM match metadata to a specific trans-unit using surgical string replacements.
        
        For 100% TM matches:
        1. Change mq:status="Translated" -> "ManuallyConfirmed"
        2. Add mq:percent="100"
        3. Add mq:translatorcommitmatchrate="100"
        4. Add mq:translatorcommitusername="System"
        5. Add mq:translatorcommittimestamp
        6. Update mq:minorversionend and mq:minorversionstart (optional, use existing values)
        """
        timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # Find the specific trans-unit opening tag for this seg_id
        # Pattern: <trans-unit[^>]*?\sid="SEG_ID"[^>]*?>
        pattern = rf'(<trans-unit\s+[^>]*?\sid="{re.escape(seg_id)}"[^>]*?)>'
        
        def modify_opening_tag(match):
            opening_tag = match.group(1)
            
            # 1. Change status to ManuallyConfirmed
            opening_tag = re.sub(
                r'mq:status="[^"]*"',
                'mq:status="ManuallyConfirmed"',
                opening_tag
            )
            
            # 2. Add/update mq:percent="100"
            if 'mq:percent=' in opening_tag:
                opening_tag = re.sub(
                    r'mq:percent="\d+"',
                    'mq:percent="100"',
                    opening_tag
                )
            else:
                opening_tag = opening_tag.rstrip() + ' mq:percent="100"'
            
            # 3. Add mq:translatorcommitmatchrate="100"
            if 'mq:translatorcommitmatchrate=' not in opening_tag:
                opening_tag = opening_tag.rstrip() + ' mq:translatorcommitmatchrate="100"'
            
            # 4. Add mq:translatorcommitusername
            if 'mq:translatorcommitusername=' not in opening_tag:
                opening_tag = opening_tag.rstrip() + ' mq:translatorcommitusername="System"'
            
            # 5. Add mq:translatorcommittimestamp
            if 'mq:translatorcommittimestamp=' not in opening_tag:
                opening_tag = opening_tag.rstrip() + f' mq:translatorcommittimestamp="{timestamp}"'
            
            # 6. Update last changed timestamp
            opening_tag = re.sub(
                r'mq:lastchangedtimestamp="[^"]*"',
                f'mq:lastchangedtimestamp="{timestamp}"',
                opening_tag
            )
            
            # 7. Update lastchanginguser
            opening_tag = re.sub(
                r'mq:lastchanginguser="[^"]*"',
                'mq:lastchanginguser="System"',
                opening_tag
            )
            
            return opening_tag + '>'
        
        xml_str = re.sub(pattern, modify_opening_tag, xml_str)
        
        return xml_str

    @staticmethod
    def parse_tmx(content: bytes) -> List[Dict]:
        """Standard TMX parsing"""
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
