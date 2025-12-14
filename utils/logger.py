import datetime
from typing import List, Dict
from models.entities import TranslationSegment, TMMatch, TermMatch

class TransactionLogger:
    def __init__(self):
        self.entries = []

    def _time(self):
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def info(self, message: str):
        """Logs a general info message"""
        self.entries.append(f"{self._time()} | INFO: {message}")

    def log_batch_start(self, batch_num: int, segments: List[TranslationSegment]):
        """Logs the segments included in the current batch"""
        msg = f"\n{self._time()} | === PROCESSING BATCH {batch_num} ==="
        msg += f"\nSegments in this batch ({len(segments)}):"
        for seg in segments:
            msg += f"\n  [{seg.id}] {seg.source}"
        self.entries.append(msg)

    def log_tm_matches(self, tm_data: Dict[str, List[TMMatch]]):
        """Logs TM matches found for the current batch"""
        count = sum(len(matches) for matches in tm_data.values())
        if count == 0:
            self.entries.append(f"{self._time()} | TM Context: No matches found.")
            return

        msg = f"{self._time()} | TM Context ({count} matches):"
        for seg_id, matches in tm_data.items():
            if matches:
                msg += f"\n  For Segment [{seg_id}]:"
                for m in matches:
                    msg += f"\n    - {m.source_text} -> {m.target_text} ({m.similarity:.0f}% {m.match_type})"
        self.entries.append(msg)

    def log_tb_matches(self, tb_data: Dict[str, List[TermMatch]]):
        """Logs Terminology matches found for the current batch"""
        count = sum(len(matches) for matches in tb_data.values())
        if count == 0:
            self.entries.append(f"{self._time()} | TB Context: No matches found.")
            return

        msg = f"{self._time()} | TB Context ({count} matches):"
        for seg_id, matches in tb_data.items():
            if matches:
                msg += f"\n  For Segment [{seg_id}]:"
                for m in matches:
                    msg += f"\n    - {m.source} = {m.target}"
        self.entries.append(msg)

    def log_llm_interaction(self, prompt: str, response: str):
        """Logs the exact prompt sent and the raw response received"""
        self.entries.append(f"\n{self._time()} | >>> INPUT TO LLM (PROMPT) >>>\n{'-'*20}\n{prompt}\n{'-'*20}")
        self.entries.append(f"\n{self._time()} | <<< OUTPUT FROM LLM (RESPONSE) <<<\n{'-'*20}\n{response}\n{'-'*20}")
        self.entries.append("=" * 80)

    def get_content(self) -> str:
        """Returns the full log as a string"""
        return "\n".join(self.entries)