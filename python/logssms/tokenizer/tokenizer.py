from dataclasses import dataclass, field

import torch
from transformers import PreTrainedTokenizerBase

from logssms.message import TokenizeMsg, DetokenizeMsg

class TokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def tokenize(self, msgs: list[TokenizeMsg]) -> list[torch.Tensor]:
        results: list[torch.Tensor] = []
        # TODO: batch tokenization
        for msg in msgs:
            prompt = self.tokenizer.apply_chat_template(
                msg.text,
                tokenize=False,
                add_generation_prompt=True,
            ) if isinstance(msg.text, list) else msg.text
            input_ids: torch.Tensor = (
                self.tokenizer.encode(prompt, return_tensors="pt")
            )
            results.append(input_ids.view(-1).to(torch.int32))
        return results


@dataclass
class DecodeStatus:
    decoded_ids : list[int] = field(default_factory=list, init=False)
    prefix_begin: int       = field(default=0, init=False) 
    prefix_end  : int       = field(default=0, init=False)
    partial_sent: int       = field(default=0, init=False)  


class DetokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        # uid -> DecodeStatus
        self.decode_map: dict[int, DecodeStatus] = {}
        self.tokenizer = tokenizer
        self.eos_token_id = self.tokenizer.eos_token_id

    def detokenize(self, msgs: list[DetokenizeMsg]) -> list[str]:
        read_ids: list[list[int]] = []
        surr_ids: list[list[int]] = []

        for msg in msgs:
            if msg.uid not in self.decode_map:
                self.decode_map[msg.uid] = DecodeStatus()
            s = self.decode_map[msg.uid]
            if isinstance(msg.next_token, int):
                if not (msg.finished and msg.next_token == self.eos_token_id):
                    s.decoded_ids.append(msg.next_token)
            else:  # verify: list of committed tokens. A trailing EOS is a stream
                # terminator, never user-visible -- strip it (the scalar EOS filter
                # above cannot catch a list).
                toks = msg.next_token
                if msg.finished and toks and toks[-1] == self.eos_token_id:
                    toks = toks[:-1]
                s.decoded_ids.extend(toks)
            read_ids.append(s.decoded_ids[s.prefix_begin:])
            surr_ids.append(s.decoded_ids[s.prefix_begin: s.prefix_end])

        read_texts = self.tokenizer.batch_decode(read_ids)
        surr_texts = self.tokenizer.batch_decode(surr_ids)
        incremental_strs: list[str] = []

        for msg, read_str, surr_str in zip(msgs, read_texts, surr_texts, strict=True):
            s = self.decode_map[msg.uid]
            new_text = read_str[len(surr_str):]
            # Streaming chunk: update the decode status
            if len(new_text) > 0 and not new_text.endswith("�"):
                incremental_strs.append(new_text[s.partial_sent:])
                s.partial_sent = 0
                s.prefix_begin = s.prefix_end
                s.prefix_end = len(s.decoded_ids)
            else:
                new_text = DetokenizeManager._find_printable_text(new_text)
                incremental_strs.append(new_text[s.partial_sent:])
                s.partial_sent = len(new_text)
            
            if msg.finished:
                del self.decode_map[msg.uid]
        return incremental_strs

    @staticmethod
    def _is_chinese_char(cp: int):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)    #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)    #
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  
            return True

        return False

    @staticmethod
    def _find_printable_text(text: str):
        """Returns the longest printable substring of text that contains only entire words."""
        # Borrowed from https://github.com/huggingface/transformers/blob/061580c82c2db1de9139528243e105953793f7a2/src/transformers/generation/streamers.py#L99

        if text.endswith("\n"):
            return text
        elif len(text) > 0 and DetokenizeManager._is_chinese_char(ord(text[-1])):
            return text
        elif len(text) > 1 and DetokenizeManager._is_chinese_char(ord(text[-2])):
            return text[:-1]
        else:
            return text[: text.rfind(" ") + 1]
