import numpy as np
from typing import List

class Tokenizer:
    def __init__(self, vocab_size: int):
        # This is still a simplified tokenizer
        self.vocab_size = vocab_size
        self.vocab = {chr(i): i for i in range(min(256, vocab_size))}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.eos_token_id = 0  # Assuming 0 is the EOS token

    def encode(self, text: str) -> List[int]:
        return [self.vocab.get(char, self.vocab_size - 1) for char in text]

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.inverse_vocab.get(id, '<unk>') for id in ids)
