import numpy as np
from typing import List, Optional, Generator
from .utils import read_gguf_file, apply_rope
from .tokenizer import Tokenizer
from .quantization import quantize_weights

class GGUFModel:
    def __init__(self, model_path: str, quantization: Optional[str] = None):
        self.weights, self.config = read_gguf_file(model_path)
        self.tokenizer = Tokenizer(self.config['vocab_size'])
        
        if quantization:
            self.weights = quantize_weights(self.weights, method=quantization)

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        batch_size, seq_len = input_ids.shape
        hidden_size = self.config['hidden_size']
        num_layers = self.config['num_layers']
        num_heads = self.config['num_heads']
        
        x = self.weights['token_embd.weight'][input_ids]
        
        for i in range(num_layers):
            q = np.matmul(x, self.weights[f'layers.{i}.attention.wq.weight'])
            k = np.matmul(x, self.weights[f'layers.{i}.attention.wk.weight'])
            v = np.matmul(x, self.weights[f'layers.{i}.attention.wv.weight'])
            
            q = q.reshape(batch_size, seq_len, num_heads, -1).transpose(0, 2, 1, 3)
            k = k.reshape(batch_size, seq_len, num_heads, -1).transpose(0, 2, 1, 3)
            v = v.reshape(batch_size, seq_len, num_heads, -1).transpose(0, 2, 1, 3)
            
            q, k = apply_rope(q, k, seq_len)
            
            attn = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(q.shape[-1])
            attn = np.exp(attn - np.max(attn, axis=-1, keepdims=True))
            attn = attn / np.sum(attn, axis=-1, keepdims=True)
            
            out = np.matmul(attn, v).transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
            out = np.matmul(out, self.weights[f'layers.{i}.attention.wo.weight'])
            
            x = x + out
            
            ff = np.maximum(np.matmul(x, self.weights[f'layers.{i}.feed_forward.w1.weight']), 0)
            ff = np.matmul(ff, self.weights[f'layers.{i}.feed_forward.w2.weight'])
            
            x = x + ff
        
        x = x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + 1e-5)
        logits = np.matmul(x, self.weights['output.weight'])
        
        return logits

    def generate(self, prompt: str, max_tokens: int = 50, temperature: float = 0.7, 
                 top_p: float = 0.9, batch_size: int = 1) -> List[str]:
        input_ids = self.tokenizer.encode(prompt)
        batch_input = np.array([input_ids] * batch_size)
        generated = [input_ids.copy() for _ in range(batch_size)]

        for _ in range(max_tokens):
            logits = self.forward(np.array(generated))
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-p sampling
            sorted_logits = np.sort(next_token_logits, axis=-1)[:, ::-1]
            cumulative_probs = np.cumsum(np.exp(sorted_logits) / np.sum(np.exp(sorted_logits), axis=-1, keepdims=True), axis=-1)
            sorted_indices = np.argsort(next_token_logits, axis=-1)[:, ::-1]
            top_p_mask = cumulative_probs <= top_p
            
            for i in range(batch_size):
                valid_indices = sorted_indices[i][top_p_mask[i]]
                next_token = np.random.choice(valid_indices)
                generated[i].append(next_token)

            if all(generated[i][-1] == self.tokenizer.eos_token_id for i in range(batch_size)):
                break

        return [self.tokenizer.decode(seq) for seq in generated]

    def generate_stream(self, prompt: str, max_tokens: int = 50, 
                        temperature: float = 0.7, top_p: float = 0.9) -> Generator[str, None, None]:
        input_ids = self.tokenizer.encode(prompt)
        generated = input_ids.copy()

        for _ in range(max_tokens):
            logits = self.forward(np.array([generated]))
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply top-p sampling
            sorted_logits = np.sort(next_token_logits)[::-1]
            cumulative_probs = np.cumsum(np.exp(sorted_logits) / np.sum(np.exp(sorted_logits)))
            sorted_indices = np.argsort(next_token_logits)[::-1]
            top_p_mask = cumulative_probs <= top_p
            
            valid_indices = sorted_indices[top_p_mask]
            next_token = np.random.choice(valid_indices)
            generated.append(next_token)

            yield self.tokenizer.decode([next_token])

            if next_token == self.tokenizer.eos_token_id:
                break
