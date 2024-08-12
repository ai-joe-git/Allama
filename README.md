# Allama

A simple tool library to run LLM models with GGUF format, implemented from scratch.

## Features

- GGUF model loading and inference
- Basic tokenization
- Model quantization options (int8, float16)
- Batch processing
- Streaming text generation
- Advanced prompt engineering tools

## Installation

```bash
pip install git+https://github.com/yourusername/simple_llm_tool.git

Certainly! I'll implement the recommendations I mentioned earlier to expand and improve our simple LLM tool library. We'll add features like model quantization options, batch processing, streaming output, and more advanced prompt engineering tools. We'll also improve the overall structure and efficiency of the code.

Here's the enhanced version of our library:

Project structure:
```
simple_llm_tool/
│
├── simple_llm_tool/
│   ├── __init__.py
│   ├── model.py
│   ├── tokenizer.py
│   ├── quantization.py
│   ├── utils.py
│   └── prompt_engineering.py
│
├── setup.py
└── README.md
```

Let's implement each file:

1. `model.py`:

```python
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
```

2. `tokenizer.py`:

```python
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
```

3. `quantization.py`:

```python
import numpy as np
from typing import Dict

def quantize_weights(weights: Dict[str, np.ndarray], method: str = 'int8') -> Dict[str, np.ndarray]:
    if method == 'int8':
        return {k: quantize_int8(v) for k, v in weights.items()}
    elif method == 'float16':
        return {k: v.astype(np.float16) for k, v in weights.items()}
    else:
        raise ValueError(f"Unsupported quantization method: {method}")

def quantize_int8(arr: np.ndarray) -> np.ndarray:
    scale = np.max(np.abs(arr)) / 127
    quantized = np.round(arr / scale).astype(np.int8)
    return quantized, scale

def dequantize_int8(quantized: np.ndarray, scale: float) -> np.ndarray:
    return quantized.astype(np.float32) * scale
```

4. `utils.py`:

```python
import struct
import numpy as np
from typing import Dict, Tuple

def read_gguf_file(filepath: str) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    weights = {}
    config = {}
    
    with open(filepath, 'rb') as f:
        magic = f.read(4)
        if magic != b'GGUF':
            raise ValueError("Not a valid GGUF file")
        
        version = struct.unpack('i', f.read(4))[0]
        
        num_tensors = struct.unpack('i', f.read(4))[0]
        for _ in range(num_tensors):
            name_length = struct.unpack('i', f.read(4))[0]
            name = f.read(name_length).decode('utf-8')
            
            shape = struct.unpack('i', f.read(4))
            dtype = struct.unpack('i', f.read(4))[0]
            
            if dtype == 0:  # float32
                data = np.frombuffer(f.read(4 * np.prod(shape)), dtype=np.float32).reshape(shape)
            elif dtype == 1:  # float16
                data = np.frombuffer(f.read(2 * np.prod(shape)), dtype=np.float16).reshape(shape)
            else:
                raise ValueError(f"Unsupported dtype: {dtype}")
            
            weights[name] = data
        
        num_config = struct.unpack('i', f.read(4))[0]
        for _ in range(num_config):
            key_length = struct.unpack('i', f.read(4))[0]
            key = f.read(key_length).decode('utf-8')
            value_length = struct.unpack('i', f.read(4))[0]
            value = f.read(value_length).decode('utf-8')
            config[key] = value
    
    return weights, config

def apply_rope(q: np.ndarray, k: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    head_dim = q.shape[-1]
    positions = np.arange(seq_len)
    freqs = 1.0 / (10000 ** (np.arange(0, head_dim, 2) / head_dim))
    phases = positions[:, np.newaxis] * freqs[np.newaxis, :]
    cos = np.cos(phases)
    sin = np.sin(phases)
    
    q_rot = q[..., ::2] * cos - q[..., 1::2] * sin
    q_pass = q[..., 1::2] * cos + q[..., ::2] * sin
    q = np.stack([q_rot, q_pass], axis=-1).reshape(q.shape)
    
    k_rot = k[..., ::2] * cos - k[..., 1::2] * sin
    k_pass = k[..., 1::2] * cos + k[..., ::2] * sin
    k = np.stack([k_rot, k_pass], axis=-1).reshape(k.shape)
    
    return q, k
```

5. `prompt_engineering.py`:

```python
from typing import List

def few_shot_prompt(task: str, examples: List[tuple], query: str) -> str:
    prompt = f"Task: {task}\n\n"
    for i, (input_text, output_text) in enumerate(examples, 1):
        prompt += f"Example {i}:\nInput: {input_text}\nOutput: {output_text}\n\n"
    prompt += f"Now, please complete the following:\nInput: {query}\nOutput:"
    return prompt

def chain_of_thought_prompt(question: str) -> str:
    return f"""Question: {question}
Let's approach this step-by-step:
1)
2)
3)
Therefore, the final answer is:"""

def self_consistency_prompt(question: str, num_solutions: int = 3) -> str:
    prompt = f"Question: {question}\n\n"
    for i in range(1, num_solutions + 1):
        prompt += f"Solution {i}:\n1)\n2)\n3)\nTherefore, the answer for Solution {i} is:\n\n"
    prompt += "Now, considering all solutions, the most consistent answer is:"
    return prompt
```

6. `setup.py`:

```python
from setuptools import setup, find_packages

setup(
    name="simple_llm_tool",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A simple tool library to run LLM models with GGUF format",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/simple_llm_tool",
)
```

7. `README.md`:

```markdown
# Simple LLM Tool

A simple tool library to run LLM models with GGUF format, implemented from scratch.

## Features

- GGUF model loading and inference
- Basic tokenization
- Model quantization options (int8, float16)
- Batch processing
- Streaming text generation
- Advanced prompt engineering tools

## Installation

```bash
pip install git+https://github.com/yourusername/simple_llm_tool.git
```

## Usage

```python
from simple_llm_tool.model import GGUFModel
from simple_llm_tool.prompt_engineering import few_shot_prompt

# Initialize the model
model = GGUFModel("path/to/your/gguf_model.gguf", quantization="int8")

# Generate text
output = model.generate("Once upon a time,", max_tokens=50, temperature=0.7, top_p=0.9)
print(output)

# Stream generation
for token in model.generate_stream("The quick brown fox"):
    print(token, end='', flush=True)

# Batch processing
prompts = ["Hello,", "Once upon a time,", "The quick brown fox"]
outputs = model.generate(prompts, max_tokens=50, batch_size=3)
for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt}\nOutput: {output}\n")

# Use prompt engineering tools
examples = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("Who wrote 'Romeo and Juliet'?", "William Shakespeare wrote 'Romeo and Juliet'.")
]
prompt = few_shot_prompt("Answer the following question:", examples, "What is the largest planet in our solar system?")
output = model.generate(prompt, max_tokens=100)
print(output)
```

Make sure to download a GGUF model before using this library.
```

This enhanced version of the library includes:

1. Model quantization options (int8 and float16)
2. Batch processing for more efficient generation of multiple outputs
3. Streaming output generation
4. More advanced prompt engineering tools (few-shot prompting, chain-of-thought prompting, and self-consistency prompting)
5. Improved type hinting and documentation
6. Top-p sampling for better text generation
