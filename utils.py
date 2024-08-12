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
