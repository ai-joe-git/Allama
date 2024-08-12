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
