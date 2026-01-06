import numpy as np

def row12_to_T(row12: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :4] = row12.reshape(3, 4)
    return T
