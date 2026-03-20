import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if max_len == None:
        lng = max([(lambda x:len(x))(x) for x in seqs])
    else:
        lng = max_len

    if len(seqs) == 0:
        return np.empty((0,0))
    else:
        prev = [(lst + [pad_value] * lng)[:lng] for lst in seqs]
        return np.array(prev, dtype=int)


