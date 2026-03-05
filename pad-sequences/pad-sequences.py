import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if len(seqs) == 0:
        return np.zeros((0,0), dtype=int)

    if max_len is None:
        L = max(len(seq) for seq in seqs)
    else:
        L = max_len

    padded = []

    for seq in seqs:

        seq = seq[:L]

        if len(seq) < L:
            seq = seq + [pad_value] * (L - len(seq))

        padded.append(seq)

    return np.array(padded)
    pass