class BlockDiagonalMask:
    """Records per-segment sequence lengths; consumed by memory_efficient_attention."""

    def __init__(self, q_seqlen, kv_seqlen):
        self.q_seqlen = [int(x) for x in q_seqlen]
        self.kv_seqlen = [int(x) for x in kv_seqlen]

    @classmethod
    def from_seqlens(cls, q_seqlen, kv_seqlen=None):
        return cls(q_seqlen, q_seqlen if kv_seqlen is None else kv_seqlen)
