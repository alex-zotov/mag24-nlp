import torch


class GPT(torch.nn.modules):

    vocab_size: int
    max_seq_len: int
    emb_size: int
    num_heads: int
    head_size: int
    num_layers: int
    dropout: float
    device: str

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        emb_size: int,
        num_heads: int,
        head_size: int,
        num_layers: int,
        dropout: float,
        device: str
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
