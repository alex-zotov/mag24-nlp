import torch
import mhatt
import ffn


class Decoder(torch.nn.Module):
    '''
    собираем вместе 
    multi-head attention
    feed forward network

    используем 
    residual connections
    нормализацию
    '''

    num_heads: int
    emb_size: int
    head_size: int
    max_seq_len: int

    multi_head_attention: mhatt.MultiHeadAttention
    feed_forward_network: ffn.FeedForward
    norm: torch.nn.LayerNorm

    def __init__(
        self,
        num_heads: int,
        emb_size: int, head_size: int, max_seq_len: int,
        dropout: float,
        device: str
    ):
        super().__init__()

        self.num_heads = num_heads
        self.emb_size = emb_size
        self.head_size = head_size
        self.max_seq_len = max_seq_len

        self.multi_head_attention = mhatt.MultiHeadAttention(
            num_heads=num_heads,
            emb_size=emb_size,
            head_size=head_size,
            max_seq_len=max_seq_len,
            dropout=dropout,
            device=device
        )

        self.feed_forward_network = ffn.FeedForward(
            emb_size=emb_size,
            dropout=dropout,
            device=device
        )

        self.norm = torch.nn.LayerNorm(emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        на входе и на выходе тензор размера
        batch_size x seq_len x emb_size
        '''

        y = self.multi_head_attention(x)
        # residual connections
        x = y+x
        x = self.norm(x)

        y = self.feed_forward_network(x)
        # residual connections
        x = y+x
        x = self.norm(x)

        return x
