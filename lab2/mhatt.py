import torch
import hatt


class MultiHeadAttention(torch.nn.Module):
    '''
    несколько (num_heads) голов внимания
    '''

    num_heads: int
    emb_size: int
    head_size: int
    max_seq_len: int
    dropout: float

    heads: torch.nn.ModuleList
    w: torch.nn.Linear

    def __init__(
        self,
        num_heads: int,
        emb_size: int, head_size: int, max_seq_len: int,
        dropout: float,
        device: str
    ):
        '''
        dropout - 0..1 вероятность слоя dropout 
        '''
        super().__init__()

        self.num_heads = num_heads
        self.emb_size = emb_size
        self.head_size = head_size
        self.max_seq_len = max_seq_len
        self.dropout = dropout

        # несколько голов
        # каждая голова на выходе batch_size x seq_len x head_size
        self.heads = torch.nn.ModuleList([
            hatt.HeadAttention(
                emb_size=emb_size,
                head_size=head_size,
                max_seq_len=max_seq_len,
                device=device
            ) for _ in range(num_heads)
        ])

        # линейный  слой с матрицей коэффициентов
        # на входе матрица сконкотенированная по head_size
        # те batch_size x seq_len x (head_size*num_heads)
        # на выходе batch_size x seq_len x emb_size
        self.w = torch.nn.Linear(head_size*num_heads, emb_size)

        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        на входе и на выходе тензор размера
        batch_size x seq_len x emb_size
        '''

        # forward каждой головы возвращает тензор размерности
        # batch_size x seq_len x head_size
        # после конкатенации по последней размерности получим
        # batch_size x seq_len x (head_size*num_heads)
        y: torch.Tensor = torch.cat(
            [h.forward(x) for h in self.heads],
            dim=-1
        )

        # пропускаем через линейный слой
        # на выходе должны получить ту же размерность
        # что и на входе
        # batch_size x seq_len x emb_size
        y = self.w(y)

        # dropout
        y = self.dropout(y)

        return y
