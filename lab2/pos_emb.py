import torch


class PositionalEmbeddings(torch.nn.Module):
    '''
    PositionalEmbeddings

    позиционные эмбендинги - сопоставляем номеру токена в последовательности вектор
    '''

    max_seq_len: int = 0
    emb_size: int = 0
    device: str
    pos_emb: torch.nn.Embedding

    def __init__(self, max_seq_len: int, emb_size: int, device: str):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.emb_size = emb_size
        self.device = device

        self.pos_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_embeddings=self.max_seq_len,
            embedding_dim=emb_size,
            device=device,
        )

    def forward(self, seq_len: int) -> torch.tensor:
        '''
        forward

        возвращает первые seq_len строк матрицы эбендингов
        '''

        # генерируем индексы позиций: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, dtype=torch.long, device=self.device)

        # получаем эмбеддинги для этих позиций
        pos_embeddings = self.pos_emb(positions)

        return pos_embeddings
