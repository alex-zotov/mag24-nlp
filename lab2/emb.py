import torch


class TokenEmbeddings(torch.nn.Module):
    '''
    TokenEmbeddings сопоставляет каждому токену словаря (vocab_size)
    вектор float размером emb_size
    '''

    vocab_size: int = 0
    emb_size: int = 0
    device: str
    vocab_emb: torch.nn.Embedding

    def __init__(self, vocab_size: int, emb_size: int, device: str):
        super().__init__()

        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.device = device

        # при создании инициализируется весами из нормального распределения
        self.vocab_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=emb_size,
            device=device,
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        '''
        преобразуем последовательность токенов в последовательность эмбендингов

        x: тензор batch_size x seq_len 
        -> тензор batch_size x seq_len x emb_size
        '''
        assert x.dtype==torch.long, 'torch.nn.Embedding требует torch.long'

        return self.vocab_emb(x)
