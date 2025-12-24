import torch


class FeedForward(torch.nn.Module):
    '''
    Feed Forward Network задача внести нелинейность
    '''

    emb_size: int
    device: str

    w1: torch.nn.Linear
    w2: torch.nn.Linear
    relu: torch.nn.ReLU
    dropout: torch.nn.Dropout

    def __init__(self, emb_size: int, device: str, dropout: float = 0.1):
        super().__init__()

        self.emb_size = emb_size
        self.device = device
        self.dropout = dropout

        # первый линейный слой
        # на входе batch_size x seq_len x emb_size
        # на выходе batch_size x seq_len x (emb_size*4)
        # множитель 4 был в оригинальной статье - можно вынести параметром
        self.w1 = torch.nn.Linear(emb_size, 4*emb_size)

        self.relu = torch.nn.ReLU()

        # второй линейный слой возвращает в исходный размер
        self.w2 = torch.nn.Linear(4*emb_size, emb_size)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.tensor) -> torch.tensor:
        '''
        на входе batch_size x seq_len x emb_size
        на выходе тот же размер
        '''

        y: torch.Tensor = self.w1(x)
        y = self.relu(y)
        y = self.w2(y)
        y = self.dropout(y)

        return y
