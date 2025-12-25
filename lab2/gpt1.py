import torch
import emb
import pos_emb
import decoder


class GPT(torch.nn.Module):
    '''
    собираем всё вместе
    TokenEmbeddings PositionalEmbeddings
    Decoder (в количестве num_layers)

    линейный слой emb_size -> vocab_size 
    для преобразования пространства эмбендингов в пространство словаря токенов
    '''

    vocab_size: int
    max_seq_len: int
    emb_size: int
    num_heads: int
    head_size: int
    num_layers: int
    device: str

    token_embeddings: emb.TokenEmbeddings
    positional_embeddings: pos_emb.PositionalEmbeddings
    dropout: torch.nn.Dropout
    decoders: torch.nn.Sequential
    w: torch.nn.Linear

    soft_max: torch.nn.Softmax

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
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device

        self.token_embeddings = emb.TokenEmbeddings(
            vocab_size=vocab_size,
            emb_size=emb_size,
            device=device
        )

        self.positional_embeddings = pos_emb.PositionalEmbeddings(
            max_seq_len=max_seq_len,
            emb_size=emb_size,
            device=device
        )

        self.dropout = torch.nn.Dropout(p=dropout)

        self.decoders = torch.nn.Sequential()
        for i in range(num_layers):
            self.decoders.add_module(
                name=f'decoder {i}',
                module=decoder.Decoder(
                    num_heads=num_heads,
                    emb_size=emb_size,
                    head_size=head_size,
                    max_seq_len=max_seq_len,
                    dropout=dropout,
                    device=device
                )
            )

        self.w = torch.nn.Linear(
            in_features=emb_size,
            out_features=vocab_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        на входе последовательность токенов
        []int размером batch_size x seq_len
        на выходе логиты (не нормированные float)
        []float размером batch_size x seq_len x vocab_size
        '''

        # на выходе batch_size x seq_len x emb_size
        token_emb: torch.Tensor = self.token_embeddings.forward(x)

        # на выходе seq_len x emb_size
        # для всех последовательностей одинакова
        seq_len: int = x.shape[-1]
        positional_emb: torch.Tensor = self.positional_embeddings.forward(
            seq_len=seq_len
        )

        embendings = token_emb + positional_emb.unsqueeze(dim=0)

        x = self.dropout(embendings)

        # с выхода одного декодера на вход другого
        # идут матрицы размерностью
        # batch_size x seq_len x emb_size
        x = self.decoders(x)

        # преобразуем в логитсы
        # размер batch_size x seq_len x vocab_size
        x = self.w(x)

        return x

    def generate(self, x: torch.Tensor, max_new_tokens: int, do_sample: bool = False) -> torch.Tensor:
        '''
        авторегрессия
        на входе последовательность токенов 
            []int batch_size x seq_len
        на выходе последовательность токенов на max_new_tokens больше
            []int batch_size x (seq_len + max_new_tokens)
        '''

        for _ in range(max_new_tokens):
            tokens_window = x[:, :-max_new_tokens]

            # batch_size x seq_len x vocab_size
            logits = self.forward(tokens_window)

            # берём последний logit (последний токен в последовательности)
            # как только взяли последний logit, сразу размерность по seq пропала
            # batch_size x vocab_size
            # он говорит какой токен будет следующим
            # применяем softmax
            # prob :  batch_size x vocab_size
            prob = torch.softmax(logits[:, -1, :], dim=-1)

            if do_sample:
                next_token = torch.multinomial(prob, num_samples=1)
            else:
                # argmax вернёт индекс из словаря (0..vocab_size-1)
                next_token = torch.argmax(prob, dim=-1, keepdim=True)

            x = torch.cat([x, next_token], dim=-1)

        return x
