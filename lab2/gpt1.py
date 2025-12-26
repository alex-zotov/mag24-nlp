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
        assert x.dtype==torch.long, 'torch.nn.Embedding требует torch.long'

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

    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None
    ) -> torch.Tensor:
        '''
        авторегрессия
        на входе последовательность токенов 
            []int batch_size x seq_len
        на выходе последовательность токенов на max_new_tokens больше
            []int batch_size x (seq_len + max_new_tokens)

        do_sample=False:
            следующий выбирается самый вероятный токен
        do_sample=True: 
            следующий токен выбирается случайно,
            но всё же более вероятные токены будут выбираться чаще
        do_sample=True, temperature:
            1 - оставляем распределение как есть
            (0..1) - при temp близких к нулю большие вероятности становятся ещё больше
                а когда возьмём softmax (там exp) и exp от самого вероятного токена улетит в бесконечность
                так что temp=0 эквивалентно do_sample=False
        do_sample=True, top_k
            сэмплируем на top_k наиболее вероятных
        do_sample=True, top_p
            сэмплируем на подвыборке, которая охватывает долю исходов суммарно top_p

        temperature и top_k/top_p 
            в принципе могут применятся последоательно, 
            но температура смещает вероятности, 
            и если применение top_k не пострадает
            то применение top_p зависит от того применять его перед применением температуры или после
            логично сначала отдельно посчитать вероятности выкинуть ненужные по top_p из логитов
            и уже потом применять softmaks
            ещё один аргумент - нельзя просто выкинуть из вероятности часть значений
            т.к сумма вероятностей будет меньше 0 - сэмплирование сломается
        top_k и top_p 
            одновременно не рекомендуется применять
        '''
        assert not (
            top_k is not None and top_p is not None
        ), 'top_k и top_p одновременно не рекомендуется применять'

        eps: float = 1e-6

        for _ in range(max_new_tokens):
            tokens_window = x[:, :-max_new_tokens]

            # batch_size x seq_len x vocab_size
            logits = self.forward(tokens_window)

            # берём последний logit (последний токен в последовательности)
            # он говорит какой токен будет следующим
            # как только взяли последний logit, сразу размерность по seq пропала
            # batch_size x vocab_size
            logits = logits[:, -1, :]

            if do_sample:
                # сначала выкидываем логитсы как того требует top_k / top_p
                # и только потом применяем температуру и считаем вероятности
                # почему? - описано в docstring к generate

                if top_k is not None:
                    _, top_k_ind = torch.topk(logits, k=top_k, dim=-1)

                    filtered = torch.full_like(logits, float('-inf'))
                    filtered.scatter_(dim=-1, index=top_k_ind, src=logits)
                    logits = filtered

                if top_p is not None:
                    prob = torch.softmax(logits, dim=-1)
                    sorted_prob, sorted_prob_ind = torch.sort(
                        prob, descending=True, dim=-1
                    )
                    cumulative_prob = torch.cumsum(sorted_prob, dim=-1)
                    sorted_to_remove = cumulative_prob > top_p
                    sorted_to_remove[:,0]=False

                    # to_remove и sorted_prob, sorted_prob_ind индексированы одинаково
                    # sorted_prob_ind указывает на элемент в исходном prob
                    # построим обраный индекс, 
                    # чтоб бежать по исходному prob и получать индекс в sorted_prob
                    inverse_ind = torch.argsort(sorted_prob_ind, dim=-1)

                    # Преобразуем to_remove из отсортированного пространства в исходное
                    mask = sorted_to_remove.gather(dim=-1, index=inverse_ind)

                    logits[mask]=float('-inf')

                if temperature < 1.0-eps:
                    logits = logits / temperature

            # применяем softmax
            # prob :  batch_size x vocab_size
            prob = torch.softmax(logits, dim=-1)

            if do_sample:
                next_token = torch.multinomial(prob, num_samples=1)
            else:
                # argmax вернёт индекс из словаря (0..vocab_size-1)
                next_token = torch.argmax(prob, dim=-1, keepdim=True)

            x = torch.cat([x, next_token], dim=-1)

        return x

    def fit(self,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: torch.utils.data.DataLoader,
        num_epoch: int,
        learning_rate: float,
        prn:bool = False
    ):
        '''
        train_loader: загрузчик тренировочной выборки
        valid_loader: загрузчик валидационной выборки

        '''

        self.to(device=self.device)

        optimizer = torch.optim.Adam(self.parameters(),lr=learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(num_epoch):
            # режим тренировки
            self.train()

            running_loss:float = 0.0

            for inputs,targets in train_loader:
                inputs.to(self.device)
                targets.to(self.device)

                # batch_size x seq_len x vocab_size
                logits = self.forward(inputs)

                batch_size, seq_len, vocab_size = logits.shape

                # для того чтоб посчитать кросэнтропию преобразуем в
                # batch_size * seq_len x vocab_size
                reshaped_loggits = torch.reshape(logits,(batch_size * seq_len, vocab_size))

                flat_targets = torch.flatten(targets,start_dim=0,end_dim=-1)

                loss = loss_fn(reshaped_loggits, flat_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # средняя потеря за эпоху
            avg_loss = running_loss/len(train_loader)

            if prn:
                print(f'Epoch [{epoch+1}/{num_epoch}], Loss on train: {avg_loss:.4f}')

            running_loss = 0.0
            # режим оценки
            self.eval()
            with torch.no_grad():
                for inputs,targets in valid_loader:
                    inputs.to(self.device)
                    targets.to(self.device)
                    # batch_size x seq_len x vocab_size
                    logits = self.forward(inputs)

                    batch_size, seq_len, vocab_size = logits.shape

                    # для того чтоб посчитать кросэнтропию преобразуем в
                    # batch_size * seq_len x vocab_size
                    reshaped_loggits = torch.reshape(logits,(batch_size * seq_len, vocab_size))

                    flat_targets = torch.flatten(targets,start_dim=0,end_dim=-1)

                    loss = loss_fn(reshaped_loggits, flat_targets)

                    running_loss += loss.item()

            # средняя потеря за эпоху
            avg_loss = running_loss/len(train_loader)

            if prn:
                print(f'Epoch [{epoch+1}/{num_epoch}], Loss on validation: {avg_loss:.4f}')

                    