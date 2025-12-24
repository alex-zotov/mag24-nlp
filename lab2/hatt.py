import torch


class HeadAttention(torch.nn.Module):
    '''
    одна голова внимания
    '''

    emb_size: int
    head_size: int
    max_seq_len: int

    w_k: torch.nn.Linear
    w_q: torch.nn.Linear
    w_v: torch.nn.Linear

    norm: torch.Tensor

    mask: torch.Tensor

    def __init__(self, emb_size: int, head_size: int, max_seq_len: int, device: str):
        super().__init__()

        self.emb_size = emb_size
        self.head_size = head_size
        self.max_seq_len = max_seq_len

        # на входе приходят ембендинги последовательности токенов
        # batch_size x seq_len x emb_size
        # для каждого токена (его эмбендинга - вектора)
        # с помощью матриц весов (w_k, w_q, w_v)
        # будет вычеслены три вектора key query value
        # про bias говорят, что для простых моделей он не сильно даёт выигрыш
        # для больших моделей он позволяет получить более выразительную модель
        # если дальше идёт нормализация то bias невилируется и не нужен
        # в оригинальной статье bias похоже был
        self.w_k = torch.nn.Linear(
            emb_size, head_size, bias=False, device=device)
        self.w_q = torch.nn.Linear(
            emb_size, head_size, bias=False, device=device)
        self.w_v = torch.nn.Linear(
            emb_size, head_size, bias=False, device=device)

        # нижнюю треугольную матрицу с размером равным
        # максимальному возможному значению длины последовательности
        # будем использовать как маску внимания
        # для того, чтоб скрыть будующие токены
        # здесь два варианта реализации либо создать матрицу и хранить её
        # либо каждый раз пользовать torch.tril

        # минус бесконечность над верхней диагональю
        self.mask = torch.full(
            (max_seq_len, max_seq_len),
            float('-inf'), device=device)
        self.mask = torch.triu(self.mask, diagonal=1)

        # генерируем индексы для нижней треугольной части (включая диагональ)
        # rows, cols = torch.tril_indices(max_seq_len, max_seq_len)
        # self.mask[rows, cols] = 1
        # для младших версий torch
        # строим элементы над диагональю
        '''
        self.mask = torch.triu(self.mask, diagonal=1)
        # строим элементы на диагонали и под ней
        tril = torch.tril(torch.full(
            (max_seq_len, max_seq_len),
            1,
            device=device))
        # и складываем
        self.mask = self.mask + tril
        '''

        # если уж думать про оптимизацию
        # тогда и нормировочный коэффициент заготовим на устройстве
        # нормировка на корень из кол-ва голов
        self.norm = torch.tensor(1/head_size**0.5, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        x : batch_size x seq_len x emb_size
        '''

        # k,q,v размерности batch_size x seq_len x head_size
        # все три слоя линейные. принимают размерность emb_size, отдают head_size
        k: torch.Tensor = self.w_k(x)
        q: torch.Tensor = self.w_q(x)
        v: torch.Tensor = self.w_v(x)

        # матрица внимания batch_size x seq_len x seq_len
        score: torch.Tensor = torch.matmul(q, k.transpose(dim0=-2, dim1=-1))

        # нормировка на корень из кол-ва голов
        score = torch.mul(self.norm, score)

        # маска
        seq_len: int = x.shape[1]
        score = score + self.mask[:seq_len, :seq_len].unsqueeze(0)

        # softmax (пробегаем по размерности эмбендингов,
        # чтоб для каждого токена по всем ембендингам получилась единица)
        score = torch.nn.functional.softmax(score, dim=-1)

        # перемножаем матрицу внимания и вектора
        return torch.matmul(score, v)
