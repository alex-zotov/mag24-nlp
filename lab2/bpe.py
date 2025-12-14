'''
токенизатор BPE разбивает текст на токены
в качестве токенов служат буквы, морфемы(части слов), слова, а может быть и целое предложение.
Всё зависит от размера корпуса на котором учим токенизатор и размера словаря.
'''

from typing import List, Dict
import dill


class BPE():
    '''
    Byte-Pair Encoding
    '''

    def __init__(self, vocab_size: int):
        # размер словаря
        self.vocab_size: int = vocab_size

        # запомним как преобразовать число в токен
        self.id2token: Dict[int, str] = {}
        # и как преобразовать токен в число
        self.token2id: Dict[str, int] = {}

    def fit(self, text: str):
        '''
        :text: str корпус текста для обучения BPE
        '''

        # в словаре как минимум должны содержаться все большие и маленькие буквы
        # чтоб если встретим слово, которое не видели на обучении могли бы разбить
        # если не на морфемы то хотя бы на буквы

        # если в словаре будут только буквы, то использовать такой декодер для генерации
        # текста - это медленно
        # на каждой итерации аввторегрессия будет добавлятся по одной букве - медленно
        # иметь в словаре морфемы или даже слова

        # в задании просят токены-буквы отсортировать по алфавиту
        # не понятно, зачем - ну отненсу это требование к особенностям проверки
        # порядок остальных токенов определяется частотой вхождения
        # если два токена имеют одинаковую частоту,
        # то токен, который встретился раньше в тексте будет иметь порядок в словаре id2token меньше

        self.id2token = {id: c for id, c in enumerate(sorted(set(text)))}

        self.token2id = {c: id for id, c in self.id2token.items()}

        assert (len(self.id2token) < self.vocab_size
                ), 'слишком маленький размер словаря'

        if len(text) == 1:
            return

        # корпус разбит на токены
        # храним начало токена в корпусе и его длину
        tokensLen: List[int] = [1 for _ in text]

        # бежим по токенам корпуса пытаемся объединить токен с рядом стоящим справа
        # у объединения считаем частоту (частота объединения не больше частоты токена)
        while len(self.id2token) < self.vocab_size:

            i_token: int = 0
            i_token_next: int = tokensLen[i_token]

            # на предыдущем шаге добавили весь корпус как токен
            assert i_token_next < len(text), 'слишком большой размер словаря'

            # если каждый токен объединить с соседом справа
            # посчитаем вхождения таких пар
            pairCnt: Dict[str, int] = {}
            while i_token_next < len(text):
                i_token_next2 = i_token_next + tokensLen[i_token_next]

                pair = text[i_token:i_token_next2]
                pairCnt[pair] = pairCnt.get(pair, 0)+1

                i_token = i_token_next
                i_token_next = i_token_next2

            # найдем пару с максимальной частотой
            max_cnt: int = 0
            max_pair: str = ''
            for pair, cnt in pairCnt.items():
                if max_cnt < cnt:
                    max_cnt = cnt
                    max_pair = pair

            # объединяем максимальную пару в tokensLen
            i_token: int = 0
            i_token_next: int = tokensLen[i_token]
            while i_token_next < len(text):
                i_token_next2 = i_token_next + tokensLen[i_token_next]
                if text[i_token:i_token_next2] == max_pair:
                    tokensLen[i_token] = i_token_next2-i_token

                    i_token = i_token_next2
                    i_token_next = i_token + tokensLen[i_token]
                else:
                    i_token = i_token_next
                    i_token_next = i_token_next2

            # добавляем максимальную пару в decoder
            id: int = len(self.id2token)
            self.id2token[id] = max_pair
            self.token2id[max_pair] = id

    def encode(self, text: str) -> List[int]:
        '''
        закодировать строку text обученным BPE
        '''

        if text == '':
            return []

        '''
        для текущего символа строки выбираем все токены,
        которые можно применить с текущего символа
        применяем самый длинный
        '''

        embending: List[int] = []
        i: int = 0
        while i < len(text):
            max_token: str = ''
            for token in self.id2token.values():
                if text[i:i+len(token)] == token and len(token) > len(max_token):
                    max_token = token
            embending.append(self.token2id[max_token])

            i += len(max_token)

        return embending

    def decode(self, token_ids: List[int]) -> str:
        '''
        декодировать эмбендинг
        '''

        text: str = ''

        for id in token_ids:
            text += self.id2token[id]

        return text

    def save(self, filename: str):
        '''
        сохранить токенайзер в файл
        '''

        with open(filename, 'wb') as f:
            dill.dump(self, f)
        print(f"Объект сохранён в {filename}")

    @classmethod
    def load(cls, filename: str):
        '''
        загрузить токенайзер из файла
        '''

        with open(filename, 'rb') as f:
            obj = dill.load(f)

        print(f"Объект загружен из {filename}")
        return obj
