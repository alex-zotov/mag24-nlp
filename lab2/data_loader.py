from torch.utils.data import Dataset
from torch import Tensor

class GetData(Dataset):
    '''
    будем при загрузке даннных использовать стандартный torch.utils.data.DataLoader
    он умеет работать с наборами данных унаследованных от torch.utils.data.DataSet
    нужно реализовать __len__ __getitem__ итератора
    '''

    data: list[int]
    seq_len:int
    device:str

    def __init__(self,data: list[int], seq_len:int, device:str):
        '''
        data: корпус текта в виде токенов
        seq_len: разбиваем на последовательности длиной seq_len
        '''

        self.data=data
        self.seq_len=seq_len
        self.device=device
        
    def __len__(self)->int:
        n = len(self.data)

        # - 1 в конце, потому что 
        # генерим пару x,y
        # y сдвинут вправо на один символ относительно x
        # нужен ещё один символ чтобы построить у для последнего х
        return n - self.seq_len - 1

    def __getitem__(self, idx:int)->tuple[Tensor,Tensor]:
        '''
        idx: 
            номер последовательности,
            он же позиция с которой начинается последовательномсть в data
        '''

        x:Tensor = Tensor(self.data[idx : idx + self.seq_len], device= self.device) 
        y:Tensor = Tensor(self.data[idx + 1: idx + 1 + self.seq_len], device= self.device) 

        return x,y