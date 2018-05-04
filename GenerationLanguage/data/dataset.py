import pickle
from torch.utils.data import DataLoader
from CuiModle.config import opt
import random
from torch.utils.data import DataLoader
class LoadData(object):
    def __init__(self,data ='../dataset/train.pkl'):
        f = open(data,'rb')
        self.data = pickle.load(f)

    def __getitem__(self, item):
        return self.data[item]
    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    data = LoadData(data ='../dataset/test.pkl')
    data_batch = DataLoader(data, batch_size=opt.batch_size, shuffle=False, drop_last=True)
    for ix, (slot_inform, DA, input_f, target_f, _, _, unpacknum) in enumerate(data_batch):
        None
