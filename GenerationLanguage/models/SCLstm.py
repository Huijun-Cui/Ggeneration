import torch.nn as nn
from torch.autograd import Variable
import torch
from torch.nn.functional import softmax
import pickle
import fire
import tqdm
from CuiModle.data import LoadData
from CuiModle.config import opt
from CuiModle import models

# class DefaultConfig(object):
#     VOC_SIZE = 7
#     alpha = 0.1
#     max_length = 137
#     input_size = 7
#     hidden_size = 5
#     output_size = 7
#     seq_size = 31
#     DA_size = 3
#     env = 'slot-generation'
#     epoch = 30
#     lr = 10.0
#     alpha = 0.5
#     batch_size = 2
#     best_acc = 0
#
# opt = DefaultConfig()


class SCLSTM_cell(nn.Module):
    def __init__(self, input_size, hidden_size,DA_size):
        super(SCLSTM_cell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.DA_size = DA_size
        self.linear = nn.Linear(input_size + hidden_size, 4 * hidden_size,bias=False)
        self.w2d = nn.Linear(self.input_size,self.DA_size,bias=False)
        self.h2d = nn.Linear(self.hidden_size,self.DA_size,bias=False)
        self.d2c = nn.Linear(self.DA_size,self.hidden_size,bias=False)
    def forward(self, input, hidden,C,DA,alpha = 0.5):
        try:
            combined = torch.cat((input, hidden), 1)
        except:
            None
        A = self.linear(combined)
        ai, af, ao, ag = torch.split(A, self.hidden_size, dim=1)
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)
        r = torch.sigmoid(self.w2d(input) + alpha * self.h2d(hidden))
        DA = r * DA
        C = f * C + i * g + torch.tanh(self.d2c(DA))
        hidden = o * (torch.tanh(C))
        return hidden,C,DA

class SCLSTM(nn.Module):
    def __init__(self,input_size, hidden_size,DA_size,num_layers =1,voc_size=opt.VOC_SIZE):
        super(SCLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.DA_size = DA_size
        self.num_layers = num_layers
        self.voc_size = voc_size
        modle_list = []
        for i in range(num_layers):
            modle_list.append(SCLSTM_cell(input_size, hidden_size,DA_size))
        self.modle_list = nn.ModuleList(modle_list)
        self.h2o = nn.Linear(self.hidden_size,self.voc_size,bias=False)
    def forward(self, input,hidden_state,C_state,DA_state,batch_size = opt.batch_size):
        loss_store = Variable(torch.FloatTensor([0]))
        sqlen = input.size()[0]
        current_input = input
        for i in range(self.num_layers):
            hidden = hidden_state
            C = C_state
            DA = DA_state
            output = []
            for j in range(sqlen):
                hidden, C, DA_ = self.modle_list[i](current_input[j],hidden,C,DA)
                store= torch.sqrt(torch.sum((DA_-DA)*(DA_-DA)))
                DA = DA_
                loss_store += opt.eta * torch.pow(Variable(torch.FloatTensor([opt.kexi])),store)
                output.append(self.h2o(hidden))
            current_input = output
        return torch.cat(output,1).view(-1).view(batch_size* sqlen,-1),hidden,C,DA,loss_store
        # return torch.cat(output,0).view(sqlen*batch_size,-1),hidden,C,DA
        # return torch.cat(output,0).view(7,2,-1).permute(1,0,2).contiguous().view(14,-1),hidden,C,DA
    def init(self,batch_size = 1):
        hidden = Variable(torch.zeros(batch_size,self.hidden_size))
        C = Variable(torch.zeros(batch_size,self.hidden_size))
        return hidden,C




if __name__ == '__main__':
    criterion = nn.CrossEntropyLoss()
    # opt = DefaultConfig()
    modle = SCLSTM(opt.input_size,opt.hidden_size,opt.DA_size)
    optimizer = torch.optim.Adadelta(modle.parameters(), lr=opt.lr)
    i2w = ['End','I','am','a','fine','Chinese','boy']
    sen1 = 'i am a Chinese boy'
    sen2 = 'i am fine'
    sen1 = [[0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
            [0,0,0,0,0,0,0]
    ]
    sen2 = [[0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0]
    ]
    data = torch.FloatTensor([sen1,sen2]).permute(1,0,2)
    data = Variable(data)
    sen1_t = [2,3,5,6,0,0]
    sen2_t = [2,4,0,0,0,0]
    target = torch.LongTensor([sen1_t,sen2_t])
    target = Variable(target)
    # print(target)
    DA_1 = [1,0,0]
    DA_2 = [0,1,0]
    # DA_1 = [0,0,0]
    # DA_2 = [0,0,0]
    target = torch.LongTensor([sen1_t, sen2_t])
    # target = Variable(target.permute(1,0))
    DA = torch.FloatTensor([DA_1, DA_2])
    DA = Variable(DA)
    mat = [[1,0,0,0,0,0,0,0,0,0,0,0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    ]
    mat = Variable(torch.FloatTensor(mat))
    for i in range(300):
        # print(DA)
        hidden,C = modle.init(batch_size=2)
        output,_,_,_ = modle(data,hidden,C,DA)
        # print(target.view(-1))
        # print(output)
        # result = torch.multinomial(torch.exp(output),1)
        result = torch.topk(torch.exp(output),1,dim = 1)
        str = ''
        for item in result[1]:
            str+= i2w[item.data[0]] + ' '
        print('epoch:',i,'the out is:',str)
        output_ = torch.mm(mat,output)
        target_ = target.contiguous().view(-1).unsqueeze(1)
        target_ = torch.mm(mat, Variable(target_.float())).contiguous().view(-1)
        loss = 0
        loss = criterion(output_,target_.long())
        print('the loss is:',loss.data[0])
        loss.backward()
        optimizer.step()
    # model_back = models.SCLSTM(opt.input_size, opt.hidden_size, opt.DA_size)
    # input = torch.randn(15,1,opt.VOC_SIZE)
    # h,c = model_back.init()
    # da = torch.randn(1,48)
    # model_back(Variable(input),h,c,Variable(da))







