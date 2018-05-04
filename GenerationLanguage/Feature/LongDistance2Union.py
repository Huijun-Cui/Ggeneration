from torch.utils.data import DataLoader
from CuiModle.config import opt
import torch
import random
import pickle
from CuiModle.config import opt
from copy import deepcopy
def Create2UnionData(path = '../dataset/My_data_v3.pkl'):
    f = open(path,'rb')
    data = pickle.load(f)
    random.shuffle(data)
    i2w = pickle.load(f)
    w2i = pickle.load(f)
    # i2w.append('AND')
    i2slot = pickle.load(f)
    slot2i = pickle.load(f)
    result = []
    max_lenth = 0
    for i in range(1, len(data)):
        if data[i - 1][-1][0] + data[i][-1][0] > max_lenth:
            max_lenth = data[i - 1][-1][0] -1 + data[i][-1][0]
    print(max_lenth)
    for item_ix in range(1,len(data)):
        num_1,num_2 = data[item_ix-1][-1][0],data[item_ix][-1][0]
        slot_list = []
        # preparation for slotlist
        slot_list.append([data[item_ix-1][0],data[item_ix][0]])
        # preparation for DA
        DA = torch.cat([data[item_ix-1][1],data[item_ix][1]])
        # preparation for input
        input1 = data[item_ix-1][2][:num_1-1]
        input2 = data[item_ix][2][:num_2]
        input_vec = torch.cat([input1,input2])
        input_vec = input_vec.numpy().tolist()
        remain_num = max_lenth - (num_1-1) - num_2
        for i in range(remain_num):
            end_vec = [0] * opt.VOC_SIZE
            end_vec[opt.END_ix] = 1
            input_vec.append(end_vec)
        input_vec = torch.FloatTensor(input_vec)
        # preparation for target sentence---------------------------------------------

        target_1_,target_2 = data[item_ix-1][3][:num_1-1],data[item_ix][3][:num_2]
        target_1 = deepcopy(target_1_) # deep copy is necessary,it almost killed me
        target_1[-1] = opt.START_ix
        target = torch.cat([target_1,target_2])
        target = target.numpy().tolist()
        remain_num = max_lenth - (num_1-1) - num_2
        target.extend([opt.END_ix] * remain_num)
        target = torch.FloatTensor(target)
        # preparation for input_b---------------------------------------------
        input1_b = data[item_ix - 1][4][:num_1]
        input2_b = data[item_ix][4][:num_2-1]
        input_vec_b = torch.cat([input2_b, input1_b])
        input_vec_b = input_vec_b.numpy().tolist()
        remain_num = max_lenth - num_1 - (num_2-1)
        for i in range(remain_num):
            start_vec = [0] * opt.VOC_SIZE
            start_vec[opt.START_ix] = 1
            input_vec_b.append(start_vec)
        input_vec_b = torch.FloatTensor(input_vec_b)
        # preparation for target_b---------------------------------------------
        target_1_b, target_2_b_ = data[item_ix - 1][5][:num_1], data[item_ix][5][:num_2-1]
        target_2_b = deepcopy(target_2_b_)
        target_2_b[-1] = opt.END_ix
        target_b = torch.cat([target_2_b, target_1_b])
        target_b = target_b.numpy().tolist()
        remain_num = max_lenth - num_1 - (num_2 -1)
        target_b.extend([opt.START_ix] * remain_num)
        target_b = torch.FloatTensor(target_b)
        # preparation for valid num---------------------------------------------
        num = torch.IntTensor([num_1 + num_2 -1])
        result.append([slot_list,DA,input_vec,target,input_vec_b,target_b,num])
    f_union = open('../dataset/Data2Union','wb')
    pickle.dump(result,f_union)
    pickle.dump(i2w,f_union)
    pickle.dump(w2i, f_union)
    pickle.dump(i2slot, f_union)
    pickle.dump(slot2i, f_union)
def create_data(path = '../dataset/Data2Union'): # for the word conver like i2w i2slot
    f = open(path, 'rb')
    result = pickle.load(f)
    i2w = pickle.load(f)
    w2i = pickle.load(f)
    i2slot = pickle.load(f)
    slot2i = pickle.load(f)
    random.shuffle(result)
    data_lenth = len(result)
    split_step = data_lenth // 10
    train = result[0:split_step * 6]
    valid = result[6 * split_step:(8 * split_step)]
    test  =  result[8 * split_step:]
    # data = LoadData(data ='../dataset/My_data_v2.pkl')
    f1 = open('../dataset/train_union.pkl', 'wb')
    pickle.dump(train, f1)
    f2 = open('../dataset/valid_union.pkl', 'wb')
    pickle.dump(valid, f2)
    f3 = open('../dataset/test_union.pkl', 'wb')
    pickle.dump(test, f3)
    f4 = open('../dataset/i2w_union.pkl', 'wb')
    pickle.dump(i2w,f4)
    f_dic = open('../dataset/i2slot_union.pkl', 'wb')
    pickle.dump(i2slot,f_dic)
    f_w2i = open('../dataset/w2i_union.pkl','wb')
    pickle.dump(w2i,f_w2i)
    f_slot2i = open('../dataset/slot2i_union.pkl','wb')
    pickle.dump(slot2i,f_slot2i)

def ConvertSlotInform(test_path = '../dataset/slot2i_union.pkl',data_path = '../dataset/test_union.pkl'): # we only convert the test data for the preparation of rerank
    f = open(test_path,'rb')
    slot2i = pickle.load(f)
    f_data = open(data_path,'rb')
    data = pickle.load(f_data)
    maxlenth = 0 # first find out the maxlenth
    for item in data:
        if len(item[0][0][0]) > maxlenth:
            maxlenth = len(item[0][0][0])
        if len(item[0][0][1]) > maxlenth:
            maxlenth = len(item[0][0][1])
    #-----------------------------------------------------------------------
    result = []
    for item in data:
        slot_inform,DA,input_f,target_f,input_b,target_b,num = item
        slot1,slot2 = slot_inform[0][0],slot_inform[0][1]
        slot1 = [slot2i[slot_name.split('=')[1]] for slot_name in slot1]
        slot2 = [slot2i[slot_name.split('=')[1]] for slot_name in slot2]
        slot1.extend([[-1,-1]] *( maxlenth - len(slot1))) # here -1 means stop notion
        slot2.extend([[-1,-1]] * (maxlenth - len(slot2))) # here 01 means stop notion
        slot_inform_converted = torch.FloatTensor([slot1,slot2])
        result.append([slot_inform_converted,DA,input_f,target_f,input_b,target_b,num])
    with open('../dataset/test_union_convert.pkl','wb') as f_convert:
        pickle.dump(result,f_convert)




if __name__ == '__main__':
    # Create2UnionData()
    # create_data()
    ConvertSlotInform()
