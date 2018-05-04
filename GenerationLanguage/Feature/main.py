from copy import deepcopy
import pickle
import pandas as pd
from collections import defaultdict
import numpy as np
import torch
from CuiModle.data import LoadData
import random
def CreatePickleFile():
    data = pd.read_csv('../dataset/My_data_ori.csv')
    dic = defaultdict(int)
    word_list = []
    i2w = []
    w2i = {}
    i2w.append('START')
    i2w.append('END')
    for sen in data['sentence']:
        for word in sen.strip().split():
            word_list.append(word)
    word_set = set(word_list)
    for word in word_set:
        i2w.append(word)
    for ix, word in enumerate(i2w):
        w2i[word] = ix
    i2slot = []
    slot2i = {}
    lexical_slot_list = ['slotaddr', 'slotcount', 'slotfood', 'slotname', 'slotphone', 'slotpostcode']
    unlex_slot_list = ['slotacttype', 'slotfoodnot', 'slotnamenot', 'slotorcare', 'slotpricerange',
                        'slotpricerangenot','slotselect2', 'slotsignature', 'slottask', 'slottype','slotarea']
    result_slot = []
    # result_slot += lexical_slot_list
    for item in lexical_slot_list:
        result_slot.append(item)
        result_slot.append(item+ '-'+'nullvalue')
    result_slot.append('slotpricerange-nullvalue')
    result_slot.append('slotarea-nullvalue')
    result_slot.append('slotsignature-nullvalue')
    for item in unlex_slot_list:
        tmp_dic = data[item].unique().tolist()
        for i in tmp_dic:
            if isinstance(i, str):
                if i not in result_slot:
                    result_slot.append(i)
    i2slot = result_slot
    for ix,i in enumerate(i2slot):
        slot2i[i] = ix
    print(i2slot)
    print(slot2i)
    df_columne = data.columns.tolist()
    df_columne.remove('Unnamed: 0')
    df_columne.remove('sentence')
    f_pickle = open('../dataset/My_data_v1.pkl','wb')
    pickle_list = []
    for item in data.itertuples():
        setence_tmp = item.sentence.split()
        sentence = [w2i[i] for i in setence_tmp]
        DA = [0] * len(i2slot)
        slot_info = []
        for columne_name in df_columne:
            print(columne_name)
            print(type(columne_name))
            columne_value = getattr(item, columne_name)
            if isinstance(columne_value, str):
                columne_value = getattr(item, columne_name)
                if columne_value == 'nullvalue':
                    columne_value = columne_name + '-' + 'nullvalue'
                elif columne_name in lexical_slot_list:
                    columne_value = columne_name
                DA[slot2i[columne_value]] = 1
                slot_info.append(columne_name + '='+ columne_value)
        pickle_list.append([slot_info,sentence,DA])
    # print(np.array(pickle_list))
    pickle.dump(pickle_list,f_pickle)
    pickle.dump(i2w,f_pickle)
    pickle.dump(w2i,f_pickle)
    pickle.dump(i2slot,f_pickle)
    pickle.dump(slot2i,f_pickle)
    f_pickle.close()


def CreatePickleForTorch():
    f = open('../dataset/My_data_v1.pkl', 'rb')
    f_target = open('../dataset/My_data_v3.pkl','wb') # create data file for Pytorch
    data = pickle.load(f)
    i2w = pickle.load(f)
    w2i = pickle.load(f)
    i2slot = pickle.load(f)
    slot2i = pickle.load(f)
    voc_size = len(i2w) # add two notation END  and START
    max_lenth = 0
    for idx, (_,sen, DA) in enumerate(data):
        if len(sen) > max_lenth:
            max_lenth = len(sen)
    max_lenth +=2 # include END and START
    START_ix = 0
    END_ix = 1
    result = []
    for slot_infrom,sen,DA in data:
        sen.insert(0,START_ix)
        sen.append(END_ix)
        sen_tmp_f = [END_ix] * max_lenth
        sen_tmp_b = [START_ix] * max_lenth
        sen_tmp_f[0:len(sen)] = sen
        sen_tmp_b[0:len(sen)] = sen[::-1]
        sen_tmp_f_shift = sen_tmp_f[1::] + [END_ix]
        sen_tmp_b_shift = sen_tmp_b[1::] + [START_ix]
        word2d_f = []
        for num in sen_tmp_f:
            word2d_tmp = [0] * voc_size
            word2d_tmp[num] = 1
            word2d_f.append(word2d_tmp)
        word2d_b = []
        for num_b in sen_tmp_b:
            word2d_tmp = [0] * voc_size
            word2d_tmp[num_b] = 1
            word2d_b.append(word2d_tmp)
        result.append([slot_infrom,
                       torch.FloatTensor(DA),
                       torch.FloatTensor(word2d_f),
                       torch.LongTensor(sen_tmp_f_shift), # Saving LongTensor type for the preparation of CrossEntropy
                       torch.FloatTensor(word2d_b),
                       torch.LongTensor(sen_tmp_b_shift),
                       torch.IntTensor([len(sen)])
                       ])
        print('the lenth of result list is{}'.format(len(result)))
    pickle.dump(result,f_target )
    pickle.dump(i2w,f_target)
    pickle.dump(w2i,f_target)
    pickle.dump(i2slot,f_target)
    pickle.dump(slot2i,f_target)
    f.close()
    f_target.close()
def create_data(path): # for the word conver like i2w i2slot
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
    test = result[8 * split_step:]
    # data = LoadData(data ='../dataset/My_data_v2.pkl')
    f1 = open('../dataset/train.pkl', 'wb')
    pickle.dump(train, f1)
    f2 = open('../dataset/valid.pkl', 'wb')
    pickle.dump(valid, f2)
    f3 = open('../dataset/test.pkl', 'wb')
    pickle.dump(test, f3)
    f4 = open('../dataset/i2w.pkl', 'wb')
    pickle.dump(i2w,f4)
    f_dic = open('../dataset/slot_dic.pkl', 'wb')
    pickle.dump(i2slot,f_dic)
    f_w2i = open('../dataset/w2i.pkl','wb')
    pickle.dump(w2i,f_w2i)
    f_slot2i = open('../dataset/slot2i.pkl','wb')
    pickle.dump(slot2i,f_slot2i)





if __name__ == '__main__':
    # CreatePickleFile()
    # CreatePickleForTorch()
    # create_data(path='../dataset/My_data_v3.pkl')