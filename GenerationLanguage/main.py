import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import tqdm
from config import opt
import models
import visdom
from os.path import exists
from data import LoadData
from torch.utils.data import DataLoader
import fire
import pickle
import copy
import nltk
# from sklearn.model_selection import train_test_split
import random
from collections import defaultdict
from multiprocessing import Pool
import multiprocessing
from math import log

def train_v2_f(**kwargs):
    for k,v in kwargs:
        setattr(opt,k,v)
    model = models.SCLSTM(opt.input_size, opt.hidden_size, opt.DA_size)
    model.load_state_dict(torch.load('params_batch_modle_f.pkl'))
    optimizer = torch.optim.Adadelta(model.parameters(), lr=opt.lr)
    criterion = nn.CrossEntropyLoss()
    train = LoadData(data='./dataset/train_union.pkl')
    valid = LoadData(data ='./dataset/valid_union.pkl')
    data_batch = DataLoader(train, batch_size=opt.batch_size, shuffle=False, drop_last=True)
    vis = visdom.Visdom()
    plt_loss = vis.line(Y=torch.Tensor([0]), X=torch.Tensor([0]), opts=dict(title='Loss Value',
                                                                       legend=['Valid Accuracy'],
                                                                       showlegend=True))
    plt_accu = vis.line(Y=torch.Tensor([0]), X=torch.Tensor([0]), opts=dict(title='Accuracy Evaluation On Validation',
                                                                            legend=['Accuracy'],
                                                                       showlegend=True))
    best_count = 0
    for e in tqdm.tqdm(range(opt.epoch)):
        epoch_loss = 0
        target_acc = 0
        flag_time = 0
        for ix, (_,DA,input_f,target_f,_,_,unpacknum) in enumerate(data_batch):
            optimizer.zero_grad()
            loss = 0
            hidden, C = model.init(opt.batch_size)
            input = input_f.float().permute(1, 0, 2)
            output, _, _, DA_,loss_store = model(Variable(input), hidden, C, Variable(DA))
            mat = torch.zeros(unpacknum.sum(), opt.seq_size * opt.batch_size)
            accumu_add = 0
            row_ix = 0
            for batch_ix in range(opt.batch_size):
                # print('in the {} thr batch,the unpack numbeer is{} '.format(batch_ix,unpacknum[batch_ix][0]))
                for batch_ix_num in range(unpacknum[batch_ix][0]):
                    mat[row_ix][accumu_add + batch_ix_num] = 1
                    row_ix += 1
                accumu_add += opt.seq_size
            mat = Variable(mat)
            # output = output.view(opt.seq_size*opt.batch_size,-1)
            output_ = torch.mm(mat, output)
            target_ = Variable(target_f.contiguous().view(-1).unsqueeze(1))
            target_ = torch.mm(mat, target_.float()).contiguous().view(-1)
            loss_cross = criterion(output_, target_.long())
            loss = loss_cross + torch.sqrt((DA_ * DA_).sum()) + loss_store
            if ix % 20 == 0:
                print('the loss is', loss.data[0])
            epoch_loss += loss
            loss.backward()
            optimizer.step()
        print('epocch: ', e, 'the loss is', epoch_loss.data[0] / ix)
        vis.line(Y=torch.FloatTensor([epoch_loss.data[0] / ix]),
                 X=np.array(np.array([1 + e])), win=plt_loss, update='append',
                 opts=dict(title='Loss Value',
                           legend=['Loss Value'],
                           showlegend=True))
        # target_acc = evaluate(model, i2w=i2w)
        accuracy = evalue_v2_f(model,valid,batch_size = 100)
        vis.line(Y=torch.FloatTensor([accuracy]),
                 X=np.array(np.array([1 + e])), win=plt_accu, update='append',
                 opts=dict(title='Accuracy Evaluation',
                           legend=['Accuracy'],
                           showlegend=True))

        if accuracy <= opt.best_acc:
            flag_time += 1
        else:
            print('Update the the best accuracy!\n New Model Saved~~')
            opt.best_acc = accuracy
            flag_time = 0
            torch.save(model.state_dict(), 'params_batch_modle_f.pkl')
        if flag_time > 3:
            break
        if accuracy > 0.84 and best_count >=3:
            setattr(opt, 'lr', opt.lr * 0.5)
            print('the learning rate is Updated as {}'.format(opt.lr))
            best_count = 0
            optimizer = torch.optim.Adadelta(model.parameters(), lr=opt.lr)
        elif accuracy > 0.83:
            best_count +=1
            print('the best count is :{}'.format(best_count))



def evalue_v2_f(model,data,batch_size):
    correct_coumt = 0
    sum_count = 0
    data_batch = DataLoader(data, batch_size=opt.batch_size, shuffle=False, drop_last=True)
    for ix, (_,DA, input_f, target_f, _, _, unpacknum) in enumerate(data_batch):
        flag = 0
        hidden, C = model.init(opt.batch_size)
        input = input_f.float().permute(1, 0, 2)
        output, _, _, _,_ = model(Variable(input), hidden, C, Variable(DA))
        mat = torch.zeros(unpacknum.sum(), opt.seq_size * opt.batch_size)
        accumu_add = 0
        row_ix = 0
        for batch_ix in range(opt.batch_size):
            # print('in the {} thr batch,the unpack numbeer is{} '.format(batch_ix,unpacknum[batch_ix][0]))
            for batch_ix_num in range(unpacknum[batch_ix][0]):
                mat[row_ix][accumu_add + batch_ix_num] = 1
                row_ix += 1
            accumu_add += opt.seq_size
        mat = Variable(mat)
        # output = output.view(opt.seq_size*opt.batch_size,-1)
        output_ = torch.mm(mat, output)
        value_tensor,index_tesor = torch.max(output_,1)
        target = []
        for idx,number in enumerate(unpacknum):
            target.extend(target_f.tolist()[idx][:number[0]])
        for x,y in zip(index_tesor.data,target):
            if x == y:
                correct_coumt +=1
                # print('the output is : {}  the target is {} \n'.format(x, y))

            sum_count += 1

    print('There is {} batch data ~~\n'.format(ix))
    print('the accuracy is: {}'.format(correct_coumt/sum_count))
    return  correct_coumt/sum_count

def train_v2_b(**kwargs):
    for k,v in kwargs:
        setattr(opt,k,v)
    model = models.SCLSTM(opt.input_size, opt.hidden_size, opt.DA_size)
    model.load_state_dict(torch.load('params_batch_modle_b.pkl'))
    optimizer = torch.optim.Adadelta(model.parameters(), lr=opt.lr)
    criterion = nn.CrossEntropyLoss()
    train = LoadData(data='./dataset/train_union.pkl')
    valid = LoadData(data='./dataset/valid_union.pkl')
    data_batch = DataLoader(train, batch_size=opt.batch_size, shuffle=False, drop_last=True)
    vis = visdom.Visdom()
    plt_loss = vis.line(Y=torch.Tensor([0]), X=torch.Tensor([0]), opts=dict(title='Loss Value',
                                                                       legend=['Valid Accuracy'],
                                                                       showlegend=True))
    plt_accu = vis.line(Y=torch.Tensor([0]), X=torch.Tensor([0]), opts=dict(title='Accuracy Evaluation On Validation',
                                                                            legend=['Accuracy'],
                                                                          showlegend=True))
    best_count = 0
    for e in tqdm.tqdm(range(opt.epoch)):
        epoch_loss = 0
        target_acc = 0
        flag_time = 0
        for ix, (_,DA,_,_,input_b,target_b,unpacknum) in enumerate(data_batch):
            optimizer.zero_grad()
            loss = 0
            hidden, C = model.init(opt.batch_size)
            input = input_b.float().permute(1, 0, 2)
            output, _, _, DA_,loss_store= model(Variable(input), hidden, C, Variable(DA))
            mat = torch.zeros(unpacknum.sum(), opt.seq_size * opt.batch_size)
            accumu_add = 0
            row_ix = 0
            for batch_ix in range(opt.batch_size):
                # print('in the {} thr batch,the unpack numbeer is{} '.format(batch_ix,unpacknum[batch_ix][0]))
                for batch_ix_num in range(unpacknum[batch_ix][0]):
                    mat[row_ix][accumu_add + batch_ix_num] = 1
                    row_ix += 1
                accumu_add += opt.seq_size
            mat = Variable(mat)
            # output = output.view(opt.seq_size*opt.batch_size,-1)
            output_ = torch.mm(mat, output)
            target_ = Variable(target_b.contiguous().view(-1).unsqueeze(1))
            target_ = torch.mm(mat, target_.float()).contiguous().view(-1)
            loss_cross = criterion(output_, target_.long())
            loss = loss_cross + torch.sqrt((DA_ * DA_).sum()) + loss_store
            if ix % 20 == 0:
                print('the loss is', loss.data[0])
            epoch_loss += loss
            loss.backward()
            optimizer.step()
        print('epocch: ', e, 'the loss is', epoch_loss.data[0] / ix)
        vis.line(Y=torch.FloatTensor([epoch_loss.data[0] / ix]),
                 X=np.array(np.array([1 + e])), win=plt_loss, update='append',
                 opts=dict(title='Loss Value',
                           legend=['Loss Value'],
                           showlegend=True))
        # target_acc = evaluate(model, i2w=i2w)
        accuracy = evalue_v2_b(model,valid,batch_size = 100)
        vis.line(Y=torch.FloatTensor([accuracy]),
                 X=np.array(np.array([1 + e])), win=plt_accu, update='append',
                 opts=dict(title='Accuracy Evaluation',
                           legend=['Accuracy'],
                           showlegend=True))

        if accuracy <= opt.best_acc:
            flag_time += 1
        else:
            print('Update the the best accuracy!\n New Model Saved~~')
            opt.best_acc = accuracy
            torch.save(model.state_dict(), 'params_batch_modle_b.pkl')
        if flag_time > 5:
            break
        if accuracy > 0.83 and best_count >=3:
            setattr(opt, 'lr', opt.lr * 0.5)
            print('the learning rate is Updated as {}'.format(opt.lr))
            best_count = 0
            optimizer = torch.optim.Adadelta(model.parameters(), lr=opt.lr)
        elif accuracy > 0.83:
            best_count +=1





def evalue_v2_b(model,data,batch_size):
    correct_coumt = 0
    sum_count = 0
    data_batch = DataLoader(data, batch_size=opt.batch_size, shuffle=False, drop_last=True)
    for ix, (_,DA, _, _, input_b, target_b, unpacknum) in enumerate(data_batch):
        hidden, C = model.init(opt.batch_size)
        input = input_b.float().permute(1, 0, 2)
        output, _, _, _,_ = model(Variable(input), hidden, C, Variable(DA))
        mat = torch.zeros(unpacknum.sum(), opt.seq_size * opt.batch_size)
        accumu_add = 0
        row_ix = 0
        for batch_ix in range(opt.batch_size):
            # print('in the {} thr batch,the unpack numbeer is{} '.format(batch_ix,unpacknum[batch_ix][0]))
            for batch_ix_num in range(unpacknum[batch_ix][0]):
                mat[row_ix][accumu_add + batch_ix_num] = 1
                row_ix += 1
            accumu_add += opt.seq_size
        mat = Variable(mat)
        # output = output.view(opt.seq_size*opt.batch_size,-1)
        output_ = torch.mm(mat, output)
        value_tensor,index_tesor = torch.max(output_,1)
        target = []
        for ix,number in enumerate(unpacknum):
            target.extend(target_b.tolist()[ix][:number[0]])
        for x,y in zip(index_tesor.data,target):
            if x == y:
                correct_coumt +=1
                print('the output is : {}  the target is {} \n'.format(x, y))
            sum_count += 1

    print('the accuracy is {}'.format(correct_coumt/sum_count))
    return  correct_coumt/sum_count









def slot_feature_batch(slot_inform):
    lexical = ['slotaddr', 'slotcount', 'slotfood', 'slotname', 'slotphone', 'slotpostcode']
    slotselect2 = ['moderate', 'east', 'cheap', 'dontcare', 'south', 'polynesian', 'chinese',\
         'turkish', 'centre', 'english', 'jamaican', 'modern european', 'crossover', 'greek',\
        'west', 'north', 'german', 'romanian', 'corsica', 'european', 'halal', 'mediterranean',\
        'gastropub', 'christmas', 'afghan', 'indian', 'world', 'brazilian', 'korean', 'belgian',\
        'basque', 'vegetarian', 'asian oriental', 'portuguese', 'cuban', 'thai', 'british', 'russian',\
        'vietnamese']
    result = []
    for item in slot_inform:
        slotname = item
        if item in lexical:
            result.append(item)
        elif 'nullvalue' in item:
            if 'slotaddr' in item:
                result.append('address')
            elif 'slotfood' in item:
                result.append(('type of food', 'kind of food'))
            elif 'slotname' in item:
                result.append('name')
            elif 'slotphone' in item:
                result.append(('phone', 'telephone'))
            elif 'slotpostcode' in item:
                result.append(('post', 'postal'))
            elif 'slotarea' in item:
                result.append(('part', 'area'))
            elif 'slotpricerange' in item:
                result.append('price range')
        elif 'slotselect2' == item:
            result.append(item)
        elif 'find'  == item:
            result.append('looking for')
        elif 'dontcare' in item:
            result.append(('care', 'any'))
        elif item in slotselect2:
            result.append(item)
    return result


def generation(modle_f,modle_b,DA,str_target,i2w,i2slot,slot_inform):
    slot_list = []   # with the usage of slotlist,wo caculate the err score
    slot_list_1 = slot_feature(slot_inform[0][0])
    slot_list_2 = slot_feature(slot_inform[0][1])
    slot_list_1 = set(slot_list_1)
    slot_list_2 = set(slot_list_2)
    slot_list = slot_list_1 | slot_list_2
    max_lenth = opt.max_length
    input = [0] * opt.VOC_SIZE
    input[opt.START_ix] = 1
    input = Variable(torch.FloatTensor([input]).unsqueeze(0))
    gen_best = -888888 # a small num ensure smaller than initial score
    gen_sen_list = []
    best_bleu = 0.0  # for choosing the best sentence to evaluate the modle
    for i in range(opt.gen_time):
        End_Flag_Count = 0
        hidden, C = modle_f.init()
        DA_ = DA
        forward_store = []
        for j in range(max_lenth):
            output, hidden, C, DA_,loss_store = modle_f(input, hidden, C, DA_, batch_size=1)
            m_ = torch.nn.Softmax()
            output_ = m_(output)
            result_ix = torch.multinomial(output_.view(-1), 1).data[0]
            input = [0] * opt.VOC_SIZE
            input[result_ix] = 1
            input = Variable(torch.FloatTensor([input]).unsqueeze(0))
            # print(result_ix)
            forward_store.append((result_ix, output_.data[0][result_ix]))
            if result_ix == opt.END_ix:
                break
        forward_store.reverse()
        backward_input_tmp = [item[0] for item in forward_store]
        backward_target = [0] * len(backward_input_tmp) #crresponding back target_output
        backward_target[0:-1],backward_target[-1] = backward_input_tmp[1:],opt.START_ix
        backward_input = []
        for item in backward_input_tmp:
            back_input = [0] * opt.VOC_SIZE
            back_input[item] = 1
            backward_input.append(back_input)
        backward_input = Variable(torch.FloatTensor(backward_input))
        try:
            backward_input = backward_input.unsqueeze(1)
        except:
            print('Attention! Here is a mistake\n')
            print(backward_input)
            return
        hidden, C = modle_b.init()
        m_b = torch.nn.Softmax()
        back_output,_,_,_,_ = modle_b(backward_input, hidden, C, DA, batch_size=1)
        back_output = m_b(back_output)
        # value_tensor, index_tesor = torch.max(back_output,1)
        value_back_list = []
        for b_ix,item in enumerate(backward_target):
            value_back_list.append(back_output.data[b_ix][item])
        # score in char level to see  what the hell is going on
        # print('the backward score is {}'.format(value_back_list))
        # print('the forward scorre is {}'.format([kk[1] for kk in forward_store]))
        forward_score = 1.0
        for _,value in forward_store:
            forward_score *=value
        back_score = 0.0 #in long distance case,i change my mind,i wanna use accumulation
        for value in value_back_list:
            back_score  +=value
        gen_sample_list = []
        for item_f in forward_store:
            if item_f[0] != opt.END_ix:
                gen_sample_list.append(i2w[item_f[0]])
        slot_dict = defaultdict(int)
        err_cost = 0
        for cmp_item in gen_sample_list:
            for slot_item in slot_list:
                if (isinstance(slot_item,str) and slot_item == cmp_item) or (isinstance(slot_item,list)\
                                                                               and cmp_item in slot_item):
                    if slot_dict[cmp_item] >=1:
                        None # in consideration of the low rate of slot repeat error we wont use it
                        # err_cost +=1
                    slot_dict[cmp_item] +=1
        for cmp_item in slot_list:
            if isinstance(cmp_item,str) and cmp_item not in gen_sample_list:
                err_cost += 1
            elif isinstance(cmp_item,list):
                flag = 0   # when it chages to 1 ,menas there exists one item inclued in  the targer sentence
                for inner_item in cmp_item:
                    if inner_item in gen_sample_list:
                        flag =1
                if flag == 0:
                    err_cost +=1
        # back_score = 0 # let the backward go ,help me !!!!
        if (forward_score + back_score) + opt.lamda * err_cost * -1 > gen_best:
            print('The target sentence is : {}'.format(str_target))
            print('The slot is :{}'.format(slot_list))
            print('Notice: In {}th gentime ,The best gen_best is updated '.format(i+1))
            gen_best = (forward_score + back_score) + opt.lamda * err_cost * -1
            forward_store.reverse()
            gen_sen_list = [item[0] for item in forward_store]
            str_test = ''
            for test_item in gen_sen_list:
                if test_item != opt.END_ix:
                    str_test += i2w[test_item] + ' '
            print('**[the new selected sentence is : {}]'.format(str_test))
            store = nltk.translate.bleu_score.sentence_bleu([str_target], str_test, weights=[1])
            if store > best_bleu:
                best_bleu = store
            print('The Bleu is {}'.format(store))
            print('The Forward Score is: {}'.format(forward_score))
            print('The Backward Score is: {}'.format(back_score))
            print('The Err Score is: {}'.format(err_cost))
        else:
            forward_store.reverse()
            gen_sen_list_no = [item[0] for item in forward_store]
            str_test = ''
            for test_item in gen_sen_list_no:
                str_test += i2w[test_item] + ' '
            print('the sentence not selected is  : {}'.format(str_test))
            store = nltk.translate.bleu_score.sentence_bleu([str_target], str_test, weights=[1])
            if store > best_bleu:
                best_bleu = store
            print('The Bleu is {}'.format(store))
            print('The Forward Score is: {}'.format(forward_score))
            print('The Backward Score is:{}'.format(back_score))
            print('The Err Score is: {}'.format(err_cost))

    return gen_sen_list,best_bleu

def generation_test(modle_f,modle_b,DA,str_target):
    max_lenth = opt.max_length
    input = [0] * opt.VOC_SIZE
    input[opt.START_ix] = 1
    input = Variable(torch.FloatTensor([input]).unsqueeze(0))
    gen_best = -10 # a negative num ensure smaller than initial score
    forward_store = []
    for i in range(1):
        hidden, C = modle_f.init()
        DA_ = DA
        for j in range(max_lenth):
            output, hidden, C, DA_,loss_store = modle_f(input, hidden, C, DA_, batch_size=1)
            m_ = torch.nn.Softmax()
            output_ = m_(output)
            # result_ix = torch.multinomial(output_.view(-1), 1).data[0]
            m_v,result_ix = torch.max(output_,1)
            result_ix = result_ix.data[0]
            input = [0] * opt.VOC_SIZE
            input[result_ix] = 1
            input = Variable(torch.FloatTensor([input]).unsqueeze(0))
            # print(result_ix)
            if result_ix == opt.END_ix:
                break
            forward_store.append((result_ix, output_.data[0][result_ix]))

    result = [k[0] for k in forward_store]

    return result



def BLEU_EV():
    vis = visdom.Visdom()
    bleu_plot = vis.line(Y=torch.FloatTensor([[0,0,0]]), X=np.column_stack((np.array([0]), np.array([0]), np.array([0]))), opts=dict(title='BLEU score',
                                                                            legend=['Score','Best Score','Max Score'],
                                                                            showlegend=True))
    f = open('./dataset/i2w_union.pkl','rb')
    i2w = pickle.load(f)
    f_dic = open('./dataset/i2slot_union.pkl','rb')
    i2slot = pickle.load(f_dic)
    # slot_dic = pickle.load(f_dic)
    modle_f = models.SCLSTM(opt.input_size, opt.hidden_size, opt.DA_size)
    modle_f.load_state_dict(torch.load('params_batch_modle_f.pkl'))
    modle_b = models.SCLSTM(opt.input_size, opt.hidden_size, opt.DA_size)
    modle_b.load_state_dict(torch.load('params_batch_modle_b.pkl'))
    f_valid = open('./dataset/test_union.pkl','rb')
    data = pickle.load(f_valid)
    # i2w.append('START')
    # i2w.append('END')
    # evalue_v2_f(modle_f, data,opt.batch_size,i2w)
    #
    BLEU_score = 0.0
    Best_BLEU_score = 0.0
    BLEU_max = 0.0
    for ix,item in enumerate(data):
        slot_inform = item[0]
        DA = Variable(torch.FloatTensor(item[1]))
        unpack_num = item[-1][0]
        target = item[3]
        target_list = []
        str_target = ''
        for i_t in target[:unpack_num].int():
            if i_t != opt.END_ix:
                str_target += i2w[i_t] + ' '
                target_list.append(i2w[i_t])
        resutl,best_bleu = generation(modle_f, modle_b, DA,str_target,i2w,i2slot,slot_inform)
        resutl_max = generation_test(modle_f, modle_b, DA, str_target)
        str_out = ''
        out_list = []
        # for sam_out,max_out in zip(resutl,resutl_max):
        #     if sam_out != opt.END_ix:
        #         str_out += i2w[sam_out] + ' '
        #     if max_out != opt.END_ix:
        #         out_list.append(i2w[max_out])
        for sam_out in resutl:
            if sam_out != opt.END_ix:
                str_out += i2w[sam_out] + ' '
        for max_out in resutl_max:
            if max_out != opt.END_ix:
                out_list.append(i2w[max_out])
        out_list = ' '.join(out_list)
        print('               Generating the {}th sentence\n'.format(ix + 1))
        print('The output is :{} \n'.format(str_out))
        print('The target is :{} \n'.format(str_target))
        store =nltk.translate.bleu_score.sentence_bleu([str_target],str_out,weights=[1])
        BLEU_score +=store
        max_store = nltk.translate.bleu_score.sentence_bleu([str_target],out_list,weights=[1])
        Best_BLEU_score += best_bleu
        BLEU_max += max_store
        if store < 0.7:
            print('the current Bleu score is {}'.format(store))
        print('The final Bleu score is updating:{} '.format(BLEU_score/(ix+1)))
        print('*'* 60)
        # vis.line(Y= torch.FloatTensor([[epoch_loss/i,target_loss.data[0]/index]]), X =np.column_stack((np.array([1+e]),np.array(1+e))), win=plt, update='append',
        #     #          opts=dict(title = 'Loss Function',
        #     #                    legend = ['Train Loss Function','Valid Loss Function'],
        #     #                    showlegend = True))
        vis.line(Y=torch.FloatTensor([[BLEU_score/(ix+1),Best_BLEU_score/(ix+1),BLEU_max/(ix+1)]]),
                 X=np.column_stack((np.array([1 + ix]),np.array([1 + ix]),np.array([1 + ix]))), win=bleu_plot, update='append',
                 opts=dict(title='BLEU score ',
                           legend=['Score','Best Score','Max Score'],
                           showlegend=True))
        # mydict['best_bleu'] = -8888
        # mydict['best_bleu_sen'] = ''
        # mydict['best_score'] = -8888
        # mydict['best_score_sen'] = ''
def mycallback(x):
    bleu_store,gen_str,gen_score,myDict,ix = x
    print('enter into mycallback\n')
    with open('call_back_log.txt', 'a+') as f:
        f.write('{} and {} th sentence is generated.\n'.format(ix//2,ix//2 + 1))
        f.write(':' + gen_str + '\n')
        f.write('The Bleu score is:{} . \n'.format(bleu_store))
        f.write('The gen_score is : {} \n'.format(gen_score))
def SigleSenGen(modle_f,modle_b,DA,str_target,i2w,i2slot,slot_inform,ix):
    print('the {} th processing is started \n'.format(ix))
    slot_list = []  # with the usage of slotlist,wo caculate the err score
    slot_list_1 = slot_feature(slot_inform[0][0])
    slot_list_2 = slot_feature(slot_inform[0][1])
    slot_list_1 = set(slot_list_1)
    slot_list_2 = set(slot_list_2)
    # slot_list = list(set(tuple(slot_list_1)) | set(tuple(slot_list_2)))
    slot_list = slot_list_1 | slot_list_2
    max_lenth = opt.max_length
    input = [0] * opt.VOC_SIZE
    input[opt.START_ix] = 1
    input = Variable(torch.FloatTensor([input]).unsqueeze(0))
    End_Flag_Count = 0
    hidden, C = modle_f.init()
    DA_ = DA
    forward_store = []
    for j in range(max_lenth):
        output, hidden, C, DA_, loss_store = modle_f(input, hidden, C, DA_, batch_size=1)
        m_ = torch.nn.Softmax()
        output_ = m_(output)
        result_ix = torch.multinomial(output_.view(-1), 1).data[0]
        input = [0] * opt.VOC_SIZE
        input[result_ix] = 1
        input = Variable(torch.FloatTensor([input]).unsqueeze(0))
        # print(result_ix)
        forward_store.append((result_ix, output_.data[0][result_ix]))
        if result_ix == opt.END_ix:
            break
    forward_store.reverse()
    backward_input_tmp = [item[0] for item in forward_store]
    backward_target = [0] * len(backward_input_tmp)  # crresponding back target_output
    backward_target[0:-1], backward_target[-1] = backward_input_tmp[1:], opt.START_ix
    backward_input = []
    for item in backward_input_tmp:
        back_input = [0] * opt.VOC_SIZE
        back_input[item] = 1
        backward_input.append(back_input)
    backward_input = Variable(torch.FloatTensor(backward_input))
    try:
        backward_input = backward_input.unsqueeze(1)
    except:
        print('Attention! Here is a mistake\n')
        print(backward_input)
        return
    hidden, C = modle_b.init()
    m_b = torch.nn.Softmax()
    back_output, _, _, _, _ = modle_b(backward_input, hidden, C, DA, batch_size=1)
    back_output = m_b(back_output)
    # value_tensor, index_tesor = torch.max(back_output,1)
    value_back_list = []

    for b_ix, item in enumerate(backward_target):
        value_back_list.append(back_output.data[b_ix][item])
    # score in char level to see  what the hell is going on
    # print('the backward score is {}'.format(value_back_list))
    # print('the forward scorre is {}'.format([kk[1] for kk in forward_store]))
    forward_score = 1.0
    for _, value in forward_store:
        forward_score *= value
    back_score = 0.0  # in long distance case,i change my mind,i wanna use accumulation
    for value in value_back_list:
        back_score += value
    gen_sample_list = []
    for item_f in forward_store:
        if item_f[0] != opt.END_ix:
            gen_sample_list.append(i2w[item_f[0]])
    slot_dict = defaultdict(int)
    err_cost = 0

    # for cmp_item in gen_sample_list:
    #     if cmp_item in slot_list and slot_dict[cmp_item] >= 1:
    #         err_cost +=1
    #     slot_dict[cmp_item] += 1
    for cmp_item in gen_sample_list:
        for slot_item in slot_list:
            if (isinstance(slot_item, str) and slot_item == cmp_item) or (isinstance(slot_item, list) \
                                                                              and cmp_item in slot_item):
                if slot_dict[cmp_item] >= 1:
                    None  # in consideration of the low rate of slot repeat error we wont use it
                    # err_cost +=1
                slot_dict[cmp_item] += 1

    # for cmp_item in slot_list:
    #     if cmp_item not in gen_sample_list:
    #         err_cost += 1
    for cmp_item in slot_list:
        if isinstance(cmp_item, str) and cmp_item not in gen_sample_list:
            err_cost += 1
        elif isinstance(cmp_item, list):
            flag = 0  # when it chages to 1 ,menas there exists one item inclued in  the targer sentence
            for inner_item in cmp_item:
                if inner_item in gen_sample_list:
                    flag = 1
            if flag == 0:
                err_cost += 1
    # back_score = 0 # let the backward go ,help me !!!!
    print('the forward score is {},the back_score is {} the opt.ambda is{}'.
          format(forward_score, back_score, opt.lamda))
    print('the {} th processing is starting ...............\n'.format(ix))
    print('fuck what is going on ')

    gen_score = (forward_score + back_score) + opt.lamda * err_cost * -1
    print('the gen score is'.format(gen_score))
    gen_sen_list = [item[0] for item in forward_store]
    gen_str = ''
    for test_item in gen_sen_list:
        if test_item != opt.END_ix:
            gen_str += i2w[test_item] + ' '

    bleu_store = nltk.translate.bleu_score.sentence_bleu([str_target], gen_str, weights=[1])
    # compare  Shared Data in this section instead of callback function
    # if bleu_store > myDict['best_bleu']:
    #     myDict['best_bleu'] = bleu_store
    #     myDict['best_bleu_sen'] = gen_str
    # if gen_score > myDict['best_score']:
    #     print('test!!!!!!!!!!!!!!!!!!!')
    #     myDict['best_score'] = gen_score
    #     myDict['best_score_sen'] = gen_str

    print('the {} th processing is ended \n'.format(ix))
    return bleu_store,gen_str,gen_score,myDict,ix

def MutiProcessBleu():
    f = open('./dataset/i2w_union.pkl', 'rb')
    i2w = pickle.load(f)
    f_dic = open('./dataset/i2slot_union.pkl', 'rb')
    i2slot = pickle.load(f_dic)
    # slot_dic = pickle.load(f_dic)
    modle_f = models.SCLSTM(opt.input_size, opt.hidden_size, opt.DA_size)
    modle_f.load_state_dict(torch.load('params_batch_modle_f.pkl'))
    modle_b = models.SCLSTM(opt.input_size, opt.hidden_size, opt.DA_size)
    modle_b.load_state_dict(torch.load('params_batch_modle_b.pkl'))
    f_valid = open('./dataset/test_union.pkl', 'rb')
    data = pickle.load(f_valid)
    for ix,item in enumerate(data):
        slot_inform = item[0]
        DA = Variable(torch.FloatTensor(item[1]))
        unpack_num = item[-1][0]
        target = item[3]
        target_list = []
        str_target = ''
        for i_t in target[:unpack_num].int():
            if i_t != opt.END_ix:
                str_target += i2w[i_t] + ' '
                target_list.append(i2w[i_t])
        p = Pool(30)
        # mydict = multiprocessing.Manager().dict()
        # mydict['best_bleu'] = -8888
        # mydict['best_bleu_sen'] = ''
        # mydict['best_score'] = -8888
        # mydict['best_score_sen'] = ''
        # SigleSenGen(modle_f, modle_b, DA, str_target, i2w, i2slot, slot_inform,mydict,ix)
        for _ in range(5):
            p.apply_async(SigleSenGen, args=(modle_f, modle_b, DA, str_target, i2w, i2slot, slot_inform,ix,),\
                          callback= mycallback)
        p.close()
        p.join()
        print('the {} th data is done\n'.format(ix))
        # with open('sentence_best.txt','a+') as f_best:
        #     f_best.write('The target sentence is: {} \n'.format(str_target))
        #     f_best.write('The {} and {} th sentence is generated \n'.format(ix,ix + 1))
        #     print('The {} and {} th sentence is generated \n'.format(ix,ix + 1))
        #     f_best.write('{}-{} Best Bleu: {} \n '.format(ix,ix+1,mydict['best_bleu_sen']))
        #     print('{}-{} Best Bleu: {} \n '.format(ix,ix+1,mydict['best_bleu_sen']))
        #     f_best.write('Bleu is : {} \n'.format(mydict['best_bleu']))
        #     print('Bleu is : {} \n'.format(mydict['best_bleu']))
        #     f_best.write('{}-{} Best Score: {} \n '.format(ix,ix+1,mydict['best_score_sen']))
        #     f_best.write('Score is : {} \n'.format(mydict['best_score_sen']))
        #     print('Score is : {} \n'.format(mydict['best_score_sen']))

def Batch_generate(modle_f,DA,i2w):
    gen_dict = defaultdict(list)
    result = []
    hidden, C = modle_f.init(opt.batch_size)
    hidden_tmp, C_tmp, DA_tmp = hidden, C, DA
    input_f = []
    for _ in range(opt.batch_size):
        tmp = [0] * opt.VOC_SIZE
        tmp[opt.START_ix] = 1
        input_f.append(tmp)
    input_f = Variable(torch.FloatTensor(input_f))
    input_f = input_f.unsqueeze(0)
    # input_f = input_f.permute(1,0,2)
    forward_score = defaultdict(lambda: 1)
    for gen_count in range(opt.max_length):
        output,hidden_tmp, C_tmp, DA_tmp, _ = \
            modle_f(input_f, hidden_tmp, C_tmp,DA_tmp, batch_size=opt.batch_size)
        output = torch.nn.functional.softmax(output,dim = 1)
        input_idx = torch.multinomial(output,1).view(-1).data
        # in this moment we calculate the forward score
        for score_ix,item in enumerate(input_idx):
            if item != opt.END_ix:
                forward_score[score_ix] *= output.data[score_ix][item]
            else:
                break
        # _, input_idx = torch.max(output, 1)
        result.append(input_idx.numpy().tolist())
        input_list = []
        for _ in range(opt.batch_size):
            input_list.append([0] * opt.VOC_SIZE)
        for item_ix,item in enumerate(input_idx):
            index_store = item
            try:
                input_list[item_ix][index_store] = 1
            except:
                None
        input_f = Variable(torch.FloatTensor(input_list)).unsqueeze(0)
    result = torch.IntTensor(result)
    for column_idx in range(opt.batch_size):
        try:
            str_tmp = map(lambda x:i2w[x],result[:,column_idx])
        except:
            None
        str_tmp = list(str_tmp)
        gen_dict[column_idx].extend(str_tmp)
    # for key in gen_dict.keys():
        # print(' '.join(gen_dict[key][0]))
    f_result = open('batch_result_test.pkl','wb')
    try:
        pickle.dump(result,f_result)
        print('the test file:batch_result_test.pkl is done\n')
    except:
        print('error! the file is not created!\n')
    return gen_dict,result,forward_score    # here result represent the batch generation of sentence


def Batch_score(modle_b,DA,result):
    sqlen,batch_size = result.shape
    result = result.numpy()
    # find out the index of END in every column
    End_ix_dict = defaultdict(int)
    Target_dict = defaultdict(list)
    for column_ix in range(opt.batch_size):
        try:
            forward_end_index = result[:, column_ix].tolist().index(opt.END_ix)
            end_ix = opt.max_length - 1 - forward_end_index
        except:
            forward_end_index = opt.max_length - 1
            end_ix = 0
        Target_dict[column_ix] =  result[:,column_ix].tolist()[:forward_end_index]
        End_ix_dict[column_ix] = end_ix  # column_ix means the order of data in a batch_size

    result = result.tolist()
    result.reverse()
    targ_sen = torch.FloatTensor(result).contiguous().view(-1)
    b_input = []

    for _ in range(opt.batch_size):
        tmp = [0] * opt.VOC_SIZE
        tmp[opt.END_ix] = 1
        b_input.append(tmp)
    for item in targ_sen.int()[:-opt.batch_size]:
        tmp = [0] * opt.VOC_SIZE
        tmp[item] = 1
        b_input.append(tmp)
    b_input = Variable(torch.FloatTensor(b_input))
    b_input = b_input.contiguous().view(opt.max_length,opt.batch_size,-1)

    hidden, C = modle_b.init(batch_size=opt.batch_size)

    backward_score = defaultdict(int)
    for item in range(opt.batch_size):
        backward_score[item] = 1
    for sqlen_ix in range(opt.max_length):
        b_input_batch = b_input[sqlen_ix].unsqueeze(0)
        output, hidden, C, DA_ori, _ = modle_b(b_input_batch, hidden, C, DA, batch_size=opt.batch_size)
        output = torch.nn.functional.softmax(output,dim = 1)
        create_DA_list = []
        for create_da_ix in range(opt.batch_size):
            if sqlen_ix < End_ix_dict[create_da_ix] + 1:
                create_DA_list.append(DA.data[create_da_ix].numpy().tolist())
            else:
                create_DA_list.append(DA_ori.data[create_da_ix].numpy().tolist())
                count = sqlen_ix - End_ix_dict[create_da_ix]
                try:
                    voc_index = Target_dict[create_da_ix][ count * -1]
                except:
                    None
                prob_score = output[create_da_ix][voc_index]
                backward_score[create_da_ix] *= prob_score.data[0]
        DA = Variable(torch.FloatTensor(create_DA_list))
    return backward_score


def Batch_error(slot_inform,gendict): # slot_inform is from the DataLoad
    f = open('./dataset/i2slot_union.pkl','rb')
    i2slot = pickle.load(f)
    slot_inform = slot_inform.int().numpy().tolist()
    slot_set_dic  = defaultdict(set)  # with the usage of slotlist,wo caculate the err score
    slot_err_dic = defaultdict(int)
    for batch_ix in range(opt.batch_size):
        slot_1, slot_2 = slot_inform[batch_ix]
        slot_1 = [i2slot[i] for i in slot_1 if i != -1]
        slot_2 = [i2slot[i] for i in slot_2 if i != -1]
        slot_set_1 = slot_feature_batch(slot_1)
        slot_set_2 = slot_feature_batch(slot_2)
        slot_set_1 = set(slot_set_1)
        slot_set_2 = set(slot_set_2)
        # slot_set = slot_set_1 | slot_set_2
        slot_set = (slot_set_1,slot_set_2)
        slot_set_dic[batch_ix] = slot_set
    print('the slot inform is {}'.format(slot_set_dic[0]))
    for batch_ix in gendict.keys():
        slot_err_dic[batch_ix] = 0
        try:
            split_index = gendict[batch_ix].index('START')
        except:
            split_index = 0
        end_index = gendict[batch_ix].index('END')
        gensen_left,gensen_right = gendict[batch_ix][:split_index],gendict[batch_ix][split_index:end_index]
        if len(gensen_left) > 0:
            for cmp_item in slot_set_dic[batch_ix][0]: # examine the missing of slot value
                if isinstance(cmp_item,str) and cmp_item not in gensen_left:
                    slot_err_dic[batch_ix] += 1
                elif isinstance(cmp_item,tuple):
                    flag = 0   # when it chages to 1 ,menas there exists one item inclued in  the targer sentence
                    for inner_item in cmp_item:
                        if inner_item in gendict[batch_ix]:
                            flag =1
                    if flag == 0:
                        slot_err_dic[batch_ix] +=1
        for cmp_item in slot_set_dic[batch_ix][1]:  # examine the missing of slot value
            if isinstance(cmp_item, str) and cmp_item not in gensen_right:
                slot_err_dic[batch_ix] += 1
            elif isinstance(cmp_item, tuple):
                flag = 0  # when it chages to 1 ,menas there exists one item inclued in  the targer sentence
                for inner_item in cmp_item:
                    if inner_item in gendict[batch_ix]:
                        flag = 1
                if flag == 0:
                    slot_err_dic[batch_ix] += 1
    print(slot_err_dic)
    return slot_err_dic











def Batch_generate_test():
    modle_f = models.SCLSTM(opt.input_size, opt.hidden_size, opt.DA_size)
    modle_f.load_state_dict(torch.load('params_batch_modle_f.pkl'))
    modle_b = models.SCLSTM(opt.input_size, opt.hidden_size, opt.DA_size)
    modle_b.load_state_dict(torch.load('params_batch_modle_b.pkl'))
    test = LoadData(data='./dataset/test_union_convert.pkl')
    f = open('./dataset/i2w_union.pkl','rb')
    i2w = pickle.load(f)
    data_batch = DataLoader(test, batch_size=opt.batch_size, shuffle=False, drop_last=True)
    for ix, (slot_inform, DA, input_f, target_f, _, _, unpacknum) in enumerate(data_batch):
        if ix == 2:
            break
        rerank_score = [-88888] * opt.batch_size # a small enough number ensure smaller than initial value
        rerank_list = defaultdict(list)
        DA = Variable(DA)
        str_list = []
        for target_str_idex in range(opt.batch_size):
            str = ' '
            for tar_item in target_f[target_str_idex]:
                str += i2w[int(tar_item)] + ' '
            str_list.append(str)
            print('the target sentence is \n {}\n'.format(str))

        for gen_time in range(opt.gen_time):
            gen_dict, result, forward_score = Batch_generate(modle_f, DA, i2w)
            # result = pickle.load(f)
            backward_score = Batch_score(modle_b, DA, result)
            slot_err_dic = Batch_error(slot_inform, gen_dict)
            print('{}th data ,the {}th gen time \n'.format(ix,gen_time))
            print('genslot_error 0  is {}'.format(slot_err_dic[0]))
            print('gen_dic 0 is {}\n'.format(gen_dict[0]))
            print('forward 0 is {}\n'.format(forward_score[0]))
            print('backward 0 is {}\n'.format(backward_score[0]))
            print('****************************the total score is {} \n'.format(forward_score[0] + backward_score[0] +opt.lamda * slot_err_dic[0] * -1))

            for re_ix in range(opt.batch_size):
                f_item,b_item,err_cost = forward_score[re_ix],backward_score[re_ix],slot_err_dic[re_ix]
                if f_item + b_item + opt.lamda * err_cost * -1> rerank_score[re_ix]:
                    rerank_score[re_ix] = f_item + b_item + opt.lamda * err_cost * -1
                    rerank_list[re_ix] = gen_dict[re_ix]
        for batch_output_index in range(opt.batch_size):
            print('the target is \n')
            print(str_list[batch_output_index])
            print('the generated is \n')
            print( ' '.join(rerank_list[batch_output_index]))
            print('-'*100)




if __name__ == '__main__':
    fire.Fire()
    # Batch_generate_test()
    # MutiProcessBleu()
    # train_v2_f()
    # train_v2_b()
    # BLEU_EV()













