import datetime
import torch
from sys import exit
import pandas as pd
import numpy as np
from NewModel import NewModel, collate, collate_test, collate_train
from dgl import load_graphs
import pickle
from utils import myFloder, load_data
import warnings
import argparse
import os
import sys
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from DGSR_utils import eval_metric, mkdir_if_not_exist, Logger
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='Beauty5', help='data name: sample')
    parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
    parser.add_argument('--hidden_size', type=int, default=50, help='hidden state size')
    parser.add_argument('--time_size', type=int, default=10, help='time encode size')
    parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0001, help='l2 penalty')
    parser.add_argument('--user_update', default='rnn')
    parser.add_argument('--item_update', default='rnn')
    parser.add_argument('--user_long', default='orgat')
    parser.add_argument('--item_long', default='orgat')
    parser.add_argument('--user_short', default='att')
    parser.add_argument('--item_short', default='att')
    parser.add_argument('--feat_drop', type=float, default=0.3, help='drop_out')
    parser.add_argument('--attn_drop', type=float, default=0.3, help='drop_out')
    parser.add_argument('--layer_num', type=int, default=3, help='GNN layer')
    parser.add_argument('--item_max_length', type=int, default=50, help='the max length of item sequence')
    parser.add_argument('--user_max_length', type=int, default=50, help='the max length of use sequence')
    parser.add_argument('--k_hop', type=int, default=3, help='sub-graph size')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--last_item', action='store_true', help='aggreate last item')
    parser.add_argument('--record', action='store_true', default=False, help='record experimental results')
    parser.add_argument('--val', action='store_true', default=False)
    parser.add_argument('--contrast_present', default='0.3')
    parser.add_argument('--is_contrast', action='store_false', default=True)
    parser.add_argument('--model_record', action='store_true', default=False, help='record model')
    parser.add_argument('--temperature', type=float, default=0.3, help='constract temperature')
    parser.add_argument('--lamda', type=float, default=1.0, help='timedisrupt')
    parser.add_argument('--alpha', type=float, default=0.3, help='user constract weight')
    parser.add_argument('--iswindows', action='store_false', default=True)
    parser.add_argument('--time_disrupt', action='store_false', default=True)
    parser.add_argument('--windows', type=int, default=5, help='windows')
    parser.add_argument('--timesteps', type=int, default=20, help='timesteps')
    parser.add_argument('--beta_start', type=float, default=0.0001, help='beta_start')
    parser.add_argument('--beta_end', type=float, default=0.02, help='beta_end')
    parser.add_argument('--diff_dim', type=int, default=64, help='diff dimension')
    parser.add_argument('--diff_loss_weight', type=float, default=0.0001, help='diff_loss_weigth')
    opt = parser.parse_args()
    args, extras = parser.parse_known_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(opt)
    if opt.record:
        log_file = f'results/{opt.data}_ba_{opt.batchSize}_userlong_{opt.user_long}_itemlong_{opt.item_long}_dim_{opt.hidden_size}_UM_{opt.user_max_length}_IM_{opt.item_max_length}_layer_{opt.layer_num}_timesize_{opt.time_size}_contrast-persent_{opt.contrast_present}_temperature_{opt.temperature}_alpha_{opt.alpha}_lambda_{opt.lamda}_windows_{opt.windows}_iswindows_{opt.iswindows}_iscontrast_{opt.is_contrast}'
        mkdir_if_not_exist(log_file)
        sys.stdout = Logger(log_file)
        print(f'Logging to {log_file}')
    if opt.model_record:
        model_file = f'{opt.data}_ba_{opt.batchSize}_G_{opt.gpu}_dim_{opt.hidden_size}_ulong_{opt.user_long}_ilong_{opt.item_long}_US_{opt.user_short}_IS_{opt.item_short}_La_{args.last_item}_UM_{opt.user_max_length}_IM_{opt.item_max_length}_K_{opt.k_hop}_layer_{opt.layer_num}_l2_{opt.l2}_timesize_{opt.time_size}_contrast-persent_{opt.contrast_present}_temperature_{opt.temperature}_alpha_{opt.alpha}_lambda_{opt.lamda}'
    torch.manual_seed(12345)
    torch.cuda.manual_seed(12345)
    data = pd.read_csv('./datasets/' + opt.data + '.csv')
    user = data['user_id'].unique()
    item = data['item_id'].unique()
    user_num = len(user)
    item_num = len(item)
    print('load_data')
    train_root = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/train/'
    test_root = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/test/'
    val_root = f'Newdata/{opt.data}_{opt.item_max_length}_{opt.user_max_length}_{opt.k_hop}/val/'
    train_set = myFloder(train_root, load_graphs)
    print('train_set')
    test_set = myFloder(test_root, load_graphs)
    if opt.val:
        val_set = myFloder(val_root, load_graphs)
    print('load_end')
    print('train number:', train_set.size)
    print('test number:', test_set.size)
    print('user number:', user_num)
    print('item number:', item_num)
    f = open('/root/data1/user/yct/newCode/' + opt.data + '_neg', 'rb')
    data_neg = pickle.load(f)
    train_data = DataLoader(dataset=train_set, batch_size=opt.batchSize, collate_fn=collate, shuffle=True, pin_memory=True, num_workers=12)
    test_data = DataLoader(dataset=test_set, batch_size=opt.batchSize, collate_fn=lambda x: collate_test(x, data_neg), pin_memory=True, num_workers=8)
    if opt.val:
        val_data = DataLoader(dataset=val_set, batch_size=opt.batchSize, collate_fn=lambda x: collate_test(x, data_neg), pin_memory=True, num_workers=2)
    model = NewModel(user_num=user_num, item_num=item_num, input_dim=opt.hidden_size, time_dim=opt.time_size, item_max_length=opt.item_max_length, user_max_length=opt.user_max_length, feat_drop=opt.feat_drop, attn_drop=opt.attn_drop, user_long=opt.user_long, user_short=opt.user_short, item_long=opt.item_long, item_short=opt.item_short, user_update=opt.user_update, item_update=opt.item_update, last_item=opt.last_item, device=device, layer_num=opt.layer_num, temperature=opt.temperature, alpha=opt.alpha, lamda=opt.lamda, windows=opt.windows, iswindows=opt.iswindows, timesteps=opt.timesteps, beta_start=opt.beta_start, beta_end=opt.beta_end, beta_sche='linear', diff_dim=opt.diff_dim).cuda()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
    loss_func = nn.CrossEntropyLoss()
    best_result = [0, 0, 0, 0, 0, 0]
    best_epoch = [0, 0, 0, 0, 0, 0]
    stop_num = 0
    for epoch in range(opt.epoch):
        stop = True
        epoch_loss = 0
        iter = 0
        print('start training: ', datetime.datetime.now())
        model.train()
        for user, batch_graph, label, last_item in train_data:
            iter += 1
            score, contrast_loss, diff_loss, item_embedding, user_embedding = model(batch_graph.to(device), user.to(device), last_item.to(device), is_training=True)
            loss = loss_func(score, label.to(device)) + float(opt.contrast_present) * contrast_loss + float(opt.diff_loss_weight) * diff_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if iter % 400 == 0:
                print('Iter {}, loss {:.4f}'.format(iter, epoch_loss / iter), datetime.datetime.now())
        epoch_loss /= iter
        model.eval()
        print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss), '=============================================')
        if opt.val:
            print('start validation: ', datetime.datetime.now())
            val_loss_all, top_val = ([], [])
            with torch.no_grad:
                for user, batch_graph, label, last_item, neg_tar in val_data:
                    score, top = model(batch_graph.to(device), user.to(device), last_item.to(device), neg_tar=torch.cat([label.unsqueeze(1), neg_tar], -1).to(device), is_training=False)
                    val_loss = loss_func(score, label.cuda())
                    val_loss_all.append(val_loss.append(val_loss.item()))
                    top_val.append(top.detach().cpu().numpy())
                recall5, recall10, recall20, ndgg5, ndgg10, ndgg20 = eval_metric(top_val)
                print('train_loss:%.4f\tval_loss:%.4f\tRecall@5:%.4f\tRecall@10:%.4f\tRecall@20:%.4f\tNDGG@5:%.4f\tNDGG10@10:%.4f\tNDGG@20:%.4f' % (epoch_loss, np.mean(val_loss_all), recall5, recall10, recall20, ndgg5, ndgg10, ndgg20))
        print('start predicting: ', datetime.datetime.now())
        all_top, all_label, all_length = ([], [], [])
        iter = 0
        all_loss = []
        with torch.no_grad():
            for user, batch_graph, label, last_item, neg_tar in test_data:
                iter += 1
                score, top = model(batch_graph.to(device), user.to(device), last_item.to(device), neg_tar=torch.cat([label.unsqueeze(1), neg_tar], -1).to(device), is_training=False)
                test_loss = loss_func(score, label.cuda())
                all_loss.append(test_loss.item())
                all_top.append(top.detach().cpu().numpy())
                all_label.append(label.numpy())
                if iter % 200 == 0:
                    print('Iter {}, test_loss {:.4f}'.format(iter, np.mean(all_loss)), datetime.datetime.now())
            recall5, recall10, recall20, ndgg5, ndgg10, ndgg20 = eval_metric(all_top)
            if recall5 > best_result[0]:
                best_result[0] = recall5
                best_epoch[0] = epoch
                stop = False
            if recall10 > best_result[1]:
                if opt.model_record:
                    torch.save(model.state_dict(), 'save_models/' + model_file + '.pkl')
                best_result[1] = recall10
                best_epoch[1] = epoch
                stop = False
            if recall20 > best_result[2]:
                best_result[2] = recall20
                best_epoch[2] = epoch
                stop = False
            if ndgg5 > best_result[3]:
                best_result[3] = ndgg5
                best_epoch[3] = epoch
                stop = False
            if ndgg10 > best_result[4]:
                best_result[4] = ndgg10
                best_epoch[4] = epoch
                stop = False
            if ndgg20 > best_result[5]:
                best_result[5] = ndgg20
                best_epoch[5] = epoch
                stop = False
            if stop:
                stop_num += 1
            else:
                stop_num = 0
            print('train_loss:%.4f\ttest_loss:%.4f\tRecall@5:%.4f\tRecall@10:%.4f\tRecall@20:%.4f\tNDGG@5:%.4f\tNDGG10@10:%.4f\tNDGG@20:%.4f\tEpoch:%d,%d,%d,%d,%d,%d' % (epoch_loss, np.mean(all_loss), best_result[0], best_result[1], best_result[2], best_result[3], best_result[4], best_result[5], best_epoch[0], best_epoch[1], best_epoch[2], best_epoch[3], best_epoch[4], best_epoch[5]))
            '保存embedding'
