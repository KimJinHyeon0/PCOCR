import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
from Layers.graph_embedding import *
import random

from utils import EarlyStopMonitor
from Layers.graph_embedding import *

random.seed(222)
np.random.seed(222)
torch.manual_seed(222)

def load_data():
    print('Loading Data')
    g_df = pd.read_csv(f'./processed/{SUBREDDIT}_structure.csv', index_col=0)
    n_feat = np.load(f'./processed/{SUBREDDIT}_node_feat_{WORD_EMBEDDING}.npy')

    g_num = g_df.g_num.values
    g_ts = g_df.g_ts.values
    src_l = g_df.u.values
    dst_l = g_df.i.values
    ts_l = g_df.ts.values
    label_l = g_df.label.values

    train_time = 3283200  # '2018-02-08 00:00:00' - '2018-01-01 00:00:00'
    test_time = 3888000  # '2018-02-15 00:00:00' - '2018-01-01 00:00:00'

    train_flag = (g_ts < train_time)
    valid_flag = (g_ts >= train_time) & (g_ts < test_time)
    test_flag = (g_ts >= test_time)

    train_g_num = g_num[train_flag]
    train_src_l = src_l[train_flag]
    train_dst_l = dst_l[train_flag]
    train_ts_l = ts_l[train_flag]
    train_label_l = label_l[train_flag]

    val_src_l = src_l[valid_flag]
    val_dst_l = dst_l[valid_flag]
    val_label_l = label_l[valid_flag]
    val_ts_l = ts_l[valid_flag]
    val_label_l = label_l[valid_flag]

    test_src_l = src_l[test_flag]
    test_dst_l = dst_l[test_flag]
    test_label_l = label_l[test_flag]
    test_ts_l = ts_l[test_flag]
    test_label_l = label_l[test_flag]

    if TRAINING_METHOD == 'SELECTIVE':
        time_cut_flag = (train_ts_l < TIME_CUT)

        train_g_num = train_g_num[time_cut_flag]
        train_src_l = train_src_l[time_cut_flag]
        train_dst_l = train_dst_l[time_cut_flag]
        train_ts_l = train_ts_l[time_cut_flag]
        train_label_l = train_label_l[time_cut_flag]

        train_index_l = np.arange(len(train_g_num))
        selective_flag = np.zeros((len(train_g_num)), dtype=bool)

        for k in set(train_g_num):
            temp_flag = (train_g_num == k)

            temp_index = train_index_l[temp_flag]
            temp_label = train_label_l[temp_flag]

            if temp_label.mean() >= 0.5:
                major, minor = temp_index[temp_label == 1], temp_index[temp_label == 0]

            else:
                major, minor = temp_index[temp_label == 0], temp_index[temp_label == 1]

            balanced = np.random.choice(major, len(minor), replace=False)
            selective_flag[balanced] = True
            selective_flag[minor] = True

        train_g_num = train_g_num[selective_flag]
        train_src_l = train_src_l[selective_flag]
        train_dst_l = train_dst_l[selective_flag]
        train_ts_l = train_ts_l[selective_flag]
        train_label_l = train_label_l[selective_flag]

    features = torch.FloatTensor(n_feat)

    labels = torch.zeros((features.shape[0], 1), dtype=torch.float)
    for index, label in zip(src_l, label_l):
        labels[index] = label

    train_graph = dgl.graph((train_src_l, train_dst_l),
                            num_nodes=features.shape[0])
    valid_graph = dgl.graph((val_src_l, val_dst_l),
                            num_nodes=features.shape[0])
    test_graph = dgl.graph((test_src_l, test_dst_l),
                           num_nodes=features.shape[0])

    train_graph = dgl.add_self_loop(train_graph)
    valid_graph = dgl.add_self_loop(valid_graph)
    test_graph = dgl.add_self_loop(test_graph)

    train_graph.ndata['features'] = features
    valid_graph.ndata['features'] = features
    test_graph.ndata['features'] = features

    train_graph.ndata['label'] = labels
    valid_graph.ndata['label'] = labels
    test_graph.ndata['label'] = labels

    return (train_graph, train_src_l), (valid_graph, val_src_l), (test_graph, test_src_l), features.shape[1]


def eval_one_epoch(model, dataloader):
    acc, ap, f1, auc = [], [], [], []
    with torch.no_grad():
        model = model.eval()

        for input_nodes, output_nodes, blocks in dataloader:
            block = blocks[0].to(device)
            input_features = block.srcdata['features']
            output_labels = block.dstdata['label'].detach().cpu().numpy()

            pred_prob = model(block, input_features).detach().cpu().numpy()
            pred_labels = pred_prob > 0.5

            acc.append(accuracy_score(output_labels, pred_labels))
            ap.append(average_precision_score(output_labels, pred_prob))
            f1.append(f1_score(output_labels, pred_labels))
            auc.append(roc_auc_score(output_labels, pred_prob))

    return np.mean(acc), np.mean(ap), np.mean(f1), np.mean(auc)


def run():
    train_set, valid_set, test_set, dim = load_data()
    train_g, train_n_ids = train_set
    valid_g, valid_n_ids = valid_set
    test_g, test_n_ids = test_set
    print('Data Loaded')

    train_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
    valid_sampler = test_sampler = train_sampler

    if EMBEDDING_METHOD == 'dgl-GCN':
        model = GCN(dim, dim, 1)
    elif EMBEDDING_METHOD == 'dgl-SAGE':
        model = SAGE(dim, dim, 1)
    elif EMBEDDING_METHOD == 'dgl-GAT':
        model = GAT(dim, dim, 1)

    train_dataloader = dgl.dataloading.pytorch.NodeDataLoader(
        train_g, train_n_ids, train_sampler,
        batch_size=200,
        shuffle=True,
        drop_last=False,
    )

    valid_dataloader = dgl.dataloading.pytorch.NodeDataLoader(
        valid_g, valid_n_ids, valid_sampler,
        batch_size=200,
        shuffle=True,
        drop_last=False,
    )

    test_dataloader = dgl.dataloading.pytorch.NodeDataLoader(
        test_g, test_n_ids, test_sampler,
        batch_size=200,
        shuffle=True,
        drop_last=False,
    )

    lr_criterion = nn.BCELoss()
    early_stopper = EarlyStopMonitor(max_round)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_g.to(device)
    valid_g.to(device)
    test_g.to(device)
    model.to(device)

    for epoch in range(NUM_EPOCH):
        acc, ap, f1, auc, m_loss = [], [], [], [], []
        print(f'Start {epoch} epoch')
        for input_nodes, output_nodes, blocks in train_dataloader:
            block = blocks[0].to(device)
            input_features = block.srcdata['features']
            output_labels = block.dstdata['label']

            pred_prob = model(block, input_features)
            optimizer.zero_grad()
            loss = lr_criterion(pred_prob, output_labels)
            loss.backward()
            optimizer.step()

        # get training results
        with torch.no_grad():
            model = model.eval()
            pred_prob = pred_prob.detach().cpu().numpy()
            pred_labels = pred_prob > 0.5
            output_labels = output_labels.detach().cpu().numpy()
            acc.append(accuracy_score(output_labels, pred_labels))
            ap.append(average_precision_score(output_labels, pred_prob))
            f1.append(f1_score(output_labels, pred_labels))
            m_loss.append(loss.item())
            auc.append(roc_auc_score(output_labels, pred_prob))

        val_acc, val_ap, val_f1, val_auc = eval_one_epoch(model, valid_dataloader)

        print(f'epoch: {epoch}')
        print(f'Epoch mean loss: {np.mean(m_loss)}')
        print(f'train acc: {np.mean(acc)}, val acc: {val_acc}')
        print(f'train ap: {np.mean(ap)}, val ap: {val_ap}')
        print(f'train f1: {np.mean(f1)}, val f1: {val_f1}')
        print(f'train auc: {np.mean(auc)}, val auc: {val_auc}')

        if early_stopper.early_stop_check(val_auc):
            print(f'No improvement over {early_stopper.max_round} epochs, stop training')
            best_epoch = early_stopper.best_epoch
            print(f'Loading the best model at epoch {best_epoch}')
            best_model_path = get_checkpoint_path(best_epoch)
            model.load_state_dict(torch.load(best_model_path))
            print(f'Loaded the best model at epoch {best_epoch} for inference')
            os.remove(best_model_path)
            break
        else:
            if early_stopper.is_best:
                torch.save(model.state_dict(), get_checkpoint_path(epoch))
                print(f'Saved {MODEL_NAME}-{early_stopper.best_epoch}.pth')
                for i in range(epoch):
                    try:
                        os.remove(get_checkpoint_path(i))
                        print(f'Deleted {MODEL_NAME}-{i}.pth')
                    except:
                        continue

    test_acc, test_ap, test_f1, test_auc = eval_one_epoch(model, test_dataloader)
    print(f'test acc: {test_acc}')
    print(f'test ap: {test_ap}')
    print(f'test f1: {test_f1}')
    print(f'test auc: {test_auc}')
    print('Saving model')
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f'{MODEL_NAME} saved')



'''
=====CONFIGS=====

        SUBREDDIT : ['iama', 'showerthoughts']

        TRAINING_METHOD = ['SELECTIVE', 'FULL']

        WORD_EMBEDDING : ['bert-base-uncased',
                         'roberta-base',
                         'deberta-base',
                         'fasttext',
                         'glove',
                         'tf-idf']

        TIME_CUT : int

        max_round : int

        BATCH_SIZE :" [200, 128, 64]

        LEARNING_RATE : [3e-4, 1e-5, 1e-4]
        
        GE : [dgl-GCN, dgl-GAT, dgl-SAGE]

=====CONFIGS=====
'''

SUBREDDIT = 'iama'
TRAINING_METHOD = 'SELECTIVE'
WORD_EMBEDDING = 'bert-base-uncased'
TIME_CUT = 309000
max_round = 10
NUM_EPOCH = 1000
BATCH_SIZE = 200
LEARNING_RATE = 3e-4
EMBEDDING_METHOD = 'dgl-GCN'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = f'{EMBEDDING_METHOD}-{SUBREDDIT}-{WORD_EMBEDDING}-{BATCH_SIZE}-{LEARNING_RATE}'
MODEL_SAVE_PATH = f'./saved_models/pretrained_models/{MODEL_NAME}.pth'
get_checkpoint_path = lambda \
        epoch: f'./saved_checkpoints/{MODEL_NAME}-{epoch}.pth'

if __name__ == "__main__":
    print(f'SUBREDDIT : {SUBREDDIT}')
    print(f'TIME_CUT :{TIME_CUT}')
    print(f'TRAINING_METHOD : {TRAINING_METHOD}')
    print(f'WORD_EMBEDDING : {WORD_EMBEDDING}')
    print(f'EMBEDDING_METHOD = {EMBEDDING_METHOD}')
    run()
