import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import time

from torch import optim, nn
from tqdm import tqdm

from data_loader_tools import *
from model.gwas_transformer_base_model import clsDNA, clsDNA_no_pos
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import random
import torch
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(1)




parser = argparse.ArgumentParser(description='PyTorch snp-seek-base-transformer Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--opt', default="adamW") # sgd adam adamW
parser.add_argument('--net', default='clsDNA')
parser.add_argument('--n_epochs', type=int, default='200')
parser.add_argument('--datasetX', type=int, default='0')
parser.add_argument('--fold', type=int, default='0')


args = parser.parse_args()

seed =1
datasetX = args.datasetX


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print("now using device:" ,device)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
min_loss = -1
high_auc = -1


import pandas as pd
# indices = pd.read_csv('./data/n_data/fold_{}/split_chromosomes/chromosome_indices.tsv'.format(args.fold), sep='\t')
print("========================>finished loading")
import pandas as pd
dataset_name = 'adni_data'
indices = pd.read_csv('./data/{}/fold_{}/split_chromosomes/chromosome_indices.tsv'.format(dataset_name, args.fold), sep='\t')

list(indices.SNP_Count)


print('==> Preparing train data..')
print("n_data_t")
all_val_dataset= None
all_test_dataset= None

from model.gwas_transformer_base_model import get_pos_embedding
chr_pos = np.loadtxt('./data/{}/fold_{}/snp_content_test_snp.txt'.format(dataset_name,args.fold), delimiter=' ', dtype=str)
t_data = np.loadtxt('./data/{}/fold_{}/genetype_test_snp.txt'.format(dataset_name, args.fold), delimiter=' ', dtype=str)


test_x_data, test_y_data = data_xy_transform_to_tensor_all(t_data)
test_x_data = test_x_data.to(torch.float32)

pos_data0 = data_pos_transform_to_tensor_all(chr_pos)
pos_data0 = pos_data0.to(device)
pos_data0 = get_pos_embedding(pos_data0, 128, device='cpu')
pos_data = pos_data0.unsqueeze(0)
pos_data = pos_data.expand(test_x_data.size(0), -1, -1)
test_x_data = torch.cat((test_x_data, pos_data), dim=2)

val_loader, test_loader = createValTestDataset(test_x_data, test_y_data, batch_size=8, seed=1, val_ratio=0.5)
# test_loader, val_loader= createValTestDataset(test_x_data, test_y_data, batch_size=8, seed=1, val_ratio=0.5)


print('==> Preparing train data..')
all_train_dataset = None
# all_val_dataset= None
train_data = np.loadtxt('./data/{}/fold_{}/genetype_train_snp.txt'.format(dataset_name, args.fold), delimiter=' ', dtype=str)
chr_pos = np.loadtxt('./data/{}/fold_{}/snp_content_train_snp.txt'.format(dataset_name, args.fold), delimiter=' ', dtype=str)

print("Now training the fold：", args.fold)
print("Now datasetx:" ,datasetX)

x_data, y_data = data_xy_transform_to_tensor_all(train_data)
x_data = x_data.to(torch.float32)

pos_data0 = data_pos_transform_to_tensor_all(chr_pos)
pos_data0 = pos_data0.to(device)
pos_data0 = get_pos_embedding(pos_data0, 128, device='cpu')
pos_data = pos_data0.unsqueeze(0)
pos_data = pos_data.expand(x_data.size(0), -1, -1)
x_data = torch.cat((x_data, pos_data), dim=2)

train_dataset = TensorDataset(x_data, y_data)
train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)


if args.net =="clsDNA":
    # transformer for DNA
    net = clsDNA(
        vocab = 40,
        seq_len= 2277,
        num_classes = 2,
        dim = int(128),
        depth = 4,
        heads = 4,
        mlp_dim = 256,
        dropout = 0.4,
        emb_dropout = 0.4,
        get_last_feature= False,
        pool = 'mean',
        use_auto_pos = False,
        snp_count_list = None,
    )
net = net.to(device)
use_pretrained = True
if use_pretrained:
    checkpoint_path = './checkpoint/new_ADNI_pre_model/{}/new_fold{}/'.format(dataset_name, args.fold)
    checkpoint_path = checkpoint_path + args.net + 'ckpt.t_best_model_chr_pre'

    checkpoint = torch.load(checkpoint_path, map_location=device)
    print("loading all model...")
    net.load_state_dict(checkpoint['model'], strict=False)

criterion = nn.CrossEntropyLoss()


if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)
elif args.opt == "adamW":
    optimizer = optim.AdamW(net.parameters(), lr=args.lr)

# use cosine scheduling
use_lr_low = True
if use_lr_low:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs, last_epoch=-1)
##### Training
# Mixing accuracy
scaler = torch.cuda.amp.GradScaler(enabled=False)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    prob_all_AUC = []
    label_all = []
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc="Training Progress")):
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.cuda.amp.autocast(enabled=False):
            outputs, _ = net(inputs[: ,: ,:40], inputs[: ,: ,40:])
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad() # step cosine scheduling

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()


        prob = torch.softmax(outputs, dim=1).cpu().detach().numpy()
        prob_all_AUC.extend(prob[:, 1])
        labels = targets.cpu()
        label_all.extend(labels)

    # Calculate AUC
    AUC = roc_auc_score(label_all, prob_all_AUC)
    print("train AUC:{:.4f}".format(AUC))
    print("Training set accuracy:", 100. * correct / total)
    print("Training set Loss: ", train_loss / (batch_idx + 1))
    return train_loss / (batch_idx + 1)

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


##### Validation
def Val():
    global best_acc
    global min_loss
    global high_auc
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    prob_all = []
    prob_all_AUC = []
    label_all = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = net(inputs[: ,: ,:40], inputs[: ,: ,40:])

            # Calculate F1
            prob = torch.softmax(outputs, dim=1).cpu().numpy()
            labels = targets.cpu()
            # prob = prob.numpy()  # Convert prob to CPU first, then to numpy, you don't need to convert to CPU first if you train on CPU itself
            prob_all.extend(np.argmax(prob, axis=1))  # Find the maximum index of each row
            label_all.extend(labels)
            # Calculate AUC
            prob_all_AUC.extend(prob[:, 1])  # prob[:,1] return to the second column of each row of the number, according to the parameters of the function can be seen, y_score represents the score of the larger label class, and therefore is the maximum index corresponding to the value of that value, rather than the maximum index value
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / (batch_idx + 1)
    # print F1-score
    f1 = f1_score(label_all, prob_all)
    AUC = roc_auc_score(label_all, prob_all_AUC)
    print("F1-Score:{:.4f}".format(f1))
    print("AUC:{:.4f}".format(AUC))

    if min_loss == -1:
        min_loss = val_loss
    if high_auc == -1:
        high_auc = AUC

    if min_loss >= val_loss:
        min_loss = val_loss
    if high_auc <= AUC:
        print('Saving the best model..')
        state = {"model": net.state_dict(),
                 "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict()}
        checkpoint_path = './checkpoint/new_ADNI_model_tune/{}/new_fold{}/'.format(dataset_name, args.fold)
        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path)
        torch.save(state, checkpoint_path + args.net + 'ckpt.t_best_model')
        high_auc = AUC


    print("val_loss:")
    print(val_loss)
    print("min_loss:")
    print(min_loss)
    print("high_auc:", high_auc)

    # Save checkpoint.
    acc = 100. * correct / total

    return val_loss, acc, f1, AUC

##### test
def test():
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    prob_all = []
    prob_all_AUC = []
    label_all = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = net(inputs[: ,: ,:40], inputs[: ,: ,40:])

            # Calculate F1
            prob = torch.softmax(outputs, dim=1).cpu().numpy()
            labels = targets.cpu()
            # prob = prob.numpy()  # Convert prob to CPU first, then to numpy, you don't need to convert to CPU first if you train on CPU itself
            prob_all.extend(np.argmax(prob, axis=1))  # Find the maximum index of each row

            label_all.extend(labels)
            # Calculate AUC
            prob_all_AUC.extend(prob[:, 1])  # prob[:,1] return to the second column of each row of the number, according to the parameters of the function can be seen, y_score represents the score of the larger label class, and therefore is the maximum index corresponding to the value of that value, rather than the maximum index value

            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # print F1-score
    f1 = f1_score(label_all, prob_all)
    AUC = roc_auc_score(label_all, prob_all_AUC)
    print("F1-Score:{:.4f}".format(f1))
    print("AUC:{:.4f}".format(AUC))

    acc = 100. * correct / total

    print("test acc:", acc)
    print("test loss:", test_loss / (batch_idx + 1))

    return acc, f1, AUC

list_loss = []
list_acc = []
list_trainloss = []
list_f1 = []
list_AUC = []


net = net.to(device)
for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloss = train(epoch)
    val_loss, acc, f1, AUC = Val()
    test()

    if use_lr_low:
        scheduler.step()  # step cosine scheduling
        # scheduler.step(AUC)  # step cosine scheduling

    list_loss.append(val_loss)
    list_acc.append(acc)
    list_trainloss.append(trainloss)
    list_f1.append(f1)
    list_AUC.append(AUC)


print("-----------------------------------------")
print("training finished!")
print("-----------------------------------------")
# Loading the Best Checkpoints file
print("loading best auc model...")
checkpoint_path = './checkpoint/new_ADNI_model_tune/{}/new_fold{}/'.format(dataset_name, args.fold)
checkpoint_path = checkpoint_path + args.net + 'ckpt.t_best_model'
checkpoint = torch.load(checkpoint_path)
net.load_state_dict(checkpoint['model'])
test_acc, test_f1, test_auc = test()
checkpoint_path = './checkpoint/new_ADNI_model_tune/{}/new_fold{}/'.format(dataset_name, args.fold)
with open(checkpoint_path + "test_auc.txt".format(args.fold, datasetX), "w") as file:
    file.write(str(test_auc))
    file.write("\n")
with open(checkpoint_path + "val_auc.txt".format(args.fold, datasetX), "w") as file:
    file.write(str(high_auc))
    file.write("\n")

# import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 8))
# plt.subplot(2, 2, 1)
# plt.plot(list_trainloss, label='Train Loss')
# plt.plot(list_loss, label='Val Loss')
# plt.legend()

# plt.subplot(2, 2, 2)
# plt.plot(list_acc, label='Val Acc')
# plt.legend()

# plt.subplot(2, 2, 3)
# plt.plot(list_f1, label='F1_score')
# plt.legend()

# plt.subplot(2, 2, 4)
# plt.plot(list_AUC, label='AUC')
# plt.legend()

# plt.savefig(checkpoint_path+'output.png')
# plt.show()