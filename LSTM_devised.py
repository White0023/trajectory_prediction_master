# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division

import time
import math
import torch.nn as nn
from torch import optim
from dataPrepare import *
from data.loader import data_loader
from args import *
from utils import *
import utils
from scipy.stats import entropy

torch.manual_seed(0)

MAX_LENGTH = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('ok')
# Training_generator, Test, Valid, WholeSet = get_dataloader()
args = get_args()


train_path = get_dset_path(args.dataset_name,"train")
test_path = get_dset_path(args.dataset_name,"test")

train_Set, Training_generator = data_loader(args, path=train_path) #
test_set, test_data_loader = data_loader(args, path=test_path)

class NNPred(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, batch_size, dropout=0.05):
        super(NNPred, self).__init__()

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = 2

        self.in2lstm = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=self.num_layers, bidirectional=False, batch_first=True,
                            dropout=0.1)
        self.in2bilstm = nn.Linear(input_size, hidden_size)
        self.bilstm = nn.LSTM(hidden_size, hidden_size // 2, num_layers=self.num_layers, bidirectional=True,
                              batch_first=True, dropout=0.1)

        self.fc0 = nn.Linear(256, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, output_size)
        # 加一个
        self.fc3 = nn.Linear(8, 8)
        self.in2out = nn.Linear(input_size, 64)
        self.tanh = nn.Tanh()

    def forward(self, input):
        # input = tensor shape[batchsize, len, num_features]

        bilstm_out, _ = self.lstm(self.in2bilstm(input))

        lstm_out, _ = self.lstm(self.in2lstm(input))
        out = self.tanh(self.fc0(lstm_out + bilstm_out))
        out = self.tanh(self.fc1(out))
        out = out + self.in2out(input)
        output = self.fc2(out)  # range [0 -> 1 ]
        return output


def trainIters(encoder, n_iters, print_every=1000, plot_every=1, learning_rate=0.001):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    # criterion = nn.SmoothL1Loss()
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction='sum')
    pltcount = 0
    prtcount = 0
    cp = 0
    losses = utils.AverageMeter("Loss", ":.6f")  #来自于train()函数
    for iter in range(1, n_iters + 1):
        if iter % 50 == 1:
            cp = cp + 1
            torch.save(encoder.state_dict(), str(cp) + 'checkpoint.pth.tar')

        for batch_idx, batch in enumerate(Training_generator):
            encoder.zero_grad()

            # local_batch = local_batch.to(device)
            # local_labels = local_labels.to(device)

            batch_size = batch[0].shape[1]

            batch = [tensor for tensor in batch]

            # batch_array = np.array(batch)

            (
                obs_traj,
                pred_traj_gt,
                obs_traj_rel,  # 8,427,2
                pred_traj_gt_rel,
                non_linear_ped,
                loss_mask,
                seq_start_end,
            ) = batch

            statue = np.zeros((batch_size, batch_size))

            obs_traj_rel = obs_traj_rel.reshape(obs_traj_rel.shape[1], obs_traj_rel.shape[0],
                                                obs_traj_rel.shape[2])  # 427,8,2
            for i in range(len(obs_traj_rel)):
                for j in range(i):
                    vector1 = obs_traj_rel[i] + 1e-2
                    vector2 = obs_traj_rel[j] + 1e-2

                    k1_dicvergence = entropy(vector1.flatten(), vector2.flatten())
                    statue[i][j] = k1_dicvergence
                    statue[j][i] = k1_dicvergence
            # 运行以上代码之后，可以得到不同轨迹之间的KL散度，得到一个二维的散度矩阵，如维度为[427, 427]
            # 那么接下来需要找出该矩阵中散度值最小时所对应的两个行人轨迹

            center_idx = np.where(np.min(statue) == statue)  # 找出矩阵中的最小值所对应的索引
            idx_a, idx_b = center_idx

            # 出现nan和inf值(等会再解决),先假设已经找到对应的索引idx_a,idx_b,接下来计算变换矩阵
            T_matrix = NNPred.fc3()  # 首先在网络中定义线性变化为fc3,维度为8×8,这样就与网络同时进行训练,tensor
            # obs_traj_rel的shape为[8,2],需要先转换为[2,8]才能使用T_matrix,但是如果直接使用维度为2×2的变换矩阵是否也可行?

            # 求scale之后batch的轨迹序列,除了idx_a对应的轨迹保留,其它的轨迹需要乘上变换矩阵T_matrix
            obs_traj_rel_scaled = []
            for idx_s in range(batch_size):
                if idx_s == idx_a:  #
                    obs_traj_rel[idx_s] = obs_traj_rel[idx_a]
                    obs_traj_rel_scaled.append(obs_traj_rel[idx_s])
                else:
                    obs_traj_rel[idx_s] = obs_traj_rel[idx_s].reshape(obs_traj_rel[idx_s].shape[1],
                                                                      obs_traj_rel[idx_s].shape[0])  # 2,8
                    obs_traj_rel[idx_s] = np.dot(np.array(obs_traj_rel[idx_s]), np.array(T_matrix))  # [2,8]×[8,8]
                    obs_traj_rel_scaled.append(obs_traj_rel[idx_s])  # [427, 2, 8]
            obs_traj_rel_scaled = torch.tensor(obs_traj_rel_scaled)  # 得到可以送入到model的input tensor
            obs_traj_rel_scaled = obs_traj_rel_scaled.reshape(obs_traj_rel_scaled.shape[2],
                                                              obs_traj_rel_scaled.shape[0],
                                                              obs_traj_rel_scaled.shape[1])
            loss = torch.zeros(1).to(pred_traj_gt)
            l2_loss_rel = []
            loss_mask = loss_mask[:, args.obs_len:]
            model_input = obs_traj_rel_scaled
            pred_traj_fake_rel = encoder.forward(  # pred_traj_fake_rel:8,43,2
                model_input, obs_traj, seq_start_end, 1) #training_step  # model_input:8,43,2


            # start_revers scaling
            T_matrix = np.array(T_matrix)
            T_matrix_inverse = np.linalg.inv(T_matrix)
            pred_traj_fake_rel_scaled = []
            pred_traj_fake_rel = pred_traj_fake_rel.reshape(pred_traj_fake_rel.shape[1],
                                                            pred_traj_fake_rel.shape[0],
                                                            pred_traj_fake_rel.shape[2])  # 427,8,2
            for idx_is in range(batch_size):
                if idx_is == idx_a:  #
                    pred_traj_fake_rel[idx_is] = pred_traj_fake_rel[idx_is]
                    pred_traj_fake_rel_scaled.append(pred_traj_fake_rel[idx_is])
                else:
                    pred_traj_fake_rel[idx_is] = pred_traj_fake_rel[idx_is].reshape(pred_traj_fake_rel[idx_is].shape[1],
                                                                                    pred_traj_fake_rel[idx_is].shape[
                                                                                        0])  # 2,8
                    pred_traj_fake_rel[idx_is] = np.dot(np.array(pred_traj_fake_rel[idx_is]),
                                                        T_matrix_inverse)  # [2,8]×[8,8]
                    pred_traj_fake_rel_scaled.append(pred_traj_fake_rel[idx_is])  # [427, 2, 8]
            obs_traj_rel_scaled = torch.tensor(obs_traj_rel_scaled)  # 得到可以送入到model的input tensor
            pred_traj_fake_rel_scaled = obs_traj_rel_scaled.reshape(obs_traj_rel_scaled.shape[2],
                                                                    obs_traj_rel_scaled.shape[0],
                                                                    obs_traj_rel_scaled.shape[1])  # 8,427,2
            # end_reverse scaling

            #损失函数
            l2_loss_rel.append(
                l2_loss(pred_traj_fake_rel_scaled, model_input, loss_mask, mode="raw") #loss_mask:43,12
            )
            l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
            l2_loss_rel = torch.stack(l2_loss_rel, dim=1)
            for start, end in seq_start_end.data:
                _l2_loss_rel = torch.narrow(l2_loss_rel, 0, start, end - start)
                _l2_loss_rel = torch.sum(_l2_loss_rel, dim=0)  # [20]
                _l2_loss_rel = torch.min(_l2_loss_rel) / (
                        (pred_traj_fake_rel_scaled.shape[0]) * (end - start)
                )
                l2_loss_sum_rel += _l2_loss_rel

            loss += l2_loss_sum_rel
            losses.update(loss.item(), obs_traj.shape[1])
            loss.backward()
            encoder_optimizer.step()


        # LSTM.py脚本的训练步骤
        # for local_batch, local_labels in Training_generator:
        #     if local_batch.shape[0] != BatchSize:
        #         continue
        #     pltcount = pltcount + 1
        #     prtcount = prtcount + 1
        #     encoder.zero_grad()
        #
        #     local_batch = local_batch.to(device)
        #     local_labels = local_labels.to(device)
    #         local_batch = pred_traj_fake_rel_scaled
    #
    #
    #         predY = encoder(local_batch)
    #         loss = criterion(predY[:, -30:, 2:4], local_labels[:, -30:, 2:4]).to(device)
    #         loss.backward()
    #         encoder_optimizer.step()
    #
    #         ls = loss.detach().item()
    #         print_loss_total += ls
    #         plot_loss_total += ls
    #     if iter % print_every == 0:
    #         print_loss_avg = print_loss_total / prtcount
    #         print_loss_total = 0
    #         prtcount = 0
    #         print('%s (%d %d%%) %f' % (timeSince(start, iter / n_iters),
    #                                    iter, iter / n_iters * 100, print_loss_avg))
    #     if iter % plot_every == 0:
    #         plot_loss_avg = plot_loss_total / pltcount
    #         plot_losses.append(plot_loss_avg)
    #         plot_loss_total = 0
    #         pltcount = 0
    # return plot_losses


# def Eval_net(encoder):
#     count = 0
#     for local_batch, local_labels in Training_generator:
#         if local_batch.shape[0] != BatchSize:
#             continue
#         count = count + 1
#         local_batch = local_batch.to(device)
#         local_labels = local_labels.to(device)
#         predY = encoder(local_batch)
#         print(WholeSet.std.repeat(BatchSize, 100, 1).shape)
#         std = WholeSet.std.repeat(BatchSize, 100, 1)
#         std = std[:, :, :4].to(device)
#         mn = WholeSet.mn.repeat(BatchSize, 100, 1)
#         mn = mn[:, :, :4].to(device)
#         rg = WholeSet.range.repeat(BatchSize, 100, 1)
#         rg = rg[:, :, :4].to(device)
#         predY = (predY * (rg * std) + mn).detach().cpu()
#         pY = np.array(predY)
#         pY = scipy.signal.savgol_filter(pY, window_length=5, polyorder=2, axis=1)
#         local_labels = (local_labels * (rg * std) + mn).detach().cpu()
#         Y = np.array(local_labels)
#         pY[:, :-30, :] = Y[:, :-30, :]
#         rst_xy = calcu_XY(pY, Y)
#         for i in range(10):
#             plt.figure(i)
#             plt.xlim(0, 80)
#             plt.ylim(0, 2000)
#             plt.plot(pY[i, :, 2], pY[i, :, 3], 'r')
#             plt.plot(Y[i, :, 2], Y[i, :, 3], 'g')
#             plt.plot(rst_xy[i, :, 0], rst_xy[i, :, 1], 'b')
#         plt.show()


def calcu_XY(predY, labelY):
    # input: [batchsize len features]; features:[velx,vely,x,y]
    '''
    deltaY = v0*delta_t + 0.5* a *delta_t^2
    a = (v - v0)/delta_t
    vo
    '''
    vels = predY[:, :, 0:2]
    rst_xy = np.zeros(predY[:, :, 0:2].shape)
    rst_xy[:, :-30, :] = predY[:, :-30, 2:4]
    delta_t = 0.1
    for i in range(30):
        a = (vels[:, -(30 - i), :] - vels[:, -(31 - i), :]) / delta_t
        delta_xy = vels[:, -(30 - i), :] * vels[:, -(30 - i), :] - vels[:, -(31 - i), :] * vels[:, -(31 - i), :]
        delta_xy = delta_xy / (2 * a)
        rst_xy[:, -(30 - i), :] = rst_xy[:, -(31 - i), :] + delta_xy

    t = rst_xy - predY[:, :, 2:4]
    print(t[1, :, :])
    return rst_xy


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.show()


train_iter = iter(Training_generator)
x, y = train_iter.__next__()
print(x.shape)
hidden_size = 256
Prednet = NNPred(x.shape[2], y.shape[2], hidden_size, BatchSize)

print(device)

TRAN_TAG = False
if TRAN_TAG:
    if path.exists("checkpoint.pth.tar"):
        Prednet.load_state_dict(torch.load('checkpoint.pth.tar'))
    Prednet = Prednet.double()
    Prednet = Prednet.to(device)
    plot_losses = trainIters(Prednet, 30, print_every=2)
    torch.save(Prednet.state_dict(), 'checkpoint.pth.tar')
    showPlot(plot_losses)
else:
    # Prednet.load_state_dict(torch.load('checkpoint.pth.tar')) 不加载权重
    Prednet = Prednet.double()
    Prednet = Prednet.to(device)
    Prednet.eval()
    # Eval_net(Prednet)
