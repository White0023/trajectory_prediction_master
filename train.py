import argparse

import numpy as np
import os
from matplotlib import pyplot as plt
import torch
import LSTM
from LSTM import *
from data.loader import data_loader
from args import *
from utils import *
import utils

args = get_args()


train_path = get_dset_path(args.dataset_name,"train")
test_path = get_dset_path(args.dataset_name,"test")

train_Set, train_data_loader = data_loader(args, path=train_path)
test_set, test_data_loader = data_loader(args, path=test_path)

# for batch_idx , batch in enumerate(train_data_loader):
#     batch_size = batch[0].shape[1]
#
#     batch = [tensor for tensor in batch]
#
#     # batch_array = np.array(batch)
#
#     (
#         obs_traj,
#         pred_traj_gt,
#         obs_traj_rel, #8,427,2
#         pred_traj_gt_rel,
#         non_linear_ped,
#         loss_mask,
#         seq_start_end,
#     ) = batch
#
#     statue = np.zeros((batch_size,batch_size))
#     from scipy.stats import entropy
#     obs_traj_rel = obs_traj_rel.reshape(obs_traj_rel.shape[1],obs_traj_rel.shape[0],obs_traj_rel.shape[2]) #427,8,2
#     for i in range(len(obs_traj_rel)):
#         for j in range(i):
#             vector1 = obs_traj_rel[i]
#             vector2 = obs_traj_rel[j]
#
#             k1_dicvergence = entropy(vector1.flatten(),vector2.flatten())
#             statue[i][j] = k1_dicvergence
#             statue[j][i] = k1_dicvergence
#     #运行以上代码之后，可以得到不同轨迹之间的KL散度，得到一个二维的散度矩阵，如维度为[427, 427]
#     #那么接下来需要找出该矩阵中散度值最小时所对应的两个行人轨迹
#     # statue_result = []
#     center_idx = np.where(np.min(statue) == statue) #找出矩阵中的最小值所对应的索引
#     idx_a, idx_b = center_idx
#
#     #出现nan和inf值(等会再解决),先假设已经找到对应的索引idx_a,idx_b,接下来计算变换矩阵
#     T_matrix = LSTM.NNPred.fc3()  #首先在网络中定义线性变化为fc3,维度为8×8,这样就与网络同时进行训练,tensor
#     #obs_traj_rel的shape为[8,2],需要先转换为[2,8]才能使用T_matrix,但是如果直接使用维度为2×2的变换矩阵是否也可行?
#     #
#     # obs_traj_rel[idx_a] = obs_traj_rel[idx_a].reshape(obs_traj_rel[idx_a].shape[1],obs_traj_rel[idx_a].shape[0])
#     # A_transed = T_matrix(obs_traj_rel[idx_a])
#     # A_transed = A_transed.reshape(A_transed.shape[1], A_transed.shape[0])
#     # obs_traj_rel[idx_b]
#     #求scale之后batch的轨迹序列,除了idx_a对应的轨迹保留,其它的轨迹需要乘上变换矩阵T_matrix
#     obs_traj_rel_scaled = []
#     for idx_s in range(batch_size):
#         if idx_s == idx_a:  #
#             obs_traj_rel[idx_s] = obs_traj_rel[idx_a]
#             obs_traj_rel_scaled.append(obs_traj_rel[idx_s])
#         else:
#             obs_traj_rel[idx_s] = obs_traj_rel[idx_s].reshape(obs_traj_rel[idx_s].shape[1],obs_traj_rel[idx_s].shape[0]) #2,8
#             obs_traj_rel[idx_s] = np.dot(np.array(obs_traj_rel[idx_s]), np.array(T_matrix)) #[2,8]×[8,8]
#             obs_traj_rel_scaled.append(obs_traj_rel[idx_s]) #[427, 2, 8]
#     obs_traj_rel_scaled = torch.tensor(obs_traj_rel_scaled) #得到可以送入到model的input tensor
#     # #这两句代码是什么作用？
#     # for i in range(batch_size):
#     #     statue_result.append(sum(batch_size[:,i]))  # 这句代码需要更改下
#     # statue_result = np.array(statue_result)
#     # center_idx = np.where(np.min(statue_result) == statue_result)
#     # statue = np.zeros((batch_size,batch_size))
#     # from scipy.stats import entropy
#     # for i in range(len(obs_traj_rel)):
#     #     for j in range(i):
#     #         vector1 = obs_traj_rel[i]
#     #         vector2 = obs_traj_rel[j]
#     #
#     #         k1_dicvergence = entropy(vector1.flatten(),vector2.flatten())
#     #         statue[i][j] = k1_dicvergence
#     #         statue[j][i] = k1_dicvergence
#     #
#     # statue_result = []
#     # for i in range(batch_size):
#     #     statue_result.append(sum[:,i])
#     # statue_result = np.array(statue_result)
#     #
#     # center_idx = np.where(np.min(statue_result) == statue_result)
#
#     #transform-> A*D =B -> D and D-1
#
#     #model
#
#     #rel+abs_lastposition = predict trajectories in real world
#
#     # metric , ADE , FDE


def train(args, model, train_loader, optimizer, epoch, training_step, writer):
    losses = utils.AverageMeter("Loss", ":.6f")
    progress = utils.ProgressMeter(
        len(train_loader), [losses], prefix="Epoch: [{}]".format(epoch)
    )
    model.train()

    for batch_idx, batch in enumerate(train_data_loader):
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
        from scipy.stats import entropy
        obs_traj_rel = obs_traj_rel.reshape(obs_traj_rel.shape[1], obs_traj_rel.shape[0],
                                            obs_traj_rel.shape[2])  # 427,8,2
        for i in range(len(obs_traj_rel)):
            for j in range(i):
                vector1 = obs_traj_rel[i]
                vector2 = obs_traj_rel[j]

                k1_dicvergence = entropy(vector1.flatten(), vector2.flatten())
                statue[i][j] = k1_dicvergence
                statue[j][i] = k1_dicvergence
        # 运行以上代码之后，可以得到不同轨迹之间的KL散度，得到一个二维的散度矩阵，如维度为[427, 427]
        # 那么接下来需要找出该矩阵中散度值最小时所对应的两个行人轨迹

        center_idx = np.where(np.min(statue) == statue)  # 找出矩阵中的最小值所对应的索引
        idx_a, idx_b = center_idx

        # 出现nan和inf值(等会再解决),先假设已经找到对应的索引idx_a,idx_b,接下来计算变换矩阵
        T_matrix = LSTM.NNPred.fc3()  # 首先在网络中定义线性变化为fc3,维度为8×8,这样就与网络同时进行训练,tensor
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
        obs_traj_rel_scaled = obs_traj_rel_scaled.reshape(obs_traj_rel_scaled.shape[2],obs_traj_rel_scaled.shape[0],obs_traj_rel_scaled.shape[1])
        optimizer.zero_grad()
        loss = torch.zeros(1).to(pred_traj_gt)
        l2_loss_rel = []
        loss_mask = loss_mask[:, args.obs_len :]

        if training_step == 1 or training_step == 2:
            model_input = obs_traj_rel_scaled #8,43,2
            pred_traj_fake_rel = model.forward( #pred_traj_fake_rel:8,43,2
                model_input, obs_traj, seq_start_end, 1, training_step #model_input:8,43,2
            )

            #start_revers scaling
            T_matrix = np.array(T_matrix)
            T_matrix_inverse = np.linalg.inv(T_matrix)
            pred_traj_fake_rel_scaled = []
            pred_traj_fake_rel = pred_traj_fake_rel.reshape(pred_traj_fake_rel.shape[1],
                                                            pred_traj_fake_rel.shape[0],
                                                            pred_traj_fake_rel.shape[2])  #427,8,2
            for idx_is in range(batch_size):
                if idx_is == idx_a:  #
                    pred_traj_fake_rel[idx_is] = pred_traj_fake_rel[idx_is]
                    pred_traj_fake_rel_scaled.append(pred_traj_fake_rel[idx_is])
                else:
                    pred_traj_fake_rel[idx_is] = pred_traj_fake_rel[idx_is].reshape(pred_traj_fake_rel[idx_is].shape[1],
                                                                      pred_traj_fake_rel[idx_is].shape[0])  # 2,8
                    pred_traj_fake_rel[idx_is] = np.dot(np.array(pred_traj_fake_rel[idx_is]), T_matrix_inverse)  # [2,8]×[8,8]
                    pred_traj_fake_rel_scaled.append(pred_traj_fake_rel[idx_is])  # [427, 2, 8]
            obs_traj_rel_scaled = torch.tensor(obs_traj_rel_scaled)  # 得到可以送入到model的input tensor
            pred_traj_fake_rel_scaled = obs_traj_rel_scaled.reshape(obs_traj_rel_scaled.shape[2],
                                                                    obs_traj_rel_scaled.shape[0],
                                                                    obs_traj_rel_scaled.shape[1])   #8,427,2
            #end_reverse scaling

            l2_loss_rel.append(
                l2_loss(pred_traj_fake_rel_scaled, model_input, loss_mask, mode="raw") #loss_mask:43,12
            )
        else:
            model_input = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)
            for _ in range(args.best_k):
                pred_traj_fake_rel = model.forward(model_input, obs_traj, seq_start_end, 0)

                #Start Reverse scaling
                pred_traj_fake_rel_scaled = []
                pred_traj_fake_rel = pred_traj_fake_rel.reshape(pred_traj_fake_rel.shape[1],
                                                                pred_traj_fake_rel.shape[0],
                                                                pred_traj_fake_rel.shape[2])  # 427,8,2
                for idx_is in range(batch_size):
                    if idx_is == idx_a:  #保留KL散度最小值所对应的索引,其索引所对应的轨迹
                        pred_traj_fake_rel[idx_is] = pred_traj_fake_rel[idx_is]
                        pred_traj_fake_rel_scaled.append(pred_traj_fake_rel[idx_is])
                    else:
                        pred_traj_fake_rel[idx_is] = pred_traj_fake_rel[idx_is].reshape(
                            pred_traj_fake_rel[idx_is].shape[1],
                            pred_traj_fake_rel[idx_is].shape[0])  # 2,8
                        pred_traj_fake_rel[idx_is] = np.dot(np.array(pred_traj_fake_rel[idx_is]),
                                                            T_matrix_inverse)  # [2,8]×[8,8]
                        pred_traj_fake_rel_scaled.append(pred_traj_fake_rel[idx_is])  # [427, 2, 8]
                obs_traj_rel_scaled = torch.tensor(obs_traj_rel_scaled)  # 得到可以送入到model的input tensor
                pred_traj_fake_rel_scaled = obs_traj_rel_scaled.reshape(obs_traj_rel_scaled.shape[2],
                                                                        obs_traj_rel_scaled.shape[0],
                                                                        obs_traj_rel_scaled.shape[1])# 8,427,2
                #end reverse scaling

                l2_loss_rel.append(
                    l2_loss(
                        pred_traj_fake_rel_scaled,
                        model_input[-args.pred_len :],
                        loss_mask,
                        mode="raw",
                    )
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
        optimizer.step()
        if batch_idx % args.print_every == 0:
            progress.display(batch_idx)
    writer.add_scalar("train_loss", losses.avg, epoch)

if "__name__"=="__main__":
    model = NNPred
    train(args=args, model=model, train_loader=train_data_loader,optimizer=optim.Adam(model.parameters(), lr=0.001),epoch=30,training_step=1,writer='')

# def train(args, model, train_loader, optimizer, epoch, training_step, writer):
#     losses = utils.AverageMeter("Loss", ":.6f")
#     progress = utils.ProgressMeter(
#         len(train_loader), [losses], prefix="Epoch: [{}]".format(epoch)
#     )
#     model.train()
#     for batch_idx, batch in enumerate(train_loader):
#         # batch = [tensor.cuda() for tensor in batch]
#         batch = [tensor for tensor in batch]
#         (
#             obs_traj,
#             pred_traj_gt,
#             obs_traj_rel,
#             pred_traj_gt_rel,
#             non_linear_ped,
#             loss_mask,
#             seq_start_end,
#         ) = batch
#         optimizer.zero_grad()
#         loss = torch.zeros(1).to(pred_traj_gt)
#         l2_loss_rel = []
#         loss_mask = loss_mask[:, args.obs_len :]
#
#         if training_step == 1 or training_step == 2:
#             model_input = obs_traj_rel #8,43,2
#             pred_traj_fake_rel = model.forward( #pred_traj_fake_rel:8,43,2
#                 model_input, obs_traj, seq_start_end, 1, training_step #model_input:8,43,2
#             )
#             l2_loss_rel.append(
#                 l2_loss(pred_traj_fake_rel, model_input, loss_mask, mode="raw") #loss_mask:43,12
#             )
#         else:
#             model_input = torch.cat((obs_traj_rel, pred_traj_gt_rel), dim=0)
#             for _ in range(args.best_k):
#                 pred_traj_fake_rel = model.forward(model_input, obs_traj, seq_start_end, 0)
#                 l2_loss_rel.append(
#                     l2_loss(
#                         pred_traj_fake_rel,
#                         model_input[-args.pred_len :],
#                         loss_mask,
#                         mode="raw",
#                     )
#                 )
#
#         l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
#         l2_loss_rel = torch.stack(l2_loss_rel, dim=1)
#         for start, end in seq_start_end.data:
#             _l2_loss_rel = torch.narrow(l2_loss_rel, 0, start, end - start)
#             _l2_loss_rel = torch.sum(_l2_loss_rel, dim=0)  # [20]
#             _l2_loss_rel = torch.min(_l2_loss_rel) / (
#                 (pred_traj_fake_rel.shape[0]) * (end - start)
#             )
#             l2_loss_sum_rel += _l2_loss_rel
#
#         loss += l2_loss_sum_rel
#         losses.update(loss.item(), obs_traj.shape[1])
#         loss.backward()
#         optimizer.step()
#         if batch_idx % args.print_every == 0:
#             progress.display(batch_idx)
#     writer.add_scalar("train_loss", losses.avg, epoch)