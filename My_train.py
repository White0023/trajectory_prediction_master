#导入必要的包
from data.loader import data_loader
from args import *
from utils import *
import utils
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from scipy.stats import entropy

#构建dataloader

args = get_args()

train_path = get_dset_path(args.dataset_name,"train")
test_path = get_dset_path(args.dataset_name,"test")

train_Set, train_data_loader = data_loader(args, path=train_path)
test_set, test_data_loader = data_loader(args, path=test_path)


#构造网络模型
class NNPred(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout=0.05):
        super(NNPred, self).__init__()
        self.input_size = input_size

        self.hidden_size = hidden_size
        self.num_layers = 2

        self.in2lstm = nn.Linear(self.input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=self.num_layers, bidirectional=False, batch_first=True,
                            dropout=0.1)
        self.in2bilstm = nn.Linear(self.input_size, hidden_size)
        self.bilstm = nn.LSTM(hidden_size, hidden_size // 2, num_layers=self.num_layers, bidirectional=True,
                              batch_first=True, dropout=0.1)

        self.fc0 = nn.Linear(256, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, output_size)
        # 加一个
        self.fc3 = nn.Linear(2, 2)
        self.in2out = nn.Linear(self.input_size, 64)
        self.tanh = nn.Tanh()

    def forward(self):
        # input = tensor shape[batchsize, len, num_features]
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
            # 对statue矩阵进行求和，找到最小值所对应的索引
            column_sums = np.sum(statue, axis=0)
            # 找到和最小值所对应的索引
            min_index = np.argmin(column_sums)

            # print("每一列的和：", column_sums)
            # print("和最小值所对应的索引：", min_index)

            # 那么接下来需要找出该矩阵中散度值最小时所对应的两个行人轨迹
            # 找到和最小的时所对应的两个轨迹分布
            # center_idx = np.where(np.min(statue) == statue)  # 找出矩阵中的最小值所对应的索引
            idx_a, idx_b = min_index, min_index+1
            """
            对K1进行求和，找到列中的最小值
            1.根据找到的A和B矩阵，求C = A^(-1) * B,而A矩阵不是方阵，需要用GMM变换等,实际用的是伪逆变换
            2.C^ = C + T_matrix(可学习矩阵)
            3. 
            """
            matrix_A = obs_traj_rel[idx_a]  #tensor
            matrix_A = np.array(matrix_A)
            matrix_B = obs_traj_rel[idx_b]
            matrix_B = np.array(matrix_B)
            matrix_A_I = np.linalg.pinv(matrix_A)
            matrix_C =  np.dot(matrix_A_I, matrix_B)
            T_matrix = torch.tensor(matrix_C) + self.fc3(torch.tensor(matrix_C))  #shape为[2, 2]

            # 求scale之后batch的轨迹序列,除了idx_a对应的轨迹保留,其它的轨迹需要乘上变换矩阵T_matrix
            # obs_traj_rel_scaled = []
            obs_traj_rel_scaled = torch.empty(0,8,2)
            for idx_s in range(batch_size):
                if idx_s == idx_a:  #
                    obs_traj_rel[idx_s] = obs_traj_rel[idx_a]
                    obs_traj_rel_scaled = torch.cat((obs_traj_rel[idx_s].unsqueeze(0),obs_traj_rel_scaled),dim=0)

                    # obs_traj_rel_scaled.append(obs_traj_rel[idx_s])
                else:
                    # obs_traj_rel[idx_s] = obs_traj_rel[idx_s].reshape(obs_traj_rel[idx_s].shape[1],
                    #                                                   obs_traj_rel[idx_s].shape[0])  # 2,8
                    obs_traj_rel[idx_s] = torch.mm(obs_traj_rel[idx_s], T_matrix)  # [8, 2]×[2,2]
                    obs_traj_rel_scaled = torch.cat((obs_traj_rel[idx_s].unsqueeze(0), obs_traj_rel_scaled), dim=0)

                    # obs_traj_rel_scaled.append(obs_traj_rel[idx_s])  # [427, 8, 2]
            # obs_traj_rel_scaled = torch.tensor(obs_traj_rel_scaled)  # 得到可以送入到model的input tensor
            # obs_traj_rel_scaled = obs_traj_rel_scaled.reshape(obs_traj_rel_scaled.shape[2],
            #                                                   obs_traj_rel_scaled.shape[0],
            #                                                   obs_traj_rel_scaled.shape[1]) #不需要变换

            input = obs_traj_rel_scaled
            self.input_size = input.shape

            bilstm_out, _ = self.lstm(self.in2bilstm(input))

            lstm_out, _ = self.lstm(self.in2lstm(input))
            out = self.tanh(self.fc0(lstm_out + bilstm_out))
            out = self.tanh(self.fc1(out))
            out = out + self.in2out(input)
            output = self.fc2(out)
            pred_traj_fake_rel = output

            #对模型的输出output进行reverse scaling
            #start reversing scaling for output


            # T_matrix = np.array(T_matrix)
            T_matrix_inverse = torch.inverse(T_matrix)
            pred_traj_fake_rel_scaled = torch.empty(0, 8, 2)
            # pred_traj_fake_rel = pred_traj_fake_rel.reshape(pred_traj_fake_rel.shape[1],
            #                                                 pred_traj_fake_rel.shape[0],
            #                                                 pred_traj_fake_rel.shape[2])  # 427,8,2

            for idx_is in range(batch_size):
                if idx_is == idx_a:  #
                    pred_traj_fake_rel[idx_is] = pred_traj_fake_rel[idx_is]
                    pred_traj_fake_rel_scaled = torch.cat((pred_traj_fake_rel[idx_is].unsqueeze(0), pred_traj_fake_rel_scaled), dim=0)
                    # pred_traj_fake_rel_scaled.append(pred_traj_fake_rel[idx_is])
                else:
                    # pred_traj_fake_rel[idx_is] = pred_traj_fake_rel[idx_is].reshape(pred_traj_fake_rel[idx_is].shape[1],
                    #                                                                 pred_traj_fake_rel[idx_is].shape[
                    #                                                                     0])  # 2,8
                    pred_traj_fake_rel[idx_is] = torch.mm(pred_traj_fake_rel[idx_is],
                                                        T_matrix_inverse)  # [8,2] × [2, 2]
                    pred_traj_fake_rel_scaled = torch.cat(
                        (pred_traj_fake_rel[idx_is].unsqueeze(0), pred_traj_fake_rel_scaled), dim=0)   #[427,8,2]
                    # pred_traj_fake_rel_scaled.append(pred_traj_fake_rel[idx_is])  # [427, 2, 8]>>>>>>[427,8,2]
            # obs_traj_rel_scaled = torch.tensor(obs_traj_rel_scaled)  # 得到可以送入到model的input tensor
            pred_traj_fake_rel_scaled = obs_traj_rel_scaled.reshape(obs_traj_rel_scaled.shape[1],
                                                                    obs_traj_rel_scaled.shape[0],
                                                                    obs_traj_rel_scaled.shape[2])  # 8,427,2

            # end_reverse scaling for encoder_output

            output = pred_traj_fake_rel_scaled
            # 计算loss
            loss = torch.zeros(1).to(pred_traj_gt)
            l2_loss_rel = []
            loss_mask = loss_mask[:, args.obs_len:]
            l2_loss_rel.append(l2_loss(pred_traj_fake_rel, input, loss_mask, mode="raw")) #loss_mask:43,12
            l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
            l2_loss_rel = torch.stack(l2_loss_rel, dim=1)
            for start, end in seq_start_end.data:
                _l2_loss_rel = torch.narrow(l2_loss_rel, 0, start, end - start)
                _l2_loss_rel = torch.sum(_l2_loss_rel, dim=0)  # [20]
                _l2_loss_rel = torch.min(_l2_loss_rel) / (
                    (pred_traj_fake_rel.shape[0]) * (end - start)
                )
                l2_loss_sum_rel += _l2_loss_rel

            loss += l2_loss_sum_rel
            print(loss)
            # 应该返回的loss
            return output,loss


#4.搭建网络模型训练步骤

hid_size = 256
output_size = 2 #暂时不清楚是什么，先假设是12
input_size = (2)   # 到底是输入的shape还是什么？
model = NNPred(input_size= input_size, output_size= output_size, hidden_size=hid_size)   #参数需要写入

optimizer = optim.Adam(model.parameters(), lr=0.01)
encoder = model
epoch = 30
cp = 0
Training_generator = train_data_loader
#4.1 定义训练epoch
for iter in range(1, epoch + 1):

    if iter % 50 == 1:
        cp = cp +1
        torch.save(model.state_dict(), str(cp) + 'checkpoint.pth.tar')
        l2_loss_rel = []
        pred_traj_fake_rel, loss = encoder.forward()  # training_step  # model_input:8,43,2
        loss.backward() #计算梯度
        optimizer.step() #更新梯度
