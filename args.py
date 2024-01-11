import argparse


def get_args():
    parser = argparse.ArgumentParser()
    # loading dataset param
    parser.add_argument('--path', type=str) #default=r'C:\Users\Admin\Desktop\LSTM-for-Trajectory-Prediction-master\data\eth\train\biwi_hotel_train.txt' 可有可无
    parser.add_argument('--dataset_name', type=str, default='eth')
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--skip', type=int, default=1)
    parser.add_argument('--delim', default='\t', type=str)

    # loading dataloader param
    parser.add_argument('--loader_num_workers', type=int, default=1)

    # training param
    parser.add_argument('--batch_size', type=int, default=128)


    # test params

    # model params

    # inferance params

    #解析命令行参数
    args = parser.parse_args()

    return args   #parser
