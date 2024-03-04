import argparse
import configparser
import os
import load_data
from load_data import CustomDataset
from model.CLUBR import *
from copy import deepcopy
from utils import *
from torch.nn.utils import clip_grad_norm_
import torch
import torch.nn as nn
import numpy as np
import time
import os
import pickle
from tqdm import tqdm
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# read hyper-param settings
parser = argparse.ArgumentParser()
parser.add_argument("--config", default='./config/model_sanya_tpp.conf', type=str, help="configuration file path")
args = parser.parse_args()
config_file = args.config
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
print('>>>>>>>  configuration   <<<<<<<')
with open(config_file, 'r') as f:
    print(f.read())
print('\n')
config.read(args.config)
data_config = config['Data']
training_config = config['Training']
model_config = config['Model']

# Data config
data_root = data_config['data_root']
city_id = data_config['city_id']
shop_num = int(data_config['shop_num'])
brand_num = int(data_config['brand_num'])
user_num = int(data_config['user_num'])
district_num = int(data_config['district_num'])
item_num = int(data_config['item_num'])
category_num = int(data_config['category_num'])
food_num = int(data_config['food_num'])
aoi_num = int(data_config['aoi_num'])
print('load dataset: data of city_', city_id)

# Training config
mode = training_config['mode'].strip()
downstream = training_config['downstream'].strip()
ctx = training_config['ctx']
os.environ["CUDA_VISIBLE_DEVICES"] = ctx
USE_CUDA = torch.cuda.is_available()
print("CUDA:", USE_CUDA, ctx)
device = torch.device("cuda" if USE_CUDA else "cpu")
print('device:', device)
use_nni = bool(int(training_config['use_nni']))
regularization = float(training_config['regularization'])
learning_rate = float(training_config['learning_rate'])
max_epochs = int(training_config['max_epochs'])
display_step = int(training_config['display_step'])
patience = int(training_config['patience'])
batch_size = int(training_config['batch_size'])
save_results = bool(int(training_config['save_results']))

specific_config = 'LSTM'

# Model Setting
emb_size = int(model_config['emb_size'])
hidden_size = int(model_config['hidden_size'])
tau = float(model_config['tau'])
margin = float(model_config['margin'])
lam_1 = float(model_config['lam_1'])
lam_2 = float(model_config['lam_2'])
delta = float(model_config['delta'])

if use_nni:
    import nni
    param = nni.get_next_parameter()
    emb_size = int(param['emb_size'])
    hidden_size = int(param['hidden_size'])
    tau = float(param['tau'])
    margin = float(param['margin'])
    lam_1 = float(param['lam_1'])
    lam_2 = float(param['lam_2'])
    delta = float(param['delta'])

# Data
print('Loading data...')
CTR_train_data, CTR_val_data, CTR_test_data, i2i, i2t, i2c, i2f, i2a = load_data.load_data(data_root, city_id=city_id, downstream = downstream, device=device)

trainY_tau_mean, trainY_tau_std = CTR_train_data.get_mean_std()

collate = load_data.collate_session_based   # padding sequence with variable len

dl_train = torch.utils.data.DataLoader(CTR_train_data, batch_size=batch_size, shuffle=True, collate_fn=collate,drop_last=True)
dl_val = torch.utils.data.DataLoader(CTR_val_data, batch_size=batch_size, shuffle=False, collate_fn=collate,drop_last=True)
dl_test = torch.utils.data.DataLoader(CTR_test_data, batch_size=batch_size, shuffle=False, collate_fn=collate,drop_last=True)

# Model setup
print('Building model...', flush=True)
# Define model
model = CLUBR(embedding_dim = emb_size, hidden_dim = hidden_size, i2i=i2i, i2t=i2t, i2c=i2c, i2f=i2f, i2a=i2a, device=device,downstream=downstream,\
                tau = tau,margin=margin, lam_1=lam_1, lam_2=lam_2, delta=delta, shift_init=19.96165667039388, scale_init=20.269409785513076,shop_num=shop_num,brand_num=brand_num,user_num=user_num,\
                district_num=district_num, item_num=item_num, category_num=category_num, food_num=food_num, aoi_num= aoi_num).to(device)
print(model, flush=True)

# params_path = os.path.join(data_root, '/experiments_city_', city_id)
params_path = data_root + '/experiments_city_' + city_id
print('params_path:', params_path)

if use_nni:
    exp_id = nni.get_experiment_id()
    trail_id = nni.get_trial_id()
    best_name = str(exp_id) + '.' + str(trail_id) + 'best.params'
    params_filename = os.path.join(params_path, best_name)
else:
    best_name = ctx + 'best.params'
    params_filename = os.path.join(params_path, best_name)

if mode == 'train':
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("total_params:", total_params, flush=True)
    print("total_trainable_params:", total_trainable_params, flush=True)

    if os.path.exists(params_path):
        print('already exist %s' % (params_path), flush=True)
    else:
        os.makedirs(params_path)
        print('create params directory %s' % (params_path), flush=True)

    print('Starting training...', flush=True)

    impatient = 0
    best_hit20 = -np.inf
    best_loss = np.inf
    best_model = deepcopy(model.state_dict())
    global_step = 0
    global_batch_step = 0
    best_epoch = -1
    

    opt = torch.optim.Adam(model.parameters(), weight_decay=regularization, lr=learning_rate, amsgrad=True)

    start = time.time()

    for epoch in range(0, max_epochs):

        model.train()
        total_loss = []

        for input in dl_train:

            opt.zero_grad()
            
            if downstream == 'pretrain':
                s_loss_score = model(input)
            elif downstream == 'CTR':
                s_loss_score, prediction, log_loss = model(input)
            elif downstream == 'TPP':
                s_loss_score, mean_time, y_label, _ = model(input)
            s_loss_score.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            global_step += 1
            s_loss_score = s_loss_score.item()
            # print(global_step)
            # print(s_loss_score)
            total_loss.append(s_loss_score)
            del input
            torch.cuda.empty_cache()

        model.eval()
        with torch.no_grad():
            if downstream == 'TPP':
                mae_val, mape_val, rmse_val, nll_t_val = get_t_for_TTP(dl_val, model)
                if (best_loss - nll_t_val) < 1e-4:
                    impatient += 1
                    if nll_t_val < best_loss:
                        best_loss = nll_t_val
                        best_model = deepcopy(model.state_dict())
                        best_epoch = epoch
                else:
                    best_loss = nll_t_val
                    best_model = deepcopy(model.state_dict())
                    best_epoch = epoch
                    impatient = 0

                if impatient >= patience:
                    print('Breaking due to early stopping at epoch %d, best epoch at %d' % (epoch, best_epoch), flush=True)
                    break

                if (epoch) % display_step == 0:
                        print('Epoch %4d, train_tnll=%.4f, val_tnll=%.4f, val_mae=%.4f, val_rmse=%.4f, val_nll=%.4f' % (epoch, np.mean(np.array(total_loss)), nll_t_val, mae_val,  rmse_val, mape_val), flush=True)
                if use_nni:
                        nni.report_intermediate_result(nll_t_val)
            elif downstream == 'CTR':
                auc, gauc, ndcg_3, ndcg_10, BCE_loss = get_t_for_CTR(dl_val, model)
                val_target = -auc
                if (best_loss - val_target) < 1e-4:
                    impatient += 1
                    if val_target < best_loss:
                        best_loss = val_target
                        best_model = deepcopy(model.state_dict())
                        best_epoch = epoch
                else:
                    best_loss = val_target
                    best_model = deepcopy(model.state_dict())
                    best_epoch = epoch
                    impatient = 0

                if impatient >= patience:
                    print('Breaking due to early stopping at epoch %d, best epoch at %d' % (epoch, best_epoch), flush=True)
                    break

                if (epoch) % display_step == 0:
                        print('Epoch %4d, train_loss=%.4f, val_loss=%.4f, val_auc=%.4f, val_gauc=%.4f, val_ndcg_10=%.4f' % (epoch, np.mean(np.array(total_loss)), BCE_loss, auc,  gauc, ndcg_10), flush=True)
                if use_nni:
                        nni.report_intermediate_result(auc)

        torch.save(best_model, params_filename)

    print("best epoch at %d" % best_epoch, flush=True)
    print('save parameters to file: %s' % params_filename, flush=True)
    print("training time: ", time.time() - start)

### Evaluation
params_filename = os.path.join(params_path, best_name)
print('load model from:', params_filename)
model.load_state_dict(torch.load(params_filename))
model.eval()
with torch.no_grad():
    if downstream == 'TPP':
        print('evaluate on the train set ... ')
        train_mae, train_mape, train_rmse, train_nll_t = get_t_for_TTP(dl_train, model, save_filename='train', params_path=params_path, use_nni=use_nni)
        print('evaluate on the val set ... ')
        val_mae, val_mape, val_rmse, val_nll_t = get_t_for_TTP(dl_val, model, save_filename='val', params_path=params_path, use_nni=use_nni)
        print('evaluate on the test set ... ')
        test_mae, test_mape, test_rmse, test_nll_t = get_t_for_TTP(dl_test, model, save_filename='test', params_path=params_path, use_nni=use_nni)


        print('Dataset\t MAE\t RMSE\t MAPE\t TNll\t\n' +
            'Train:\t %.4f\t %.4f\t %.4f\t %.4f\t\n' % (train_mae, train_rmse, train_mape, train_nll_t) +
            'Val:\t %.4f\t %.4f\t %.4f\t %.4f\t\n' % (val_mae, val_rmse, val_mape, val_nll_t) +
            'Test:\t %.4f\t %.4f\t %.4f\t %.4f\t\n' % (test_mae, test_rmse, test_mape, test_nll_t), flush=True)

        if use_nni:
            nni.report_final_result(test_nll_t)
    else:
        print('evaluate on the train set ... ')
        train_auc, train_gauc, train_ndcg_3, train_ndcg_10, train_BCE_loss = get_t_for_CTR(dl_train, model, save_filename='train', params_path=params_path, use_nni=use_nni)
        print('evaluate on the val set ... ')
        val_auc, val_gauc, val_ndcg_3, val_ndcg_10, val_BCE_loss = get_t_for_CTR(dl_val, model, save_filename='val', params_path=params_path, use_nni=use_nni)
        print('evaluate on the test set ... ')
        test_auc, test_gauc, test_ndcg_3, test_ndcg_10, test_BCE_loss = get_t_for_CTR(dl_test, model, save_filename='test', params_path=params_path, use_nni=use_nni)


        print('Dataset\t AUC\t GAUC\t NDCG_3\t NDCG_10\t BCE\t\n' +
            'Train:\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t\n' % (train_auc, train_gauc, train_ndcg_3, train_ndcg_10, train_BCE_loss) +
            'Val:\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t\n' % (val_auc, val_gauc, val_ndcg_3, val_ndcg_10, val_BCE_loss) +
            'Test:\t %.4f\t %.4f\t %.4f\t %.4f\t %.4f\t\n' % (test_auc, test_gauc, test_ndcg_3, test_ndcg_10, test_BCE_loss), flush=True)

        if use_nni:
            nni.report_final_result(test_auc)
    


