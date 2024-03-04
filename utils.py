import numpy as np
import os
import torch.nn.functional as F
import torch
from sklearn import metrics
import random
import mz2geohash as geohash

def get_hash_tensor(geohash_list):
    cur_list = []
    for i in geohash_list:
        cur_list.append(get_hash(i))
    return torch.tensor(cur_list)
         
def get_hash(geohash):
    if geohash == '0':
        return 0
    else:
        return hash(geohash)/1e17
        
def label_get(x,his_len):
    batch_size=x.shape[0]
    y = torch.zeros((batch_size,1)).to(x.device)
    new_x = []
    for i in range(batch_size):
        hl=his_len[i]
        if hl == 1:
            y[i]=x[i][0][hl-1]
            new_x.append(x[i,:,:-1].reshape(1, 5,-1))
        elif hl == 0:
            y[i]=x[i-1][0][his_len[i-1]-1]
            new_x.append(x[i-1,:,:-1].reshape(1, 5,-1))
        else:
            y[i]=x[i][0][hl-1]
            new_x.append(torch.concat((x[i,:,:hl-1],x[i,:,hl:]),dim=-1).reshape(1, 5,-1))
    x = torch.concat(new_x,dim = 0)
    return x, y
    
def geo_label_get(x,his_len):
    batch_size=len(x)
    y = []
    new_x = []
    for i in range(batch_size):
        hl=his_len[i]
        if hl == 1:
            y.append(x[i][0][hl-1])
            cur_x = []
            for j in range(5):
                cur_x.append(x[i][j][:-1])
            new_x.append(cur_x)
        elif hl == 0:
            y.append(x[i-1][0][his_len[i-1]-1])
            cur_x = []
            for j in range(5):
                cur_x.append(x[i-1][j][:-1])
            new_x.append(cur_x)
        else:
            y.append(x[i][0][hl-1])
            cur_x = []
            for j in range(5):
                cur_x.append(x[i][j][:hl-1] + x[i][j][hl:])
            new_x.append(cur_x)
    return new_x, y

def label_get_r(x,his_len):
    batch_size=x.shape[0]
    y = torch.zeros((batch_size,1)).to(x.device)
    new_x = []
    for i in range(batch_size):
        hl=his_len[i]
        if hl == 2:
            y[i]=x[i][hl-2]
            new_x.append(x[i][:-2].reshape(1,-1))
        elif hl == 0:
            y[i]=x[i-1][his_len[i-1]-2]
            new_x.append(x[i-1][:-2].reshape(1,-1))
        else:
            y[i]=x[i][hl-2]
            new_x.append(torch.concat((x[i][:hl-2],x[i][hl:]),dim=0).reshape(1,-1))
    x = torch.concat(new_x,dim = 0)
    return x, y
    
def geo_label_get_r(x,his_len):
    batch_size=len(x)
    y = []
    new_x = []
    for i in range(batch_size):
        hl=his_len[i]
        if hl == 2:
            y.append(x[i][hl-2])
            new_x.append(x[i][:-2])
        elif hl == 0:
            y.append(x[i-1][his_len[i-1]-2])
            new_x.append(x[i-1][:-2])
        else:
            y.append(x[i][hl-2])
            new_x.append(x[i][:hl-2]+x[i][hl:])
    return new_x, y

def get_t_for_TTP(loader, model, save_filename=None, params_path=None, experiment_base_dir=None, use_nni=False):
    '''
    calculates the loss, mae and mape for the entire data loader
    :param loader:
    :param save:
    :return:
    '''
    ground_truth_Y_tau = []
    predicted_Y_tau = []
    all_X_length = []
    all_nll_t = []

    for input in loader:
        nll, mean,truth_Y_tau,loss = model(input)  # (batch_size,), (batch_size,)
        all_X_length.append(np.array(input.his_length))
        ground_truth_Y_tau.append(truth_Y_tau)
        predicted_Y_tau.append(mean.detach().cpu().numpy())
        all_nll_t.append(loss.detach().cpu().numpy())
        del input
        torch.cuda.empty_cache()

    all_nll_t = np.array(all_nll_t)
    ground_truth_Y_tau = np.concatenate(ground_truth_Y_tau).flatten()
    predicted_Y_tau = np.concatenate(predicted_Y_tau).flatten()
    mae = np.mean(abs(ground_truth_Y_tau - predicted_Y_tau))
    rmse = np.sqrt(((ground_truth_Y_tau - predicted_Y_tau) ** 2).mean())
    cur_ground_truth_y_tau = np.maximum(ground_truth_Y_tau,1)
    mape = np.mean(abs(ground_truth_Y_tau - predicted_Y_tau) / np.mean(cur_ground_truth_y_tau))
    nll_t = np.mean(all_nll_t)

    if (save_filename is not None) and (not use_nni):
        filename = os.path.join(params_path, save_filename+'_results.npz')
        np.savez(filename, ground_truth_Y_tau=ground_truth_Y_tau, predicted_Y_tau=predicted_Y_tau)

    return mae, mape, rmse, nll_t

def get_t_for_CTR(loader, model, save_filename=None, params_path=None, experiment_base_dir=None, use_nni=False):
    '''
    calculates the loss, mae and mape for the entire data loader
    :param loader:
    :param save:
    :return:
    '''
    ground_truth_label = []
    predicted = []
    all_BCE = []

    for input in loader:
        loss, prediction, logloss = model(input)  # (batch_size,), (batch_size,)
        ground_truth_label.append(input.label.detach().cpu().numpy())
        predicted.append(prediction.detach().cpu().numpy())
        all_BCE.append(logloss.detach().cpu().numpy())
        del input
        torch.cuda.empty_cache()

    ground_truth_label = np.concatenate(ground_truth_label).flatten()
    predicted = np.concatenate(predicted).flatten()
    auc = metrics.roc_auc_score(ground_truth_label, predicted)
    fpr, tpr, thresholds = metrics.roc_curve(ground_truth_label, predicted, pos_label=1)
    gauc = metrics.auc(fpr, tpr) * 2 - 1
    ndcg_3 = ndcg_at_k(ground_truth_label, predicted, k=3)
    ndcg_10 = ndcg_at_k(ground_truth_label, predicted, k=10)
    BCE_loss = np.mean(np.array(all_BCE))


    if (save_filename is not None) and (not use_nni):
        filename = os.path.join(params_path, save_filename+'_results.npz')
        np.savez(filename, ground_truth_label=ground_truth_label, predicted=predicted)

    return auc, gauc, ndcg_3, ndcg_10, BCE_loss

def enrichment(geo_shop_dict,shop_item,shop_id_, item_id_, category_1_id_, merge_standard_food_id_, brand_id_, price_, shop_aoi_id_, shop_geohash6_, timediff_,
                            week_hours_, time_type_, user_geohash6_,his_length):
    for i in range(his_length):
        rshop,goehash6 = shop_near_shop_func(geo_shop_dict,shop_id_[2*i],shop_geohash6_[2*i])
        item = int(random.choice(list(shop_item[rshop].keys())))
        [category_1_id,merge_standard_food_id,brand_id,price,shop_aoi_id] = shop_item[rshop][item]
        shop_id_.insert(i*2+1,rshop)
        item_id_.insert(i*2+1,item)
        category_1_id_.insert(i*2+1,category_1_id)
        merge_standard_food_id_.insert(i*2+1,merge_standard_food_id)
        brand_id_.insert(i*2+1,brand_id)
        price_.insert(i*2+1,price)
        shop_aoi_id_.insert(i*2+1,shop_aoi_id)
        shop_geohash6_.insert(i*2+1,goehash6)
        timediff_.insert(i*2+1,timediff_[2*i])
        week_hours_.insert(i*2+1,week_hours_[2*i])
        time_type_.insert(i*2+1,time_type_[2*i])
        user_geohash6_.insert(i*2+1,user_geohash6_[2*i])
    if his_length < 50:
        for i in range(his_length,50):
            shop_id_.insert(i*2+1,0)
            item_id_.insert(i*2+1,0)
            category_1_id_.insert(i*2+1,0)
            merge_standard_food_id_.insert(i*2+1,0)
            brand_id_.insert(i*2+1,0)
            price_.insert(i*2+1,0.)
            shop_aoi_id_.insert(i*2+1,0)
            shop_geohash6_.insert(i*2+1,'0')
            timediff_.insert(i*2+1,-1.)
            week_hours_.insert(i*2+1,0)
            time_type_.insert(i*2+1,0)
            user_geohash6_.insert(i*2+1,'0')
    return shop_id_, item_id_, category_1_id_, merge_standard_food_id_, brand_id_, price_, shop_aoi_id_, shop_geohash6_, timediff_, week_hours_, time_type_, user_geohash6_

def augmentation_1(geo_shop_dict, shop_item, user_shop, hour_shop, shop_geo_dict, shop_id_1, item_id_1, category_1_id_1, merge_standard_food_id_1, brand_id_1, price_1, shop_aoi_id_1, shop_geohash6_1, user_geohash6_1, his_length):
    selected_numbers = select_random_numbers(his_length, his_length//5)
    for i in selected_numbers:
        rshop,goehash6 = shop_near_shop_func(geo_shop_dict,shop_id_1[i],user_geohash6_1[i])
        item = int(random.choice(list(shop_item[rshop].keys())))
        [category_1_id,merge_standard_food_id,brand_id,price,shop_aoi_id] = shop_item[rshop][item]
        shop_id_1[i] = rshop
        item_id_1[i] = item
        category_1_id_1[i] = category_1_id
        merge_standard_food_id_1[i] = merge_standard_food_id
        brand_id_1[i] = brand_id
        price_1[i] = price
        shop_aoi_id_1[i] = shop_aoi_id
        shop_geohash6_1[i] = goehash6
    return shop_id_1, item_id_1, category_1_id_1, merge_standard_food_id_1, brand_id_1, price_1, shop_aoi_id_1, shop_geohash6_1

def augmentation_2(geo_shop_dict, shop_item, user_shop, hour_shop, shop_geo_dict, user_id, shop_id_2, item_id_2, category_1_id_2, merge_standard_food_id_2, brand_id_2, price_2, shop_aoi_id_2, shop_geohash6_2, his_length):
    selected_numbers = select_random_numbers(his_length, his_length//5)
    for i in selected_numbers:
        if user_id not in user_shop:
            rshop = shop_id_2[i]
        else:
            rshop = user_shop[user_id][int(i%len(user_shop[user_id]))][0]
        # rshop = shop_id_2[i]
        goehash6 = shop_geo_dict[rshop]
        item = int(random.choice(list(shop_item[rshop].keys())))
        [category_1_id,merge_standard_food_id,brand_id,price,shop_aoi_id] = shop_item[rshop][item]
        shop_id_2[i] = rshop
        item_id_2[i] = item
        category_1_id_2[i] = category_1_id
        merge_standard_food_id_2[i] = merge_standard_food_id
        brand_id_2[i] = brand_id
        price_2[i] = price
        shop_aoi_id_2[i] = shop_aoi_id
        shop_geohash6_2[i] = goehash6
    return shop_id_2, item_id_2, category_1_id_2, merge_standard_food_id_2, brand_id_2, price_2, shop_aoi_id_2, shop_geohash6_2

def augmentation_3(geo_shop_dict, shop_item, user_shop, hour_shop, shop_geo_dict, shop_id_3, item_id_3, category_1_id_3, merge_standard_food_id_3, brand_id_3, price_3, shop_aoi_id_3, shop_geohash6_3, week_hours_3, his_length):
    selected_numbers = select_random_numbers(his_length, his_length//5)
    for i in selected_numbers:
        rshop = hour_shop[week_hours_3[i]-1][int(i%len(hour_shop[week_hours_3[i]-1]))][0]
        # rshop = shop_id_3[i]
        goehash6 = shop_geo_dict[rshop]
        item = int(random.choice(list(shop_item[rshop].keys())))
        [category_1_id,merge_standard_food_id,brand_id,price,shop_aoi_id] = shop_item[rshop][item]
        shop_id_3[i] = rshop
        item_id_3[i] = item
        category_1_id_3[i] = category_1_id
        merge_standard_food_id_3[i] = merge_standard_food_id
        brand_id_3[i] = brand_id
        price_3[i] = price
        shop_aoi_id_3[i] = shop_aoi_id
        shop_geohash6_3[i] = goehash6
    return shop_id_3, item_id_3, category_1_id_3, merge_standard_food_id_3, brand_id_3, price_3, shop_aoi_id_3, shop_geohash6_3


def select_random_numbers(max_value, count):
    return random.sample(range(max_value), count)


def shop_near_shop_func(geo_shop_dict,shop1,geo1):
    rshop=0
    near_geo=near_1(geo1)
    for g in near_geo:
        if g in geo_shop_dict:
            geo_list=geo_shop_dict[g]
            random.shuffle(geo_list)
            for shop2 in geo_list:
                if shop2!=shop1 and shop2!=0:
                    rshop=shop2
                    break
    if rshop == 0:
        g = geo1
        rshop = shop1
    return rshop,g

def add_near(geo_list,geo,key):
        near = geohash.neighbors(geo)
        for i in key:
            geo_list.append(near[i])
        return geo_list

def near_1(geo):
    near_list = []
    key1 = ['n','nw','w','sw','s','se','e','ne']
    near_list = add_near(near_list,geo,key1)
    key2 = ['n','nw','ne']
    near_list = add_near(near_list,near_list[1],key2)
    key2 = ['s','se','sw']
    near_list = add_near(near_list,near_list[5],key2)
    return near_list

def dcg_at_k(scores, k):
    return sum((2**s - 1) / np.log2(idx + 2) for idx, s in enumerate(scores[:k]))

def ndcg_at_k(predicted_scores, true_labels, k):
    order = np.argsort(predicted_scores)[::-1]
    true_scores = np.take(true_labels, order)
    dcg = dcg_at_k(true_scores, k)
    ideal_order = np.argsort(true_labels)[::-1]
    ideal_scores = np.take(true_labels, ideal_order)
    idcg = dcg_at_k(ideal_scores, k)

    return dcg / idcg if idcg > 0 else 0
