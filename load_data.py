import torch
from torch.utils import data
import numpy as np
from scipy.sparse import csr_matrix
import copy
from utils import *

'''
['label','user_id','gender','visit_city','avg_price','is_supervip','ctr_30','ord_30','total_amt_30','shop_id','item_id','city_id','district_id','shop_aoi_id','shop_geohash_6','shop_geohash_12','brand_id',
'category_1_id','merge_standard_food_id','rank_7','rank_30','rank_90','shop_id_list','item_id_list','category_1_id_list','merge_standard_food_id_list','brand_id_list','price_list','shop_aoi_id_list',
'shop_geohash6_list','timediff_list','hours_list','time_type_list','weekdays_list','times','hours','time_type','weekdays','geohash12']
'''

def load_data(data_root, city_id, downstream, device):
    path = data_root + '/' + city_id + '_data_split.npz'
    data_all = np.load(path,allow_pickle=True)
    train_all = data_all['train_data']
    val_all = data_all['valid_data']
    test_all = data_all['test_data']
    user_shop=np.load(data_root+ '/' + city_id +'_shophour_usershop.npz',allow_pickle=True)['user_shop'].item()
    hour_shop=np.load(data_root+ '/' + city_id +'_shophour_usershop.npz',allow_pickle=True)['hour_shop']
    shop_geo_dict = np.load(data_root+ '/' + city_id +'_shop_geohash.npy',allow_pickle=True).item()
    geo_shop_dict=np.load(data_root+ '/' + city_id +'_geohash_shop.npy',allow_pickle=True).item()
    shop_item = np.load(data_root+ '/' + city_id +'_shop_item_dict.npy',allow_pickle=True).item()

    CTR_train_data = CTR_Dataset(downstream,train_all,device,geo_shop_dict,shop_item,user_shop,hour_shop,shop_geo_dict)
    CTR_val_data = CTR_Dataset(downstream,val_all,device,geo_shop_dict,shop_item,user_shop,hour_shop,shop_geo_dict)
    CTR_test_data = CTR_Dataset(downstream,test_all,device,geo_shop_dict,shop_item,user_shop,hour_shop,shop_geo_dict)

    i2i = np.load(data_root + '/' + city_id + '_i2i_matrix.npz')
    i2c = np.load(data_root + '/' + city_id + '_i2c_matrix.npz')
    i2f = np.load(data_root + '/' + city_id + '_i2f_matrix.npz')
    i2t = np.load(data_root + '/' + city_id + '_i2t_matrix.npz')
    i2a = np.load(data_root + '/' + city_id + '_i2a_matrix.npz')
    i2c = csr_matrix((i2c['data'], i2c['indices'], i2c['indptr']), shape=i2c['shape'])
    i2f = csr_matrix((i2f['data'], i2f['indices'], i2f['indptr']), shape=i2f['shape'])
    i2t = csr_matrix((i2t['data'], i2t['indices'], i2t['indptr']), shape=i2t['shape'])
    i2a = csr_matrix((i2a['data'], i2a['indices'], i2a['indptr']), shape=i2a['shape'])
    i2i = csr_matrix((i2i['data'], i2i['indices'], i2i['indptr']), shape=i2i['shape'])
    
    
    return CTR_train_data, CTR_val_data, CTR_test_data, i2c, i2f, i2t, i2a, i2i

class CustomDataset(data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]



class CTR_Dataset(data.Dataset):
    def __init__(self, downstream, data, device, geo_shop_dict, shop_item, user_shop, hour_shop, shop_geo_dict):
        self.data = data
        self.device = device
        self.downstream = downstream
        self.geo_shop_dict = geo_shop_dict
        self.shop_item = shop_item
        self.user_shop = user_shop
        self.hour_shop = hour_shop
        self.shop_geo_dict = shop_geo_dict

    def __getitem__(self, index):
        label = self.data[index][0]
        user_id = self.data[index][1]
        gender = self.data[index][2] 
        visit_city = self.data[index][3] 
        avg_price = self.data[index][4] 
        is_supervip = self.data[index][5] 
        ctr_30 = self.data[index][6] 
        ord_30 = self.data[index][7] 
        total_amt_30 = self.data[index][8] 
        shop_id = self.data[index][9] 
        item_id = self.data[index][10] 
        city_id = self.data[index][11] 
        district_id = self.data[index][12] 
        shop_aoi_id = self.data[index][13] 
        shop_geohash_6 = self.data[index][14] 
        shop_geohash_12 = self.data[index][15] 
        brand_id = self.data[index][16] 
        category_1_id = self.data[index][17] 
        merge_standard_food_id = self.data[index][18] 
        rank_7 = self.data[index][19] 
        rank_30 = self.data[index][20] 
        rank_90 = self.data[index][21] 
  
        shop_id_list = self.data[index][22].copy() 
        item_id_list = self.data[index][23].copy() 
        category_1_id_list = self.data[index][24].copy() 
        merge_standard_food_id_list = self.data[index][25].copy() 
        brand_id_list = self.data[index][26].copy() 
        price_list = self.data[index][27].copy() 
        shop_aoi_id_list = self.data[index][28].copy() 
        shop_geohash6_list = self.data[index][29].copy() 
        timediff_list = [-1 if i == -1 else i/60000 for i in self.data[index][30]].copy() 
        week_hours_list = self.data[index][31].copy() 
        time_type_list = self.data[index][32].copy() 
        user_geohash6_list = self.data[index][38].copy() 
        shop_id_list_r, item_id_list_r, category_1_id_list_r, merge_standard_food_id_list_r, brand_id_list_r, price_list_r, shop_aoi_id_list_r, shop_geohash6_list_r, timediff_list_r,week_hours_list_r, time_type_list_r, user_geohash6_list_r = enrichment(self.geo_shop_dict,self.shop_item,shop_id_list.copy(), item_id_list.copy(), category_1_id_list.copy(), merge_standard_food_id_list.copy(), brand_id_list.copy(), price_list.copy(), shop_aoi_id_list.copy(), shop_geohash6_list.copy(), timediff_list.copy(),
                            week_hours_list.copy(), time_type_list.copy(), user_geohash6_list.copy(),self.data[index][39])
        shop_id_list, item_id_list, category_1_id_list, merge_standard_food_id_list, brand_id_list, price_list, shop_aoi_id_list, shop_geohash6_list, timediff_list, week_hours_list, time_type_list, user_geohash6_list = self.get_augmentation(user_id, shop_id_list.copy(), item_id_list.copy(), category_1_id_list.copy(), merge_standard_food_id_list.copy(), brand_id_list.copy(), price_list.copy(), shop_aoi_id_list.copy(), shop_geohash6_list.copy(), timediff_list.copy(), user_geohash6_list.copy(), week_hours_list.copy(), time_type_list.copy(), self.data[index][39])

  
        times = self.data[index][33] 
        hours = self.data[index][34]
        time_type = self.data[index][35] 
        weekdays = self.data[index][36] 
        geohash12 = self.data[index][37] 
        his_length_r = self.data[index][39] * 2
        his_length = self.data[index][39]

        return label,user_id, gender, visit_city, avg_price, is_supervip, ctr_30, ord_30, total_amt_30, shop_id_list, item_id_list, category_1_id_list, merge_standard_food_id_list, brand_id_list, price_list, shop_aoi_id_list, shop_geohash6_list, timediff_list, \
               week_hours_list, time_type_list, user_geohash6_list , shop_id, item_id, city_id,district_id, shop_aoi_id, shop_geohash_6, shop_geohash_12, brand_id, category_1_id, merge_standard_food_id, rank_7, rank_30, rank_90, \
               times, hours, time_type, weekdays, geohash12, his_length, shop_id_list_r, item_id_list_r, category_1_id_list_r, merge_standard_food_id_list_r, brand_id_list_r, price_list_r, shop_aoi_id_list_r, shop_geohash6_list_r, timediff_list_r,week_hours_list_r, time_type_list_r, user_geohash6_list_r, his_length_r, self.device

    def __len__(self):
        return len(self.data)
    
    def get_mean_std(self):
        timediff = self.data[:,30]
        l = len(timediff)
        w = len(timediff[0])
        timediff_all=[]
        for i in range(l):
            for j in range(w):
                if timediff[i][j]==-1:
                    continue
                else:
                    timediff_all.append(timediff[i][j]/60000)
        mean = np.mean(timediff_all)
        std = np.std(timediff_all)
        return mean,std

    def get_augmentation(self,user_id, shop_id_list, item_id_list, category_1_id_list, merge_standard_food_id_list, brand_id_list, price_list, shop_aoi_id_list, shop_geohash6_list, timediff_list, user_geohash6_list, week_hours_list, time_type_list, his_length):
        shop_id_list_1, item_id_list_1, category_1_id_list_1, merge_standard_food_id_list_1, brand_id_list_1, price_list_1, shop_aoi_id_list_1, shop_geohash6_list_1 = augmentation_1(self.geo_shop_dict, self.shop_item, self.user_shop, self.hour_shop, self.shop_geo_dict, shop_id_list, item_id_list, category_1_id_list, merge_standard_food_id_list, brand_id_list, price_list, shop_aoi_id_list, shop_geohash6_list, user_geohash6_list, his_length)
        shop_id_list_2, item_id_list_2, category_1_id_list_2, merge_standard_food_id_list_2, brand_id_list_2, price_list_2, shop_aoi_id_list_2, shop_geohash6_list_2 = augmentation_2(self.geo_shop_dict, self.shop_item, self.user_shop, self.hour_shop, self.shop_geo_dict, user_id, shop_id_list, item_id_list, category_1_id_list, merge_standard_food_id_list, brand_id_list, price_list, shop_aoi_id_list, shop_geohash6_list, his_length)
        shop_id_list_3, item_id_list_3, category_1_id_list_3, merge_standard_food_id_list_3, brand_id_list_3, price_list_3, shop_aoi_id_list_3, shop_geohash6_list_3 = augmentation_3(self.geo_shop_dict, self.shop_item, self.user_shop, self.hour_shop, self.shop_geo_dict, shop_id_list, item_id_list, category_1_id_list, merge_standard_food_id_list, brand_id_list, price_list, shop_aoi_id_list, shop_geohash6_list, week_hours_list, his_length)
        shop_id_list = [shop_id_list, shop_id_list_1, shop_id_list_2, shop_id_list_3]
        item_id_list = [item_id_list, item_id_list_1, item_id_list_2, item_id_list_3]
        category_1_id_list = [category_1_id_list, category_1_id_list_1, category_1_id_list_2, category_1_id_list_3]
        merge_standard_food_id_list = [merge_standard_food_id_list, merge_standard_food_id_list_1, merge_standard_food_id_list_2, merge_standard_food_id_list_3]
        brand_id_list = [brand_id_list, brand_id_list_1, brand_id_list_2, brand_id_list_3]
        price_list = [price_list, price_list_1, price_list_2, price_list_3]
        shop_aoi_id_list = [shop_aoi_id_list, shop_aoi_id_list_1, shop_aoi_id_list_2, shop_aoi_id_list_3]
        shop_geohash6_list = [shop_geohash6_list, shop_geohash6_list_1, shop_geohash6_list_2, shop_geohash6_list_3]
        timediff_list = [timediff_list, timediff_list, timediff_list, timediff_list]
        week_hours_list = [week_hours_list, week_hours_list, week_hours_list, week_hours_list]
        time_type_list = [time_type_list, time_type_list, time_type_list, time_type_list]
        user_geohash6_list = [user_geohash6_list, user_geohash6_list, user_geohash6_list, user_geohash6_list]

        return shop_id_list, item_id_list, category_1_id_list, merge_standard_food_id_list, brand_id_list, price_list, shop_aoi_id_list, shop_geohash6_list, timediff_list, week_hours_list, time_type_list, user_geohash6_list


def collate_session_based(batch):
    '''
    get the output of dataset.__getitem__, and perform padding
    :param batch:
    :return:
    '''
    device = batch[0][-1]

    batch_size = len(batch)
    label = [item[0] for item in batch]
    user_id = [item[1] for item in batch]
    gender = [item[2] for item in batch]
    visit_city = [item[3] for item in batch]
    avg_price = [item[4] for item in batch]
    is_supervip = [item[5] for item in batch]
    ctr_30 = [item[6] for item in batch]
    ord_30 = [item[7] for item in batch]
    total_amt_30 = [item[8] for item in batch]
    shop_id_list = [item[9] + [batch[(index+15) % batch_size][9][0]] for index, item in enumerate(batch)]
    item_id_list = [item[10] + [batch[(index+15) % batch_size][10][0]] for index, item in enumerate(batch)]
    category_1_id_list = [item[11] + [batch[(index+15) % batch_size][11][0]] for index, item in enumerate(batch)]
    merge_standard_food_id_list = [item[12] + [batch[(index+15) % batch_size][12][0]] for index, item in enumerate(batch)]
    brand_id_list = [item[13] + [batch[(index+15) % batch_size][13][0]] for index, item in enumerate(batch)]
    price_list = [item[14] + [batch[(index+15) % batch_size][14][0]] for index, item in enumerate(batch)]
    shop_aoi_id_list = [item[15] + [batch[(index+15) % batch_size][15][0]] for index, item in enumerate(batch)]
    shop_geohash6_list = [item[16] + [batch[(index+15) % batch_size][16][0]] for index, item in enumerate(batch)]
    timediff_list = [item[17] + [batch[(index+15) % batch_size][17][0]] for index, item in enumerate(batch)]
    week_hours_list = [item[18] + [batch[(index+15) % batch_size][18][0]] for index, item in enumerate(batch)]
    time_type_list = [item[19] + [batch[(index+15) % batch_size][19][0]] for index, item in enumerate(batch)]
    user_geohash6_list = [item[20] + [batch[(index+15) % batch_size][20][0]] for index, item in enumerate(batch)]
    shop_id = [item[21] for item in batch]
    item_id = [item[22] for item in batch]
    city_id = [item[23] for item in batch]
    district_id = [item[24] for item in batch]
    shop_aoi_id = [item[25] for item in batch]
    shop_geohash_6 = [item[26] for item in batch]
    shop_geohash_12 = [item[27] for item in batch]
    brand_id = [item[28] for item in batch]
    category_1_id = [item[29] for item in batch]
    merge_standard_food_id = [item[30] for item in batch]
    rank_7 = [item[31] for item in batch]
    rank_30 = [item[32] for item in batch]
    rank_90 = [item[33] for item in batch]
    times = [item[34] for item in batch]
    hours = [item[35] for item in batch]
    time_type = [item[36] for item in batch]
    weekdays = [item[37] for item in batch]
    geohash12 = [item[38] for item in batch]
    his_length = [item[39] for item in batch]
    shop_id_list_r = [item[40] for item in batch]
    item_id_list_r = [item[41] for item in batch]
    category_1_id_list_r = [item[42] for item in batch]
    merge_standard_food_id_list_r = [item[43] for item in batch]
    brand_id_list_r = [item[44] for item in batch]
    price_list_r = [item[45] for item in batch]
    shop_aoi_id_list_r = [item[46] for item in batch]
    shop_geohash6_list_r = [item[47] for item in batch]
    timediff_list_r = [item[48] for item in batch]
    week_hours_list_r = [item[49] for item in batch]
    time_type_list_r = [item[50] for item in batch]
    user_geohash6_list_r = [item[51] for item in batch]
    his_length_r = [item[52] for item in batch], 
    return session_Batch(label,user_id, gender, visit_city, avg_price, is_supervip, ctr_30, ord_30, total_amt_30, shop_id_list, item_id_list, \
                         category_1_id_list, merge_standard_food_id_list, brand_id_list, price_list, shop_aoi_id_list, shop_geohash6_list, timediff_list, \
                         week_hours_list, time_type_list, user_geohash6_list , shop_id, item_id, city_id,district_id, shop_aoi_id, shop_geohash_6,\
                         shop_geohash_12, brand_id, category_1_id, merge_standard_food_id, rank_7, rank_30, rank_90, \
                         times, hours, time_type, weekdays, geohash12,his_length, shop_id_list_r, item_id_list_r, category_1_id_list_r, merge_standard_food_id_list_r,\
                         brand_id_list_r, price_list_r, shop_aoi_id_list_r, shop_geohash6_list_r, timediff_list_r,week_hours_list_r, time_type_list_r, user_geohash6_list_r, his_length_r, device)


class session_Batch():
    def __init__(self, label,user_id, gender, visit_city, avg_price, is_supervip, ctr_30, ord_30, total_amt_30, shop_id_list, item_id_list, \
                         category_1_id_list, merge_standard_food_id_list, brand_id_list, price_list, shop_aoi_id_list, shop_geohash6_list, timediff_list, \
                         week_hours_list, time_type_list, user_geohash6_list , shop_id, item_id, city_id,district_id, shop_aoi_id, shop_geohash_6,\
                         shop_geohash_12, brand_id, category_1_id, merge_standard_food_id, rank_7, rank_30, rank_90, \
                         times, hours, time_type, weekdays, geohash12, his_length,shop_id_list_r, item_id_list_r, category_1_id_list_r, merge_standard_food_id_list_r,\
                         brand_id_list_r, price_list_r, shop_aoi_id_list_r, shop_geohash6_list_r, timediff_list_r,week_hours_list_r, time_type_list_r, user_geohash6_list_r, his_length_r,  device):
        self.label = torch.LongTensor(label).to(device)
        self.user_id = torch.LongTensor(user_id).to(device)
        self.gender = torch.Tensor(gender).to(device)
        self.visit_city = torch.LongTensor(visit_city).to(device)
        self.avg_price = torch.Tensor(avg_price).to(device)
        self.is_supervip = torch.LongTensor(is_supervip).to(device)  
        self.ctr_30 = torch.Tensor(ctr_30).to(device) 
        self.ord_30 = torch.Tensor(ord_30).to(device) 
        self.total_amt_30 = torch.Tensor(total_amt_30).to(device)
        self.shop_id_list = torch.LongTensor(shop_id_list).to(device)
        self.item_id_list = torch.LongTensor(item_id_list).to(device)
        self.category_1_id_list = torch.LongTensor(category_1_id_list).to(device)
        self.merge_standard_food_id_list = torch.LongTensor(merge_standard_food_id_list).to(device)
        self.brand_id_list = torch.LongTensor(brand_id_list).to(device)  
        self.price_list = torch.Tensor(price_list).to(device)
        self.shop_aoi_id_list = torch.LongTensor(shop_aoi_id_list).to(device)  
        self.shop_geohash6_list = shop_geohash6_list
        self.timediff_list = torch.Tensor(timediff_list).to(device)
        self.week_hours_list = torch.LongTensor(week_hours_list).to(device)  
        self.time_type_list = torch.LongTensor(time_type_list).to(device) 
        self.user_geohash6_list = user_geohash6_list
        self.shop_id = torch.LongTensor(shop_id).to(device)  
        self.item_id = torch.LongTensor(item_id).to(device) 
        self.city_id = city_id  
        self.visit_city = visit_city 
        self.district_id = torch.LongTensor(district_id).to(device)  
        self.shop_aoi_id = torch.LongTensor(shop_aoi_id).to(device)  
        self.shop_geohash_6 = shop_geohash_6 
        self.shop_geohash_12 = shop_geohash_12
        self.brand_id = torch.LongTensor(brand_id).to(device) 
        self.category_1_id = torch.LongTensor(category_1_id).to(device) 
        self.merge_standard_food_id = torch.LongTensor(merge_standard_food_id).to(device)  
        self.rank_7 = torch.Tensor(rank_7).to(device) 
        self.rank_30 = torch.Tensor(rank_30).to(device)  
        self.rank_90 = torch.Tensor(rank_90).to(device)  
        self.times = torch.Tensor(times).to(device)  
        self.hours = torch.LongTensor(hours).to(device) 
        self.time_type = torch.LongTensor(time_type).to(device)  
        self.weekdays = torch.LongTensor(weekdays).to(device) 
        self.geohash12 = geohash12
        self.his_length = his_length
        self.shop_id_list_r = torch.LongTensor(shop_id_list_r).to(device)
        self.item_id_list_r = torch.LongTensor(item_id_list_r).to(device)
        self.category_1_id_list_r = torch.LongTensor(category_1_id_list_r).to(device)
        self.merge_standard_food_id_list_r = torch.LongTensor(merge_standard_food_id_list_r).to(device)
        self.brand_id_list_r = torch.LongTensor(brand_id_list_r).to(device)  
        self.price_list_r = torch.Tensor(price_list_r).to(device)
        self.shop_aoi_id_list_r = torch.LongTensor(shop_aoi_id_list_r).to(device)  
        self.shop_geohash6_list_r = shop_geohash6_list_r
        self.timediff_list_r = torch.Tensor(timediff_list_r).to(device)
        self.week_hours_list_r = torch.LongTensor(week_hours_list_r).to(device)  
        self.time_type_list_r = torch.LongTensor(time_type_list_r).to(device) 
        self.user_geohash6_list_r = user_geohash6_list_r
        self.his_length_r = his_length_r
