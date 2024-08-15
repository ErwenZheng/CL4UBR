import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from utils import *

class CL4UBR(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, i2i, i2t, i2c, i2f, i2a, device, downstream, tau, margin, lam_1, lam_2, delta, shift_init, scale_init, \
                 shop_num, brand_num, user_num, district_num, item_num, category_num, food_num, aoi_num):
        super(CL4UBR,self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.down_stream = downstream
        self.shift_init=torch.tensor(shift_init)
        self.scale_init=torch.tensor(scale_init)
        self.margin = margin
        self.lam_1 = lam_1
        self.lam_2 = lam_2
        self.delta = delta
        self.L = 1024
        self.k = 1024
        self.shop_num = shop_num
        self.brand_num = brand_num
        self.user_num = user_num
        self.district_num = district_num
        self.item_num = item_num
        self.category_num = category_num
        self.food_num = food_num
        self.aoi_num = aoi_num

        self.label_embedding = nn.Embedding(num_embeddings=3,embedding_dim=self.embedding_dim,padding_idx=0)

        self.i2i = self.get_sparse(i2i)
        self.i2t = self.get_sparse(i2t)
        self.i2c = self.get_sparse(i2c)
        self.i2f = self.get_sparse(i2f)
        self.i2a = self.get_sparse(i2a)

        # self.W_l = nn.Parameter(torch.randn(self.i2i.shape[1],self.L))
        self.W_l = nn.Parameter(torch.randn(self.i2i.shape[1],self.k)).to(self.device)
        # # self.b_l = nn.Parameter(torch.randn(1,self.L))
        self.b_l = nn.Parameter(torch.randn(self.i2i.shape[0],self.embedding_dim)).to(self.device)
        self.b_a = nn.Parameter(torch.randn(self.L,1)).to(self.device)
        self.W_e = nn.Parameter(torch.randn(self.L,self.k)).to(self.device)
        self.W_t = nn.Parameter(torch.randn(self.i2t.shape[1],self.k)).to(self.device)
        self.W_c = nn.Parameter(torch.randn(self.i2c.shape[1],self.k)).to(self.device)
        self.W_f = nn.Parameter(torch.randn(self.i2f.shape[1],self.k)).to(self.device)
        self.W_a = nn.Parameter(torch.randn(self.i2a.shape[1],self.k)).to(self.device)
        self.W_em = nn.Parameter(torch.randn(self.k,self.embedding_dim)).to(self.device)
        self.b_em = nn.Parameter(torch.randn(self.i2a.shape[0],self.embedding_dim)).to(self.device)
        self.W_s = nn.Parameter(torch.randn(self.shop_num,self.i2i.shape[0])).to(self.device)

        self.leakyrulu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # # self.e_l = []
        # # for i in tqdm(range(self.i2i.shape[0])):
        # #     _sum = 0
        # #     for j in range(self.i2i.shape[0]):
        # #         _sum = _sum + self.alpha(self.s(i),self.s(j)) @ self.s(j) @ self.W_e 
        # #     self.e_l_i = self.sigmoid(_sum)
        # #     self.e_l.append(self.e_l_i)
        # # self.e_l = torch.concat(self.e_l)

        self.e_l = self.tanh(self.i2i @ self.W_l).to_dense()
        self.e_t = self.tanh(self.i2t @ self.W_t).to_dense()
        self.e_c = self.tanh(self.i2c @ self.W_c).to_dense()
        self.e_f = self.tanh(self.i2f @ self.W_f).to_dense()
        self.e_a = self.tanh(self.i2a @ self.W_a).to_dense()
        self.item_embed = (self.e_l + self.e_t + self.e_c + self.e_f + self.e_a) @ self.W_em + self.b_em
        self.shop_embed = self.W_s @ self.item_embed 
        self.item_embedding = nn.Embedding.from_pretrained(self.item_embed,freeze=False,padding_idx=0)
        self.shop_embedding = nn.Embedding.from_pretrained(self.shop_embed,freeze=False,padding_idx=0)
        # self.item_embedding = nn.Embedding(self.item_num,embedding_dim=self.embedding_dim,padding_idx=0)
        # self.shop_embedding = nn.Embedding(self.shop_num,embedding_dim=self.embedding_dim,padding_idx=0)
        self.time_embedding = nn.Embedding(169,embedding_dim=self.embedding_dim,padding_idx=0)
        self.brand_embedding = nn.Embedding(self.brand_num, embedding_dim=self.embedding_dim,padding_idx=0)
        self.c_embedding = nn.Embedding(self.category_num,embedding_dim=self.embedding_dim,padding_idx=0)
        self.f_embedding = nn.Embedding(self.food_num,embedding_dim=self.embedding_dim,padding_idx=0)
        self.aoi_embedding = nn.Embedding(self.aoi_num,embedding_dim=self.embedding_dim,padding_idx=0)
        self.time_type_embedding = nn.Embedding(6, embedding_dim=self.embedding_dim,padding_idx=0)
        self.user_embedding = nn.Embedding(self.user_num, embedding_dim=self.embedding_dim,padding_idx=0)
        self.district_embedding = nn.Embedding(self.district_num, embedding_dim=self.embedding_dim,padding_idx=0)
        self.gender_proj = nn.Linear(in_features=1,out_features=self.embedding_dim)
        self.gender_emnbedding = nn.Embedding(4, embedding_dim=self.embedding_dim,padding_idx=0)
        self.vip_embedding = nn.Embedding(3, embedding_dim=self.embedding_dim,padding_idx=0)

        self.price_proj = nn.Linear(in_features=1, out_features=self.embedding_dim)
        self.location_proj = nn.Linear(in_features=1, out_features=self.embedding_dim)
        self.time_diff_proj = nn.Linear(in_features=1, out_features=self.embedding_dim)
        self.rank_proj = nn.Linear(in_features=1, out_features=embedding_dim)
        self.ctr_proj = nn.Linear(in_features=1, out_features=embedding_dim)
        self.ord_proj = nn.Linear(in_features=1, out_features=embedding_dim)

        self.cs_proj = nn.Linear(in_features=self.embedding_dim*4, out_features=self.hidden_dim*1)

        self.h_Contrast = h_Contrast(hidden_dim=embedding_dim,tau=tau,down_stream=self.down_stream)

        self.shop_rnn = nn.LSTM(input_size = self.embedding_dim, hidden_size=self.hidden_dim, batch_first = True)
        self.item_rnn = nn.LSTM(input_size = self.embedding_dim, hidden_size=self.hidden_dim, batch_first = True)
        self.time_rnn = nn.LSTM(input_size = self.embedding_dim, hidden_size=self.hidden_dim, batch_first = True)
        self.location_rnn = nn.LSTM(input_size = self.embedding_dim, hidden_size=self.hidden_dim, batch_first = True)
        self.user_proj = nn.Linear(in_features=embedding_dim,out_features=hidden_dim)

        self.p_mask = nn.Parameter(torch.randn(self.hidden_dim * 5, self.hidden_dim * 5))
        self.n_mask = nn.Parameter(torch.randn(self.hidden_dim * 5, self.hidden_dim * 5))
        self.MLP_p = nn.Sequential(
                       nn.Linear(in_features=self.hidden_dim * 11, out_features=self.hidden_dim),
                       nn.Sigmoid(),
                       nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim // 8),
                       nn.Sigmoid(),
                       nn.Linear(in_features=self.hidden_dim // 8, out_features=1),
                      )
        self.MLP_n = nn.Sequential(
                       nn.Linear(in_features=self.hidden_dim * 11, out_features=self.hidden_dim),
                       nn.Sigmoid(),
                       nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim // 8),
                       nn.Sigmoid(),
                       nn.Linear(in_features=self.hidden_dim // 8, out_features=1),
                      )
        
        self.mae = nn.L1Loss()

        self.p_proj = nn.Linear(in_features=1, out_features=self.hidden_dim*5)
        self.n_proj = nn.Linear(in_features=1, out_features=self.hidden_dim*5)
        self.label_linear = nn.Linear(in_features=self.embedding_dim,out_features=hidden_dim)

        #CTR
        self.CTR_MLP = nn.Sequential(
                       nn.Linear(in_features=self.hidden_dim * 4, out_features=self.hidden_dim),
                       nn.BatchNorm1d(num_features=self.hidden_dim),
                       nn.Sigmoid(),
                       nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim // 8),
                       nn.BatchNorm1d(num_features=self.hidden_dim // 8),
                       nn.Sigmoid(),
                       nn.Linear(in_features=self.hidden_dim // 8, out_features=1),
                      )
        self.pos_weight = torch.tensor([1])
        self.BCEloss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        #TPP
        self.TPP_MLP = nn.Sequential(
                       nn.Linear(in_features=self.hidden_dim * 4, out_features=self.hidden_dim),
                       nn.Sigmoid(),
                       nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim // 8),
                       nn.Sigmoid(),
                       nn.Linear(in_features=self.hidden_dim // 8, out_features=1),
                      )

    def get_sparse(self,matrix):
        matrix = matrix.tocoo()
        values = matrix.data
        indices = np.vstack((matrix.row, matrix.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = torch.Size(matrix.shape)
        return torch.sparse.FloatTensor(i, v, shape).to(self.device)
    
    def his_item_feature(self,batch):
        item = batch.item_id_list.to(self.device)
        price = batch.price_list.to(self.device)
        brand_id = batch.brand_id_list.to(self.device)
        category_1_id_list = batch.category_1_id_list.to(self.device)
        standard_food_id = batch.merge_standard_food_id_list.to(self.device)
        return item, price, brand_id, category_1_id_list, standard_food_id

    def his_shop_feature(self,batch):
        shop = batch.shop_id_list.to(self.device)
        shop_aoi = batch.shop_aoi_id_list.to(self.device)
        shop_geohash = batch.shop_geohash6_list
        return shop, shop_aoi, shop_geohash

    def his_st_feature(self,batch):
        time = batch.week_hours_list.to(self.device)
        time_type = batch.time_type_list.to(self.device)
        time_diff = batch.timediff_list.to(self.device)
        user_geohash = batch.user_geohash6_list
        return time, time_type, time_diff, user_geohash
    
    def his_item_feature_r(self,batch):
        item = batch.item_id_list_r.to(self.device)
        price = batch.price_list_r.to(self.device)
        brand_id = batch.brand_id_list_r.to(self.device)
        category_1_id_list = batch.category_1_id_list_r.to(self.device)
        standard_food_id = batch.merge_standard_food_id_list_r.to(self.device)
        return item, price, brand_id, category_1_id_list, standard_food_id

    def his_shop_feature_r(self,batch):
        shop = batch.shop_id_list_r.to(self.device)
        shop_aoi = batch.shop_aoi_id_list_r.to(self.device)
        shop_geohash = batch.shop_geohash6_list_r
        return shop, shop_aoi, shop_geohash

    def his_st_feature_r(self,batch):
        time = batch.week_hours_list_r.to(self.device)
        time_type = batch.time_type_list_r.to(self.device)
        time_diff = batch.timediff_list_r.to(self.device)
        user_geohash = batch.user_geohash6_list_r
        return time, time_type, time_diff, user_geohash
    
    def cur_st_feature(self,batch):
        times = batch.times.to(self.device)
        hours = batch.hours.to(self.device)
        time_type = batch.time_type.to(self.device)
        weekdays = batch.weekdays.to(self.device)
        geohash12 = batch.geohash12
        return times, hours, time_type, weekdays, geohash12

    def cur_shop_feature(self,batch):
        shop_id = batch.shop_id.to(self.device)
        shop_aoi_id = batch.shop_aoi_id.to(self.device)
        shop_geohash_6 = batch.shop_geohash_6
        return shop_id, shop_aoi_id, shop_geohash_6
    
    def cur_item_feature(self,batch):
        item_id = batch.item_id.to(self.device)
        district_id = batch.district_id.to(self.device)
        category_1_id = batch.category_1_id.to(self.device)
        merge_standard_food_id = batch.merge_standard_food_id.to(self.device)
        rank_7 = batch.rank_7.to(self.device)
        rank_30 = batch.rank_30.to(self.device)
        rank_90 = batch.rank_90.to(self.device)
        return item_id, district_id, category_1_id, merge_standard_food_id, rank_7, rank_30, rank_90
    
    def user_static_feature(self,batch):
        user_id = batch.user_id.to(self.device)
        gender = batch.gender.to(self.device)
        avg_price = batch.avg_price.to(self.device)
        is_supervip = batch.is_supervip.to(self.device)
        ctr_30 = batch.ctr_30.to(self.device)
        ord_30 = batch.ord_30.to(self.device)
        total_amt_30 = batch.total_amt_30.to(self.device)
        return user_id, gender, avg_price, is_supervip, ctr_30, ord_30, total_amt_30

    def embed_r(self,batch):
        lenth = batch.his_length_r[0]
        item, price, brand_id, category_1_id_list, standard_food_id = self.his_item_feature_r(batch)
        shop, shop_aoi, shop_geohash = self.his_shop_feature_r(batch)
        time, time_type, time_diff, user_geohash = self.his_st_feature_r(batch)
        user_id = self.user_static_feature(batch)[0]

        if self.down_stream == 'TPP':
            shop,_ = label_get_r(shop,lenth)
            shop_aoi,_ = label_get_r(shop_aoi,lenth)
            item,_ = label_get_r(item,lenth)
            price,_ = label_get_r(price,lenth)
            brand_id,_ = label_get_r(brand_id,lenth)
            category_1_id_list,_ = label_get_r(category_1_id_list,lenth)
            standard_food_id,_ = label_get_r(standard_food_id,lenth)
            time,_ = label_get_r(time,lenth)
            user_geohash,_ = geo_label_get_r(user_geohash,lenth)
            shop_geohash,_ = geo_label_get_r(shop_geohash,lenth)
            
            time_type,_ = label_get_r(time_type,lenth)
            time_diff,_ = label_get_r(time_diff,lenth)
            his_label = []
            for i in lenth:
                if i == 1:
                    cur_label = [1,2] + [0] * (96)
                else:
                    cur_label = [1,2] * int(i/2 - 1) + [0] * (100-i)
                his_label.append(cur_label)
            
            cur_lenth = []
            for i,j in enumerate(lenth):
                if j == 0:
                    cur_lenth.append(lenth[i-1]-2)
                elif j == 2:
                    cur_lenth.append(j)
                else:
                    cur_lenth.append(j-2)
            lenth = cur_lenth
        else:
            his_label = []
            for i in lenth:
                cur_label = [1,2] * int(i/2) + [0] * (100-i)
                his_label.append(cur_label)

        his_label = torch.LongTensor(his_label).to(self.device)
        his_label = self.label_embedding(his_label)

        user_embedding = self.user_embedding(user_id.reshape(user_id.shape[0],1).repeat(1,100))

        shop_location = []
        for i in shop_geohash:
            loaction_list = []
            for j in i:
                loaction_list.append(get_hash(j))
            shop_location.append(loaction_list)
        shop_location =  torch.tensor(shop_location).reshape(len(shop_location),-1,1).to(self.device)
        shop_location = self.tanh(self.location_proj(shop_location))

        item_embedding = self.item_embedding(item) + self.tanh(self.price_proj(price.reshape(price.shape[0],-1,1))) + self.brand_embedding(brand_id) + self.c_embedding(category_1_id_list) + self.f_embedding(standard_food_id) + his_label
        shop_embedding = self.shop_embedding(shop) + self.aoi_embedding(shop_aoi) + shop_location + his_label
        time_embedding = self.time_embedding(time) + self.time_type_embedding(time_type) + self.tanh(self.price_proj(time_diff.reshape(time_diff.shape[0],-1,1)))

        user_location = []
        for i in user_geohash:
            loaction_list = []
            for j in i:
                loaction_list.append(get_hash(j))
            user_location.append(loaction_list)
        user_location =  torch.tensor(user_location).reshape(len(user_location),-1,1).to(self.device)
        user_location = self.tanh(self.location_proj(user_location))
        
        return shop_embedding, item_embedding, time_embedding, user_location, user_embedding, lenth
    
    def embed(self,batch):
        lenth = batch.his_length
        item, price, brand_id, category_1_id_list, standard_food_id = self.his_item_feature(batch)
        shop, shop_aoi, shop_geohash = self.his_shop_feature(batch)
        time, time_type, time_diff, user_geohash = self.his_st_feature(batch)
        user_id = self.user_static_feature(batch)[0]

        if self.down_stream == 'TPP':
            shop,_ = label_get(shop,lenth)
            shop_aoi,_ = label_get(shop_aoi,lenth)
            item,_ = label_get(item,lenth)
            price,_ = label_get(price,lenth)
            brand_id,_ = label_get(brand_id,lenth)
            category_1_id_list,_ = label_get(category_1_id_list,lenth)
            standard_food_id,_ = label_get(standard_food_id,lenth)
            time,_ = label_get(time,lenth)
            user_geohash,_ = geo_label_get(user_geohash,lenth)
            shop_geohash,_ = geo_label_get(shop_geohash,lenth)
            
            time_type,_ = label_get(time_type,lenth)
            time_diff,_ = label_get(time_diff,lenth)

            cur_lenth = []
            for i,j in enumerate(lenth):
                if j == 0:
                    cur_lenth.append(lenth[i-1]-1)
                elif j == 1:
                    cur_lenth.append(j)
                else:
                    cur_lenth.append(j-1)
            lenth = cur_lenth

        user_embedding = self.user_embedding(user_id.reshape(user_id.shape[0],1).repeat(1,50))

        shop_location = []
        for i in shop_geohash:
            loaction_list = []
            for j in i:
                loaction_list.append([get_hash(k) for k in j])
            shop_location.append(loaction_list)
        shop_location =  torch.tensor(shop_location).reshape(len(shop_location),5,-1,1).to(self.device)
        shop_location = self.tanh(self.location_proj(shop_location))

        item_embedding = self.item_embedding(item) + self.tanh(self.price_proj(price.reshape(price.shape[0],5,-1,1))) + self.brand_embedding(brand_id) + self.c_embedding(category_1_id_list) + self.f_embedding(standard_food_id)
        shop_embedding = self.shop_embedding(shop) + self.aoi_embedding(shop_aoi) + shop_location
        time_embedding = self.time_embedding(time) + self.time_type_embedding(time_type) + self.tanh(self.price_proj(time_diff.reshape(time_diff.shape[0],5,-1,1)))

        user_location = []
        for i in user_geohash:
            loaction_list = []
            for j in i:
                loaction_list.append([get_hash(k) for k in j])
            user_location.append(loaction_list)
        user_location =  torch.tensor(user_location).reshape(len(user_location),5,-1,1).to(self.device)
        user_location = self.tanh(self.location_proj(user_location))
        
        return shop_embedding, item_embedding, time_embedding, user_location, user_embedding, lenth
    
    def s(self,i):
        s_i = self.i2i[i] @ self.W_l + self.b_l
        return s_i.to_dense()
    
    def a(self,s_i,s_j):
        return (s_i + s_j) @ self.b_a
    
    def alpha(self,s_i,s_j):
        un = torch.tensor(0.)
        for i in range(self.i2i.shape[0]):
            un =un + torch.exp(self.leakyrulu(self.a(s_i,self.s(i))))
        return torch.exp(self.leakyrulu(self.a(s_i,s_j)))/un

    def h_r_encoder(self,shop_embedding, item_embedding, time_embedding, user_location, user_embedding, lenth):
        pack_shop = pack_padded_sequence(shop_embedding.permute(1, 0, 2), lengths=lenth, enforce_sorted=False)
        pack_item = pack_padded_sequence(item_embedding.permute(1, 0, 2), lengths=lenth, enforce_sorted=False)
        pack_time = pack_padded_sequence(time_embedding.permute(1, 0, 2), lengths=lenth, enforce_sorted=False)
        pack_location = pack_padded_sequence(user_location.permute(1, 0, 2), lengths=lenth, enforce_sorted=False)

        shop_out, (h_n, c_n) = self.shop_rnn(pack_shop)
        item_out, (h_n, c_n) = self.item_rnn(pack_item)
        time_out, (h_n, c_n) = self.time_rnn(pack_time)
        location_out, (h_n, c_n) = self.location_rnn(pack_location)

        shop_out, out_len = pad_packed_sequence(shop_out, batch_first=True)
        item_out, out_len = pad_packed_sequence(item_out, batch_first=True)
        time_out, out_len = pad_packed_sequence(time_out, batch_first=True)
        location_out, out_len = pad_packed_sequence(location_out, batch_first=True)
        user_embedding = self.sigmoid(self.user_proj(user_embedding))

        rep = torch.concat((shop_out[:,-1,:],item_out[:,-1,:],time_out[:,-1,:],location_out[:,-1,:],user_embedding[:,-1,:]),dim=-1)

        return rep 
    
    def h_encoder(self,shop_embedding, item_embedding, time_embedding, user_location, user_embedding, lenth):
        rep_list = []
        user_embedding = self.sigmoid(self.user_proj(user_embedding))
        for i in range(5):
            shop = shop_embedding[:,i,:,:]
            item = item_embedding[:,i,:,:]
            time = time_embedding[:,i,:,:]
            user_loc = user_location[:,i,:,:]

            pack_shop = pack_padded_sequence(shop.permute(1, 0, 2), lengths=lenth, enforce_sorted=False)
            pack_item = pack_padded_sequence(item.permute(1, 0, 2), lengths=lenth, enforce_sorted=False)
            pack_time = pack_padded_sequence(time.permute(1, 0, 2), lengths=lenth, enforce_sorted=False)
            pack_location = pack_padded_sequence(user_loc.permute(1, 0, 2), lengths=lenth, enforce_sorted=False)

            shop_out, (h_n, c_n) = self.shop_rnn(pack_shop)
            item_out, (h_n, c_n) = self.item_rnn(pack_item)
            time_out, (h_n, c_n) = self.time_rnn(pack_time)
            location_out, (h_n, c_n) = self.location_rnn(pack_location)

            shop_out, out_len = pad_packed_sequence(shop_out, batch_first=True)
            item_out, out_len = pad_packed_sequence(item_out, batch_first=True)
            time_out, out_len = pad_packed_sequence(time_out, batch_first=True)
            location_out, out_len = pad_packed_sequence(location_out, batch_first=True)
            

            rep = torch.concat((shop_out[:,-1,:],item_out[:,-1,:],time_out[:,-1,:],location_out[:,-1,:],user_embedding[:,-1,:]),dim=-1)
            rep_list.append(rep.reshape(rep.shape[0],1,rep.shape[1]))

        return torch.concat(rep_list,dim=1) 
    
    def disentanglement(self,rep):
        rep_p = self.tanh(rep @ self.p_mask)
        rep_n = self.tanh(rep @ self.n_mask)
        return rep_p, rep_n
    
    def get_cs_feature(self,batch):
        item_id, district_id, category_1_id, merge_standard_food_id, rank_7, rank_30, rank_90 = self.cur_item_feature(batch)
        shop_id, shop_aoi_id, shop_geohash_6 = self.cur_shop_feature(batch)
        times, hours, time_type, weekdays, geohash12 = self.cur_st_feature(batch)
        user_id, gender, avg_price, is_supervip, ctr_30, ord_30, total_amt_30 = self.user_static_feature(batch)

        cur_item = self.item_embedding(item_id) + self.district_embedding(district_id) + self.c_embedding(category_1_id) + self.f_embedding(merge_standard_food_id) + self.tanh(self.rank_proj(rank_7.reshape(-1,1))) + self.tanh(self.rank_proj(rank_30.reshape(-1,1))) + self.tanh(self.rank_proj(rank_90.reshape(-1,1)))
        cur_shop = self.shop_embedding(shop_id) + self.aoi_embedding(shop_aoi_id) + self.tanh(self.location_proj(get_hash_tensor(shop_geohash_6).to(self.device).reshape(-1,1)))
        cur_st = self.time_embedding(weekdays * 24 + hours + 1) + self.time_type_embedding(time_type) + self.tanh(self.location_proj(get_hash_tensor(geohash12).to(self.device).reshape(-1,1)))
        s_user = self.user_embedding(user_id) + self.tanh(self.gender_proj(gender.reshape(-1,1))) + self.tanh(self.price_proj(avg_price.reshape(-1,1))) + self.vip_embedding(is_supervip) + self.tanh(self.ctr_proj(ctr_30.reshape(-1,1))) + self.tanh(self.ord_proj(ord_30.reshape(-1,1))) + self.tanh(self.price_proj(total_amt_30.reshape(-1,1)))
        
        return self.tanh(self.cs_proj(torch.concat((cur_item, cur_shop, cur_st, s_user), dim=-1)))

    def gate(self, cs_featrue, rep, rep_p, rep_n):
        p_gate = self.sigmoid(self.p_proj(cs_featrue.reshape(cs_featrue.shape[0],self.hidden_dim,1)))
        n_gate = self.sigmoid(self.n_proj(cs_featrue.reshape(cs_featrue.shape[0],self.hidden_dim,1)))
        return torch.concat((torch.einsum('bkh, bh -> bk',(1 - p_gate - n_gate), rep), + torch.einsum('bkh, bh -> bk', p_gate, rep_p), torch.einsum('bkh, bh -> bk', n_gate, rep_n)), dim=-1)

    def predict_tower(self, rep, rep_p, rep_n, cs_featrue):
        logit_p = self.MLP_p(torch.concat((rep_p, rep, self.label_linear(self.label_embedding(torch.tensor(1).to(self.device))) + cs_featrue),dim=-1))
        logit_n = self.MLP_n(torch.concat((rep_n, rep, self.label_linear(self.label_embedding(torch.tensor(0).to(self.device))) + cs_featrue),dim=-1))
        return logit_p, logit_n

    def ST_Contrast(self, rep):
        loss = 0.0
        anchor = torch.concat([item[0] for item in rep],dim=-1)
        positives = [torch.concat([item[1] for item in rep],dim=-1),torch.concat([item[2] for item in rep],dim=-1),torch.concat([item[3] for item in rep],dim=-1)]
        negative = torch.concat([item[-1] for item in rep],dim=-1)
        neg_sim = F.cosine_similarity(anchor, negative,dim=0)
        for positive in positives:
            pos_sim = F.cosine_similarity(anchor, positive,dim=0)
            cur_loss = torch.exp(pos_sim/self.margin)/(torch.exp(pos_sim/self.margin) + torch.exp(neg_sim/self.margin))
            loss += torch.log(cur_loss)
        
        return -loss / len(positives)

    def BPR_Contrast(self,logit_p, logit_n):
        loss = -torch.log(self.sigmoid(torch.abs(logit_p - logit_n)))
        return torch.mean(loss)
    
    def normalized(self,x):
        return (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    
    def get_prediction(self,x):
        x = self.sigmoid(x.reshape(x.shape[0]))
        predictions = (x >= 0.5).float()
        return predictions

    def forward(self,batch):
        shop_embedding, item_embedding, time_embedding, user_location, user_embedding, lenth = self.embed(batch)
        shop_r_embedding, item_r_embedding, time_r_embedding, user_r_location, user_r_embedding, lenth_r = self.embed_r(batch)
        rep = self.h_encoder(shop_embedding, item_embedding, time_embedding, user_location, user_embedding, lenth)# b * 5h
        rep_r = self.h_r_encoder(shop_r_embedding, item_r_embedding, time_r_embedding, user_r_location, user_r_embedding, lenth_r)# b * 5h
        rep_p, rep_n = self.disentanglement(rep_r)# b * 5h
        cs_featrue = self.get_cs_feature(batch)# b * h
        gated_rep = self.gate(cs_featrue, rep_r, rep_p, rep_n)# b * 3h
        if self.down_stream == 'pretrain':
            shop_view_embedding = shop_embedding[:,0,:,:] + time_embedding[:,0,:,:] + user_location[:,0,:,:] + user_embedding
            item_view_embedding = item_embedding[:,0,:,:] + time_embedding[:,0,:,:] + user_location[:,0,:,:] + user_embedding
            ch_loss = self.h_Contrast(shop_view_embedding, item_view_embedding)
            st_loss = self.ST_Contrast(rep)
            logit_p, logit_n = self.predict_tower(rep_r, rep_p, rep_n, cs_featrue)
            bpr_loss = self.BPR_Contrast(logit_p, logit_n)
            return (1 - self.lam_1 - self.lam_2) * bpr_loss + self.lam_1 * ch_loss + self.lam_2 * st_loss
        if self.down_stream == 'TPP':
            shop_view_embedding = shop_embedding[:,0,:,:] + time_embedding[:,0,:,:] + user_location[:,0,:,:] + user_embedding[:,1:,:]
            item_view_embedding = item_embedding[:,0,:,:] + time_embedding[:,0,:,:] + user_location[:,0,:,:] + user_embedding[:,1:,:]
        elif self.down_stream == 'CTR':
            shop_view_embedding = shop_embedding[:,0,:,:] + time_embedding[:,0,:,:] + user_location[:,0,:,:] + user_embedding
            item_view_embedding = item_embedding[:,0,:,:] + time_embedding[:,0,:,:] + user_location[:,0,:,:] + user_embedding
        ch_loss = self.h_Contrast(shop_view_embedding, item_view_embedding)
        st_loss = self.ST_Contrast(rep)
        logit_p, logit_n = self.predict_tower(rep_r, rep_p, rep_n, cs_featrue)
        bpr_loss = self.BPR_Contrast(logit_p, logit_n)
        if self.down_stream == 'CTR':
            prediction = self.get_prediction(self.CTR_MLP(torch.concat((gated_rep, cs_featrue), dim=-1)))
            loss = self.BCEloss(prediction, batch.label.to(self.device).float())/prediction.shape[0]
            total_loss = (1 - self.lam_1 - self.lam_2) * bpr_loss + self.lam_1 * ch_loss + self.lam_2 * st_loss + self.delta * loss #联合训练
            return total_loss/(self.delta + 1), prediction, loss
        elif self.down_stream == 'TPP':
            time_diff = batch.timediff_list.to(self.device)
            _,y_label = label_get(time_diff,lenth)
            y = torch.log(y_label + 1e-2).unsqueeze(-1) 
            y = torch.flatten((y - self.shift_init.to(y.device)) / self.scale_init.to(y.device))
            mean_time = torch.flatten(self.TPP_MLP(torch.concat((gated_rep, cs_featrue), dim=-1)))
            loss = self.mae(mean_time, (y))
            a = self.scale_init.to(y.device)
            b = self.shift_init.to(y.device)
            mean_time = torch.exp(a * mean_time + b )
            total_loss = (1 - self.lam_1 - self.lam_2) * bpr_loss + self.lam_1 * ch_loss + self.lam_2 * st_loss + self.delta * loss #联合训练
            return total_loss/(self.delta + 1), mean_time, y_label.detach().cpu().numpy(), loss





class h_Contrast(nn.Module):
    def __init__(self, hidden_dim, tau, down_stream):
        super(h_Contrast, self).__init__()
        if down_stream == 'CTR':
            self.proj = nn.Sequential(
                nn.Linear(50* hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        else:
            self.proj = nn.Sequential(
                nn.Linear(49* hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        self.tau = tau
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, v1, v2):
        v1_norm = torch.norm(v1, dim=-1, keepdim=True)
        v2_norm = torch.norm(v2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(v1, v2.t())
        dot_denominator = torch.mm(v1_norm, v2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def forward(self, v_s, v_i):
        v_proj_s = self.proj(v_s.reshape(v_s.shape[0],-1))
        v_proj_i = self.proj(v_i.reshape(v_s.shape[0],-1))
        matrix_s2i = F.cosine_similarity(v_proj_s, v_proj_i) / (self.tau * 10)
        # matrix_s2i = matrix_s2i/(torch.sum(matrix_s2i, dim=1).view(-1, 1) + 1e-8)
        h_loss = -torch.log(matrix_s2i.sum(dim=-1)).mean()
        return h_loss
