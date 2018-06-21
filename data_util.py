import numpy as np
import pandas as pd
import os
import json
from itertools import groupby
from util import *
from pprint import pprint


class DataLoader:

    def __init__(self, data_folder, data_set, split_rates, **opts):

        self.data_folder = data_folder
        self.data_set = data_set        
        self.split_rates = split_rates
        self.verbosity = opts.get('verbosity', 0)
        self.resplit_data = opts.get('verbosity', True)

        self.entites = {'user':0, 'business':1, 'category':2}
        
        self.tables = self._load_tables()

    def _load_tables(self):


        if self.data_set == 'yelp_debug' or self.data_set == 'yelp':


            save_path = os.path.join(self.data_folder, self.data_set, 'numpy')
            if not os.path.isdir(save_path):
                os.makedirs(save_path)

            save_path = os.path.join(save_path, 'yelp_data.npz')
            if os.path.isfile(save_path):            
                print(".... loading data from .npz file ....")
                data = np.load(save_path)
                review = data['review'].item()
                friend = data['friend'].item()
                category = data['category'].item()
                n_users = review['shape'][0]
                n_businesses = review['shape'][1]
                # n_reviews = review['shape'][2]
                n_categories = category['shape'][1]
                print(".... finished ....\n")

            else:
                businesses = self._load_csv("yelp_business.csv")
                users = self._load_csv("yelp_user.csv")
                reviews = self._load_csv("yelp_review.csv")
                
                friend_indices, friend_values, user_ids = self._get_friend_table_data(users) #generalize: '_get_self_self_interaction'                               
                category_indices, category_values, business_ids, category_ids = self._get_category_table_data(businesses)
                review_indices, review_values, review_ids = self._get_reviews_table_data(reviews, business_ids, user_ids)
                
                n_users = len(user_ids)
                n_businesses = len(business_ids)
                # n_reviews = len(review_ids)
                n_categories = len(category_ids)
                
                review = {'indices':review_indices,
                          'values':review_values,
                          # 'shape':[n_users, n_businesses, n_reviews]}
                          'shape':[n_users, n_businesses]}
                friend = {'indices':friend_indices,
                          'values':friend_values,
                          'shape':[n_users, n_users]}
                category = {'indices':category_indices,
                            'values':category_values,
                            'shape':[n_businesses, n_categories]}

                data = {'review':review,
                        'friend':friend,
                        'category':category}
                
                print(".... saving data to .npz ....")
                np.savez_compressed(save_path, **data)
                print(".... finished ....\n")

            print(".... splitting data into train/validation/test ....")
            review['split'] = self._get_split(review['indices'].shape[0])
            friend['split'] = self._get_split(friend['indices'].shape[0])
            category['split'] = self._get_split(category['indices'].shape[0])
            print(".... finished ....\n")

            table_review = Table(0, review, [0, 1], predict=True)   # user x business
            table_friend = Table(1, friend, [0, 0])     # user x user
            table_category = Table(2, category, [1, 2])     # business x category 

            print(".... data loaded:", n_users, "users,",  n_businesses, "businesses,",  n_categories, "categories")
            print(".... ", table_review.num_obs, "review-relationships,", table_friend.num_obs, "friend-relationships,", table_category.num_obs, "categories-relationships ....\n")

            return {'table_0':table_review, 'table_1':table_friend, 'table_2':table_category}
            # return {'table_0':table_review}



    def _update_id_dict(self, id_dict, key):
        if key not in id_dict:
            id_dict[key] = len(id_dict)


    def _get_category_table_data(self, businesses):
        print(".... creating categories table ....")
        category_indices = []
        business_ids = {}
        category_ids = {}        
        for _, business in businesses.iterrows():
            self._update_id_dict(business_ids, business['business_id'])
            for category in business['categories'].split(';'):
                self._update_id_dict(category_ids, category.strip())
                category_indices.append([ business_ids[business['business_id']], category_ids[category] ])
        category_indices.sort()
        category_indices = np.array(category_indices)
        n = category_indices.shape[0]
        category_values = np.ones(n)
        print(".... finished ....\n")
        return category_indices, category_values, business_ids, category_ids


    def _get_friend_table_data(self, users):
        print(".... creating friends table ....")      
        friend_indices = []
        user_ids = {}
        for _, user in users.iterrows():
            self._update_id_dict(user_ids, user['user_id'])
            if user['friends'] != 'None':
                for friend in user['friends'].split(','):                
                    self._update_id_dict(user_ids, friend.strip())
                    friend_indices.append([ user_ids[user['user_id']], user_ids[friend.strip()] ])
        friend_indices.sort()
        friend_indices = np.array(friend_indices)
        n = friend_indices.shape[0]
        friend_values = np.ones(n)
        print(".... finished ....\n")
        return friend_indices, friend_values, user_ids


    # def _get_table_data(self, data_frame, index_names, value_names):
    #     n_dims = len(index_names)
    #     n_features = len(value_names)
    #     data_frame = data_frame[index_names + value_names].copy()        
    #     data_frame.sort_values(by=index_names, inplace=True)
    #     data_frame = np.array(data_frame)
    #     return data_frame[:,0:n_dims], np.squeeze(data_frame[:,n_dims:(n_dims+n_features)])


    def _get_reviews_table_data(self, reviews, business_ids, user_ids):
        print(".... creating reviews table ....")                
        reviews_data = []       
        review_ids = {}
        for _, review in reviews.iterrows():
            self._update_id_dict(review_ids, review['review_id'])            
            # data = [ user_ids[review['user_id']], business_ids[review['business_id']], review_ids[review['review_id']], review['stars'] ]
            data = [ user_ids[review['user_id']], business_ids[review['business_id']] , review['stars'] ]
            reviews_data.append(data)
        reviews_data.sort()
        reviews_data = np.array(reviews_data)
        print(".... finished ....\n")
        # return reviews_data[:,0:3], reviews_data[:,3], review_ids
        return reviews_data[:,0:2], reviews_data[:,2], review_ids


    def _get_split(self, n, randomize=True):
        n_tr, n_vl, n_ts = self._get_split_sizes(n)
        split = np.concatenate( (np.zeros(n_tr, np.int32), np.ones(n_vl, np.int32), 2*np.ones(n_ts, np.int32)), axis=0 )
        if randomize:
            np.random.shuffle(split)
        return split


    def _get_split_sizes(self, n):
        n_ts = int(n * self.split_rates[2])
        n_vl = int(n * self.split_rates[1])
        n_tr = n - n_vl - n_ts
        return n_tr, n_vl, n_ts



    def _map_ids(self, data_frame, id_name):
        df_ids = data_frame[id_name].unique()
        n = df_ids.shape[0]
        return dict(zip(df_ids, range(n)))



    def _load_csv(self, file_name, do_split=False):
        print(".... loading", file_name, "....")
        table = pd.read_csv(os.path.join(self.data_folder, self.data_set, file_name))
        n = table.shape[0]
        if do_split:            
            n_ts = int(self.split_rates[2] * n)
            n_vl = int(self.split_rates[1] * n)
            n_tr = n - n_vl - n_ts
            split = np.concatenate((np.zeros(n_tr, np.int32), np.ones(n_vl, np.int32), 2*np.ones(n_ts, np.int32)), axis=0)
            table['split'] = split
        print("....", n, "records loaded from", file_name, "....\n")
        return table.sample(frac=1.).reset_index(drop=True)


class Table:
    def __init__(self, tid, data, entities, num_features=1, predict=False):
        self.tid = tid
        self.name = 'table_' + str(tid)
        self.indices = data['indices']
        self.values = data['values']
        self.shape = data['shape']
        self.split = data['split']
        self.entities = entities
        self.num_values = self.values.shape[0]
        self.num_features = num_features
        self.num_obs = int(self.num_values / self.num_features)
        self.num_obs_tr = np.sum(self.split == 0)
        self.num_obs_vl = np.sum(self.split == 1)
        self.num_obs_ts = np.sum(self.split == 2)
        self.predict = predict


    def print_top(self, top=5):

        print("indices: \n", self.indices[0:top,:])
        print("values: \n", self.values[0:top])
        print("split: \n", self.split[0:top])
        print("shape: \n", self.shape)
        print("\n")

    # def __init__(self, tid, indices, values, shape, split_sizes, num_features, one_hot=False, split_mode='uniform'):
    #     self.tid = tid
    #     tr, vl, ts = split_sizes
    #     assert tr+vl+ts == 1, "(" + str(tr) + ", " + str(vl) + ", " + str(ts) + ") is not a valid train/valid/test split"
    #     n = values.shape[0]
    #     self.goal_dist_tr = None
    #     self.goal_dist_vl = None

    #     if split_mode == 'uniform':
    #         n_ts = int(n*ts)
    #         n_vl = int(n*vl)
    #         n_tr = n - n_ts - n_vl
    #         split = np.concatenate((np.zeros(n_tr, np.int32), np.ones(n_vl, np.int32), 2*np.ones(n_ts, np.int32)))
    #         np.random.shuffle(split)
        
    #     elif split_mode == 'by_col': # So that the two entries for each match are in the same tr/vl/ts split 
    #         n_cols = shape[1]
    #         n_cols_ts = int(n_cols*ts)
    #         n_cols_vl = int(n_cols*vl)
    #         n_cols_tr = n_cols - n_cols_ts - n_cols_vl
    #         col_split = np.concatenate((np.zeros(n_cols_tr, np.int32), np.ones(n_cols_vl, np.int32), 2*np.ones(n_cols_ts, np.int32)))
    #         np.random.shuffle(col_split)
    #         split = np.take(col_split, indices[:,1])

    #         n_ts = np.sum(split == 2)
    #         n_vl = np.sum(split == 1)
    #         n_tr = n - n_ts - n_vl


    #     if num_features > 1:            
    #         new_vals = []
    #         for val in values:                
    #             new_val = np.zeros(num_features)
    #             if one_hot: 
    #                 new_val[min(val[0], num_features-2)] = 1
    #                 new_val[num_features-1] = val[1]
    #             else:
    #                 new_val[0] = val

    #             new_vals.append(new_val)

    #         new_vals = np.reshape(np.array(new_vals), [-1])

    #     self._set_data_splits(indices, values, new_vals, split, num_features)
    #     self.shape = shape
    #     self.mean_tr = np.mean(self.values_tr)
    #     self.num_values_tr = n_tr
    #     self.num_values = n
        

    # def _set_data_splits(self, indices, values, one_hot_values, split, num_features):
    #     self.indices_all = indices
    #     self.indices_tr = indices[split == 0]
    #     self.indices_vl = indices[split == 1]
    #     self.indices_tr_vl = indices[split <= 1]
    #     self.indices_ts = indices[split == 2]
    #     self.split = split

    #     if self.tid == 1: # team_match table            
    #         values = np.minimum(values, (num_features-1)*np.ones_like(values))
    #         goal_probs_tr = np.unique(values[split == 0], return_counts=True)[1]
    #         goal_probs_vl = np.unique(values[split == 1], return_counts=True)[1]
    #         goal_probs_tr = goal_probs_tr / np.sum(goal_probs_tr)
    #         goal_probs_vl = goal_probs_vl / np.sum(goal_probs_vl)
    #         self.goal_dist_tr = goal_probs_tr
    #         self.goal_dist_vl = goal_probs_vl

        
    #     split = np.array([i for i in split for _ in range(num_features)])

    #     self.values_all = one_hot_values
    #     self.values_tr = one_hot_values[split == 0]
    #     self.values_vl = one_hot_values[split == 1]
    #     self.values_tr_vl = one_hot_values[split <= 1]
    #     self.values_ts = one_hot_values[split == 2]



        




