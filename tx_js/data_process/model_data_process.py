from tx_js.utils import utils
import pandas as pd
import json
class model_data_process:

    def __init__(self, config):
        self.fea_config = config['fea_config']
        self.data = pd.read_csv(config['data_path'])
    def deal_min_max_(self):
        print('*' * 10, 'min_max', '*' * 10)
        min_max_config = self.fea_config['min_max_config']
        for key in min_max_config:
            print('=' * 10, '>' + key)
            self.data[key] = utils.min_max_feature_(self.data[key])
    def multihot_encoder_for_train(self, sparse_features_multi_value):
        print('=' * 10, '> multihot_encoder_for_train')
        multihot_feature_lists = []
        multihot_maxlen_lists = []
        multihot_key2index_lists = []
        for col in (sparse_features_multi_value):
            if col == 'ad_idclick_list90':
                col_lis, max_len = utils.multihot_encoder_for_train_key(self.data[col], 'ad_id')
                key2_index = []
            elif col == 'creative_idclick_list90':
                col_lis, max_len = utils.multihot_encoder_for_train_key(self.data[col], 'creative_id')
                key2_index = []
            else:
                col_lis, max_len, key2_index = utils.multihot_encoder_for_train(self.data[col])
            multihot_feature_lists.append(col_lis)
            multihot_maxlen_lists.append(max_len)
            multihot_key2index_lists.append(key2_index)
        return multihot_feature_lists, multihot_maxlen_lists, multihot_key2index_lists
    def multihot_encoder_for_test_age(self, sparse_features_multi_value):
        if 'creative_idclick_list90' in sparse_features_multi_value and 'ad_idclick_list90' in sparse_features_multi_value:
            with open('/nfs/project/xiongfeng/xf_data/multihot_dict_age_a_id_c_id.json', 'r', encoding='utf-8') as f:
                multihot_key2index_lists = json.load(f)
        elif 'creative_idclick_list90' in sparse_features_multi_value:
            with open('/nfs/project/xiongfeng/xf_data/multihot_dict_age_c_id_pro.json', 'r', encoding='utf-8') as f:
                multihot_key2index_lists = json.load(f)
        else:
            with open('/nfs/project/xiongfeng/xf_data/multihot_dict_age_ad_id_pro.json', 'r', encoding='utf-8') as f:
                multihot_key2index_lists = json.load(f)
        print('=' * 10, '> multihot_encoder_for_test')

        multihot_feature_lists = []
        count = 0
        for col in (sparse_features_multi_value):
            if col == 'creative_idclick_list90':
                col_lis, max_len = utils.multihot_encoder_for_train_key(self.data[col], 'creative_id')
            elif col == 'ad_idclick_list90':
                col_lis, max_len = utils.multihot_encoder_for_train_key(self.data[col], 'ad_id')
            else:
                col_lis = utils.multihot_encoder_for_test(self.data[col], count, multihot_key2index_lists)
            count += 1
            multihot_feature_lists.append(col_lis)
        return multihot_feature_lists