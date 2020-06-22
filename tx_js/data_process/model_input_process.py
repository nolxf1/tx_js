from tx_js.data_process.model_data_process import model_data_process

from tx_js.utils.utils import SparseFeat, DenseFeat, VarLenSparseFeat

from tx_js.data_process.dl_embedding import get_fixlen_feature_names

import numpy as np
import keras
import json
import pandas as pd

class model_input_process(model_data_process):

    def __init__(self, config):
        super(model_input_process, self).__init__(config)
    def deepfm_get_standard_input_test(self):
        # mix_max
        sparse_fea = self.fea_config['sparse_fea_config']
        dense_fea = self.fea_config['dense_fea_config']
        sparse_mul_fea = self.fea_config['sparse_mul_fea']
        self.data = self.data.loc[:, sparse_fea + dense_fea + sparse_mul_fea + ['user_id', 'age']]
        # self.data['gender'] = self.data['gender']-1

        self.deal_min_max_()
        # 处理空值
        # self.deal_nan()
        self.data.fillna(0, inplace=True)
        # catencoder
        self.categlory_encoder()
        # mix_max
        # self.deal_min_max()
        sparse_fea = self.fea_config['sparse_fea_config']
        dense_fea = self.fea_config['dense_fea_config']
        sparse_mul_fea = self.fea_config['sparse_mul_fea']
        par = self.fea_config['par']
        # print(par)
        fixlen_feature_columns = [SparseFeat(col, self.data[col].nunique()) for col in
                                  sparse_fea] + [DenseFeat(col, 1) for col in dense_fea]
        multihot_feature_lists = self.multihot_encoder_for_test_age(sparse_mul_fea)
        fixlen_feature_columns = [SparseFeat(col, data[col].nunique()) for col in sparse_fea] + [DenseFeat(col, 1, ) for
                                                                                                 col in dense_fea]
        # varlen_feature_columns = [
        # VarLenSparseFeat(sparse_mul_fea[i], len(multihot_key2index_lists[i]) + 1,
        #                     multihot_maxlen_lists[i], 'mean') for i in range(len(sparse_mul_fea))]
        linear_feature_columns = fixlen_feature_columns
        # linear_feature_columns = fixlen_feature_columns
        dnn_feature_columns = linear_feature_columns
        fixlen_feature_names = get_fixlen_feature_names(linear_feature_columns + dnn_feature_columns)

        data_model_input_train = [self.data[name].values for name in fixlen_feature_names] + [col for col in
                                                                                              multihot_feature_lists]
        # del self.data
        return data_model_input_train

    def deepfm_get_standard_input_train_and_val_xf_model_s(self):
        # mix_max
        self.data['age'] = self.data['age'] - 1
        self.data['gender'] = self.data['gender'] - 1
        self.deal_min_max_()
        self.categlory_encoder()
        # mix_max
        # self.deal_min_max()
        sparse_fea = self.fea_config['sparse_fea_config']
        dense_fea = self.fea_config['dense_fea_config']
        sparse_mul_fea = self.fea_config['sparse_mul_fea']
        par = self.fea_config['par']
        # print(par)
        fixlen_feature_columns = [SparseFeat(col, self.data[col].nunique()) for col in
                                  sparse_fea] + [DenseFeat(col, 1) for col in dense_fea]
        multihot_feature_lists, multihot_maxlen_lists, multihot_key2index_lists = self.multihot_encoder_for_train(
            self.data, sparse_mul_fea)
        fixlen_feature_columns = [SparseFeat(col, self.data[col].nunique()) for col in sparse_fea] + [
            DenseFeat(col, 1, ) for col in dense_fea]
        # print(len(fixlen_feature_columns))
        # print(len(multihot_key2index_lists[0]))
        varlen_feature_columns = [
            VarLenSparseFeat(sparse_mul_fea[i], len(multihot_key2index_lists[i]) + 1,
                             multihot_maxlen_lists[i], 'sum') for i in range(len(sparse_mul_fea))]
        linear_feature_columns = fixlen_feature_columns + varlen_feature_columns
        # linear_feature_columns = fixlen_feature_columns
        dnn_feature_columns = linear_feature_columns
        fixlen_feature_names = get_fixlen_feature_names(linear_feature_columns + dnn_feature_columns)
        # 此处遗留var len特征未处理
        # 此处划分训练集 验证集

        train_data = self.data.sample(frac=0.95, random_state=2021)
        # print(train_data.index.values)
        test_data = self.data[~self.data.user_id.isin(train_data.user_id)]

        del self.data
        # print("len",len(fixlen_feature_names))
        data_model_input_train = [train_data[name].values for name in fixlen_feature_names] + [
            col[train_data.index.values] for col in multihot_feature_lists]
        data_model_input_test = [test_data[name].values for name in fixlen_feature_names] + [col[test_data.index.values]
                                                                                             for col in
                                                                                             multihot_feature_lists]

        train_label = keras.utils.to_categorical(train_data['age'])
        # print(type(train_label))
        test_label = keras.utils.to_categorical(test_data['age'])
        # data_model_input_test = []
        # test_label = []
        # test_data = []
        # print(data_model_input_train)
        # print(np.array(data_model_input_train).shape)
        # print(np.array(data_model_input_test).shape)
        # print(test_data['label'].sum())
        # del self.data
        # with open('/nfs/project/xiongfeng/xf_data/multihot_dict_age_c_id_pro.json', 'w', encoding='utf-8') as f:
        #    json.dump(multihot_key2index_lists, f)
        del multihot_key2index_lists, multihot_feature_lists, multihot_maxlen_lists
        return test_data, data_model_input_train, data_model_input_test, train_label, test_label, linear_feature_columns, dnn_feature_columns
