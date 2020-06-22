from collections import OrderedDict
from keras.initializers import RandomNormal, TruncatedNormal
from keras.layers import Concatenate, Dense, Embedding, Input, add, Flatten, multiply, RepeatVector, Reshape, Conv1D, \
    MaxPool1D, AveragePooling1D, LSTM, Dropout, Bidirectional, GRU, BatchNormalization
from keras.regularizers import l2
from tx_js.utils.utils import concat_fun, SparseFeat, DenseFeat, VarLenSparseFeat
import keras
from tx_js.model.model_dl_base import PredictionLayer, DNN, PredictionLayer_S, InteractingLayer, Transformer, \
    LayerNormalization, AttentionSequencePoolingLayer,SequencePoolingLayer
from keras.layers.core import Lambda
import numpy as np
import tensorflow as tf
import json
from gensim.models import Word2Vec


def get_fixlen_feature_names(feature_columns, prefix=''):
    features = build_input_features(feature_columns, include_varlen=False, include_fixlen=True, prefix='')
    return features.keys()


def get_varlen_feature_names(feature_columns, prefix=''):
    features = build_input_features(feature_columns, include_varlen=True, include_fixlen=False, prefix='')
    return features.keys()


def build_input_features(feature_columns, include_varlen=True, mask_zero=True, prefix='', include_fixlen=True):
    input_features = OrderedDict()
    if include_fixlen:
        for fc in feature_columns:
            if isinstance(fc, SparseFeat):
                input_features[fc.name] = Input(shape=(1,), name=prefix + fc.name, dtype=fc.dtype)
            elif isinstance(fc, DenseFeat):
                input_features[fc.name] = Input(shape=(1,), name=prefix + fc.name, dtype=fc.dtype)
    if include_varlen:
        for fc in feature_columns:
            if isinstance(fc, VarLenSparseFeat):
                input_features[fc.name] = Input(shape=(fc.maxlen,), name=prefix + 'seq_' + fc.name, dtype=fc.dtype)
        if not mask_zero:
            for fc in feature_columns:
                input_features[fc.name + "_seq_length"] = Input(shape=(1,), name=prefix + 'seq_length_' + fc.name)
                input_features[fc.name + "_seq_max_length"] = fc.maxlen

    return input_features


def create_sparse_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, embedding_size, init_std, seed,
                                 l2_reg, prefix='sparse_', seq_mask_zero=True):
    feat_embed_dict = {}
    if sparse_feature_columns and len(sparse_feature_columns):
        for feat in sparse_feature_columns:
            feat_embed_dict[feat.embedding_name] = Embedding(feat.dimension, embedding_size,
                                                             embeddings_initializer=RandomNormal(mean=0.0,
                                                                                                 stddev=init_std,
                                                                                                 seed=seed),
                                                             embeddings_regularizer=l2(l2_reg),
                                                             name=prefix + '_emb_' + feat.name)

    if varlen_sparse_feature_columns and len(varlen_sparse_feature_columns) > 0:
        for feat in varlen_sparse_feature_columns:
            embed_size_dic = {
                'creative_idclick_list90': 128,
                'creative_idclick_list901': 128,
                'ad_idclick_list90': 128,
                'product_idclick_list90': 20,
                'product_categoryclick_list90': 20,
                'advertiser_idclick_list90': 128,
                'industryclick_list90': 20,
                'timeclick_list90': 20,
            }
            feat_embed_dict[feat.embedding_name] = Embedding(feat.dimension, embed_size_dic[feat.name],
                                                             embeddings_initializer=RandomNormal(mean=0.0,
                                                                                                 stddev=init_std,
                                                                                                 seed=seed),
                                                             embeddings_regularizer=l2(l2_reg),
                                                             name=prefix + '_seq_emb_' + feat.name,
                                                             mask_zero=seq_mask_zero)

    return feat_embed_dict


def w2c_create_sparse_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, embedding_size, init_std,
                                     seed, l2_reg, prefix='sparse_', seq_mask_zero=True):
    feat_embed_dict = {}
    if sparse_feature_columns and len(sparse_feature_columns):
        for feat in sparse_feature_columns:
            print("aaaa")
            feat_embed_dict[feat.embedding_name] = Embedding(feat.dimension, 40,
                                                             embeddings_initializer=RandomNormal(mean=0.0,
                                                                                                 stddev=init_std,
                                                                                                 seed=seed),
                                                             embeddings_regularizer=l2(l2_reg),
                                                             name=prefix + '_emb_' + feat.name)

    if varlen_sparse_feature_columns and len(varlen_sparse_feature_columns) > 0:
        for feat in varlen_sparse_feature_columns:
            embed_size_dic = {
                'creative_idclick_list90': 128,
                'creative_idclick_list901': 128,
                'ad_idclick_list90': 128,
                'product_idclick_list90': 20,
                'product_categoryclick_list90': 8,
                'advertiser_idclick_list90': 20,
                'industryclick_list90': 8,
                'timeclick_list90': 8,
                'advertiser_idclick_list901': 128
            }
            if feat.name == 'creative_idclick_list90':
                # 先加载配置文件
                print("ad_create")
                with open('/nfs/project/xiongfeng/xf_data/creative_id.json', 'r', encoding='utf-8') as f:
                    key2index = json.load(f)
                print(len(key2index))
                with open('/nfs/project/xiongfeng/xf_data/c_type.json', 'r', encoding='utf-8') as f:
                    temp_key2index = json.load(f)
                model = Word2Vec.load('/nfs/project/xiongfeng/xf_data/w2c_c_type')
                embedding_matrix = np.zeros(((len(key2index) + 1), 128))
                for word, i in key2index.items():
                    try:
                        ii = temp_key2index[word]
                        embedding_vector = model[ii]
                        embedding_matrix[int(i)] = embedding_vector
                    except KeyError:
                        continue
                feat_embed_dict[feat.embedding_name] = Embedding(len(key2index) + 1, embed_size_dic[feat.name],
                                                                 weights=[embedding_matrix],
                                                                 input_length=100,
                                                                 trainable=False, mask_zero=False)
                del key2index
                del embedding_matrix
            elif feat.name == 'ad_idclick_list90':
                print("ad_create")
                with open('/nfs/project/xiongfeng/xf_data/ad_id_1201.json', 'r', encoding='utf-8') as f:
                    key2index = json.load(f)
                print(len(key2index))
                model = Word2Vec.load('/nfs/project/xiongfeng/xf_data/w2c_ad_id_128')
                embedding_matrix = np.zeros(((len(key2index) + 1), 128))
                for word, i in key2index.items():
                    try:
                        embedding_vector = model[i]
                        embedding_matrix[int(i)] = embedding_vector
                    except KeyError:
                        continue
                feat_embed_dict[feat.embedding_name] = Embedding(len(key2index) + 1, embed_size_dic[feat.name],
                                                                 weights=[embedding_matrix],
                                                                 input_length=100,
                                                                 trainable=False, mask_zero=True)
                del key2index
                del embedding_matrix
            else:
                feat_embed_dict[feat.embedding_name] = Embedding(feat.dimension, embed_size_dic[feat.name],
                                                                 embeddings_initializer=RandomNormal(mean=0.0,
                                                                                                     stddev=init_std,
                                                                                                     seed=seed),
                                                                 embeddings_regularizer=l2(l2_reg),
                                                                 name=prefix + '_seq_emb_' + feat.name,
                                                                 mask_zero=True)


    return feat_embed_dict




def create_dense_embedding_dict(dense_feature_columns, embedding_size, init_std, seed, l2_reg, prefix='dense_',
                                seq_mask_zero=True):
    feat_embed_dict = {}

    if dense_feature_columns and len(dense_feature_columns) > 0:
        for feat in dense_feature_columns:
            feat_embed_dict[feat.embedding_name] = (Dense(embedding_size, activation=None, use_bias=False,
                                                          kernel_initializer=RandomNormal(mean=0.0, stddev=init_std,
                                                                                          seed=seed),
                                                          kernel_regularizer=l2(l2_reg),
                                                          name=prefix + '_emb_' + feat.name),
                                                    NanLayer(embedding_size, init_std, seed, l2_reg))

            # feat_embed_dict[feat.embedding_name] = MyLayer(embedding_size)
    return feat_embed_dict


def sparse_embedding_lookup(sparse_embedding_dict, sparse_input_dict, sparse_feature_columns, return_feat_list=(),
                            mask_feat_list=()):
    embedding_vec_list = []
    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if len(return_feat_list) == 0 or feature_name in return_feat_list and fc.embedding:
            if fc.use_hash:
                lookup_idx = Hash(fc.dimension, mask_zero=(feature_name in mask_feat_list))(
                    sparse_input_dict[feature_name])
            else:
                lookup_idx = sparse_input_dict[feature_name]
            embedding_vec_list.append(sparse_embedding_dict[embedding_name](lookup_idx))
    return embedding_vec_list
def dense_embedding_lookup(dense_embedding_dict, dense_input_dict, dense_feature_columns, embedding_size,
                           return_feat_list=(), mask_feat_list=()):
    embedding_vec_list = []
    for fc in dense_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if len(return_feat_list) == 0 or feature_name in return_feat_list and fc.embedding:
            lookup_idx = dense_input_dict[feature_name]
            x = dense_embedding_dict[embedding_name][0](lookup_idx)
            embedding_vec_list.append(RepeatVector(1)(x))
    return embedding_vec_list


def varlen_embedding_lookup(embedding_dict, sequence_input_dict, varlen_sparse_feature_columns):
    varlen_embedding_vec_dict = {}
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.use_hash:
            lookup_idx = Hash(fc.dimension, mask_zero=True)(sequence_input_dict[feature_name])
        else:
            lookup_idx = sequence_input_dict[feature_name]
        varlen_embedding_vec_dict[feature_name] = embedding_dict[embedding_name](lookup_idx)

    return varlen_embedding_vec_dict


def get_varlen_pooling_list(embedding_dict, features, varlen_sparse_feature_columns):
    pooling_vec_list = []
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        combiner = fc.combiner
        feature_length_name = feature_name + '_seq_length'
        if feature_length_name in features:
            vec = SequencePoolingLayer(combiner, supports_masking=False)(
                [embedding_dict[feature_name], features[feature_length_name]])
        else:
            vec = SequencePoolingLayer(combiner, supports_masking=True)(embedding_dict[feature_name])
        pooling_vec_list.append(vec)
    return pooling_vec_list


def sum_(att_output):
    return tf.reduce_mean(att_output,1,keep_dims=True)

def max_(att_output):
    return tf.reduce_max(att_output, 1, keepdims=True)


def cc(att):
    return tf.concat(att, axis=-1)


def get_add_varlen_att_cnn_list(embedding_dict, features, varlen_sparse_feature_columns, aa):
    temp_lis1 = []
    for fc in varlen_sparse_feature_columns:
        if fc.name not in ['advertiser_idclick_list901', 'creative_idclick_list90']:
            temp_lis1.append(embedding_dict[fc.name])
    final_embed1 = concat_fun([p for p in temp_lis1], axis=-1)
    final_embed1 = keras.layers.BatchNormalization()(final_embed1)

    final_embed1 = LSTM(units=600, input_shape=(100, 192), activation='relu', return_sequences=True, dropout=0.1,
                        recurrent_dropout=0.2)(final_embed1)

    att_vec1 = final_embed1
    # att_vec1 = Transformer(att_embedding_size=184,head_num=1,dropout_rate=0.3,
    #                         use_feed_forward=True,use_layer_norm=True,use_res=True)([att_vec1,att_vec1])
    att_vec2 = LSTM(units=600, input_shape=(100, 192), activation='relu', return_sequences=True, dropout=0.1,
                        recurrent_dropout=0.2)(final_embed1)(embedding_dict['creative_idclick_list90'])
    att_vec2 = Lambda(sum_)(att_vec2)

    att_vec1 = AttentionSequencePoolingLayer()([att_vec2, att_vec1])
    att_vec = att_vec1

    '''
    pool_output = []
    kernel_sizes = [2,3,4,7,14,30,45] 
    for kernel_size in kernel_sizes:
        c = Conv1D(filters=128, kernel_size=kernel_size, strides=1)(att_vec)
        p = MaxPool1D(pool_size=int(c.shape[1]))(c)
        pool_output.append(p)
    cnn_vec = concat_fun([p for p in pool_output])
    '''

    final_vec = Dropout(0.2)(att_vec)

    return [final_vec]
def get_dense_input(features, feature_columns, embedding_size, l2_reg, init_std, seed, prefix='', seq_mask_zero=True):
    dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if feature_columns else []
    dense_input_list = []
    for fc in dense_feature_columns:
        lookup_idx = features[fc.name]
        dense_input_list.append(lookup_idx)
    return dense_input_list


def embedding_input_from_feature_columns(features, feature_columns, embedding_size, l2_reg, init_std, seed, prefix='',
                                         seq_mask_zero=True):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []
    dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if feature_columns else []

    sparse_embedding_dict = create_sparse_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns,
                                                         embedding_size,
                                                         init_std, seed, l2_reg, prefix=prefix + 'sparse',
                                                         seq_mask_zero=seq_mask_zero)
    sparse_embedding_list = sparse_embedding_lookup(sparse_embedding_dict, features, sparse_feature_columns)

    sequence_embed_dict = varlen_embedding_lookup(sparse_embedding_dict, features, varlen_sparse_feature_columns)
    sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, features, varlen_sparse_feature_columns)
    sparse_embedding_list += sequence_embed_list

    dense_embedding_dict = create_dense_embedding_dict(dense_feature_columns, embedding_size,
                                                       init_std, seed, l2_reg, prefix=prefix + 'dense',
                                                       seq_mask_zero=seq_mask_zero)

    dense_embedding_list = dense_embedding_lookup(dense_embedding_dict, features, dense_feature_columns, embedding_size)

    dense_input_list = get_dense_input(features, feature_columns, 1, l2_reg, init_std, seed, prefix='',
                                       seq_mask_zero=True)
    return sparse_embedding_list, dense_embedding_list, dense_input_list
def add_att_cnn_embedding_input_from_feature_columns(features, feature_columns, embedding_size, l2_reg, init_std, seed,
                                                     prefix='', seq_mask_zero=True):
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

    sparse_embedding_dict = w2c_create_sparse_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns,
                                                             embedding_size,
                                                             init_std, seed, l2_reg, prefix=prefix + 'sparse',
                                                             seq_mask_zero=False)

    sequence_embed_dict = varlen_embedding_lookup(sparse_embedding_dict, features, varlen_sparse_feature_columns)
    sequence_embed_list = get_add_varlen_att_cnn_list(sequence_embed_dict, features, varlen_sparse_feature_columns,
                                                      sparse_feature_columns)
    sparse_embedding_list = []
    sparse_embedding_list += sequence_embed_list
    dense_input_list = get_dense_input(features, feature_columns, 1, l2_reg, init_std, seed, prefix='',
                                       seq_mask_zero=True)
    return sparse_embedding_list,dense_input_list


def r_dim(idx):
    return keras.backend.squeeze(idx, axis=1)


def combined_dnn_input_dense_not_embed(sparse_embedding_list, dense_value_list):
    if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
        sparse_dnn_input = Flatten()(concat_fun(sparse_embedding_list))
        x = concat_fun(dense_value_list)
        x = Flatten()(RepeatVector(1)(x))
        return concat_fun([sparse_dnn_input, x])
    elif len(sparse_embedding_list) > 0:
        return concat_fun(sparse_embedding_list)
    elif len(dense_value_list) > 0:
        return Flatten()(concat_fun(dense_value_list))
    else:
        raise NotImplementedError


def combined_dnn_input_dense_embed(sparse_embedding_list, dense_embedding_list):
    if len(sparse_embedding_list) > 0 and len(dense_embedding_list) > 0:
        sparse_dnn_input = Flatten()(concat_fun(sparse_embedding_list))
        dense_dnn_input = Flatten()(concat_fun(dense_embedding_list))
        return concat_fun([sparse_dnn_input, dense_dnn_input])
    elif len(sparse_embedding_list) > 0:
        return Flatten()(concat_fun(sparse_embedding_list))
    elif len(dense_embedding_list) > 0:
        return Flatten()(concat_fun(dense_embedding_list))
    else:
        raise NotImplementedError
