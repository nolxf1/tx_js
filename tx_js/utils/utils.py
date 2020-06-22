from collections import namedtuple
from keras.engine.topology import Layer
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
import json
import keras

'''
稀疏特征类
'''
class SparseFeat(namedtuple('SparseFeat', ['name', 'dimension', 'use_hash', 'dtype', 'embedding_name', 'embedding'])):
    __slots__ = ()

    def __new__(cls, name, dimension, use_hash=False, dtype="int32", embedding_name=None, embedding=True):
        if embedding and embedding_name is None:
            embedding_name = name
        return super(SparseFeat, cls).__new__(cls, name, dimension, use_hash, dtype, embedding_name, embedding)

'''
稠密特征类
'''
class DenseFeat(namedtuple('SparseFeat', ['name', 'dimension', 'use_hash', 'dtype', 'embedding_name', 'embedding'])):
    __slots__ = ()

    def __new__(cls, name, dimension, use_hash=False,dtype="float32", embedding_name=None, embedding=True):
        if embedding and embedding_name is None:
            embedding_name = name
        return super(DenseFeat, cls).__new__(cls, name, dimension, use_hash, dtype, embedding_name, embedding)

'''
序列特征类
'''
class VarLenSparseFeat(namedtuple('VarLenFeat',
                                  ['name', 'dimension', 'maxlen', 'combiner', 'use_hash', 'dtype', 'embedding_name',
                                   'embedding'])):
    __slots__ = ()

    def __new__(cls, name, dimension, maxlen, combiner="mean", use_hash=False, dtype="float32", embedding_name=None,
                embedding=True):
        if embedding_name is None:
            embedding_name = name
        return super(VarLenSparseFeat, cls).__new__(cls, name, dimension, maxlen, combiner, use_hash, dtype,
                                                    embedding_name, embedding)
'''
归一化
'''
def min_max_feature_(fea):
    s = (fea - fea.min())/(fea.max() - fea.min())
    return s


'''
序列编码操作
'''
def multihot_encoder_for_train_key(fea, key, flag=0):
    print('=' * 10, '> multihot_encoder_for_train_for_key')

    def toset(row):
        row = row[1:-1].split(',')
        click_buid = []
        for buid in row:
            buid = buid.strip()
            click_buid.append(buid)
        return click_buid

    if key == 'ad_id':
        with open('./' + key + '_1201.json', 'r', encoding='utf-8') as f:
            key2index = json.load(f)
    else:
        with open('./' + key + '.json', 'r', encoding='utf-8') as f:
            key2index = json.load(f)
    def split(x):
        key_ans = x
        return list(map(lambda x: int(key2index[x]), key_ans))
    fea = fea.apply(toset)
    col_list = list(map(split, fea.values))
    max_length = 100
    if flag == 0:
        col_list = pad_sequences(col_list, maxlen=max_length, padding='post', truncating='post')
    else:
        col_list = pad_sequences(col_list, maxlen=max_length, padding='post')
    return col_list, max_length
def multihot_encoder_for_train_key1(fea, key, flag=0):
    print('=' * 10, '> multihot_encoder_for_train_for_key1')

    def toset(row):
        row = row[1:-1].split(',')
        click_buid = []
        for buid in row:
            buid = buid.strip()
            buid = buid.split('*')
            click_buid.append(buid[1][:-1])
            click_buid.append(buid[0][1:])
        return click_buid

    if key == 'ad_id':
        with open('./' + key + '_1201.json', 'r', encoding='utf-8') as f:
            key2index = json.load(f)
    else:
        with open('./' + key + '.json', 'r', encoding='utf-8') as f:
            key2index = json.load(f)
    def split(x):
        key_ans = x
        return list(map(lambda x: int(key2index[x]), key_ans))
    fea = fea.apply(toset)
    col_list = list(map(split, fea.values))
    max_length = 100
    if flag == 0:
        col_list = pad_sequences(col_list, maxlen=max_length, padding='post', truncating='post')
    else:
        col_list = pad_sequences(col_list, maxlen=max_length, padding='post')
    return col_list, max_length


def multihot_encoder_for_train(fea, flag=0):
    print('=' * 10, '> multihot_encoder_for_train')
    def toset(row):
        row = row[1:-1].split(',')
        click_buid = []
        for buid in row:
            buid = buid.strip()
            click_buid.append(buid)
        return click_buid

    def split(x):
        key_ans = x
        for key in key_ans:
            if key not in key2index:
                # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for multi-hot input
                key2index[key] = len(key2index) + 1
        return list(map(lambda x: key2index[x], key_ans))
    key2index = dict()
    fea = fea.apply(toset)
    col_list = list(map(split, fea.values))
    max_length = 100
    if flag == 0:
        col_list = pad_sequences(col_list, maxlen=max_length, padding='post', truncating='post')
    else:
        col_list = pad_sequences(col_list, maxlen=max_length, padding='post')

    return col_list, max_length, key2index


def multihot_encoder_for_test(fea, count, multihot_key2index_lists, flag=0):
    print('=' * 10, '> multihot_encoder_for_test')

    def toset(row):
        row = row[1:-1].split(',')
        click_buid = []
        for buid in row:
            buid = buid.strip()
            click_buid.append(buid)
        return click_buid

    def match(x):
        key_ans = x
        for key in key_ans:
            if key not in multihot_key2index_lists[count]:
                multihot_key2index_lists[count][key] = 0
        return list(map(lambda x: multihot_key2index_lists[count][x], key_ans))

    fea = fea.apply(toset)
    col_list = list(map(match, fea.values))
    max_length = 100
    if flag == 0:
        col_list = pad_sequences(col_list, maxlen=max_length, padding='post', truncating='post')
    else:
        col_list = pad_sequences(col_list, maxlen=max_length, padding='post')
    return col_list


def concat_fun(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        # concatenate 与 concat作用类似
        print(inputs)
        return keras.layers.Concatenate(axis=axis)(inputs)
