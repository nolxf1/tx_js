from tx_js.model.model_dl_base import PredictionLayer,FM,DNN,PredictionLayer_S,InteractingLayer
from tx_js.data_process.dl_embedding import build_input_features,combined_dnn_input_dense_embed,\
embedding_input_from_feature_columns,get_dense_input,combined_dnn_input_dense_not_embed,add_att_cnn_embedding_input_from_feature_columns
from keras.layers import Concatenate, Dense, Embedding, Input, add, Flatten,multiply,RepeatVector,Reshape,Conv1D, MaxPool1D,Dropout
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop, Adagrad
import keras
import tensorflow as tf
class LSTM_AT_CNN_EM:
    def __init__(self, model_config):

        # 初始化模型参数
        self.linear_feature_columns = model_config['linear_feature_columns']
        self.dnn_feature_columns = model_config['dnn_feature_columns']
        self.embedding_size = model_config['embedding_size']
        self.dnn_hidden_units = model_config['dnn_hidden_units']
        self.l2_reg_linear = model_config['l2_reg_linear']
        self.l2_reg_embedding = model_config['l2_reg_embedding']
        self.l2_reg_dnn = model_config['l2_reg_dnn']
        self.init_std = model_config['init_std']
        self.seed = model_config['seed']
        self.dnn_dropout = model_config['dnn_dropout']
        self.dnn_activation = model_config['dnn_activation']
        self.dnn_use_bn = model_config['dnn_use_bn']
        self.Optimizer = model_config['Optimizer']
        self.learning_rate = model_config['learning_rate']
        self.use_global_epochs = model_config['use_global_epochs']
        self.epochs = model_config['epochs']
        self.verbose = model_config['verbose']
        self.batch_size = model_config['batch_size']
        self.prefix = model_config['prefix']
        self.task = model_config['task']

        print('=' * 10, '> construct Model')
        # 模型输入特征
        self.features = build_input_features(self.linear_feature_columns + self.dnn_feature_columns, prefix=self.prefix)
        # 模型输入向量
        self.inputs_list = list(self.features.values())
        sparse_embedding_list,  dense_value_list = add_att_cnn_embedding_input_from_feature_columns(
            self.features, self.dnn_feature_columns,
            self.embedding_size, self.l2_reg_embedding, self.init_std,
            self.seed, prefix=self.prefix)
        self.sparse_embedding_list = sparse_embedding_list
        self.dense_value_list = dense_value_list

        dnn_input = combined_dnn_input_dense_not_embed(self.sparse_embedding_list, self.dense_value_list)
        self.dnn_input = dnn_input
        dnn_out = DNN(self.dnn_hidden_units, self.dnn_activation, self.l2_reg_dnn, self.dnn_dropout, self.dnn_use_bn,
                      self.seed)(dnn_input)
        self.final_logit = dnn_out

    def model_age_gender(self):
        self.final_logit1 = keras.layers.Dense(1, use_bias=False, activation=None)(self.final_logit)
        self.output1 = PredictionLayer('binary', name='task1')(self.final_logit1)
        self.output2 = keras.layers.Dense(6, use_bias=True, activation='softmax')(self.final_logit)
        self.output2 = PredictionLayer_S('softmax', name='task2')(self.output2)
        model = Model(inputs=self.inputs_list, outputs=[self.output1, self.output2])
        self.model = model
        print(model.summary())

    def model_gender(self):
        self.final_logit = keras.layers.Dense(1, use_bias=False, activation=None)(self.final_logit)
        self.output = PredictionLayer(self.task, name=self.prefix)(self.final_logit)
        model = Model(inputs=self.inputs_list, outputs=self.output)
        self.model = model
        print(model.summary())

    def model_age(self):
        self.output = keras.layers.Dense(10, use_bias=True, activation='softmax', name='softmax')(self.final_logit)
        self.output = PredictionLayer_S(self.task, name=self.prefix)(self.output)
        model = Model(inputs=self.inputs_list, outputs=self.output)
        self.model = model
        print(model.summary())
    def model_age_add_gender(self):
        self.output = keras.layers.Dense(20, use_bias=True, activation='softmax', name='softmax')(self.final_logit)
        self.output = PredictionLayer_S(self.task, name=self.prefix,nums=20)(self.output)
        model = Model(inputs=self.inputs_list, outputs=self.output)
        self.model = model
        print(model.summary())


    def train(self, train_model_input, test_model_input, train_target, test_target,key='age'):

        if self.Optimizer == 'Adam':
            optimizer = Adam(lr=self.learning_rate)
        elif self.Optimizer == 'Adagrad':
            optimizer = Adagrad(lr=self.learning_rate)
        elif self.Optimizer == 'RMSprop':
            optimizer = RMSprop(lr=self.learning_rate)
        elif self.Optimizer == 'SGD':
            optimizer = SGD(lr=self.learning_rate)
        print('=' * 10, '> compile Model')
        if key=='age' or key=='age_add_gender':
            self.model.compile(optimizer=optimizer, loss="categorical_crossentropy",
                               metrics=['categorical_crossentropy', 'accuracy'])
        elif key=='gender':
            self.model.compile(optimizer=optimizer, loss="binary_crossentropy",
                               metrics=['binary_crossentropy', 'accuracy'])
        else:
            self.model.compile(optimizer=optimizer, loss=['binary_crossentropy', 'categorical_crossentropy'],
                               loss_weights=[0.5, 0.5], metrics={'task1': 'accuracy', 'task2': 'accuracy'})
        if self.use_global_epochs:
            for i in range(self.epochs):
                print('=' * 10, '> global_epoch_{}'.format(i + 1))
                self.model.fit(
                    train_model_input,
                    train_target,
                    validation_data=(test_model_input, test_target),
                    batch_size=self.batch_size,
                    epochs=1,
                    verbose=self.verbose
                )
        else:
            self.model.fit(
                train_model_input,
                train_target,
                validation_data=(test_model_input, test_target),
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=self.verbose
            )