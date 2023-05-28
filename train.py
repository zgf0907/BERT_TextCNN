import json
import pandas as pd
import numpy as np

from bert4keras.backend import keras
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding ,DataGenerator
from sklearn.metrics import classification_report
from bert4keras.optimizers import Adam

from bert_model import build_bert_model
from data_helper import load_data # 加载数据

# 定义超参数和配置文件
class_nums = 13
maxlen = 128
batch_size = 32

config_path = './chinese_rbt3_L-3_H-768_A-12/bert_config_rbt3.json'
checkpoint_path = './chinese_rbt3_L-3_H-768_A-12/bert_model.ckpt'
dict_path = './chinese_rbt3_L-3_H-768_A-12/vocab.txt'
tokenizer = Tokenizer(dict_path)

# 定义数据生成器 将数据传递到模型中
class data_generator(DataGenerator) :
    """
    数据生成器
    """
    def __iter__(self , random = False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], [] # 对于每一个batchsize的训练，包括 token  分隔符segment 标签label三者的序列
        for is_end, (text , label ) in self.sample(random):
            token_ids , segments_ids = tokenizer.encode(text , maxlen=maxlen) # [1,3,2,5,9,12,243,0,0,0]  编码token和分隔符segment序列，按照最大长度进行padding
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segments_ids)
            batch_labels.append([label])

            if len(batch_token_ids) == self.batch_size or is_end :
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield  [batch_token_ids , batch_segment_ids] ,batch_labels
                batch_token_ids,batch_segment_ids,batch_labels = [],[],[]

if __name__ == '__main__':
    # 加载数据集
    train_data = load_data('./dataset/train.csv')
    test_data = load_data('./dataset/test.csv')
    # 转换数据集
    train_generator = data_generator(train_data,batch_size)
    test_generator = data_generator(test_data,batch_size)

    model = build_bert_model(config_path, checkpoint_path ,class_nums)
    print(model.summary())

    model.compile(
        #'''
       # binary_crossentropy通常与sigmoid激活函数一起使用，可用于二分类，也可用于多分类，在多分类时，每个类别的预测时相互独立的
       # categorical_crossentropy和sparse_categorical_crossentropy都只会计算每个样本的交叉熵，最后需要将所有样本的交叉熵相加求平均。
       # 不同的是前者需要将y_true转化为one_hot编码，后者不需要，可以使用整数编码

        loss='sparse_categorical_crossentropy', # 离散值损失函数 交叉熵损失
        optimizer=Adam(5e-6),
        metrics=['accuracy']
    )
    earlystop = keras.callbacks.EarlyStopping(
        monitor='var_loss',
        patience= 3,
        verbose=2,
        mode='min'
    )

    bast_model_filepath = './model/best_model.weights'

    checkpoint = keras.callbacks.ModelCheckpoint(
        bast_model_filepath ,
        monitor = 'val_loss',
        verbose= 1,
        save_best_only=True,
        mode='min'
    )
    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=10,
        validation_data=test_generator.forfit(),
        validation_steps=len(test_generator),
        shuffle=True,
        callbacks=[earlystop,checkpoint]
    )

    # bast_model_filepath = './model/best_model.weights'
    model.save_weights('best_model.weights')
    model.load_weights(bast_model_filepath)
    test_pred = []
    test_true = []
    for x, y in test_generator:
        p = model.predict(x).argmax(axis=1)
        test_pred.extend(p)

    test_true = test_data[:, 1].tolist()
    print(set(test_true))
    print(set(test_pred))

    target_names = [line.strip() for line in open('./dataset/label', 'r', encoding='utf8')]
    print(classification_report(test_true, test_pred, target_names=target_names))
