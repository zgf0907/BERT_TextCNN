# 使用苏建林开发的bert4keras深度学习框架加载BERT模型
from bert4keras.backend import keras,set_gelu
from bert4keras.models import build_transformer_model # 加载BERT的方法
from bert4keras.optimizers import Adam # 优化器
set_gelu('tanh')

# 实现textcnn
def textcnn(input,kernel_initializer) :
    # 3,4,5
    cnn1 = keras.layers.Conv1D(
        256, # 卷积核数量
        3, # 卷积核大小
        strides= 1, # 步长

        padding= 'same', # 输出与输入维度一致
        activation='relu',  # 激活函数
        kernel_initializer = kernel_initializer # 初始化器
    )(input) # shape = [batch_size ,maxlen-2,256]
    cnn1 = keras.layers.GlobalAvgPool1D()(cnn1) # 全局最大池化操作 shape = [batch_size ,256]

    cnn2 = keras.layers.Conv1D(
        256,  # 卷积核数量
        4,  # 卷积核大小
        strides=1,  # 步长
        padding='same',  # 输出与输入维度一致
        activation='relu',  # 激活函数
        kernel_initializer=kernel_initializer  # 初始化器
    )(input)
    cnn2 = keras.layers.GlobalAvgPool1D()(cnn2)  # 全局最大池化操作 shape = [batch_size ,256]

    cnn3 = keras.layers.Conv1D(
        256,  # 卷积核数量
        5,  # 卷积核大小
        strides=1,  # 步长
        padding='same',  # 输出与输入维度一致
        kernel_initializer=kernel_initializer  # 初始化器
    )(input)
    cnn3 = keras.layers.GlobalAvgPool1D()(cnn3) # 全局最大池化操作 shape = [batch_size ,256]

    # 将三个卷积结果进行拼接
    output = keras.layers.concatenate([cnn1,cnn2,cnn3],
                                   axis= -1)
    output = keras.layers.Dropout(0.2)(output) # 最后接Dropout

    return output

# 定义函数加载BERT
def build_bert_model(config_path , checkpoint_path , class_nums) : # config_path配置文件的路径 checkpoint_path预训练路径 class_nums类别的数量
    bert = build_transformer_model(
        config_path = config_path ,
        checkpoint_path = checkpoint_path ,
        model = 'bert' ,
        return_keras_model= False)
    # 在BERT模型输出中抽取[CLS]
    cls_features = keras.layers.Lambda(lambda x:x[:,0],name='cls-token')(bert.model.output) # [:,0]选取输出的第一列，BERT模型的输出中[CLS]在第一个位置 shape = [batch_size ,768]
    all_token_embedding = keras.layers.Lambda(lambda x:x[:,1:-1],name='all-token')(bert.model.output) # 获取第2列至倒数第二列的所有token  shape = [batch_size ,maxlen-2,768] 除去CLS、SEP

    # textcnn抽取特征
    cnn_features = textcnn(all_token_embedding, bert.initializer) # 输入all_token_embedding  shape = [batch_size,cnn_output_dim]
    # 将cls_features 与 cnn_features 进行拼接
    concat_features = keras.layers.concatenate([cls_features,cnn_features] ,axis= -1)

    # 全连接层
    dense = keras.layers.Dense (
        units= 512, # 输出维度
        activation = 'relu' , # 激活函数
        kernel_initializer= bert.initializer # bert权重初始化
    )(concat_features) # 输入

    # 输出
    output = keras.layers.Dense (
        units= class_nums, # 输出类别数量
        activation= 'softmax', # 激活函数 (多分类输出层最常用的激活函数)
        kernel_initializer= bert.initializer # bert权重初始化
    )(dense) # 输入

    model = keras.models.Model(bert.model.input,output) # (bert.model.input输入，output输出)
    print(model.summary())

    return model

if __name__ == '__main__':
    config_path = '.\chinese_L-12_H-768_A-12\\bert_config.json'
    checkpoint_path = '.\chinese_L-12_H-768_A-12\\bert_model.ckpt'
    class_nums = 13
    build_bert_model(config_path , checkpoint_path , class_nums)
