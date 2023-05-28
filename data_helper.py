import json
import pandas as pd
import matplotlib.pyplot as plt
def gen_training_data(row_data_path) :
    label_list = [line.strip() for line in open('./dataset/label', 'r' ,encoding='utf8')]
    print(label_list)

    # 映射id，为每一条数据添加id
    label2id = {label : idx for idx, label in enumerate(label_list)}
    data = []
    with open('./dataset/CMID.json','r',encoding='utf8') as f :
        origin_data = f.read()
        origin_data = eval(origin_data)

    label_set = set()
    for item in origin_data :
        text = item['originalText']

        label_class = item['label_4class'][0].strip("'")
        if label_class == '其他' :
            data.append([text , label_class ,label2id[label_class]])
            continue
        label_class = item["label_36class"][0].strip("'") # 所有的意图标签都从label_36class中取出
        label_set.add(label_class)
        if label_class not in label_list:
            continue
        data.append([text, label_class ,label2id[label_class]])
    print(label_set)

    data = pd.DataFrame(data , columns=['text','label_class','label'])
    print(data['label_class'].value_counts())
    data['text_len'] = data['text'].map(lambda x : len(x)) # 序列长度
    print(data['text_len'].describe())
    plt.hist(data['text_len'], bins=30, rwidth= 0.9, density=True)
    plt.show()

    del data['text_len']

    data = data.sample(frac = 1.0)
    # 将数据集拆分为测试集和训练集
    train_num = int(0.9*len(data))
    train , test = data[:train_num],data[train_num:]
    train.to_csv('./dataset/train.csv', index=False)
    test.to_csv('./dataset/test.csv', index = False)

# 加载训练数据集
def load_data(filename) :
    df = pd.read_csv(filename , header= 0 )
    return df[['text','label']].values

if __name__ == '__main__':
    data_path = './dataset/CMID.json'
    gen_training_data(data_path)

