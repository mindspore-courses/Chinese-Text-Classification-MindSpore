import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore import Tensor, CSRTensor, COOTensor

class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextCNN'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = mindspore.Tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype(Tensor.float32))\
            if embedding != 'random' else None                                # 预训练词向量
        
        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)


'''Convolutional Neural Networks for Sentence Classification'''


class Model(nn.Cell):
    def __init__(self, config):
        super(Model, self).__init__()        # 继承自nn.Cell类 创建一个模型类实例(self)
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.CellList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes], has_bias=True)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Dense(config.num_filters * len(config.filter_sizes), config.num_classes)


    def conv_and_pool(self, x, conv):
        x = ops.relu(conv(x))                #relu:激活函数，它将所有负值变为零，并保持正值不变
        x = ops.squeeze(x, axis = 3)         #squeeze:去掉维度,axis = 删除指定 axis 中大小为1的维度
        x = ops.adaptive_max_pool1d(x, x.size(2))              #应用一维自适应最大池化操作
        x = ops.squeeze(x, axis = 2)
        return x
    
    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = ops.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)      #拼接卷积核池化操作后的张量
        out = self.dropout(out)
        out = self.fc(out)
        return out 
