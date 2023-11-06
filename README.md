# Chinese-Text-Classification-Mindspore


中文文本分类，TextCNN，TextRNN，FastText，TextRCNN，BiLSTM_Attention, DPCNN, Transformer, 基于mindspore，开箱即用。

## 介绍

### 声明：
本项目是通过原[Chinese-Text-Classification-Pytorch](https://github.com/649453932/Chinese-Text-Classification-Pytorch)项目针对pytorch进行mindspore框架的模型迁移与训练，旨在让更多感兴趣的人能够上手mindspore并了解、支持mindspore的研发，做出贡献加速mindspore社区成长与完善,欢迎star:star2::blush::two_hearts:


### Mindspore： 
昇思[MindSpore](https://www.mindspore.cn/)通过社区合作，面向全场景构建最佳昇腾匹配、支持多处理器架构的开放AI架构，为算法工程师和数据科学家提供开发友好、运行高效、部署灵活的体验，帮助人工智能软硬件应用生态繁荣发展。

数据以字为单位输入模型，预训练词向量使用 [搜狗新闻 Word+Character 300d](https://github.com/Embedding/Chinese-Word-Vectors)，[点这里下载](https://pan.baidu.com/s/14k-9jsspp43ZhMxqPmsWMQ)  

## 环境
python 3.7  
mindspore 2.0.0 
tqdm  
sklearn  


## 中文数据集
从[THUCNews](http://thuctc.thunlp.org/)中抽取了20万条新闻标题，文本长度在20到30之间。一共10个类别，每类2万条。

类别：财经、房产、股票、教育、科技、社会、时政、体育、游戏、娱乐。

数据集划分：

数据集|数据量
--|--
训练集|18万
验证集|1万
测试集|1万


### 更换自己的数据集
 - 如果用字，按照数据集的格式来格式化你的数据。  
 - 如果用词，提前分好词，词之间用空格隔开，`python run.py --model TextCNN --word True`  
 - 使用预训练词向量：utils.py的main函数可以提取词表对应的预训练词向量。  



## 使用说明
```
# 训练并测试：
# TextCNN
python run.py --model TextCNN

# TextRNN
python run.py --model TextRNN

# TextRNN_Att
python run.py --model TextRNN_Att

# TextRCNN
python run.py --model TextRCNN

# FastText, embedding层是随机初始化的
python run.py --model FastText --embedding random 

# DPCNN
python run.py --model DPCNN

# Transformer
python run.py --model Transformer
```

### 参数
模型都在models目录下，超参定义和模型定义在同一文件中。  


### API对应及出处
[PyTorch与MindSpore API映射表](https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/note/api_mapping/pytorch_api_mapping.html)

