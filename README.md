# fasttext-based-on-pytorch
带有2-gram特征和3-gram特征的fasttext模型

新闻文本分类，运行run.py开始训练

data_processing.py用于处理数据，构建词表及数据迭代器

fasttext.py 包含模型参数

train.py    训练模型并评估**

------------------------------

算法流程：

1.模型输入：                [batch_size, seq_len]

2.embedding层：（随机初始化）   

用哈希算法将2-gram、3-gram信息分别映射到两张表内。

word：                      [batch_size, seq_len, embed_size]

2-gram：                    [batch_size, seq_len, embed_size]

3-gram：                    [batch_size, seq_len, embed_size]

3.拼接embedding层： [batch_size, seq_len， embed_size * 3]

4.求所有seq_len个词的均值：[batch_size, embed_size * 3]

5.全连接+非线性激活：       [batch_size, hidden_size]

6.全连接：                  [batch_size, num_class]

7.softmax.归一化：          [batch_size, 1]
