# Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
import torch.nn as nn
import torch.optim as optim
from datasets import *
from transformer_cuda import Transformer

# PyTorch由4个主要包装组成：
# 1.Torch：类似于Numpy的通用数组库，可以在将张量类型转换为（torch.cuda.TensorFloat）并在GPU上进行计算。
# 2.torch.autograd：用于构建计算图形并自动获取渐变的包
# 3.torch.nn：具有共同层和成本函数的神经网络库
# 4.torch.optim：具有通用优化算法（如SGD，Adam等）的优化包


if __name__ == "__main__":

    enc_inputs, dec_inputs, dec_outputs = make_data()
    loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)

    # 定义网络
    # model = Transformer().cuda()
    model = Transformer()
    # nn.CrossEntropyLoss() 计算交叉熵损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=0)         # 忽略 占位符 索引为0.
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

    # 训练 transformer
    for epoch in range(50):
        for enc_inputs, dec_inputs, dec_outputs in loader:  # enc_inputs : [batch_size, src_len]
                                                            # dec_inputs : [batch_size, tgt_len]
                                                            # dec_outputs: [batch_size, tgt_len]

            # enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()
            enc_inputs, dec_inputs, dec_outputs = enc_inputs, dec_inputs, dec_outputs
            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
                                                            # outputs: [batch_size * tgt_len, tgt_vocab_size]
            loss = criterion(outputs, dec_outputs.view(-1))
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    torch.save(model, 'model.pth')
    print("保存模型")
