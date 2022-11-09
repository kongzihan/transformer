import numpy as np
import torch
import torch.nn as nn
from datasets import *

# 设置参数
d_model = 512   # 字 Embedding 的维度
d_ff = 2048     # 前向传播隐藏层维度
d_k = d_v = 64  # K(=Q), V的维度
n_layers = 6    # 有多少个encoder和decoder
n_heads = 8     # Multi-Head Attention设置为8

test_arr = np.array([[1, 2, 3, 4, 5],
            [11, 12, 13, 14, 15],
            [21, 22, 23, 24, 25]])
# 从第二行开始，从每行的第0个元素开始，每隔三个元素输出一个
# print(test_arr[2:, 0::3])
test1 = torch.Tensor(test_arr)
print(test1)
print(test_arr.shape)
test2 = test1.data.eq(25).unsqueeze(1)
print(test2)

# 定义位置信息
class PositionalEncoding(nn.Module):
    # 初始化函数 init
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # nn.dropout()是为了防止或减轻过拟合而使用的函数，
        # 它一般用在全连接层 Dropout就是在不同的训练过程中随机扔掉一部分神经元。
        self.dropout = nn.Dropout(p=dropout)
        pos_table = np.array([
            [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
            if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        print(pos_table)
        # pos_table 为一个固定矩阵
        # 字嵌入维度为偶数时,np.sin (a)函数：对a中元素取正弦值。
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])
        # 字嵌入维度为奇数时
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])
        # enc_inputs: [seq_len, d_model]
        self.pos_table = torch.FloatTensor(pos_table)

    def forward(self, enc_inputs):
        # enc_inputs: [batch_size, seq_len, d_model]
        enc_inputs += self.pos_table[:enc_inputs.size(1), :]
        return self.dropout(enc_inputs)
        # 生成位置信息矩阵pos_table，直接加上输入的enc_inputs上，得到带有位置信息的字向量，
        # pos_table是一个固定值的矩阵。
        # 这里矩阵加法利用到了广播机制


# mask掉停用词
# Mask句子中没有实际意义的占位符，
# 例如'我 是 学 生 P' ，P对应句子没有实际意义，所以需要被Mask，
# Encoder_input 和 Decoder_input 占位符都需要被Mask。
def get_attn_pad_mask(seq_q, seq_k):
    # seq_q: [batch_size, seq_len] ,seq_k: [batch_size, seq_len]
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # 判断 输入那些含有P(=0),用1标记 ,[batch_size, 1, len_k]
    # unsqueeze ()函数用于扩充维度，它有一个参数unsqueeze (dim)，表示在第dim维上扩充维度。
    # eq函数是留下seq_k等于0的坐标
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    # 扩展成多维度,将1维扩展到 len_q 维
    return pad_attn_mask.expand(batch_size, len_q, len_k)


# decoder 输入 mask
# 用来Mask未来输入信息，返回的是一个上三角矩阵。
# 比如我们在中英文翻译时候，会先把"我是学生"整个句子输入到Encoder中，
# 得到最后一层的输出后，才会在Decoder输入"S I am a student"（s表示开始）,
# 但是"S I am a student"这个句子我们不会一起输入，
# 而是在T0时刻先输入"S"预测，预测第一个词"I"；
# 在下一个T1时刻，同时输入"S"和"I"到Decoder预测下一个单词"am"；
# 然后在T2时刻把"S,I,am"同时输入到Decoder预测下一个单词"a"，
# 依次把整个句子输入到Decoder,预测出"I am a student E"。
def get_attn_subsequence_mask(seq):
    # seq: [batch_size, tgt_len]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # 生成上三角矩阵,[batch_size, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    # [batch_size, tgt_len, tgt_len]
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask


# 计算注意力信息、残差和归一化
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # Q: [batch_size, n_heads, len_q, d_k]
        # K: [batch_size, n_heads, len_k, d_k]
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        # attn_mask: [batch_size, n_heads, seq_len, seq_len]

        # scores : [batch_size, n_heads, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)

        # 如果时停用词P就等于 0
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        # [batch_size, n_heads, len_q, d_v]
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        # input_Q: [batch_size, len_q, d_model]
        # input_K: [batch_size, len_k, d_model]
        # input_V: [batch_size, len_v(=len_k), d_model]
        # attn_mask: [batch_size, seq_len, seq_len]
        residual, batch_size = input_Q, input_Q.size(0)
        # Q: [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        # attn_mask : [batch_size, n_heads, seq_len, seq_len]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        # context: [batch_size, n_heads, len_q, d_v]
        # attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        # context: [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        # [batch_size, len_q, d_model]
        output = self.fc(context)
        return nn.LayerNorm(d_model)(output + residual), attn
# 计算注意力信息， W^Q，W^K，W^V 矩阵会拆分成8个小矩阵。
# 细节请看2.2章。
# 注意传入的input_Q, input_K, input_V，
# 在Encoder和Decoder的第一次调用传入的三个矩阵是相同的，
# 但Decoder的第二次调用传入的三个矩阵input_Q 等于 input_K 不等于 input_V。


# 前馈神经网络
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False))

    def forward(self, inputs):
        # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        # [batch_size, seq_len, d_model]
        return nn.LayerNorm(d_model)(output + residual)


# 输入inputs ，经过两个全连接成，得到的结果再加上 inputs ，再做LayerNorm归一化。
# LayerNorm归一化可以理解层是把Batch中每一句话进行归一化。


# 单个 encoder
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        # 多头注意力机制
        self.enc_self_attn = MultiHeadAttention()
        # 前馈神经网络
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        # enc_inputs: [batch_size, src_len, d_model]
        # enc_self_attn_mask: [batch_size, src_len, src_len]

        # 输入3个enc_inputs分别与W_q、W_k、W_v相乘得到Q、K、V
        # enc_outputs: [batch_size, src_len, d_model],
        # attn: [batch_size, n_heads, src_len, src_len]
        # enc_outputs: [batch_size, src_len, d_model]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


# 整个 encoder
# 第一步，中文字索引进行Embedding，转换成512维度的字向量。
# 第二步，在字向量上面加上位置信息。
# 第三步，Mask掉句子中的占位符号。
# 第四步，通过6层的encoder（上一层的输出作为下一层的输入）。
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # 把字转换字向量
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        # 加入位置信息
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        # enc_inputs: [batch_size, src_len]
        # enc_outputs: [batch_size, src_len, d_model]
        enc_outputs = self.src_emb(enc_inputs)
        # enc_outputs: [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs)
        # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs :   [batch_size, src_len, d_model],
            # enc_self_attn : [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


# 单个 decoder
# decoder两次调用MultiHeadAttention时，
# 第一次调用传入的 Q，K，V 的值是相同的，都等于dec_inputs，
# 第二次调用 Q 矩阵是来自Decoder的输入。K，V 两个矩阵是来自Encoder的输出，等于enc_outputs。
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        # dec_inputs: [batch_size, tgt_len, d_model]
        # enc_outputs: [batch_size, src_len, d_model]
        # dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        # dec_enc_attn_mask: [batch_size, tgt_len, src_len]

        # dec_outputs: [batch_size, tgt_len, d_model]
        # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs,
                                                        dec_inputs,
                                                        dec_self_attn_mask)

        # dec_outputs: [batch_size, tgt_len, d_model]
        # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs,
                                                      enc_outputs,
                                                      dec_enc_attn_mask)

        # dec_outputs: [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


# 整个 decoder
# 第一步，英文字索引进行Embedding，转换成512维度的字向量。
# 第二步，在子向量上面加上位置信息。
# 第三步，Mask掉句子中的占位符号和输出顺序细节见3.1。
# 第四步，通过6层的decoder（上一层的输出作为下一层的输入）。
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):                         # dec_inputs: [batch_size, tgt_len]
                                                                                    # enc_intpus: [batch_size, src_len]
                                                                                    # enc_outputs: [batsh_size, src_len, d_model]
        dec_outputs = self.tgt_emb(dec_inputs)                                      # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs)                               # [batch_size, tgt_len, d_model]
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)    # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)   # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask +
                                       dec_self_attn_subsequence_mask), 0)    # [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)               # [batc_size, tgt_len, src_len]
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:                                                   # dec_outputs: [batch_size, tgt_len, d_model]
                                                                                    # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
                                                                                    # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


# Trasformer的整体结构，
# 输入数据先通过Encoder，再通过Decoder，
# 最后把输出进行多分类，分类数为英文字典长度，也就是判断每一个字的概率。
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.Encoder = Encoder()
        self.Decoder = Decoder()
        # nn.linear ()是用来设置网络中的全连接层的，
        # 而在全连接层中的输入与输出都是二维张量，一般形状为 [batch_size, size]，
        # 与卷积层要求输入输出是4维张量不同
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):                          # enc_inputs: [batch_size, src_len]
                                                                        # dec_inputs: [batch_size, tgt_len]
        enc_outputs, enc_self_attns = self.Encoder(enc_inputs)          # enc_outputs: [batch_size, src_len, d_model],
                                                                        # enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.Decoder(
            dec_inputs, enc_inputs, enc_outputs)                        # dec_outpus    : [batch_size, tgt_len, d_model],
                                                                        # dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len],
                                                                        # dec_enc_attn  : [n_layers, batch_size, tgt_len, src_len]
        dec_logits = self.projection(dec_outputs)                       # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns
        # view()函数有两种用法：
        # 1.普通用法，手动调整 size
        # view()相当于reshape、resize，重新调整Tensor的形状
        # 2.特殊用法，参数 -1 ，自动调整 size
        # view中一个参数定为-1，代表自动调整这个维度上的元素个数，以保证元素的总数不变。
