# Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
import torch
import torch.utils.data as Data

# 数据准备

# sentences 的长度为 3
#               Encoder_input    Decoder_input        Decoder_output
sentences = [['我 是 学 生 P', 'S I am a student', 'I am a student E'],         # S: 开始符号
             ['我 喜 欢 学 习', 'S I like learning P', 'I like learning P E'],  # E: 结束符号
             ['我 是 男 生 P', 'S I am a boy', 'I am a boy E']]                 # P: 占位符号，如果当前句子不足固定长度用P占位

# 词源字典  字：索引
src_vocab = {'P': 0, '我': 1, '是': 2, '学': 3, '生': 4, '喜': 5, '欢': 6, '习': 7, '男': 8}
src_idx2word = {src_vocab[key]: key for key in src_vocab}
# print(src_idx2word) # 输出为 索引：字
# 字典字的个数
src_vocab_size = len(src_vocab)

tgt_vocab = {'P': 0, 'S': 1, 'E': 2, 'I': 3, 'am': 4, 'a': 5, 'student': 6, 'like': 7, 'learning': 8, 'boy': 9}
# 把目标字典转换成 索引：字的形式
idx2word = {tgt_vocab[key]: key for key in tgt_vocab}
# 目标字典尺寸
tgt_vocab_size = len(tgt_vocab)

# Encoder输入的最大长度
src_len = len(sentences[0][0].split(" "))
# Decoder输入输出最大长度
tgt_len = len(sentences[0][1].split(" "))


# 把sentences 转换成字典索引
def make_data():
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    # len(sentences) = 3
    for i in range(len(sentences)):
        # print("第%d次循环" % i)
        # sentences的第一列为 encoder 输入
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]
        # print(enc_input)
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]

        # extend () 函数用于在列表末尾一次性追加另一个序列中的多个值
        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)

    # torch.LongTensor 生成64位整型的张量tensor
    # print(enc_inputs)
    # print(torch.LongTensor(enc_inputs))
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)


# make_data()


# 自定义数据集函数
class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

# sentences 里一共有三个训练数据，中文->英文。
# 把Encoder_input、Decoder_input、Decoder_output转换成字典索引，
# 例如"学"->3、"student"->6。
# 再把数据转换成batch大小为2的分组数据，3句话一共可以分成两组，一组2句话、一组1句话。
# src_len表示中文句子固定最大长度，tgt_len 表示英文句子固定最大长度。
