import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 4, 'b': 4}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    """
    标准数据嵌入层 (Standard Data Embedding)
    将时间步 (L) 作为 Tokens 处理，与传统 Transformer 相似。
    """
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        # 1. 值嵌入：将输入特征维度 c_in (即 N) 映射到 d_model
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        # 2. 位置嵌入：添加时间步 L 的绝对/相对位置信息
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        # 3. 时间嵌入：处理 x_mark (时间标记/协变量)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                  freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # x 的形状通常是 (B, L, N)
        if x_mark is None:
            # 仅值嵌入 + 位置嵌入
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            # 值嵌入 + 时间嵌入 + 位置嵌入 (三者相加)
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        # 输出形状: (B, L, E) -> L 是 Tokens
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):#重点看
    """
    反转数据嵌入层 (Inverted Data Embedding) - 用于 iTransformer
    将变量 (N) 视为 Tokens,时间步 (L) 视为特征维度。
    """
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        # 在反转后，c_in 实际上是原始的序列长度 L (seq_len)
        # 线性层用于将 L 维度的特征投影到 d_model 维（需要嵌入的维度，可以不只一维，可以很多维）
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # 原始输入 x 形状: (B, L, N)
        
        # 核心步骤 1: 维度转置 (Permutation)
        x = x.permute(0, 2, 1)
        # 形状变为: (B, N, L) -> [Batch(每次输入的数据量), Variate (N变量个数), Time (L时间长度)]
        # 此时，N 变成了序列维度 (Tokens)，L 变成了特征维度
        
        if x_mark is None:
            # 直接将 L 维特征投影到 d_model
            x = self.value_embedding(x)
        else:
            # 协变量 (x_mark) 也需要转置: (B, L, MarkDim) -> (B, MarkDim, L)
            # 然后沿着 Token 维度 (维度 1) 与 x 拼接
            # 拼接后形状: (B, N + MarkDim, L)
            # 这样做是为了将时间标记作为额外的“变量 Token”来处理
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1)) 
        
        # 核心步骤 2: 线性投影
        # 形状变为: (B, N, d_model) -> [Batch, Variate (Tokens), Embedding (E)]
        # 注意：这里的线性层作用于维度 L 上，将其映射到 d_model (E)
        
        # 输出形状: (B, N, E) -> N 是 Tokens，E 是特征维度
        return self.dropout(x)

