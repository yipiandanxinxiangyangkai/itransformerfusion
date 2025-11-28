import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    # --- [修复 1/2]: 将 __init__ 里的 selfself 改回 self ---
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
    # --- [修复结束] ---
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# 假设 time_features 是从 utils.timefeatures 导入的
# 如果你没有这个文件，请确保你有对应的函数定义
from utils.timefeatures import time_features 

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', 
                 text_data_path='text.csv'):
        
        # size [seq_len, label_len, pred_len]
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.text_data_path = text_data_path 
        
        self.flag = flag # 存储 flag 以判断是否为 test 模式
        
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        
        # 1. 读取价格和文本文件
        price_path = os.path.join(self.root_path, self.data_path)
        df_price = pd.read_csv(price_path, encoding='latin1')
        
        text_path = os.path.join(self.root_path, self.text_data_path)
        df_text = pd.read_csv(text_path, encoding='utf-8')

        # 2. 日期转换与合并
        df_price['date'] = pd.to_datetime(df_price['date'], errors='coerce')
        df_text['date'] = pd.to_datetime(df_text['date'], errors='coerce')
        df_raw = pd.merge(df_price, df_text, on='date', how='left')
        
        # --- [新增] 存储完整的日期列，并格式化为字符串 ---
        self.raw_dates_all = df_raw['date'].dt.strftime('%Y-%m-%d %H:%M:%S').values 
        
        # 3. 分离 股价 (data) 和 文本 (text)
        cols_data = ['Open', 'High', 'Low', 'Close'] 
        cols_text = [col for col in df_raw.columns if 'dim_' in col]
            
        df_data = df_raw[cols_data]
        df_text_vectors = df_raw[cols_text]
        df_stamp = df_raw[['date']]
        
        # 4. 按比例切分数据
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_val = len(df_raw) - num_train - num_test
        
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(df_raw)]
        self.border1 = border1s[self.set_type] # --- 保存 border1
        border2 = border2s[self.set_type]

        # 5. 标准化“股价” (data_x)
        data_x_raw = df_data.values # --- 保存原始数据
        
        if self.scale:
            train_data = df_data.values[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data_x_scaled = self.scaler.transform(df_data.values)
        else:
            data_x_scaled = data_x_raw

        # 6. 处理“文本向量” (data_text)
        data_text = df_text_vectors.values
        data_text = np.nan_to_num(data_text) 
        
        # 7. 处理“时间标记” (data_stamp)
        if self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            data_stamp = data_stamp.transpose(1, 0) if self.timeenc == 1 else np.zeros((len(df_raw), 1))
            # 注意：如果 timeenc=0，上面的 zeros 只是占位
            if self.timeenc == 0:
                 data_stamp = np.zeros((len(df_raw), 4)) # 通常 mark 有4个维度，这里暂设为4防报错

        # --- 保存完整的 (raw) 和缩放后的数据
        self.data_x_full_scaled = data_x_scaled 
        self.data_x_full_raw = data_x_raw 

        # 8. 赋值 (基于 train/val/test 划分)
        self.data_x = data_x_scaled[self.border1:border2] # 缩放后的数据用于输入
        self.data_y = data_x_scaled[self.border1:border2] # 缩放后的数据用于解码器输入
        
        self.data_text = data_text[self.border1:border2] 
        self.data_stamp = data_stamp[self.border1:border2] 
        
        # --- [新增] 存储切片后的日期列 (用于 __getitem__ 的日期返回) ---
        self.raw_dates_border = self.raw_dates_all[self.border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        # 1. 取出价格数据 
        # 形状: [seq_len, 4]，假设 seq_len=30，这里就是 [30, 4]
        seq_x_prices = self.data_x[s_begin:s_end]
        seq_y_prices = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # 2. 取出并处理文本向量
        pred_day_index = r_begin 
        if pred_day_index >= len(self.data_text):
            pred_day_index = len(self.data_text) - 1
            
        # 原始形状: (30,) 也就是一维数组
        text_vec_day_T = self.data_text[pred_day_index] 
        

        # 第一步：强制把文本变成竖着的列向量 (30, 1)
        # 这一步要求 text_vec_day_T 的长度必须和 seq_len 一致
        text_vec_reshaped = text_vec_day_T.reshape(-1, 1) 

        # 安全检查：防止 seq_len 不等于文本维度导致报错
        if seq_x_prices.shape[0] != text_vec_reshaped.shape[0]:
            raise ValueError(f"强制拼接失败！要求 seq_len ({seq_x_prices.shape[0]}) 必须等于文本维度 ({text_vec_reshaped.shape[0]})。请检查参数设置。")

        # 第二步：直接拼接
        # 左边 [30, 4] + 右边 [30, 1] = 结果 [30, 5]
        seq_x = np.concatenate([seq_x_prices, text_vec_reshaped], axis=1)
        
        # ---------------------------------------

        # 3. 创建分类标签 (保持原逻辑不变)
        global_r_begin = self.border1 + r_begin
        global_r_end = self.border1 + r_end
        
        if global_r_begin == 0:
            first_label = np.array([0]) 
            prices_t = self.data_x_full_raw[global_r_begin + 1 : global_r_end, 3]
            prices_t_minus_1 = self.data_x_full_raw[global_r_begin : global_r_end - 1, 3]
            diff = prices_t - prices_t_minus_1
            subsequent_labels = (diff > 0).astype(np.int64)
            seq_y_labels = np.concatenate([first_label, subsequent_labels])
        else:
            prices_t = self.data_x_full_raw[global_r_begin : global_r_end, 3] 
            prices_t_minus_1 = self.data_x_full_raw[global_r_begin - 1 : global_r_end - 1, 3]
            diff = prices_t - prices_t_minus_1
            seq_y_labels = (diff > 0).astype(np.int64) 

        # 4. 返回结果
        if self.flag == 'test':
            # 转换为 list 避免 collate 错误
            seq_dates_np = self.raw_dates_border[r_begin:r_end]
            seq_dates_list = list(seq_dates_np) 
            return seq_x, seq_y_prices, seq_x_mark, seq_y_mark, seq_dates_list, seq_y_labels
        else:
            return seq_x, seq_y_prices, seq_x_mark, seq_y_mark, seq_y_labels

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
class Dataset_PEMS(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)
        data = data['data'][:, :, 0]

        train_ratio = 0.6
        valid_ratio = 0.2
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)): int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]
        total_data = [train_data, valid_data, test_data]
        data = total_data[self.set_type]

        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        df = pd.DataFrame(data)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values

        self.data_x = df
        self.data_y = df

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', 
                 text_data_path='text.csv', cols=None): # [新] 添加文本路径参数
        # size [seq_len, label_len, pred_len]
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.text_data_path = text_data_path # [新] 保存文本路径
        
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        
        # 1. 读取价格和文本文件
        price_path = os.path.join(self.root_path, self.data_path)
        df_price = pd.read_csv(price_path, encoding='latin1')
        text_path = os.path.join(self.root_path, self.text_data_path)
        df_text = pd.read_csv(text_path, encoding='utf-8')

        # 2. 合并数据
        df_price['date'] = pd.to_datetime(df_price['date'], errors='coerce')
        df_text['date'] = pd.to_datetime(df_text['date'], errors='coerce')
        df_raw = pd.merge(df_price, df_text, on='date', how='left')

        # 3. 分离数据和文本
        cols_data = ['Open', 'High', 'Low', 'Close'] 
        cols_text = [col for col in df_raw.columns if 'dim_' in col]
        df_data = df_raw[cols_data]
        df_text_vectors = df_raw[cols_text]
        df_text_vectors = df_text_vectors.fillna(0) # 填充 NaN

        # 4. 边界和缩放 (只取最后 seq_len 个数据用于预测)
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.scale:
            # 仅用训练集数据进行拟合（这里简化为 border1 之前的数据）
            if border1 > 0:
                train_data = df_data.values[:border1] 
                self.scaler.fit(train_data)
            else: # 如果整个数据集都是预测数据，就用所有数据拟合（不推荐，但安全）
                self.scaler.fit(df_data.values)
                
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        # 5. 时间标记处理 
        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date.values, errors='coerce')
        
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            if self.freq == 't' or self.freq == '15min':
                df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
                df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            # [修复] 确保在 timeenc=0 时 data_stamp 被定义
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            data_stamp = np.zeros((len(df_raw) + self.pred_len, 1))

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_text = df_text_vectors.values[border1:border2] # 文本数据
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        text_vec_day_T = self.data_text[s_end - 1] 
        
        text_feature_to_smuggle = text_vec_day_T[0] # 取 'dim_1'
        text_vec_smuggled = np.full((self.seq_len, 1), text_feature_to_smuggle)
        seq_x = np.concatenate([seq_x, text_vec_smuggled], axis=1)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
