import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from utils.timefeatures import time_features 

class MyFusionDataset(Dataset):
    def __init__(self, text_csv_path, stock_csv_path, tokenizer, 
                 stock_seq_len, text_seq_len, pred_len, # [!! 新增 pred_len !!]
                 freq='d', stock_features=None, flag='train'):
        
        self.tokenizer = tokenizer
        self.stock_seq_len = stock_seq_len
        self.text_seq_len = text_seq_len
        self.pred_len = pred_len # [!! 新增 !!]
        self.freq = freq
        self.flag = flag 

        if stock_features is None:
            # [!! 关键 !!] 确保 'Close' 是最后一个, 以便在模型中用 index=3 提取
            self.stock_features = ['Open', 'High', 'Low', 'Close'] 
        else:
            self.stock_features = stock_features
            
        self.load_and_align_data(text_csv_path, stock_csv_path)

    def load_and_align_data(self, text_csv_path, stock_csv_path):
        try:
            text_df = pd.read_csv(text_csv_path, encoding='gbk')
            stock_df = pd.read_csv(stock_csv_path, encoding='gbk')
        except UnicodeDecodeError:
            print("警告: 'gbk' 编码失败, 正在尝试 'latin-1'...")
            text_df = pd.read_csv(text_csv_path, encoding='latin-1')
            stock_df = pd.read_csv(stock_csv_path, encoding='latin-1')
        except FileNotFoundError:
            print(f"警告：正在使用虚拟数据...")
            text_df = pd.DataFrame(eval(text_csv_path))
            stock_df = pd.DataFrame(eval(stock_csv_path))

        text_df['date'] = pd.to_datetime(text_df['date'])
        stock_df['date'] = pd.to_datetime(stock_df['date'])

        print("按日期聚合每日新闻...")
        text_agg_df = text_df.groupby('date')['news_text'].apply(' '.join).reset_index()

        self.data = pd.merge(stock_df, text_agg_df, on='date', how='left')
        self.data['news_text'] = self.data['news_text'].fillna("[PAD]")
        self.data = self.data.sort_values(by='date').reset_index(drop=True)

        print(f"原始日期对齐后，共 {len(self.data)} 条有效股价日。")

        num_total = len(self.data)
        num_train = int(num_total * 0.7)
        num_val = int(num_total * 0.1)
        
        # [!! 修改 !!] 数据分割边界需要考虑 pred_len
        if self.flag == 'train':
            self.data = self.data.iloc[0 : num_train]
        elif self.flag == 'val':
            # 验证集需要 L_stock 的历史
            self.data = self.data.iloc[num_train - self.stock_seq_len : num_train + num_val]
        elif self.flag == 'test':
            # 测试集需要 L_stock 的历史
            self.data = self.data.iloc[num_train + num_val - self.stock_seq_len : ]
        
        print(f"为 {self.flag} 加载了 {len(self.data)} 条数据。")

    def __len__(self):
        # [!! 核心修改 !!] 总长度现在要减去历史和未来
        return len(self.data) - self.stock_seq_len - self.pred_len

    def __getitem__(self, idx):
        # [!! 核心修改 !!] 定义三个窗口
        hist_start = idx
        hist_end = idx + self.stock_seq_len
        label_start = hist_end
        label_end = label_start + self.pred_len

        # 1. 历史窗口 (x_enc)
        hist_slice = self.data.iloc[hist_start:hist_end]
        x_enc_values = hist_slice[self.stock_features].values.astype(np.float32)
        
        # 2. 历史时间特征 (x_mark_enc)
        hist_dates = hist_slice['date']
        x_mark_enc = time_features(hist_dates, timeenc=1, freq=self.freq).astype(np.float32)

        # 3. 历史文本 (text_ids_enc)
        hist_texts_list = hist_slice['news_text'].tolist()
        tokenized = self.tokenizer(
            hist_texts_list,
            max_length=self.text_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        text_ids_enc = tokenized['input_ids'].long() 

        # 4. 标签窗口 (y)
        label_slice = self.data.iloc[label_start:label_end]
        y_labels = label_slice[self.stock_features].values.astype(np.float32)

        return {
            'x_enc': torch.tensor(x_enc_values, dtype=torch.float32),         # [L_stock, c_in]
            'x_mark_enc': torch.tensor(x_mark_enc, dtype=torch.float32),     # [L_stock, time_features]
            'text_ids_enc': text_ids_enc,                                   # [L_stock, L_text]
            'y': torch.tensor(y_labels, dtype=torch.float32)                 # [L_pred, c_in]
        }