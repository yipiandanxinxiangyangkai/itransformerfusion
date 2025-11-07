# main_run.py (已更新为动态索引)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from types import SimpleNamespace 
import pandas as pd
import numpy as np
import os 
from transformers import AutoTokenizer

# 1. 导入 "积木"
from layers.Transformer_EncDec import DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding, PositionalEmbedding, DataEmbedding_inverted

# 2. 导入自定义模块
from MyTextStockFusionModel import TextStockFusionModel
from MyDataset import MyFusionDataset

# (get_dummy_data 函数... 在这里省略以保持简洁)
def get_dummy_data():
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
    stock_data = {'date': dates, 'Open': np.random.rand(200) * 100 + 100, 'High': np.random.rand(200) * 10 + 100, 'Low': np.random.rand(200) * -10 + 100, 'Close': np.random.rand(200) * 10 + 100, 'Volume': np.random.rand(200) * 1000}
    stock_df = pd.DataFrame(stock_data)
    stock_df = stock_df[stock_df['date'].dt.dayofweek < 5] 
    news_dates = dates.repeat(np.random.randint(1, 4, size=200))
    news_data = {'date': news_dates, 'news_text': [f"这是一条关于 {d.dayofweek} 的新闻" for d in news_dates]}
    news_df = pd.DataFrame(news_data)
    news_df = news_df.sample(frac=0.9) 
    return str(news_df.to_dict('records')), str(stock_df.to_dict('records'))


# --- 主程序开始 ---
if __name__ == '__main__':
    
    # 步骤 3: 定义配置 (Configs)
    stock_seq_len = 30  
    text_seq_len = 50   
    pred_len = 30 # 预测未来 30 天
    batch_size = 16
    
    # [!! 核心修改 !!] 在这里统一定义特征
    # 这样 'Close' 就是你说的 "倒数第二列" 了
    stock_features_list = ['Open', 'High', 'Low', 'Close', 'Volume'] 
    
    # 如果你不想用 'Volume', 就用下面这行:
    # stock_features_list = ['Open', 'High', 'Low', 'Close'] 
    
    configs = SimpleNamespace(
        # 序列长度
        stock_seq_len=stock_seq_len, 
        text_seq_len=text_seq_len,   
        pred_len=pred_len,
        
        # iTransformer Encoder 层数
        e_layers=2,
        
        # [!! 核心修改 !!] 动态设置
        stock_c_in=len(stock_features_list), # 自动设为 4 或 5
        stock_features=stock_features_list, # 传递给 Dataset
        close_idx=stock_features_list.index('Close'), # 自动找到 'Close' 的索引 (e.g., 3)
        
        # 股价/Embed
        embed_type='timeF', freq='d',          
        
        # 文本
        vocab_size=30522,  
        
        # Transformer
        d_model=64, n_heads=8, d_ff=128,
        factor=5, dropout=0.1, activation='relu',
        text_hidden_dim=32,
    )

    # 步骤 4: 创建模型
    model = TextStockFusionModel(configs)
    
    # 步骤 5: 创建数据集和加载器
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    
    real_news_path = '.\\data\\AAPL_text_processed.csv'
    real_stock_path = '.\\data\\AAPL_price_processed.csv'
    
    # [!! 核心修改 !!] 传入 pred_len 和 stock_features
    train_dataset = MyFusionDataset(
        real_news_path, real_stock_path, tokenizer,
        stock_seq_len=configs.stock_seq_len, 
        text_seq_len=configs.text_seq_len, 
        pred_len=configs.pred_len,
        stock_features=configs.stock_features, # [!! 新增 !!]
        freq=configs.freq, flag='train'
    )
    val_dataset = MyFusionDataset(
        real_news_path, real_stock_path, tokenizer,
        stock_seq_len=configs.stock_seq_len, 
        text_seq_len=configs.text_seq_len, 
        pred_len=configs.pred_len,
        stock_features=configs.stock_features, # [!! 新增 !!]
        freq=configs.freq, flag='val'
    )
    test_dataset = MyFusionDataset(
        real_news_path, real_stock_path, tokenizer,
        stock_seq_len=configs.stock_seq_len, 
        text_seq_len=configs.text_seq_len, 
        pred_len=configs.pred_len,
        stock_features=configs.stock_features, # [!! 新增 !!]
        freq=configs.freq, flag='test'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 步骤 6: 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = torch.nn.MSELoss() 
    
    num_epochs = 10 
    best_val_loss = float('inf') 
    model_save_path = './best_fusion_it_model.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 步骤 7: 训练循环
    print(">>>>>>> 开始训练融合 iTransformer 模型 >>>>>>>")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss_total = 0.0
        
        for batch in train_loader:
            x_enc = batch['x_enc'].to(device)
            x_mark_enc = batch['x_mark_enc'].to(device)
            text_ids_enc = batch['text_ids_enc'].to(device)
            y = batch['y'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(x_enc, x_mark_enc, text_ids_enc) # [B, L_pred, 1]
            
            # [!! 核心修改 !!] 使用动态索引
            labels = y[:, :, configs.close_idx : configs.close_idx + 1] # [B, L_pred, 1]
            
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()
        
        avg_train_loss = train_loss_total / len(train_loader)
        
        # --- 验证循环 ---
        model.eval() 
        val_loss_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_enc = batch['x_enc'].to(device)
                x_mark_enc = batch['x_mark_enc'].to(device)
                text_ids_enc = batch['text_ids_enc'].to(device)
                y = batch['y'].to(device)
                
                outputs = model(x_enc, x_mark_enc, text_ids_enc)
                # [!! 核心修改 !!] 使用动态索引
                labels = y[:, :, configs.close_idx : configs.close_idx + 1]
                
                val_loss = loss_fn(outputs, labels)
                val_loss_total += val_loss.item()
        
        avg_val_loss = val_loss_total / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, 训练 Loss: {avg_train_loss:.6f}, 验证 Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"模型已保存 (Val Loss 降低至 {avg_val_loss:.6f})")

    print(">>>>>>> 训练完成 >>>>>>>")

    # --- 步骤 8：测试模型 ---
    print("\n>>>>>>> 开始测试模型 >>>>>>>")
    
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            x_enc = batch['x_enc'].to(device)
            x_mark_enc = batch['x_mark_enc'].to(device)
            text_ids_enc = batch['text_ids_enc'].to(device)
            y = batch['y'].to(device)
            
            outputs = model(x_enc, x_mark_enc, text_ids_enc) # [B, L_pred, 1]
            # [!! 核心修改 !!] 使用动态索引
            labels = y[:, :, configs.close_idx : configs.close_idx + 1]
            
            all_preds.append(outputs.cpu())
            all_labels.append(labels.cpu())

    all_preds_tensor = torch.cat(all_preds, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)
    
    test_mse = loss_fn(all_preds_tensor, all_labels_tensor)
    test_rmse = torch.sqrt(test_mse)
    
    print(f"测试集 MSE: {test_mse.item():.6f}")
    print(f"测试集 RMSE: {test_rmse.item():.6f}")
    
    # --- 保存预测结果到 CSV ---
    print("\n>>>>>>> 正在保存预测结果到 CSV >>>>>>>")
    
    predictions_np = all_preds_tensor.flatten().numpy()
    labels_np = all_labels_tensor.flatten().numpy()
    
    results_df = pd.DataFrame({
        'Predicted_Close': predictions_np,
        'Actual_Close': labels_np
    })
    
    output_csv_path = '.\\data\\prediction_results.csv'
    results_df.to_csv(output_csv_path, index=False)
    
    print(f"预测结果已成功保存到: {output_csv_path}")
    print(">>>>>>> 测试完成 >>>>>>>")