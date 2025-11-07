# MyTextStockFusionModel.py (已修正 DataEmbedding_inverted 的调用)

import torch
import torch.nn as nn

# [!! 新增 !!] 导入 iTransformer 的核心 "积木"
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.Embed import DataEmbedding_inverted # iTransformer 的反转 Embedding

# 导入你原有的 "积木"
from layers.Transformer_EncDec import DecoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding, PositionalEmbedding

class TextStockFusionModel(nn.Module):
    def __init__(self, configs):
        super(TextStockFusionModel, self).__init__()
        
        self.configs = configs
        self.stock_seq_len = configs.stock_seq_len
        self.text_seq_len = configs.text_seq_len
        self.pred_len = configs.pred_len 
        self.close_idx = configs.close_idx 
        
        # --- 1. 融合特征提取部分 ---
        
        self.stock_embed = DataEmbedding(
            c_in=configs.stock_c_in, 
            d_model=configs.d_model,
            embed_type=configs.embed_type,
            freq=configs.freq,
            dropout=configs.dropout
        )
        
        self.text_embed = nn.Embedding(
            configs.vocab_size, 
            configs.d_model, 
            padding_idx=0
        )
        self.text_pos_embed = PositionalEmbedding(d_model=configs.d_model)
        
        self.fusion_decoder = DecoderLayer(
            self_attention=AttentionLayer(
                FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                configs.d_model, configs.n_heads
            ),
            cross_attention=AttentionLayer(
                FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                configs.d_model, configs.n_heads
            ),
            d_model=configs.d_model,
            d_ff=configs.d_ff,
            dropout=configs.dropout,
            activation=configs.activation
        )
        self.dropout = nn.Dropout(configs.dropout)

        # --- 2. iTransformer 预测部分 ---
        
        # 我们的输入 N = d_model (融合特征) + 1 (Close价格)
        it_c_in = configs.d_model + 1  # This is N (e.g., 65)
        it_d_model = configs.d_model # This is E (e.g., 64)
        
        # iTransformer 的反转 Embedding
        self.it_embedding = DataEmbedding_inverted(
            # [!! 核心 T_T!!]
            # c_in 应该传入 L (序列长度), 而不是 N (特征数)
            c_in=configs.stock_seq_len, # <--- [!! 修正 !!] 传入 L (e.g., 30)
            d_model=it_d_model,         # <--- 传入 E (e.g., 64)
            embed_type=configs.embed_type,
            freq=configs.freq,
            dropout=configs.dropout
        )
        
        self.it_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), it_d_model, configs.n_heads),
                    it_d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(it_d_model)
        )
        
        self.it_projector = nn.Linear(it_d_model, self.pred_len, bias=True)


    def forward(self, x_enc, x_mark_enc, text_ids_enc, x_dec=None, x_mark_dec=None):
        # x_enc:        [B, L_stock, c_in] (e.g., 4 or 5)
        # x_mark_enc:   [B, L_stock, time_features]
        # text_ids_enc: [B, L_stock, L_text]
        
        B = x_enc.shape[0]
        
        # --- 步骤 1: 运行融合模型 (特征提取) ---
        
        stock_vec = self.stock_embed(x_enc, x_mark_enc) # [B, L_stock, d_model]
        
        text_ids_flat = text_ids_enc.view(B * self.stock_seq_len, self.text_seq_len)
        text_vec_flat = self.text_embed(text_ids_flat) + self.text_pos_embed(text_ids_flat)
        day_text_vec = torch.mean(text_vec_flat, dim=1) 
        text_vec = day_text_vec.view(B, self.stock_seq_len, self.configs.d_model)
        text_vec = self.dropout(text_vec)

        fused_output = self.fusion_decoder(stock_vec, text_vec) # [B, L_stock, d_model=64]

        # --- 步骤 2: 准备 iTransformer 的输入 ---
        
        close_price = x_enc[:, :, self.close_idx : self.close_idx + 1] # [B, L_stock, 1]
        
        it_input = torch.cat((fused_output, close_price), dim=-1) # [B, L_stock, d_model+1]
        
        # --- 步骤 3: 运行 iTransformer 预测 ---
        
        # (B, L, N) -> (B, N, E)
        it_enc_out = self.it_embedding(it_input, None) 
        
        it_enc_out, attns = self.it_encoder(it_enc_out, attn_mask=None) # [B, N, E]
        
        dec_out = self.it_projector(it_enc_out) # [B, N, E] -> [B, N, L_pred]
        
        dec_out = dec_out.permute(0, 2, 1) # [B, N, L_pred] -> [B, L_pred, N]
        
        # 提取 'Close' 价格的预测
        final_prediction = dec_out[:, :, -1:] # [B, L_pred, 1]
        
        return final_prediction
    
    