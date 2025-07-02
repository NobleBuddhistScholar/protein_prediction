import torch
import torch.nn as nn
import torch.nn.functional as F
class HybridModel(nn.Module):
    """CNN-BiLSTM混合模型"""
    def __init__(self, vocab_size: int, embedding_dim: int, num_classes: int):
        super().__init__()
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # CNN部分
        self.conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(2)
        
        # BiLSTM部分
        self.lstm1 = nn.LSTM(256, 256, bidirectional=True, batch_first=True)
        self.dropout1 = nn.Dropout(0.5)
        
        self.lstm2 = nn.LSTM(512, 128, bidirectional=True, batch_first=True)
        self.dropout2 = nn.Dropout(0.5)
        
        # 全连接层
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # 添加层初始化
        self._init_weights()
    
    def _init_weights(self):
        for layer in [self.fc1, self.fc2]:
            nn.init.kaiming_normal_(layer.weight)
            nn.init.constant_(layer.bias, 0.1)
        
    def forward(self, x):
        # 输入形状: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # CNN处理
        x = x.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.pool1(x)  # (batch_size, 128, seq_len//2)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.pool2(x)  # (batch_size, 256, seq_len//4)
        
        # LSTM处理
        x = x.permute(0, 2, 1)  # (batch_size, seq_len//4, 256)
        x, _ = self.lstm1(x)    # (batch_size, seq_len//4, 512)
        x = self.dropout1(x)
        
        # 获取最后一个时间步的输出
        _, (h_n, _) = self.lstm2(x)  # h_n: (2, batch_size, 128)
        x = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch_size, 256)
        x = self.dropout2(x)
        
        # 全连接层
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

class MSA_ResGRUNet(nn.Module):
    def __init__(self, vocab_size=5, embedding_dim=64, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 多尺度卷积模块
        self.multi_scale = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embedding_dim, 64, k, padding=k//2),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.MaxPool1d(kernel_size=3, stride=2)  # 长度减半
            ) for k in [3, 5, 7]
        ])
        
        # 残差卷积块
        self.res_blocks = nn.Sequential(
            ResidualBlock(192, 256, dilation=1),
            ResidualBlock(256, 256, dilation=2),
            nn.MaxPool1d(kernel_size=3, stride=2)  # 长度再减半
        )
        
        # 自适应压缩层
        self.adaptive_compress = nn.Sequential(
            nn.AdaptiveAvgPool1d(512),  # 动态调整到固定长度
            nn.Conv1d(256, 128, 1),     # 通道压缩
            nn.GELU()
        )
        
        # 高效注意力模块
        self.attention = nn.Sequential(
            LocalAttention(128, window_size=32),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        # 双向GRU
        self.gru = nn.GRU(128, 64, bidirectional=True, batch_first=True, num_layers=2)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # 输入形状: (B, L)
        x = self.embedding(x)  # (B, L, E)
        x = x.permute(0, 2, 1)  # (B, E, L)
        
        # 多尺度特征提取
        features = []
        for module in self.multi_scale:
            features.append(module(x))
        x = torch.cat(features, dim=1)  # (B, 192, L//2)
        
        # 残差卷积处理
        x = self.res_blocks(x)  # (B, 256, L//8)
        
        # 自适应压缩
        x = self.adaptive_compress(x)  # (B, 128, 512)
        x = x.permute(0, 2, 1)  # (B, 512, 128)
        
        # 局部注意力增强
        x = self.attention(x)  # (B, 512, 128)
        
        # 双向GRU
        x, _ = self.gru(x)  # (B, 512, 128)
        
        # 动态池化
        x = F.adaptive_avg_pool1d(x.permute(0, 2, 1), 1)  # (B, 128, 1)
        x = x.squeeze(-1)  # (B, 128)
        
        return self.classifier(x)

class HyperFusionCortex(nn.Module):
    def __init__(self, vocab_size=5, embedding_dim=64, num_classes=2):
        super().__init__()
        
        # 增强的嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        
        # 卷积模块
        self.conv_block = nn.Sequential(
            DepthwiseSeparableConv(embedding_dim, 64, 3),
            nn.MaxPool1d(2),
            
            DepthwiseSeparableConv(64, 128, 3),
            nn.MaxPool1d(2)
        )
        
        # 保留BiLSTM并优化配置
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            bidirectional=True,
            batch_first=True,
            dropout=0.2
        )
        
        # Transformer编码器
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=256, 
                nhead=4,
                dim_feedforward=512,
                dropout=0.1,
                activation='gelu'
            ),
            num_layers=2
        )
        
        # 动态特征融合模块
        self.feature_fusion = nn.Sequential(
            nn.Linear(256 + 256, 128),  # 双向特征拼接
            nn.LayerNorm(128),
            nn.Dropout(0.3),
            nn.GELU()
        )
        
        # 轻量级分类器
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes))
        
        self._init_weights()
        
    def _init_weights(self):
        """改进的参数初始化"""
        for name, param in self.bilstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param)
            elif 'weight_hh' in name:
                nn.init.kaiming_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 输入形状: (B, L)
        x = self.embedding(x)  # (B, L, E)
        x = self.pos_encoder(x)
        x = x.permute(0, 2, 1)  # (B, E, L)
        
        # 轻量化卷积特征提取
        conv_features = self.conv_block(x)  # (B, 128, L//4)
        conv_features = conv_features.permute(0, 2, 1)  # (B, L//4, 128)
        
        # BiLSTM处理
        bilstm_out, (h_n, c_n) = self.bilstm(conv_features)  # (B, L//4, 256)
        
        # 双向状态拼接
        lstm_features = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, 256)
        
        # Transformer处理
        transformer_out = self.transformer(bilstm_out)  # (B, L//4, 256)
        transformer_features = torch.mean(transformer_out, dim=1)  # (B, 256)
        
        # 特征融合
        combined = torch.cat([lstm_features, transformer_features], dim=1)
        fused = self.feature_fusion(combined)  # (B, 128)
        
        return self.classifier(fused.unsqueeze(-1))
class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积块"""
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, 
                                  kernel_size, padding=kernel_size//2,
                                  groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.gelu(self.bn(x))

class PositionalEncoding(nn.Module):
    """改进的位置编码"""
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ResidualBlock(nn.Module):
    """带扩张卷积的残差块"""
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, 
                             padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3,
                             padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.gelu(x)

class LocalAttention(nn.Module):
    """高效局部注意力机制"""
    def __init__(self, dim, window_size=31):
        super().__init__()
        self.qkv = nn.Linear(dim, dim*3)
        self.proj = nn.Linear(dim, dim)
        self.window_size = window_size
        self.dim = dim

    def forward(self, x):
        B, L, C = x.shape
        # 动态填充至窗口整数倍
        pad_len = (self.window_size - L % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_len, 0, 0))  # 在序列末尾填充
        
        # 分块计算
        new_L = L + pad_len
        x = x.view(B, new_L // self.window_size, self.window_size, C)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        
        attn = (q @ k.transpose(-2, -1)) / (C ** 0.5)
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).view(B, L, C)
        return self.proj(x[:, :L, :])
