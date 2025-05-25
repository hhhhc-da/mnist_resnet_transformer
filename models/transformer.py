# coding=utf-8
import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(114514)
torch.cuda.manual_seed(1919810)

# 自回归编码器
class AutoEncoderLayer(nn.Module):
    def __init__(self, max_length:int=5, emb:int=1024, heads:int=8):
        super(AutoEncoderLayer, self).__init__()
        
        self.max_length = max_length
        self.emb = emb
        self.heads = heads

        self.Wq = nn.ModuleList([nn.Linear(self.emb, self.emb) for _ in range(self.heads)])
        self.Wk = nn.ModuleList([nn.Linear(self.emb, self.emb) for _ in range(self.heads)])
        self.Wv = nn.ModuleList([nn.Linear(self.emb, self.emb) for _ in range(self.heads)])
        self.Wz = nn.Linear(self.emb * self.heads, self.emb)
        self.ln_1 = nn.LayerNorm(normalized_shape=self.emb)
        self.Wf = nn.Linear(self.emb, self.emb)
        self.ln_2 = nn.LayerNorm(normalized_shape=self.emb)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, embdding):
        # 自回归
        Z = torch.cat([_Wv(embdding) @ self.softmax(_Wk(embdding).T @ _Wq(embdding) / (self.emb**0.5)) for _Wq, _Wk, _Wv in zip(self.Wq, self.Wk, self.Wv)], dim=1)
        O = self.Wz(Z)
        LN1 = self.ln_1(O) + embdding

        FN = self.Wf(LN1)
        LN2 = self.ln_2(FN) + LN1
        return LN2
        
# 自回归解码器
class AutoDecoderLayer(nn.Module):
    def __init__(self, emb:int=1024, heads:int=8):
        super(AutoDecoderLayer, self).__init__()
        
        self.emb = emb
        self.heads = heads

        self.Wq = nn.ModuleList([nn.Linear(self.emb, self.emb) for _ in range(self.heads)])
        self.Wk = nn.ModuleList([nn.Linear(self.emb, self.emb) for _ in range(self.heads)])
        self.Wv = nn.ModuleList([nn.Linear(self.emb, self.emb) for _ in range(self.heads)])
        self.Wz = nn.Linear(self.emb*self.heads, self.emb)
        self.ln_1 = nn.LayerNorm(normalized_shape=self.emb)
        self.Wf = nn.Linear(self.emb, self.emb)
        self.ln_2 = nn.LayerNorm(normalized_shape=self.emb)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, history, embdding):
        # 自回归
        Z = torch.cat([_Wv(history) @ self.softmax(_Wk(embdding).T @ _Wq(embdding) / (self.emb**0.5)) for _Wq, _Wk, _Wv in zip(self.Wq, self.Wk, self.Wv)], dim=1)
        O = self.Wz(Z)
        LN1 = self.ln_1(O) + embdding

        FN = self.Wf(LN1)
        LN2 = self.ln_2(FN) + LN1
        return LN2

# Transformer 推理模型
class TransformerModule(nn.Module):
    def __init__(self, encoder_layer_num:int=3, decoder_layer_num:int=3, heads:int=4,
                 img_length:int=28, emb_length:int=256, layer_num=10, device:str='cuda:0'):
        super(TransformerModule, self).__init__()

        self.encoder_layer_num = encoder_layer_num
        self.decoder_layer_num = decoder_layer_num
        self.heads = heads
        self.img_length = img_length
        self.layer_num = layer_num
        self.emb = emb_length
        self.device = device
        
        self.history_transform_layer = nn.Linear(self.img_length, self.emb)

        self.emb_encode_layer = nn.Linear(self.img_length, self.emb)
        self.emb_decode_layer = nn.Linear(self.emb * self.img_length, self.layer_num)

        self.encoder = nn.ModuleList([AutoEncoderLayer(emb=self.emb, heads=self.heads) for _ in range(self.encoder_layer_num)])
        self.decoder = nn.ModuleList([AutoDecoderLayer(emb=self.emb, heads=self.heads) for _ in range(self.decoder_layer_num)])

    # 输入是 [28, 28]
    def forward(self, x):
        # 形状变换
        x = self.emb_encode_layer(x)
        # 循环放入我们的自编码器层
        for encoder in self.encoder:
            x = encoder(x)
        # 然后循环放入我们的解码器层
        history = torch.zeros((self.img_length, self.emb), device=self.device)
        for decoder in self.decoder:
            history = decoder(history, x)
        # Flatten 操作
        history = history.view(1, self.emb * self.img_length)
        return history
    
    # 输入是 [B, 28, 28]
    def parallel_forward(self, x):
        if len(x.shape) == 4:
            x = x.squeeze(1) # 应对 [B,1,28,28] 情况, 变换为 [B,28,28]
            
        # 平行计算并线性变换
        x = torch.cat([self.forward(x[i, :, :]) for i in range(x.shape[0])], dim=0)
        x = self.emb_decode_layer(x)
        return F.softmax(x, dim=1)

    @torch.no_grad
    def predict(self, x):
        return torch.argmax(self.parallel_forward(x), dim=1)
    
if __name__ == '__main__':
    model = TransformerModule(encoder_layer_num=3, decoder_layer_num=3, heads=4, img_length=28, emb_length=256, device='cuda:0').to('cuda:0')
    x = torch.randn((5, 28, 28)).to('cuda:0')
    y = model.predict(x)
    print(model, '\n最终结果为:', y)