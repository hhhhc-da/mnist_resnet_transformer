#coding=utf-8
import torch
import argparse
import os

class Config():
    def __init__(self, options=None):
        self.models = ['resnet18', 'transformer']
        
        # Mode
        self.mode = options.mode
        
        # Data
        self.data_dir = options.data_dir
        self.batch_size = options.batch_size
        self.num_workers = options.num_workers

        # Model
        self.model = options.model
        self.model_name = options.model_name
        if self.model not in self.models:
            raise ValueError(f"Model {self.model} not in {self.models}.")

        self.encoder_layer_num = options.encoder_layer_num
        self.decoder_layer_num = options.decoder_layer_num
        self.heads = options.heads
        self.img_length = options.img_length
        self.emb_length = options.emb_length        
        self.num_classes = options.num_classes
        self.last_model_name = os.path.join(options.model, self.model_name) if str(options.model_name).endswith('.pth') else os.path.join(options.model, self.model_name + ".pth")
        self.best_model_name = os.path.join(options.model, 'best_' + self.model_name) if str(options.model_name).endswith('.pth') else os.path.join(options.model, 'best_' + self.model_name + ".pth")

        # Training
        self.epochs = options.epochs
        self.learning_rate = options.learning_rate
        self.bare_rate = options.bare_rate
        self.save_freq = options.save_freq

        # Device
        self.device = options.device
        
    def selectable_models(self):
        return '可以选择的模型有: ' + ' / '.join(self.models)

    def __str__(self):
        return str(self.__dict__)
    
def get_options():
    parser = argparse.ArgumentParser(description="训练需要的模型超参数, 可以使用 sklearn 的 GridSearchCV 进行超参数调优")
    parser.add_argument('--mode', type=str, default='train', help='训练模式, 可选 train, test')
    parser.add_argument('--model', type=str, default='resnet18', help='需要使用的模型, 模型可选 resnet18, transformer')
    parser.add_argument('--data_dir', type=str, default=os.path.join('datasets', 'mnist'), help='训练集路径, 需要满足条件')
    parser.add_argument('--batch_size', type=int, default=128, help='批处理大小')
    parser.add_argument('--bare_rate', type=int, default=3, help='提前停止容忍次数')
    parser.add_argument('--num_workers', type=int, default=4, help='main 函数内 Dataloader 协作线程个数')
    parser.add_argument('--model_name', type=str, default="model", help='模型名称')
    parser.add_argument('--encoder_layer_num', type=int, default=2, help='Transformer 编码器层数')
    parser.add_argument('--decoder_layer_num', type=int, default=2, help='Transformer 解码器层数')
    parser.add_argument('--heads', type=int, default=4, help='Transformer 注意力头数')
    parser.add_argument('--img_length', type=int, default=28, help='Transformer 输入图片长度')
    parser.add_argument('--emb_length', type=int, default=32, help='Transformer 嵌入长度')
    parser.add_argument('--num_classes', type=int, default=10, help='最终分类个数')
    parser.add_argument('--epochs', type=int, default=500, help='训练次数, 往大选就可以了')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--save_freq', type=int, default=1, help='每隔几次保存一次模型')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='使用的设备')
    args = parser.parse_args()
    
    config = Config(options=args)
    return config

if __name__ == '__main__':
    options = get_options()
    print(options.selectable_models())