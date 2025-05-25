#coding=utf-8
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import os
import matplotlib.pyplot as plt

from utils.config import Config, get_options
from utils.mnist_dataset import MnistTrainDataset, MnistValDataset, MnistTestDataset
from utils.earlystop import EarlyStopping

from models.resnet18 import ResnetModule
from models.transformer import TransformerModule

torch.manual_seed(114514)
torch.cuda.manual_seed(1919810)

def train(config, model, train_loader, val_loader, criterion, optimizer, earlystopping, epoch_num=10, save_freq=5):
    # 计算参数存储体
    all_labels = []
    all_predictions = []
    
    for epoch in range(epoch_num):
        model.train()
        
        # 清空迭代器
        optimizer.zero_grad()
        
        # 进度条训练
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}') as tbar:
            for i, (imgs, labels) in enumerate(tbar, start=0):
                inputs = imgs.to(config.device)
                labels = labels.to(config.device)

                # 前向传播
                outputs = None
                if config.model == 'resnet18':
                    outputs = model(inputs)
                elif config.model == 'transformer':
                    outputs = model.parallel_forward(inputs)
                
                # 反向传播
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # 记录训练信息
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                tbar.set_postfix(loss=loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            with tqdm(val_loader, desc='Valid') as tbar:
                for i, (imgs, labels) in enumerate(tbar, start=0):
                    inputs = imgs.to(config.device)
                    labels = labels.to(config.device)

                    outputs = None
                    if config.model == 'resnet18':
                        outputs = model(inputs)
                    elif config.model == 'transformer':
                        outputs = model.parallel_forward(inputs)
                        
                    loss = criterion(outputs, labels)
                    tbar.set_postfix(loss=loss.item())
                    
                    val_loss += loss.item()
                    
                    all_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())

            val_loss /= len(val_loader)
            stop, best = earlystopping(val_loss)
            
            if best:
                print("当前模型效果优于历史最优")
                torch.save(model.state_dict(), os.path.join('weights', config.best_model_name))
            if stop:
                print('触发 EarlyStop, 保存退出')
                torch.save(model.state_dict(), os.path.join('weights', config.last_model_name))
                break

            if (epoch+1) % save_freq == 0 or epoch == epoch_num - 1:
                print("最新模型保存中...")
                torch.save(model.state_dict(), os.path.join('weights', config.last_model_name))
    
    # 计算分类报告
    class_report = classification_report(all_labels, all_predictions)
    print(class_report)
    
     # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    print(conf_matrix)
    
def test(config, model, test_loader):
    # 计算参数存储体
    all_images = []
    all_predictions = []

    model.eval()
    with torch.no_grad():
        with tqdm(test_loader, desc='Test') as tbar:
            for i, imgs in enumerate(tbar, start=0):
                inputs = imgs.to(config.device)

                outputs = None
                if config.model == 'resnet18':
                    outputs = model(inputs)
                elif config.model == 'transformer':
                    outputs = model.parallel_forward(inputs)

                all_images.extend(imgs.cpu().numpy())
                all_predictions.extend(torch.argmax(outputs, dim=1).cpu().numpy())

                # 只抽取二十个
                if (i+1)*config.batch_size // 20 > 0:
                    print("已达到测试样例数限制, 停止测试")
                    break

    # 展示推片
    fig, axes = plt.subplots(4, 5, figsize=(10, 8))
    for i, ax in enumerate(axes.flatten()):
        ax.axis('off')
        ax.imshow(all_images[i].reshape(28, 28), cmap='gray')
        ax.set_title(f'Pred: {all_predictions[i]}')
    plt.tight_layout()
    plt.savefig(os.path.join('images', f'test_{config.model}.png'))
    plt.show()

if __name__ == '__main__':
    # 准备模型配置
    config = Config(options=get_options())
    config.selectable_models()
    print("当前选择的模型为:", config.model)
    
    if config.mode == 'train':
        # 准备训练集和验证集
        train_dataset = MnistTrainDataset(root=os.path.join("datasets", "mnist"))
        val_dataset = MnistValDataset(root=os.path.join("datasets", "mnist"))
        train_loader = DataLoader(train_dataset, config.batch_size, shuffle=True, num_workers=config.num_workers)
        val_loader = DataLoader(val_dataset, config.batch_size, shuffle=True, num_workers=config.num_workers)
        # 初始化模型
        model = None
        if config.model == 'resnet18':
            model = ResnetModule(config=config).to(config.device)
        elif config.model == 'transformer':
            model = TransformerModule(encoder_layer_num=config.encoder_layer_num, decoder_layer_num=config.decoder_layer_num, heads=config.heads, img_length=config.img_length, emb_length=config.emb_length, device=config.device).to(config.device)
        # 初始化其他工具
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9,0.9))
        earlystopping = EarlyStopping(mode='min', patience=config.bare_rate)
        
        # 开始训练模型
        train(config=config, model=model, train_loader=train_loader, val_loader=val_loader, criterion=criterion, optimizer=optimizer, earlystopping=earlystopping, epoch_num=config.epochs)
    
    elif config.mode == 'test':
        # 准备测试集
        test_dataset = MnistTestDataset(root=os.path.join("datasets", "mnist"))
        test_loader = DataLoader(test_dataset, config.batch_size, shuffle=True, num_workers=config.num_workers)
        # 初始化模型
        model = None
        if config.model == 'resnet18':
            model = ResnetModule(config=config).to(config.device)
        elif config.model == 'transformer':
            model = TransformerModule(encoder_layer_num=config.encoder_layer_num, decoder_layer_num=config.decoder_layer_num, heads=config.heads, img_length=config.img_length, emb_length=config.emb_length, device=config.device).to(config.device)
        
        # 装载参数
        model.load_state_dict(torch.load(os.path.join('weights', config.best_model_name), weights_only=True))
        # 开始测试模型
        test(config=config, model=model, test_loader=test_loader)