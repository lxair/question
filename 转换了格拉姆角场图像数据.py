from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import copy
import math
from ...utils import get_or_create_path
from ...log import get_module_logger

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.dates as mdates
import torch.nn.functional as F
from torch import Tensor
from absl import app, flags
from easydict import EasyDict
import torchvision

# from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhansmaster.cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhansmaster.cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

from torch.utils.tensorboard import SummaryWriter
import logging
import operator
import os
from copy import deepcopy

from imageio import imsave


from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary


from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from qlib.contrib.model.pytorch_transformer_ts import Transformer
from qlib.contrib.model.pytorch_transformer_ts import TransformerModel
import numpy as np
from torch.optim.lr_scheduler import StepLR
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class TransGANModel(Model):
    def __init__(
        self,
        d_feat: int = 4,
        d_model: int = 4,
        nhead: int = 2,
        rank: int = -1,
        n_critic : int = 1,
        num_layers: int = 2,
        dropout: float = 0,
        activation: str = "gelu",
        batch_size: int = 32,
        early_stop=400,
        g_accumulated_times = 2,
        iter_idx = 0,
        world_size = 1,
        ema = 0.995,  
        latent_dim = 158,
        n_epochs: int = 20,
        load_path : str= None ,
        learning_rate: float = 0.002,
        g_lr : float =0.004,
        d_lr: float =0.002,
        weight_decay: float = 1e-3,
        evaluation_epoch_num: int = 10,
        n_jobs: int = 10,
        ema_kimg:int = 100,
        ema_warmup = 0,
        beta1=0,
        beta2 = 0.9,
        dis_batch_size = 64,
        max_iter =  500,
        global_steps : int = 0,
        accumulated_times: int=1,
        hidden_size:int = 5,
        loss="mse",
        metric="",
        optimizer_betas :float= (0.9,0.999),
        optimizer="adam",
        seed=999,
        GPU=3,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.nhead = nhead
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.n_epochs = n_epochs
        self.ema_warmup =ema_warmup
        self.global_steps =global_steps
        self.n_critic =n_critic
        self.rank = rank
        self.latent_dim = latent_dim
        self.world_size = world_size
        self.metric = metric
        self.d_feat = d_feat
        self.beta1 = beta1
        self.beta2 = beta2
        self.ema =ema
        self.num_layers = num_layers
        self.optimizer_betas = optimizer_betas
        self.dropout = dropout
        self.ema_kimg = ema_kimg
        self.iter_idx  = iter_idx  
        self.load_path = load_path
        self.g_accumulated_times =g_accumulated_times
        self.loss = loss
        self.accumulated_times = accumulated_times
        self.activation = activation
        self.batch_size = batch_size
        self.dis_batch_size = dis_batch_size
        self.early_stop = early_stop
        self.seed = seed
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.evaluation_epoch_num = evaluation_epoch_num
        self.n_jobs = n_jobs
        self.optimizer = optimizer.lower()
        self.device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.logger = get_module_logger("TransGANModel")
        self.logger.info("Naive TransGAN:" "\nbatch_size : {}" "\ndevice : {}".format(self.batch_size, self.device))

        # Create model
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)

        # gen_optimizer = optim.Adam(generator.parameters(), lr=self.lr, betas=self.optimizer_betas) # 优化器，adam优化generator的参数，学习率为lr,动量参数beats
        # dis_optimizer = optim.Adam(discriminator.parameters(), lr=self.lr, betas=self.optimizer_betas)
        
        # Optimizer
        if optimizer.lower() == "adam":
            self.gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.generator.parameters()),
                                        self.g_lr, (self.beta1, self.beta2))
            self.dis_optimizer =torch.optim.Adam(filter(lambda p: p.requires_grad, self.discriminator.parameters()),
                                        self.d_lr, (self.beta1, self.beta2))

        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))
        
        # 定义生成器和判别器的学习率调度器
        gen_scheduler = LinearLrDecay(self.gen_optimizer, self.g_lr, 0.0, 0, self.max_iter * self.n_critic) ## 定义了一个学习率衰减器对象，自动调整生成器先训练。
        dis_scheduler = LinearLrDecay(self.dis_optimizer, self.d_lr, 0.0, 0, self.max_iter * self.n_critic)
        self.schedulers = (gen_scheduler, dis_scheduler)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred.float() - label.float()) ** 2 
        return torch.mean(loss)  

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])

        raise ValueError("unknown loss `%s`" % self.loss)
    
    def metric_fn(self, pred, label):

        mask = torch.isfinite(label)

        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])

        raise ValueError("unknown metric `%s`" % self.metric)
    

    def train_epoch_gan(self, data_loader,epoch,gen_avg_param,writer):
        # train mode
        print("Generator and discriminator are initialized")

        self.generator.train()
        self.discriminator.train()
        self.dis_optimizer.zero_grad()
        self.gen_optimizer.zero_grad()
        generator_losses = []
        discriminator_losses = []
        criterion = nn.BCELoss() # 二分类任务的损失函数 ，输出层激活函数为sigmoid，输出代表正例的概率，与真实标签比较。
        real_data_list = []
        predicted_data_list = []
        epsilon = 0.02
        real_label = 0.9
        fake_label = 0.
        col_set=["KMID","KLOW","OPEN0",]
        
        for iter_idx,sequence_batch in enumerate(tqdm(data_loader)):
            
            real_sequence = sequence_batch[:, :, 0:-1].to(self.device) # 取feature时间特征。 [64,30,158]
            
            # 转换数据为格拉姆图像
            GAF_data = []
            for i in range(len(real_sequence)):
                series = real_sequence[i].cpu().numpy()
                GAF = self.compute_gramian_angle_field(series)
                GAF_data.append(GAF)

            # GAF_data = np.array(GAF_data)

            # 将格拉姆图像转换为张量
            real_sequence = torch.tensor(GAF_data, dtype=torch.float32)
            print("将时间序列的股票数据转换为格拉姆图像完成")
            
            batch_size = real_sequence.size(0) # 读取batch_size,后面比较的时候需要用到生成了batch_size的数据。
            real_labels = torch.full((batch_size,), real_label, dtype=torch.float, device=self.device) #111111.....
            fake_labels = torch.full((batch_size,), fake_label, dtype=torch.float, device=self.device)
            
            real_sequence = real_sequence.reshape(real_sequence.shape[0],1,real_sequence.shape[2],real_sequence.shape[1],) #把[64,30,158]变为[64,158,1,30]
            real_data_list.append(real_sequence)
            
            # 真实的数据放到判别器
            discriminator_output_real = self.discriminator(real_sequence)# 将真实的值传到鉴别器中，先学习真实的数据。discriminator_output_real.shape(64,1)
            # discriminator_output_real = torch.sigmoid(discriminator_output_real)# 这里的值好像不在0-1之间，用sigmod函数，然后与真实的值进行比较。

            z = torch.cuda.FloatTensor(np.random.normal(0, 1,(real_sequence.shape[0],158,30)))
            generator_input_sequence = z  # generator_input_sequence.shape(256,158,30)
            _ = self.generator(generator_input_sequence)  #generator_input_sequence 是一个三维的数据
            generator_outputs_sequence = self.generator.saved_output #generator_outputs_sequence.shape(512,158,1,30)
            
        # generator_result_concat.shape(64,158,1,30*2) 正常情况下拼接是时间步拼接,前半部分是真实,后面的生成.这里时间步是第3维.
            # generator_result_concat = torch.cat((generator_input_sequence, generator_outputs_sequence.detach()), 3) #generator_outputs_sequence.shape(64,158,1,30)
            discriminator_output_fake = self.discriminator(generator_outputs_sequence) # discriminator_output_fake.shape(512,2)
            # discriminator_output_fake = torch.sigmoid(discriminator_output_fake)
            
            ##-----------------------------------------------------------------------------------这是'hinge'的损失函数
            # discriminator_error_real = criterion(discriminator_output_real, real_labels)
            # discriminator_error_fake = criterion(discriminator_output_fake, fake_labels)
            # discriminator_error = discriminator_error_real + discriminator_error_fake
            # 计算(1.0 - real_validity)的每个元素的值，将小于等于0的值变为0。这表示真实样本被判别为假的程度。
            #计算(1 + fake_validity)的每个元素的值，将小于等于0的值变为0。这表示生成的样本被判别为真的程度。
            discriminator_error = torch.mean(nn.ReLU(inplace=True)(1.0 - discriminator_output_real)) + \
                    torch.mean(nn.ReLU(inplace=True)(1 + discriminator_output_fake)) # \ 表示连接符。 1.0 - discriminator_output_real趋向1.真实样本。1 + discriminator_output_fake虚假的样本0
            
            discriminator_error = discriminator_error/float(self.accumulated_times)  # args.accumulated_times = 1
            discriminator_error.backward()
            
            torch.nn.utils.clip_grad_value_(self.discriminator.parameters(), 3.0)
            self.dis_optimizer.step()
            discriminator_losses.append(discriminator_error)
            self.dis_optimizer.zero_grad()
            
            writer.add_scalar("Discriminator Loss", discriminator_error, self.global_steps)
    
            ##---------------------------------------------------------
            
            # if self.global_steps % (self.n_critic * self.accumulated_times) == 0:  # global_steps运行初始为0
            # if self.global_steps % (self.accumulated_times +1) == 0:
            for accumulated_idx in range(self.g_accumulated_times):
            # z用于保持生成的虚假图像与真实图像一致，而gen_z用于生成多样化的虚假图像样本。(真实数据的batch的，生成的虚假的图片第一维有的不一定相同？)
                gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1,(64,158,30))) # 注意gen_z和gen的区别。
                gen_z.requires_grad = True
                _ = self.generator(gen_z)  # gen_imgs(64,3,1,150)
                generated_data = self.generator.saved_output  #generated_data(512,158,1,30)
                
                predicted_data_list.append(generated_data)

                generated_data = projected_gradient_descent(self.discriminator, generated_data, 0.05, 0.02, 40, np.inf)
                
                discriminator_output = self.discriminator(generated_data) # 作为输入到鉴别器
                # discriminator_output = torch.sigmoid(discriminator_output)
                
                #----------------------------------------------------------当判别器对生成样本的输出结果越小，说明生成器生成的样本越逼真
                # generator_error = criterion(discriminator_output, real_labels)
                #对于生成器而言，discriminator_output趋向于1更好，因为这意味着生成器生成的样本越逼真，能够欺骗判别器，使其将生成样本误判为真实样本。
                generator_error = -torch.mean(discriminator_output)
                # generator_error = generator_error/float(self.g_accumulated_times)
                generator_error.backward()
                torch.nn.utils.clip_grad_value_(self.generator.parameters(), 3.0)
                self.gen_optimizer.step()
                generator_losses.append(generator_error)
                self.gen_optimizer.zero_grad()
                writer.add_scalar("Generator Loss", generator_error, self.global_steps) 
                writer.close()
                
            # 记录或保存学习率的变化
            if self.schedulers:
                gen_scheduler, dis_scheduler = self.schedulers
                g_lr = gen_scheduler.step(self.global_steps)
                d_lr = dis_scheduler.step(self.global_steps)
                writer.add_scalar("Generator Learning Rate", g_lr, self.global_steps)
                writer.add_scalar("Discriminator Learning Rate", d_lr, self.global_steps)
                writer.close()
            
            self.global_steps = +1 

        # # plot可视化真实的数据，选取的一两个feature.
        dis_loss = torch.tensor(discriminator_losses).mean().item()
        gen_loss = torch.tensor(generator_losses).mean().item()
        
        real_data = torch.cat(real_data_list, dim=0)  # 将真实数据列表合并为一个张量 real_data.shape(1536,158,1,30)

        real_data = real_data.detach().cpu().numpy() # 将张量的值转换为NumPy数组
        real_data = real_data.squeeze()
        real_data = real_data.transpose(0,2,1)
        real_data = real_data[:,-1,:]
        real_data_copy =np.copy(real_data)

        real_data = np.concatenate((real_data,real_data_copy), axis = 0)
        
        df_real = pd.DataFrame(real_data.reshape(-1,158))  # real_data的数据类型是array
        

        # predicted_data = torch.cat(predicted_data_list, dim=0)  # 将生成的数据列表合并为一个张量 predicted_data.shape(1536,158,1,30):
        predicted_data = torch.cat(predicted_data_list, dim=0)
        predicted_data = predicted_data.detach().cpu().numpy() # 将张量的值转换为NumPy数组
        predicted_data = predicted_data.squeeze()
        predicted_data = predicted_data.transpose(0,2,1)
        predicted_data = predicted_data[:,-1,:]
        
        df_pred = pd.DataFrame(predicted_data.reshape(-1,158))
        
        # TODO: get x values and plot prediction of multiple columns
        fig = plt.figure(figsize=(16,8))
        plt.xlabel("stock_number")
        plt.ylabel("KLOW")
        plt.title("GAN_train")
        plt.plot( df_real[1], label="Real")
        plt.plot(df_pred[1],label="Predicted")
        # plt.ylim(bottom=0)
        plt.legend()
        fig.savefig('./KLOW/plt_epoch_{}.png'.format(epoch))
        plt.close(fig)

        # TODO: get x values and plot prediction of multiple columns
        fig2 = plt.figure(figsize=(16,8))
        plt.xlabel("stock_number")
        plt.ylabel("KOPEN")
        plt.title("GAN_train")
        plt.plot( df_real[2], label="Real")
        plt.plot(df_pred[2],label="Predicted")
        # plt.ylim(bottom=0)
        plt.legend()
        fig2.savefig('./kopen/plt_epoch_{}.png'.format(epoch))
        plt.close(fig)
        
        
        # # 计算均方误差（MSE）
        # mse = torch.tensor((real_data - predicted_data)**2).mean().item()
        # print('\n[{}/{}]\tDiscriminator Loss: {:.4f}\tGenerator Loss: {:.4f}   \tRMSE:{:.4f}'
        #         .format(epoch+1, self.n_epochs, dis_loss, gen_loss, mse))
        
        return dis_loss, gen_loss

    def test_epoch_train_data(self, data_loader):

        self.generator.eval()

        scores = []
        losses = []

        for data in data_loader:

            feature = data[:, :, 0:-1].to(self.device).to(self.device) #feature.shape (64,30,3)
            
            label = data[:, -1, -1].to(self.device) #label.shape(64)

            feature = feature.to("cuda:3")
            label = label.to("cuda:3") 

            with torch.no_grad():
                feature = feature.permute(0,2,1)
                _ = self.generator(feature.float())
                pred = self.generator.pre_out # .float(),这里跳进去的函数是”mse“那段/ 这里用transgan的是(64,3,1,30)
                pred = pred.squeeze()
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())

                score = self.metric_fn(pred, label)
                scores.append(score.item())
            return np.mean(losses), np.mean(scores)
    
    def test_epoch_valid_data(self, data_loader,):
        
            # global generator
            self.discriminator.eval()
            self.generator.eval()

            scores = []
            losses = []

            report = EasyDict(nb_test=0, correct=0, correct_fgm=0, correct_pgd=0)
            
            for iter_idx,data in enumerate(tqdm(data_loader)):
            # for data in data_loader:

                feature = data[:, :, 0:-1].to(self.device).to(self.device) #feature.shape (64,30,158)
                feature = feature.reshape(feature.shape[0],feature.shape[1], 1 ,feature.shape[2],) #把[64,30,158]变为[64,30,1,158]
                feature = feature.permute(0,3,2,1)
                label = data[:, -1, -1].to(self.device) #label.shape(64)

                feature = feature.to("cuda:3")
                label = label.to("cuda:3") 
                y = label
                for i in range(y.size(0)):   
                    if(y[i]>0):
                        y[i] = 1      #涨，记作1.
                    else:
                        y[i] = 0  
                _, y_pred = self.discriminator(feature).max(1)  # model prediction on clean examples  y_pred.shape=(128)  y_pred_fgm.shape(128)

                x_pgd = projected_gradient_descent(self.discriminator, feature, 0.3, 0.01, 40, np.inf)
                # model prediction on FGM adversarial examples
                # model prediction on PGD adversarial examples

                _, y_pred_pgd = self.discriminator(x_pgd).max(1)
                report.nb_test += y.size(0)  # report.nb_test = 64
                report.correct += y_pred.eq(y).sum().item()

                report.correct_pgd += y_pred_pgd.eq(y).sum().item()  # eq是比较。通过 y_pred.eq(y) 来比较预测标签和真实标签是否相等，得到的是一个布尔值的 Tensor，其中值为 1 的位置表示预测正确，值为 0 的位置表示预测错误。

                with torch.no_grad():
                    _ = self.generator(feature.float())
                    pred = self.generator.pre_out # .float(),这里跳进去的函数是”mse“那段/ 这里用transgan的是(64,3,1,30)
                    pred = pred.squeeze()
                    loss = self.loss_fn(pred, label)
                    losses.append(loss.item())

                    score = self.metric_fn(pred, label)
                    scores.append(score.item())
            print("test acc on clean examples (%): {:.3f}".format(report.correct / report.nb_test * 100.0))

            print("test acc on PGD adversarial examples (%): {:.3f}".format(report.correct_pgd / report.nb_test * 100.0)) 
            return np.mean(losses), np.mean(scores)
    
    def predict(self, dataset):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        
        dl_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        dl_test.config(fillna_type="ffill+bfill")
        test_loader = DataLoader(dl_test, batch_size=self.batch_size, num_workers=self.n_jobs)
        
        # global generator
        self.generator.eval()
        
        preds = []

        for data in test_loader:

            feature = data[:, :, 0:-1].to(self.device)

            with torch.no_grad():
                feature = feature.permute(0,2,1)
                _ = self.generator(feature.float())
                pred = self.generator.pre_out
                pred = pred.squeeze()
                pred= pred.detach().cpu().numpy()

            preds.append(pred)
        return pd.Series(np.concatenate(preds), index=dl_test.get_index())
    
    # 定义格拉姆角场转换函数
    def compute_gramian_angle_field(self,series):
        normalized_series = (series - np.min(series)) / (np.max(series) - np.min(series))
        x = np.outer(normalized_series, normalized_series)
        gaf = np.arccos(x) / np.pi
        return gaf

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        save_path=None,
    ):
        
        dl_train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        dl_valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        
        if dl_train.empty or dl_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        dl_train.config(fillna_type="ffill+bfill")  # process nan brought by dataloader
        dl_valid.config(fillna_type="ffill+bfill")  # process nan brought by dataloader

        # Dataloader是pytorch的数据处理，并用batch_sampler将指定batch分配给指定worker，worker将它负责的batch加载进RAM。
        # num_worker设置得大，好处是寻batch速度快，因为下一轮迭代的batch很可能在上一轮/上上一轮...迭代时已经加载好了。坏处是内存开销大，也加重了CPU负担

        train_loader = DataLoader(
            dl_train, batch_size=self.batch_size, shuffle=True, num_workers=self.n_jobs, drop_last=True
        )
        valid_loader = DataLoader(
            dl_valid, batch_size=self.batch_size,shuffle=False, num_workers=self.n_jobs, drop_last=True
        )

        if self.max_iter:
            self.max_epoch = np.ceil(self.max_iter * self.n_critic / (len(train_loader)/self.batch_size))  # 将最大迭代次数转换为最大训练轮数。
        
        save_path = get_or_create_path(save_path)
        evaluation_metrics = {"gen_loss":[], "disc_loss":[], "rmse":[]}
        gen_avg_param = copy_params(self.generator, mode='gpu')
        stop_steps = 0
        train_loss = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []
        # 定义空列表
        dis_losses = []
        gen_losses = []
        writer = SummaryWriter(log_dir="logs")

        # train
        self.logger.info("training...")
        self.fitted = True

        params_before_train = copy.deepcopy(self.generator.state_dict())
        ##  训练GAN网络的transformer模型
        for epoch in range(self.n_epochs):
            self.logger.info("Epoch%d:", epoch)
            self.logger.info("Training...")
            train_result = self.train_epoch_gan(train_loader,epoch,gen_avg_param,writer)
            dis_losses.append(train_result[0])
            gen_losses.append(train_result[1])

            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch_train_data(train_loader)
            val_loss, val_score = self.test_epoch_valid_data(valid_loader)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = epoch
                best_param = copy.deepcopy(self.generator.state_dict())
                best_param1 = copy.deepcopy(self.discriminator.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

            self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
            self.generator.load_state_dict(best_param)
            self.discriminator.load_state_dict(best_param1)
            torch.save(best_param,save_path)
        
        # 绘制损失曲线图
        fig = plt.figure(figsize=(16,8))
        plt.plot(range(self.n_epochs), dis_losses, label='Discriminator Loss')
        plt.plot(range(self.n_epochs), gen_losses, label='Generator Loss')
        # 设置x轴刻度
        plt.xticks(range(0, self.n_epochs, 1))  # 设置每隔5个epoch显示一个刻度
        # 设置y轴刻度
        plt.yticks(np.arange(0, max(max(dis_losses), max(gen_losses)), 0.1))  # 根据损失值的范围设置刻度

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('GAN Training Loss')
        # fig.savefig('./plots/plt_epoch_{}.png'.format(epoch))
        fig.savefig('./epoch_loss/plt_epoch_{}.png'.format(epoch))
        plt.close(fig)

        if self.use_gpu:
            torch.cuda.empty_cache()

class Generator(nn.Module):
    def __init__(self, seq_len=30, patch_size=15, channels=10, num_classes=9, embed_dim=158, depth=3,
                 num_heads=5, forward_drop_rate=0.5, attn_drop_rate=0.5):
        super(Generator, self).__init__()
        self.channels = channels
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.depth = depth
        self.attn_drop_rate = attn_drop_rate
        self.forward_drop_rate = forward_drop_rate
        
        # self.l1 = nn.Linear(self.latent_dim, self.seq_len * self.embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.embed_dim)) #位置编码张量，用于引入序列中每个位置的位置信息
        self.blocks = Gen_TransformerEncoder( # 生成器的 Transformer 编码器部分。
                         depth=self.depth, #编码器层数
                         emb_size = self.embed_dim, #嵌入维度
                         drop_p = self.attn_drop_rate, #注意力丢弃率
                         forward_drop_p=self.forward_drop_rate #前向传播丢弃率
                        )

        self.deconv = nn.Sequential( # 反卷积层，将嵌入维度映射回原始的通道数.
            nn.Conv2d(self.embed_dim, self.channels, 1, 1, 0)
        )

        self.decoder_layer = nn.Linear(158,1) #用于预测的线性层，将嵌入维度映射为一个维度，用于生成时间序列数据的预测。
        self.linear = nn.Linear(158,2)  # 用于分类的线性层，将嵌入维度映射为二维，用于辅助任务（如 FGM）的格式。

    def forward(self, z):
        # x = self.l1(z).view(-1, self.seq_len, self.embed_dim)
        z = z.to(dtype=torch.float32,device="cuda:3")  #(64,158,1,30)
        z = z.squeeze()
        z = z.permute(0,2,1)
        x = (z + self.pos_embed).to(dtype=torch.float32,device="cuda:3") #x.shape(64,30,158)
        H, W = 1, self.seq_len
        x = self.blocks(x)
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        output = self.deconv(x.permute(0, 3, 1, 2))
        output = output.view(-1, self.channels, H, W)  #output.shape为[64,10,1,30]
        self.saved_output = output
        
        out = output.squeeze()
        out = out.permute(0,2,1)
        
        self.pre_out = self.decoder_layer(out[:, -1, :]) # 预测的值,是一个一维的.
        fenlei_out = self.linear(out[:, -1, :])  # 二维的值,用作fgm的格式.
        
        return fenlei_out
    
    
class Gen_TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=2,#多头注意力里面的头数
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

        
class Gen_TransformerEncoder(nn.Sequential):
    def __init__(self, depth=8, **kwargs):
        super().__init__(*[Gen_TransformerEncoderBlock(**kwargs) for _ in range(depth)])       
        
        
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor: #qkv经过线形层之后，用rearrange进行重排，以适应多头注意的计算。
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        #张量乘法 torch.einsum 计算注意力权重 att。注意力权重的计算过程包括计算查询与键的点积得到能量张量 energy，并进行缩放和 softmax 操作得到注意力权重。如果提供了掩码 mask，则将其应用于能量张量。
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

    
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
    
    
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

        
        
class Dis_TransformerEncoderBlock(nn.Sequential):#nn.Sequential意味着是按顺序执行的。
    def __init__(self,
                 emb_size=100,
                 num_heads=5,
                 drop_p=0.,
                 forward_expansion=4,
                 forward_drop_p=0.):
        super().__init__( #两个子模块都通过残差连接（ResidualAdd）的方式与输入进行相加，保留了输入的原始信息，并通过层归一化和丢弃层进行规范化和正则化操作。
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),#这个子模块用于对输入进行自注意力计算，并学习输入中的上下文关系。
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(  #这个子模块用于对经过自注意力计算的结果进行非线性变换和特征提取。
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class Dis_TransformerEncoder(nn.Sequential):
    def __init__(self, depth=8, **kwargs):
        super().__init__(*[Dis_TransformerEncoderBlock(**kwargs) for _ in range(depth)])
        
        
class ClassificationHead(nn.Sequential):  #用于将模型的输出特征进行分类，实现了一个简单的分类头，用于将输入特征映射为分类结果。
    def __init__(self, emb_size=100, n_classes=2):
        super().__init__()
        self.clshead = nn.Sequential( #首先对特征进行平均池化，然后通过层归一化和线性映射得到最终的分类输出。
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        out = self.clshead(x)
        return out

    
class PatchEmbedding_Linear(nn.Module):
    #what are the proper parameters set here?
    def __init__(self, in_channels = 474, patch_size = 474, emb_size = 15, seq_length = 30):
        # self.patch_size = patch_size
        super().__init__()
        #change the conv2d parameters here 将模型嵌入到指定的向量空间，可以帮助模型学习到时间序列数据中的局部关系和特征。
        self.projection = nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)',s1 = 474, s2 = patch_size), # 这里(h h1) (w w1)就相当于h与w变为原来的1/h1,1/w1倍
            nn.Linear(patch_size*in_channels, emb_size,dtype=torch.float32)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))#引入一个特殊的标记，用于表示整个序列的分类信息。在序列分类任务中，这个标记可以帮助模型区分序列的整体性质和类别。
        # self.positions = nn.Parameter(torch.randn(3, emb_size))
        self.positions = nn.Parameter(torch.randn(1, 1, emb_size))  # 引入位置信息。

    def forward(self, x: Tensor) -> Tensor:  # (64,158,1,30)  格拉姆是（64，1，4740，4740）
        # x = x.reshape(x.shape[0],x.shape[2], 1 ,x.shape[1], ) # x变成四维(64,158,1,30)  原论文的作者是（64，3，1，150）
        b, _, _, _ = x.shape
        x = x.to(dtype=torch.float32,device="cuda:3")
        x = self.projection(x)  # 经过了projection之后，x变为(64,2,15)  格拉姆是（32,100,15）
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)  #cls+tokens(64,1,15)
        #prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)  # x又变成了(512,2,15)   格拉姆是（32,101,15）
        # position
        x += self.positions.expand_as(x)
        return x         #x.shape(512,3,15)  格拉姆是（64，47401，15）
        
        
class Discriminator(nn.Sequential):
    def __init__(self, 
                 in_channels=474,
                 patch_size=474,
                 emb_size=15, 
                 seq_length = 30,
                 depth=3, 
                 n_classes=2, 
                 **kwargs):
        super().__init__(
            PatchEmbedding_Linear(in_channels, patch_size, emb_size, seq_length),
            Dis_TransformerEncoder(depth, emb_size=emb_size, drop_p=0.5, forward_drop_p=0.5, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )
    
def copy_params(model, mode='cpu'):
    if mode == 'gpu':
        flatten = []
        for p in model.parameters():
            cpu_p = deepcopy(p).cpu()
            flatten.append(cpu_p.data)
    else:
        flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


class LinearLrDecay(object):  # 线性学习率衰减器
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step) #学习率的衰减量(delta)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:# 当前步数小于等于衰减起始步数，则学习率保持为初始学习率。
            lr = self.start_lr
        elif current_step >= self.decay_end_step: #当前步数大于等于衰减结束步数decay_end_step，则学习率为结束学习率。
            lr = self.end_lr
        else: # 学习率按线性衰减公式计算，即学习率等于初始学习率减去衰减量乘以当前步数与衰减起始步数的差值。
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups: #学习率需要更新的情况下，通过遍历优化器的参数组，
                param_group['lr'] = lr
        return lr