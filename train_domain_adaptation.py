import torch
import torchvision
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
 


import wandb

from MnistResnet18RevGrad import *
from data_generation_cifar_color import *

batch_size = 256#64
if __name__=="__main__":
    
    """Fashion Dataset"""  ###### comment if you want to use Digits
    # Load base datasets
    fmnist_train_pil = datasets.FashionMNIST("./data", train=True, download=True, transform=torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(p=0.5),torchvision.transforms.RandomAffine(5, (0.05,0.05))]))
    fmnist_test_pil  = datasets.FashionMNIST("./data", train=False, download=True, transform=None)
    # Labels/names come from the data file:
    CLASS_NAMES = fmnist_train_pil.classes
    print("CLASS_NAMES:", CLASS_NAMES)
    # Background textures
    cifar_train_pil = datasets.CIFAR10("./data", train=True, download=True, transform=None)
     # Source and Target datasets
    src_train = FashionMNIST_RGB(fmnist_train_pil)
    src_test  = FashionMNIST_RGB(fmnist_test_pil)
    tgt_train =FashionMNISTM_EvenBackground(fmnist_train_pil,seed=0,mask_power=1.4) #FashionMNISTM_ColorDigit(fmnist_train_pil, cifar_train_pil, seed=0, mask_power=1.4)
    tgt_test  =FashionMNISTM_EvenBackground(fmnist_test_pil,seed=0,mask_power=1.4) #FashionMNISTM_ColorDigit(fmnist_test_pil,  cifar_train_pil, seed=1, mask_power=1.4)
    print("Source train/test:", len(src_train), len(src_test))
    print("Target train/test:", len(tgt_train), len(tgt_test))

    half = batch_size // 2

    src_loader = DataLoader(src_train, batch_size=half, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    tgt_loader = DataLoader(tgt_train, batch_size=half, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)

    src_test_loader = DataLoader(src_test, batch_size=256, shuffle=False, num_workers=2)
    tgt_test_loader = DataLoader(tgt_test, batch_size=256, shuffle=False, num_workers=2)

    lambda_hp = 0.5

    # lambda_scheduler = None
    gamma = 10
    lambda_scheduler = lambda p: 2 / (1 + np.exp(-gamma * p)) - 1
    model=SmallStemResNet18(
        len(CLASS_NAMES),
        in_channels=3,
        lambda_hp=lambda_hp, 
        lambda_scheduler=lambda_scheduler
        )

    wandb.login()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model.to(device)
    
    config = {'epochs': 100,'lr': 3e-2, "weight":0.0,'batch':batch_size, 'lambda':lambda_hp}#, 'momentum': 0.8
    
    iter = 0
    with wandb.init(config = config,project="DomainAdaptation", id="grad-reversal-evenbackground") as run:


        wandb.define_metric("train/iter_loss", step_metric="global_step")
        wandb.define_metric("train/iter_accuracy", step_metric="global_step")
        wandb.define_metric("val/epoch_loss", step_metric="epoch")
        wandb.define_metric("val/epoch_accuracy", step_metric="epoch")
        wandb.define_metric("val/target_accuracy",step_metric="epoch")

        optimizer = torch.optim.AdamW(model.parameters(), lr=run.config['lr'],weight_decay=run.config['weight'])#, weight_decay=0.0, momentum=run.config['momentum'])
        scheduler = lr_scheduler.StepLR(optimizer, step_size=run.config['epochs']/4, gamma=0.5)
        criterion = torch.nn.CrossEntropyLoss()
        domain_criterion=torch.nn.BCEWithLogitsLoss()
        for i in range(run.config['epochs']):
            model.train()
            print("Epoch {}".format(i))
            for (x,y),(z,_) in zip(src_loader,tgt_loader):
                iter+=1



                x = x.to(device)
                y = y.to(device)
                z= z.to(device)
                if iter == 0:
                    image = wandb.Image(x[0,:,:,:])
                    run.log({"example": image})

                out,domains1 = model(x)
                _, domains2=model(z)
                loss_label = criterion(out,y)
                domain_loss= domain_criterion(torch.cat([domains1,domains2]).squeeze(), torch.cat([torch.ones(x.shape[0]), torch.zeros(z.shape[0])]).to(device))
                loss=loss_label+domain_loss
                model.zero_grad()
                loss.backward()

                optimizer.step()
                # model.clip_weights(0.5)

                _, predicted = torch.max(out.data, 1)
                correct = (predicted == y).float().mean().item()

                run.log({"train/train_loss": loss_label.item(), "train/train_accuracy": correct}, step = iter)

            model.step_scheduler()

            model.eval()

            running_loss = 0
            running_acc = 0

            with torch.no_grad():
                for j,input in enumerate(src_test_loader,0):
                    x = input[0].to(device)
                    y = input[1].to(device)

                    out,_ = model(x)
                    loss = criterion(out,y)

                    _, predicted = torch.max(out.data, 1)
                    correct = (predicted == y).sum().item()

                    running_loss += loss.item()
                    running_acc += correct

                run.log({"epoch": i, "val/epoch_loss": running_loss / len(src_test), "val/epoch_accuracy": running_acc / len(src_test)}, step = iter + 1)
                running_acc = 0
                for j,input in enumerate(tgt_test_loader,0):
                    x = input[0].to(device)
                    y = input[1].to(device)

                    out,_ = model(x)

                    _, predicted = torch.max(out.data, 1)
                    correct = (predicted == y).sum().item()

                    running_acc += correct

            run.log({"epoch": i, "val/target_accuracy": running_acc / len(tgt_test)}, step = iter + 1)
           
    wandb.finish()
