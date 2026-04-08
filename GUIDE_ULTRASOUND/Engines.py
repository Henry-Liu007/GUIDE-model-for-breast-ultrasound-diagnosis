import datetime
import time
import torch
import torchvision
from sklearn.metrics import confusion_matrix, accuracy_score
from torch import nn as nn
from torch.nn import functional as F

def KL(alpha, c, device):
    beta = torch.ones((1, c)).to(device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def ce_loss(p, alpha, c, global_step, annealing_step, device):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c, device)
    return (A + B)

class Visualize_train(nn.Module):
    def __init__(self):
        super().__init__()

    def save_image(self, image, tag, epoch, writer):
        if tag == 'img':
            image = (image - image.min()) / (image.max() - image.min() + 1e-6)
            grid = torchvision.utils.make_grid(image, nrow=1, pad_value=1)
        else:
            image = (image - image.min()) / (image.max() - image.min() + 1e-6)
            grid = torchvision.utils.make_grid(image, nrow=1, pad_value=1)
        writer.add_image(tag, grid, epoch)

    def forward(self, img_list, gaze_list, gaze_pred_list,
                epoch, writer):
        self.save_image(img_list.float(), 'img', epoch, writer)
        self.save_image(gaze_list.float(), 'gaze', epoch, writer)
        self.save_image(gaze_pred_list.float(), 'gaze_pred', epoch, writer)

def train_one_epoch(model, dataloader_train, optimizer, device, epoch, args, writer):
    model.train()
    print_freq = 50
    total_steps = len(dataloader_train)
    start_time = time.time()

    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    img_list, gaze_list, gaze_pred_list = [], [], []
    step = 0
    pred_list = []
    label_list = []
    for img, gaze, label, _ in dataloader_train:
        start = time.time()

        img = img.to(device)
        gaze = gaze.to(device)
        label = label.to(device)

        cls_pred = model(img)
        gaze_pred = model.get_gaze()

        evidences = F.softplus(cls_pred)
        alpha = evidences + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        E = alpha - 1
        b = E / (S.expand(E.shape))
        Tem_Coef = epoch*(0.99/args.epochs)+0.01
        loss_un = 0
        loss_un += ce_loss(label, alpha, args.num_classes, epoch, args.epochs, device)
        loss_ACE = torch.mean(loss_un)
        loss_cls = criterion_cls(b / Tem_Coef, label)
        loss_reg = criterion_reg(gaze_pred, gaze)
        loss =  loss_cls +  loss_reg + loss_ACE
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 0:
            img_list.append(img[0].detach())
            gaze_list.append(gaze[0].detach())
            gaze_pred_list.append(gaze_pred[0].detach())

        _, pred = torch.max(cls_pred, 1)
        pred_list.append(pred.cpu().detach().numpy().tolist())
        label_list.append(label.cpu().detach().numpy().tolist())
        if step % print_freq == 0:
            itertime = time.time() - start
            print('    lr: {:.6f}'.format(optimizer.param_groups[0]["lr"]))
            print('    loss: {:.4f}'.format(loss.item()))
            print('    [{} / {}] iter time: {:.4f}'.format(step, total_steps, itertime))
            print('    ------------------------------------')
        step = step + 1
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    pred_list = [b for a in pred_list for b in a]
    label_list = [b for a in label_list for b in a]
    acc = accuracy_score(pred_list, label_list)
    cm = confusion_matrix(pred_list, label_list)
    print(acc)
    print(cm)
    print('Epoch: [{}] Total time: {} ({:.4f}  / it )'.format(epoch, total_time_str, total_time / total_steps))

    writer.add_scalar('train_loss', loss.item(), epoch)
    writer.add_scalar('train_acc', acc, epoch)

    visual_train = Visualize_train()
    visual_train(torch.stack(img_list), torch.stack(gaze_list), torch.stack(gaze_pred_list),
                 epoch, writer)

    
def val_one_epoch(model, dataloader_val, device, epoch, writer):
    model.eval()
    print_freq = 50
    total_steps = len(dataloader_val)
    start_time = time.time()
    step = 0
    pred_list = []
    label_list = []

    criterion_cls = nn.CrossEntropyLoss()
    val_loss_sum = 0.0
    val_count = 0

    for img, label, _ in dataloader_val:
        start = time.time()

        img = img.to(device)
        label = label.to(device)

        with torch.no_grad():
            cls_pred = model(img)
            evidences = F.softplus(cls_pred)
            alpha = evidences + 1
            S = torch.sum(alpha, dim=1, keepdim=True)
            E = alpha - 1
            b = E / (S.expand(E.shape))

            loss_cls = criterion_cls(b, label)

        val_loss_sum += loss_cls.item() * img.size(0)
        val_count += img.size(0)

        _, pred = torch.max(b, 1)
        pred_list.append(pred.cpu().detach().numpy().tolist())
        label_list.append(label.cpu().detach().numpy().tolist())

        if step % print_freq == 0:
            itertime = time.time() - start
            print('    [{} / {}] iter time: {:.4f}'.format(step, total_steps, itertime))
            print('    ------------------------------------')
        step = step + 1

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    pred_list = [b for a in pred_list for b in a]
    label_list = [b for a in label_list for b in a]
    acc = accuracy_score(pred_list, label_list)

    val_loss = val_loss_sum / max(val_count, 1)

    writer.add_scalar('val_acc', acc, epoch)
    writer.add_scalar('val_loss', val_loss, epoch)

    return val_loss, acc

