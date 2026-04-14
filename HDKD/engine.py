from typing import Iterable, Optional
from losses import *
import timm

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device, epoch: int,
                    use_distillation: bool = True, _lambda = 10): 
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = 0
        if use_distillation==True:
            distillation_loss = criterion['Total_distillation_loss']
            loss = distillation_loss(inputs, outputs, labels)

        else:
            CE_loss = criterion['CE_loss']
            loss = CE_loss(outputs, labels)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if use_distillation==True:
            out_cls,out_distill = outputs
            # simple averaging of outputs by 2 tokens
            # TODO: try some parameter learning for this or just change this averaging thing to more weightage 
            output = (out_cls + out_distill) / 2
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        else:
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    train_loss /= len(data_loader)
    train_accuracy = correct / total

    return {'loss':train_loss, 'accuracy':train_accuracy}



@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, data_loader: Iterable,
            device: torch.device ,use_distillation: bool = True):
    val_loss = 0.0
    correct = 0
    total = 0
    model.eval()
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = None
        if use_distillation==True:
            out_cls,out_distill = model(inputs)
            # TODO : Tweak the weighting 
            outputs = (out_cls + out_distill)/2
        else:
            outputs = model(inputs)

        loss = criterion(outputs, labels)
        val_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    val_loss /= len(data_loader)
    val_accuracy = correct / total

    return {'loss':val_loss, 'accuracy':val_accuracy}

