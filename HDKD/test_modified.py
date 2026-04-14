import argparse
from typing import Iterable, Optional
import torch
import torch.nn as nn
import timm
from timm.models import create_model
from torchvision import datasets
from dataset import get_transform
# from engine import  evaluate
from models import *


@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, data_loader: Iterable,
            device: torch.device ,use_distillation: bool = True,weighting=0.5):
    val_loss = 0.0
    correct = 0
    total = 0
    print(f"{weighting} applied on distill token")
    model.eval()
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = None
        if use_distillation==True:
            out_cls,out_distill = model(inputs)
            # TODO!!! changed this
            # TODO : Tweak the weighting 
            outputs = ((1-weighting)*out_cls + weighting* out_distill)
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



def build_dataset(folder,args):
    data_transforms = get_transform(False,args)
    image_datasets = datasets.ImageFolder(folder, data_transforms)
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size = args.batch_size,shuffle=False,num_workers=2)

    return dataloaders

def get_args_parser():
    parser = argparse.ArgumentParser('HDKD training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=16, type=int)

    # Model parameters
    parser.add_argument('--model', default='HDKD', type=str, metavar='MODEL',
                        help='Name of model to train whether the teacher model, student model or HDKD model')
    parser.add_argument('--input-size', default=[224,224], type=int, help='images input size')
    parser.add_argument('--checkpoint', default='best_model.pth', type=str, help='Path to the model checkpoint')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    parser.add_argument('--data-path', default="/content/HAM-10000/test", type=str,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=7, type=int,
                        help='number of the classes within the dataset')
    parser.add_argument('--batch_size', default=16, type=int, 
                        help='number of the classification types')


    return parser




args,unknown = get_args_parser().parse_known_args()
assert args.model in ['HDKD', 'student_model', 'teacher_model']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


test_loader = build_dataset(args.data_path, args)

print(f"Creating model: {args.model}")
model = create_model(
    args.model,
    num_classes=args.nb_classes,
    drop=args.drop
)
model.to(device)

if device == "cuda":
    checkpoint = torch.load(args.checkpoint)
else:
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

model.load_state_dict(checkpoint)

use_distillation=False
if args.model=="HDKD":
    use_distillation=True

test_criterion=nn.CrossEntropyLoss()
for i in [0.2,0.4,0.5,0.6,0.7,0.8,0.9]:
    print("-"*100)
    test_stats = evaluate(model, test_criterion, test_loader, device, use_distillation = use_distillation,weighting=i)

    accuracy = test_stats['accuracy']
    print("Testing Accuracy is {}".format(accuracy))
