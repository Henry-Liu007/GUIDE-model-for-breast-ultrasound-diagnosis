import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")
import datetime
import argparse
import os
import random
import time
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import torch
from tensorboardX import SummaryWriter
from Dataset import DatasetGaze, Dataset_nogaze
from Engines import train_one_epoch, val_one_epoch
from models.GUIDE.GUIDE import GUIDE
from utils import misc


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--lr_drop', default=500, type=int)
    parser.add_argument('--in_channels', default=1, type=int)
    parser.add_argument('--size', default=224, type=int)
    parser.add_argument('--output_dir', default='output/weights', help='path where to save, empty for no saving')
    parser.add_argument('--data_dir', default='Data') #The “Data” directory contains two folders: an “img” folder that stores ultrasound images, and a “gaze” folder that stores the corresponding gaze maps.
    parser.add_argument('--csv_path', default='Data/image_names.csv')  # image_names.csv contains three columns: image_id, class_id, and train_test. The class_id column uses 0 to indicate benign and 1 to indicate malignant
    parser.add_argument('--device', default='cuda', type=str, help='device to use for training / testing')
    parser.add_argument('--GPU_ids', type=str, default='0', help='Ids of GPUs')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--num_classes', default=2, type=int)
    return parser


def main(args):

    writer = SummaryWriter(log_dir=args.output_dir + '/summary')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(args.device)

    model = GUIDE(num_classes=2)
    model.to(device)
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [{"params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad]}]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    print('Building training dataset...')
    dataset_train = DatasetGaze(args.data_dir, args.csv_path, 'train', args.size)
    print('Number of training images: {}'.format(len(dataset_train)))

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    output_dir = Path(args.output_dir)

    print("Start training")
    start_time = time.time()

    for epoch in range(0, args.epochs):
        print('-' * 40)
        train_one_epoch(model, dataloader_train, optimizer, device, epoch, args, writer)
        lr_scheduler.step()

    final_checkpoint_path = output_dir / 'checkpoint_final.pth'
    misc.save_on_master({
        'model': model_without_ddp.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'epoch': args.epochs,
        'args': vars(args),
    }, final_checkpoint_path)
    print(f"Final model saved to: {final_checkpoint_path}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Classification training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.GPU_ids)
    main(args)
