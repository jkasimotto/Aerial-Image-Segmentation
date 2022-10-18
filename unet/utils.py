import argparse
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import os
from dataset import PlanesDataset
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP, DataParallel as DP
import torch.distributed as dist
import numpy
import random
from model import UNET

def is_main_node(rank):
    return rank is None or rank == 0


def get_model(args):
    if args.get('config').get('distributed'):
        dist_args = args.get('distributed')
        model = UNET(in_channels=3, out_channels=1).cuda(dist_args.get('gpu'))
        model = DDP(model, device_ids=[dist_args.get('gpu')])
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_ids = [i for i in range(torch.cuda.device_count())]
        model = UNET(in_channels=3, out_channels=1).to(device)
        model = DP(model, device_ids=device_ids)

    model = model.to(memory_format=get_memory_format(args))

    return model

def get_memory_format(args):
    if args.get('config').get('channels-last'):
        return torch.channels_last
    else:
        return torch.contiguous_format


def get_device(args):
    if args.get('config').get('distributed'):
        gpu = args.get('distributed').get('gpu')
        device = f'cuda:{gpu}'
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return device


def dist_env_setup(args):
    if args.get("distributed").get('ngpus') is None:
        args['distributed']['ngpus'] = torch.cuda.device_count()
    args['distributed']['world-size'] = args.get("distributed").get('nodes') * args.get("distributed").get('ngpus')
    os.environ['MASTER_ADDR'] = args.get("distributed").get('ip-address')
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = str(args.get("distributed").get('world-size'))


def dist_process_setup(args, gpu):
    args['distributed']['gpu'] = gpu
    dist_args = args.get('distributed')
    rank = dist_args.get('local-ranks') * dist_args.get('ngpus') + gpu

    dist.init_process_group(
        backend='nccl',
        world_size=dist_args.get('world-size'),
        rank=rank,
    )

    torch.cuda.set_device(dist_args.get('gpu'))

    return rank


def read_config_file():
    cla = _command_line_args()
    with open(cla.config_file, "r") as stream:
        return yaml.safe_load(stream)


def _command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    args = parser.parse_args()
    return args

def _augmentations():
    # Augmentations to training set
    train_transforms = A.Compose([
        A.Rotate(limit=35, p=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2()])

    # Augmentations to test set
    test_transforms = A.Compose([
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2()])

    return train_transforms, test_transforms

def _collate_fn(batch):
    images, labels = [], []
    for img, mask in batch:
        images.append(img)
        labels.append(mask)
    images = torch.stack(images)
    labels = torch.stack(labels)
    return images, labels


def _get_datasets(args):
    data_dir = args.get('config').get('data-dir')

    img_dir = os.path.join(data_dir, 'train/images_tiled')
    mask_dir = os.path.join(data_dir, 'train/masks_tiled')
    test_img_dir = os.path.join(data_dir, 'test/images_tiled')
    test_mask_dir = os.path.join(data_dir, 'test/masks_tiled')

    train_transform, test_transform = _augmentations()

    train_dataset = PlanesDataset(
        img_dir=img_dir, mask_dir=mask_dir,
        num_classes=args.get('config').get('classes'), transforms=train_transform,
    )
    test_dataset = PlanesDataset(
        img_dir=test_img_dir, mask_dir=test_mask_dir,
        num_classes=args.get('config').get('classes'), transforms=test_transform,
    )

    return train_dataset, test_dataset


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def get_data_loaders(args, rank=None):
    train_dataset, test_dataset = _get_datasets(args)

    dist_args, hyper_params = args.get('distributed'), args.get('hyper-params')

    train_sampler, test_sampler, batch_size = None, None, hyper_params.get('batch-size')
    if args.get('config').get('distributed'):
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=dist_args.get('world-size'), rank=rank
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, num_replicas=dist_args.get('world-size'), rank=rank
        )
        batch_size = int(hyper_params.get('batch-size') / dist_args.get('ngpus'))

    g = torch.Generator()
    g.manual_seed(0)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=hyper_params.get('workers'),
        collate_fn=_collate_fn,
        pin_memory=True,
        sampler=train_sampler,
        worker_init_fn=_seed_worker,
        generator=g,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=hyper_params.get('workers'),
        collate_fn=_collate_fn,
        pin_memory=True,
        sampler=test_sampler,
        worker_init_fn=_seed_worker,
        generator=g,
    )

    return train_loader, test_loader




# def get_loaders(
#     train_dir,
#     train_maskdir,
#     val_dir,
#     val_maskdir,
#     batch_size,
#     train_transform,
#     val_transform,
#     num_gpus,
#     rank,
#     num_workers=4,
#     pin_memory=True,
# ):
#     train_ds = PlanesDataset(
#         img_dir=train_dir,
#         mask_dir=train_maskdir,
#         transform=train_transform
#     )

#     train_sampler = torch.utils.data.distributed.DistributedSampler(
#         train_ds, num_replicas=num_gpus, rank=rank
#     )

#     train_loader = DataLoader(
#         train_ds,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#         shuffle=False,
#         sampler=train_sampler
#     )

#     val_ds = PlanesDataset(
#         img_dir=val_dir,
#         mask_dir=val_maskdir,
#         transform=val_transform
#     )

#     test_sampler = torch.utils.data.distributed.DistributedSampler(
#         val_ds, num_replicas=num_gpus, rank=rank
#     )

#     val_loader = DataLoader(
#         val_ds,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#         shuffle=False,
#         sampler=test_sampler
#     )

#     return train_loader, val_loader
