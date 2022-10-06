import argparse
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import os
from dataset import PlanesDataset
from torch.utils.data import DataLoader
from torchvision.models.segmentation import fcn_resnet101
from torch.nn.parallel import DistributedDataParallel as DDP, DataParallel as DP
import torch.distributed as dist


def is_main_node(rank):
    return rank is None or rank == 0


def get_model(args):
    if args.get('config').get('distributed'):
        dist_args = args.get('distributed')
        model = fcn_resnet101(num_classes=args.get('config').get('classes')).cuda(dist_args.get('gpu'))
        model = DDP(model, device_ids=[dist_args.get('gpu')])
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_ids = [i for i in range(torch.cuda.device_count())]
        model = fcn_resnet101(num_classes=args.get('config').get('classes')).to(device)
        model = DP(model, device_ids=device_ids).to(device)

    return model


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
    torch.manual_seed(0)

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
    train_transforms = A.Compose([
        A.Rotate(limit=35, p=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
        ToTensorV2()])

    test_transforms = A.Compose([
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=hyper_params.get('workers'),
        collate_fn=_collate_fn,
        pin_memory=True,
        sampler=train_sampler,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=hyper_params.get('workers'),
        collate_fn=_collate_fn,
        pin_memory=True,
        sampler=test_sampler,
    )

    return train_loader, test_loader
