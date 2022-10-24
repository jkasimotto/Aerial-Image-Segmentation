import argparse
import yaml
import torch
import os
import numpy
import random
import torch.distributed as dist
from dataset import PlanesDataset
import transforms as T
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torch.nn.parallel import DistributedDataParallel as DDP, DataParallel as DP


def is_main_node(rank):
    return rank is None or rank == 0


def get_model(args):
    if args.get('distributed').get('enabled'):
        dist_args = args.get('distributed')
        model = maskrcnn_resnet50_fpn_v2(num_classes=args.get('config').get('classes')).cuda(dist_args.get('gpu'))
        model = DDP(model, device_ids=[dist_args.get('gpu')])
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_ids = [0]
        model = maskrcnn_resnet50_fpn_v2(num_classes=args.get('config').get('classes')).to(device)
        model = DP(model, device_ids=device_ids)

    model = model.to(memory_format=get_memory_format(args))

    return model


def get_memory_format(args):
    if args.get('channels-last').get('enabled'):
        return torch.channels_last
    else:
        return torch.contiguous_format


def get_device(args):
    if args.get('distributed').get('enabled'):
        gpu = args.get('distributed').get('gpu')
        device = f'cuda:{gpu}'
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return device


def collate_fn(batch):
    return tuple(zip(*batch))


def read_config_file():
    cla = _command_line_args()
    with open(cla.config_file, "r") as stream:
        return yaml.safe_load(stream)


def _command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="path to config file")
    args = parser.parse_args()
    return args


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

    if args.get('cuda-graphs').get('enabled'):
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"

    dist.init_process_group(
        backend='nccl',
        world_size=dist_args.get('world-size'),
        rank=rank,
    )

    torch.cuda.set_device(dist_args.get('gpu'))

    return rank


def _get_datasets(args):
    data_dir = args.get('config').get('data-dir')

    img_dir = os.path.join(data_dir, 'train/images_tiled')
    mask_dir = os.path.join(data_dir, 'train/masks_tiled')
    test_img_dir = os.path.join(data_dir, 'test/images_tiled')
    test_mask_dir = os.path.join(data_dir, 'test/masks_tiled')

    train_transform, test_transform = get_transform(train=True), get_transform(train=False)

    train_dataset = PlanesDataset(img_dir=img_dir, mask_dir=mask_dir, transforms=train_transform)
    test_dataset = PlanesDataset(img_dir=test_img_dir, mask_dir=test_mask_dir, transforms=test_transform)

    return train_dataset, test_dataset


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        # transforms.append(T.RandomVerticalFlip(0.5))
    return T.Compose(transforms)


def get_data_loaders(args, rank=None):
    train_dataset, test_dataset = _get_datasets(args)

    dist_args, hyper_params = args.get('distributed'), args.get('hyper-params')

    train_sampler, test_sampler, batch_size = None, None, hyper_params.get('batch-size')
    if args.get('distributed').get('enabled'):
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=dist_args.get('world-size'), rank=rank
        )
        test_sampler = DistributedSampler(
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
        collate_fn=collate_fn,
        pin_memory=True,
        sampler=train_sampler,
        worker_init_fn=_seed_worker,
        generator=g,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=hyper_params.get('workers'),
        collate_fn=collate_fn,
        pin_memory=True,
        sampler=test_sampler,
        worker_init_fn=_seed_worker,
        generator=g,
    )

    return train_loader, test_loader


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict
