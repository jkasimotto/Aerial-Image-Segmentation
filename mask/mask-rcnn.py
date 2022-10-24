import torch
import wandb
from analyser import ModelAnalyser
from torch.cuda.amp import GradScaler
import torch.multiprocessing as mp
import torch.distributed as dist
import engine
from utils import (
    read_config_file,
    dist_env_setup,
    dist_process_setup,
    get_data_loaders,
    get_model,
    is_main_node,
)


def training_setup(gpu, args):
    torch.manual_seed(0)

    # Setup process if distributed is enabled
    rank = None
    if args.get('distributed').get('enabled'):
        rank = dist_process_setup(args, gpu)

    hyper_params = args.get('hyper-params')

    # Setup dataloaders, model, criterion and optimiser
    train_loader, test_loader = get_data_loaders(args, rank)
    model = get_model(args)
    capture = args.get('cuda-graphs').get('enabled')
    optimizer = torch.optim.Adam(model.parameters(), lr=hyper_params.get('learning-rate'), capturable=capture)

    if args.get('wandb').get('enabled') and is_main_node(rank):
        wandb.init(project=args.get('wandb').get('project-name'), entity="usyd-04a", config=args, dir="./wandb_data")
        wandb.watch(model)
        wandb.run.name = args.get('config').get('run')

    # Create analyser for saving model checkpoints
    analyser = ModelAnalyser(
        checkpoint_dir=args.get('config').get('checkpoint-dir'),
        run_name=args.get('config').get('run'),
    )

    # Setup Gradient Scaler if AMP is enabled
    scaler = None
    if args.get('amp').get('enabled'):
        scaler = GradScaler()

    try:
        model = engine.train(model=model,
                             optimizer=optimizer,
                             train_loader=train_loader,
                             test_loader=test_loader,
                             scaler=scaler,
                             analyser=analyser,
                             args=args,
                             rank=rank)
    except:
        dist.destroy_process_group()
        return

    if is_main_node(rank):
        analyser.save_model(model=model,
                            epochs=hyper_params.get('epochs'),
                            optimizer=optimizer,
                            batch_size=hyper_params.get('batch-size'),
                            lr=hyper_params.get('learning-rate'))

    # Clean up distributed process
    if args.get('distributed').get('enabled'):
        dist.destroy_process_group()


def main():
    args = read_config_file()

    print(f'Starting run: {args.get("config").get("run")}\n')
    print(f'AMP: {args.get("amp").get("enabled")}')
    print(f'Channels Last: {args.get("channels-last").get("enabled")}')
    print(f'Distributed: {args.get("distributed").get("enabled")}')
    print(f'CUDA Graphs: {args.get("cuda-graphs").get("enabled")}')
    print(f'GPU available: {torch.cuda.is_available()} ({torch.cuda.device_count()})')

    torch.cuda.empty_cache()

    if args.get('distributed').get('enabled'):
        dist_env_setup(args)
        mp.spawn(training_setup, nprocs=args.get("distributed").get('ngpus'), args=(args,))
    else:
        training_setup(None, args)


if __name__ == "__main__":
    main()
