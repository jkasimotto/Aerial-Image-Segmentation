import copy
import time
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
from utils import get_memory_format, is_main_node, get_device


def run(args, model, criterion, optimizer, warmup_loader, train_loader, scaler, rank):
    device = get_device(args)

    if is_main_node(rank):
        print("\nWarming up")

    model, input_shape, label_shape = _warm_up(
        args=args,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        dataloader=warmup_loader,
        rank=rank,
    )

    if is_main_node(rank):
        print("\nCapturing")

    g, capture_input, capture_target = _capture(
        input_shape=input_shape,
        label_shape=label_shape,
        device=device,
        optimizer=optimizer,
        model=model,
        criterion=criterion,
        scaler=scaler,
        args=args,
    )

    if is_main_node(rank):
        print("\nTraining")

    start = time.time()

    _train(
        args=args,
        optimizer=optimizer,
        train_loader=train_loader,
        rank=rank,
        capture_input=capture_input,
        capture_target=capture_target,
        g=g,
        scaler=scaler,
    )

    end = time.time()

    if is_main_node(rank):
        print(f"\nTraining took: {end - start:.2f}s")

    return model


def _warm_up(args, model, criterion, optimizer, scaler, device, dataloader, rank):
    warmup_iters = args.get('cuda-graphs').get('warmup-iters')
    use_amp = args.get('amp').get('enabled')

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        model = copy.deepcopy(model)
        model.train()

        for batch, (images, labels) in enumerate(tqdm(dataloader, total=warmup_iters, disable=not is_main_node(rank))):
            if batch == warmup_iters:
                break

            images = images.to(device, memory_format=get_memory_format(args))
            labels = labels.to(device, memory_format=get_memory_format(args))

            with autocast(enabled=use_amp):
                y_pred = model(images)['out']
                loss = criterion(y_pred, labels)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

    torch.cuda.current_stream().wait_stream(s)

    return model, images.shape, labels.shape


def _capture(input_shape, label_shape, device, optimizer, model, criterion, scaler, args):
    use_amp = args.get('amp').get('enabled')

    capture_input = torch.empty(input_shape, device=device)
    capture_target = torch.empty(label_shape, device=device)

    g = torch.cuda.CUDAGraph()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.graph(g):
        with autocast(enabled=use_amp):
            capture_y_pred = model(capture_input)['out']
            capture_loss = criterion(capture_y_pred, capture_target)

        if use_amp:
            scaler.scale(capture_loss).backward()
        else:
            capture_loss.backward()
            optimizer.step()

    return g, capture_input, capture_target


def _train(args, optimizer, train_loader, rank, capture_input, capture_target, g, scaler):
    use_amp = args.get('amp').get('enabled')

    for epoch in range(args.get('hyper-params').get('epochs')):
        for batch, (images, labels) in enumerate(tqdm(train_loader, disable=not is_main_node(rank))):
            capture_input.copy_(images)
            capture_target.copy_(labels)
            g.replay()
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
