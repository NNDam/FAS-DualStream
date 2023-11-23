import os
import copy
import time
import tqdm

import numpy as np
import torch
from torch import autocast
from torch.utils.data import Dataset, DataLoader
from args import parse_arguments
from models import FaceEncoder
from utils import cosine_lr, torch_load, LabelSmoothing, AverageMeter, get_lr
from datasets.base_dataset import FASData

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def calc_accuracy(logits, labels):
    pred = sigmoid(logits)
    pred = np.where(pred >= 0.5, 1.0, 0.0)
    pred = np.squeeze(pred)
    labels = np.where(labels >= 0.5, 1.0, 0.0)
    labels = np.squeeze(labels)

    assert len(pred) == len(labels)
    tp = np.sum(np.where(pred == labels, 1, 0))
    return float(tp) / float(len(pred))

def eval(model, val_loader):
    model.eval()
    acc = 0
    count = 0
    with torch.no_grad():
        for i, (inputs, inputs_bpf, labels) in enumerate(val_loader):
            inputs = inputs.cuda()
            inputs_bpf = inputs_bpf.cuda()
            labels = labels.cuda()
            logits_fused, logits_org, logit_bpf = model(inputs, inputs_bpf)
            v = calc_accuracy(logits_fused.detach().cpu().numpy(), labels.detach().cpu().numpy())
            count += len(logits_fused)
            acc += v*len(logits_fused)
    return acc/count

def finetune(args, is_load = False):
    assert args.save is not None, "args.save must be define for saving checkpoint"
    # assert args.load is not None, "Please provide the patch to a checkpoint through --load."
    # assert args.train_dataset is not None, "Please provide a training dataset."
    use_cuda = True if args.device == "cuda" else False

    model = FaceEncoder('r18_imagenet', '', image_size = 224, feature_dim = 512)
    if args.load is not None:
        print('  - Loading ', args.load)
        model = model.load(args.load)


    print(model)
    print('  - Init train dataloader')
    train_dataset = FASData(args.train_txt, input_size = 224, is_train=True)
    val_dataset = FASData(args.val_txt, input_size = 224, is_train=False)
    train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8
        )
    val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
        )
    num_batches = len(train_loader)
    print('    + Number of batches per epoch: {}'.format(num_batches))

    if use_cuda:
        model = model.cuda()
        devices = list(range(torch.cuda.device_count()))
        print('  - Using device_ids: ', devices)
        model = torch.nn.DataParallel(model, device_ids=devices)



    loss_fn = torch.nn.BCEWithLogitsLoss()
    print('  - Init BCEWithLogitsLoss')

    if args.freeze_encoder:
        print('  - Freeze backbone')
        model.module.backbone.requires_grad_(False)
        model.module.head.requires_grad_(True)
        model.module.fc_fused.requires_grad_(True)
        model.module.fc_org.requires_grad_(True)
        model.module.fc_bpf.requires_grad_(True)
    else:
        model.module.backbone.requires_grad_(True)
        model.module.head.requires_grad_(True)
        model.module.fc_fused.requires_grad_(True)
        model.module.fc_org.requires_grad_(True)
        model.module.fc_bpf.requires_grad_(True)

    params      = [p for name, p in model.named_parameters() if p.requires_grad]
    params_name = [name for name, p in model.named_parameters() if p.requires_grad]
    print('  - Total {} params to training: {}'.format(len(params_name), [pn for pn in params_name]))
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)
    print('  - Init AdamW with cosine learning rate scheduler')

    if args.fp16:
        print('  - Using Auto mixed precision')
        scaler = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)
    start_epoch = 0
    best_acc = 0.0
    input_key = 'images'
    print_every = 100
    save_every = 1000
    for epoch in range(start_epoch, args.epochs):
        print(f"Start epoch: {epoch}")
        model.train()
    
        if args.freeze_encoder:
            print('  - Freeze backbone')
            model.module.backbone.requires_grad_(False)
            model.module.head.requires_grad_(True)
            model.module.fc_fused.requires_grad_(True)
            model.module.fc_org.requires_grad_(True)
            model.module.fc_bpf.requires_grad_(True)
        else:
            model.module.backbone.requires_grad_(True)
            model.module.head.requires_grad_(True)
            model.module.fc_fused.requires_grad_(True)
            model.module.fc_org.requires_grad_(True)
            model.module.fc_bpf.requires_grad_(True)
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        

        for i, (inputs, inputs_bpf, labels) in enumerate(train_loader):
            start_time = time.time()
            step = i + epoch * num_batches
            lr = scheduler(step)
            optimizer.zero_grad()
            inputs = inputs.cuda()
            inputs_bpf = inputs_bpf.cuda()
            labels = labels.cuda()
            data_time.update(time.time() - start_time)
            # data_time = time.time() - start_time
            start_time = time.time()
            # compute output
            if args.fp16:
                with autocast(device_type="cuda", dtype=torch.float16):
                    logits_fused, logits_org, logit_bpf = model(inputs, inputs_bpf)
                    loss = loss_fn(logits_fused, labels) + 0.5*loss_fn(logits_org, labels) + 0.5*loss_fn(logit_bpf, labels)
                    # print('='*50)
                    # print(logits_fused, labels, loss_fn(logits_fused, labels))
                    # print(logits_org, labels, loss_fn(logits_org, labels))
                    # print(logit_bpf, labels, loss_fn(logit_bpf, labels))
                losses.update(loss.item(), inputs.size(0))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                # optimizer.zero_grad()
            else:
                logits_fused, logits_org, logit_bpf = model(inputs, inputs_bpf)
                loss = loss_fn(logits_fused, labels) + 0.5*loss_fn(logits_org, labels) + 0.5*loss_fn(logit_bpf, labels)

                losses.update(loss.item(), inputs.size(0))

                # compute gradient and do SGD step
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

            batch_time.update(time.time() - start_time)
            # start_time = time.time()

            if i % print_every == 0:
                logits_np = logits_fused.detach().cpu().numpy()
                labels_np = labels.detach().cpu().numpy()
                train_acc = calc_accuracy(logits_np, labels_np)
                percent_complete = 100 * i / len(train_loader)
                print(
                    f"Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(train_loader)}]\t",
                    "Lr: {}\tLoss: {}\tAccuracy: {}\tData (t) {}\tBatch (t) {}".format(get_lr(optimizer), loss.item(), train_acc, data_time.avg, batch_time.avg), 
                    flush=True
                )

            if i % save_every == 0:
                if args.save is not None:
                    os.makedirs(args.save, exist_ok=True)
                model_path = os.path.join(args.save, f'checkpoint_iters_{i}.pt')
                image_classifier = model.module if use_cuda else model
                image_classifier.save(model_path)
              


        print(f"Epoch {epoch}:\t Loss: {losses.avg:.5f}\t"
              f"Data(t): {data_time.avg:.3f}\t Batch(t): {batch_time.avg:.3f}")

        # if args.freeze_encoder:
        #     image_classifier = ImageClassifier(image_classifier.image_encoder, model.module) if use_cuda \
        #     else ImageClassifier(image_classifier.image_encoder, model)
        # else:
        image_classifier = model.module if use_cuda else model

        tik = time.time()
        val_acc = eval(image_classifier, val_loader)
        tok = time.time()
        print('Eval done in', tok - tik, val_acc, best_acc)
        is_best = val_acc > best_acc

        # Saving model
        if args.save is not None:
            os.makedirs(args.save, exist_ok=True)
            if epoch % args.save_interval == 0:
                model_path = os.path.join(args.save, f'checkpoint_{epoch+1}.pt')
                image_classifier.save(model_path)
              
            if is_best:
                print('  - Saving as best checkpoint')
                image_classifier.save(os.path.join(args.save, f'checkpoint_model_best.pt'))
                best_acc = val_acc
            
    if args.save is not None:
        return model_path

if __name__ == '__main__':
    args = parse_arguments()
    finetune(args)