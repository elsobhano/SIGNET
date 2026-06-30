import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from models import SignetGate, SignetExpert
import utils as utils
from data import S2T_Dataset
from collections import OrderedDict
import os
import time
import argparse, json, datetime
from pathlib import Path
import math
import sys
from timm.optim import create_optimizer
from models import get_requires_grad_dict
from metrics import translation_performance, islr_performance, wer_list
from transformers import get_scheduler
import datetime
import wandb
import os
from tqdm import tqdm
import importlib
from torch.cuda.amp import autocast
# from augment import augment_skeleton
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 


from config import *

def load_dataset(module_name, dataset_name):
    datasets = {
        "CSL_News": "S2T_Large_Scale_dataset",
        "YT_ASL": "S2T_Large_Scale_dataset",
        "BOBSL": "S2T_Large_Scale_dataset",
        "CSL_Daily": "S2T_Dataset",
        "Phoenix": "S2T_Dataset_phoenix",
        "How2Sign": "S2T_Dataset_how2sign",
        "WLASL": "S2T_Dataset_wlasl",
        "MeinDGS": "S2T_Dataset_meindgs",
    }
    module = importlib.import_module(module_name)
    return getattr(module, datasets[dataset_name])

def setupWandB(wandb_config, storage=None):
    os.environ.update(wandb_config)
    if storage is not None:
        os.environ['WANDB_CACHE_DIR'] = storage+'/wandb/cache'
        os.environ['WANDB_CONFIG_DIR'] = storage+'/wandb/config'

def setup_logging(args, config, wandb_config, current_time=None, mode="pre"):
    """
    Sets up the logging directory based on the provided arguments.
    Creates the directory if it does not exist.
    """
    if current_time is None:
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir=f'{args.output_dir}/log_{current_time}'
    setupWandB(wandb_config, storage=save_dir)
    wandb.login()
    wandb.init(
    project=f"local-fine-{config['logging']['project_name']}",
    config=vars(args),
    reinit=True  # ensures new run instead of reusing Lightning's run
    )
    
    return wandb


def main(args):
    utils.init_distributed_mode_ds(args)

    print(args)
    utils.set_seed(args.seed)
    
    print("Initializing WandB...")
    logger_config = {
        "logging": {
            "project_name": f"{args.dataset}--gating-contrastive",
            "run_name": f"distill-{args.dataset}-{args.task}",
            "save_dir": args.output_dir,
        }}
    logger = setup_logging(args, logger_config, WANDB_CONFIG, current_time=None, mode="fine")
    

    print("Creating dataset:")
    
    Dataset = load_dataset('data', args.dataset)
    
    train_data = Dataset(path=train_label_paths[args.dataset], 
                            args=args, phase='train')
    print(train_data)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,shuffle=True)
    train_dataloader = DataLoader(train_data,
                                batch_size=args.batch_size, 
                                num_workers=args.num_workers, 
                                collate_fn=train_data.collate_fn,
                                sampler=train_sampler, 
                                pin_memory=args.pin_mem,
                                drop_last=True)
    
    dev_data = Dataset(path=dev_label_paths[args.dataset], 
                        args=args, phase='dev')
    print(dev_data)
    # dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_data,shuffle=False)
    dev_sampler = torch.utils.data.SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers, 
                                collate_fn=dev_data.collate_fn,
                                sampler=dev_sampler, 
                                pin_memory=args.pin_mem)
        
    test_data = Dataset(path=test_label_paths[args.dataset], 
                            args=args, phase='test')
    print(test_data)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_data,shuffle=False)
    test_sampler = torch.utils.data.SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers, 
                                collate_fn=test_data.collate_fn,
                                sampler=test_sampler, 
                                pin_memory=args.pin_mem)
    experts = []
    for i, expert_ckpt in enumerate(args.expert_model_paths):
        print(f"Creating Expert {i + 1} model from {expert_ckpt}:")
        expert_model = SignetExpert(args=args)
        expert_model.cuda()
        expert_model.eval()
        expert_model = utils.expert_weights_from_ckpt(expert_model, model_path=expert_ckpt)
        for name, param in expert_model.named_parameters():
            param.data = param.data.to(torch.float32)
            param.requires_grad = False
        experts.append(expert_model)
    print(f"Loaded {len(experts)} experts.")
    print("Creating Gate model:")
    gate_model = SignetGate(args=args, num_experts=len(experts), k=args.K)
    gate_model.cuda()
    gate_model.train()
    gate_model = utils.load_weights_from_ckpt(gate_model, args)
    # exit(0)
    for name, param in gate_model.named_parameters():
        # print(name, param.requires_grad)
        param.data = param.data.to(torch.float32)
    # exit(0)
    # for param in gate_model.mt5_model.encoder.parameters():
    #     param.requires_grad = False
    model_without_ddp = gate_model
    if args.distributed:
        gate_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(gate_model)
        gate_model = torch.nn.parallel.DistributedDataParallel(gate_model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = gate_model.module
    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    print(f'number of params: {n_parameters}M')
    
    router_params, other_params = [], []
    
    for n, p in model_without_ddp.named_parameters():
        if not p.requires_grad:
            continue
        if "router" in n:
            router_params.append(p)
        else:
            other_params.append(p)
    
    param_groups = [
        {"params": other_params, "lr": args.lr, "weight_decay": 0.01},   # slower for LLM/LoRA
        {"params": router_params, "lr": args.router_lr, "weight_decay": 0.01},  # router fast LR
    ]

    optimizer = create_optimizer(args, param_groups)
    lr_scheduler = get_scheduler(
                name='cosine',
                optimizer=optimizer,
                num_warmup_steps=int(args.warmup_epochs * len(train_dataloader)/args.gradient_accumulation_steps),
                num_training_steps=int(args.epochs * len(train_dataloader)/args.gradient_accumulation_steps),
            )
    
    gate_model, optimizer, lr_scheduler = utils.init_deepspeed(args, gate_model, optimizer, lr_scheduler)
    model_without_ddp = gate_model.module.module
    # print(model_without_ddp)
    print(optimizer)

    output_dir = Path(args.output_dir)

    start_time = time.time()
    max_accuracy = 0
    if args.task == "CSLR":
        max_accuracy = 1000
    elif "loss" in args.task :
        min_loss = float("inf")
    
    if args.eval:
        if utils.is_main_process():
            if args.task != "ISLR":
                print("📄 dev result")
                evaluate(args, dev_dataloader, gate_model, model_without_ddp, phase='dev')
            print("📄 test result")
            evaluate(args, test_dataloader, gate_model, model_without_ddp, phase='test')

        return 
    print(f"Start training for {args.epochs} epochs")
    # Initialize once

    for epoch in tqdm(range(0, args.epochs)):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(args, gate_model, experts, train_dataloader, optimizer, epoch)


        # single gpu inference
        if utils.is_main_process():
            test_stats = evaluate(args, dev_dataloader, gate_model, model_without_ddp, experts, phase='dev')
            dev_states = evaluate(args, test_dataloader, gate_model, model_without_ddp, experts, phase='test')

            # Save last checkpoint (always overwrite)
            last_checkpoint_path = output_dir / 'last_checkpoint.pth'
            utils.save_on_master({
                'model': get_requires_grad_dict(model_without_ddp),
                'epoch': epoch,  # optionally save epoch info
            }, last_checkpoint_path)

            if "loss" in args.task:
                current_loss = test_stats["loss"]

                if current_loss < min_loss:
                    # Delete previous best checkpoint if it exists
                    prev_best_path = output_dir / f'best_checkpoint_loss_{min_loss:.4f}.pth'
                    if prev_best_path.exists():
                        prev_best_path.unlink()
                    
                    # Update min_loss and save new best checkpoint
                    min_loss = current_loss
                    best_checkpoint_path = output_dir / f'best_checkpoint_loss_{min_loss:.4f}.pth'
                    utils.save_on_master({
                        'model': get_requires_grad_dict(model_without_ddp),
                        'epoch': epoch,
                        'loss': min_loss,
                    }, best_checkpoint_path)

                print(f"Validation loss of the network on the {len(dev_dataloader)} dev videos: {current_loss:.4f}")
                print(f"Best (lowest) validation loss so far: {min_loss:.4f}")

            if args.task == "SLT":
                if max_accuracy < test_stats["bleu4"]:
                    max_accuracy = test_stats["bleu4"]
                    if args.output_dir and utils.is_main_process():
                        checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                        for checkpoint_path in checkpoint_paths:
                            utils.save_on_master({
                                'model': get_requires_grad_dict(model_without_ddp),
                            }, checkpoint_path)

                print(f"BLEU-4 of the network on the {len(dev_dataloader)} dev videos: {test_stats['bleu4']:.2f}")
                print(f'Max BLEU-4: {max_accuracy:.2f}%')
            
            elif args.task == "ISLR":
                if max_accuracy < test_stats["top1_acc_pi"]:
                    max_accuracy = test_stats["top1_acc_pi"]
                    if args.output_dir and utils.is_main_process():
                        checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                        for checkpoint_path in checkpoint_paths:
                            utils.save_on_master({
                                'model': get_requires_grad_dict(model_without_ddp),
                            }, checkpoint_path)

                print(f"PI accuracy of the network on the {len(dev_dataloader)} dev videos: {test_stats['top1_acc_pi']:.2f}")
                print(f'Max PI accuracy: {max_accuracy:.2f}%')
            
            elif args.task == "CSLR":
                if max_accuracy > test_stats["wer"]:
                    max_accuracy = test_stats["wer"]
                    if args.output_dir and utils.is_main_process():
                        checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                        for checkpoint_path in checkpoint_paths:
                            utils.save_on_master({
                                'model': get_requires_grad_dict(model_without_ddp),
                            }, checkpoint_path)
                            
                print(f"WER of the network on the {len(dev_dataloader)} dev videos: {test_stats['wer']:.2f}")
                print(f'Min WER: {max_accuracy:.2f}%')
        
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        **{f'dev_{k}': v for k, v in dev_states.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
            
            logger.log(log_stats)
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def train_one_epoch(args, model, experts, data_loader, optimizer, epoch):
    model.train()
    experts = [expert_model.eval() for expert_model in experts]

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 10
    optimizer.zero_grad()

    target_dtype = None
    if model.bfloat16_enabled():
        target_dtype = torch.bfloat16
    
    # warmup_steps = int(0.1 * len(data_loader) * args.epochs)  # e.g. 10% of total steps
    for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if target_dtype is not None:
            for k in src_input:
                if isinstance(src_input[k], torch.Tensor):
                    src_input[k] = src_input[k].to(target_dtype).cuda(non_blocking=True)
        else:
            for k in src_input:
                if isinstance(src_input[k], torch.Tensor):
                    src_input[k] = src_input[k].cuda(non_blocking=True)

        # === expert forwards (no grad) ===
        expert_outputs_lists = []
        with torch.no_grad():
            expert_input = {k: (v.to(torch.float32).cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v)
                            for k, v in src_input.items()}
            for Tk in experts:
                expert_outputs = Tk(expert_input, tgt_input)
                for k in expert_outputs:
                    if isinstance(expert_outputs[k], torch.Tensor):
                        expert_outputs[k] = expert_outputs[k].to(target_dtype).cuda(non_blocking=True)
                expert_outputs_lists.append(expert_outputs)
                    
        gate_out = model(expert_outputs_lists, tgt_input, src_input['attention_mask'], task=args.task)
        total_loss = gate_out['loss']
        # stack into [K,B,T,D]

        
        model.backward(total_loss)
        model.step()

        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
            
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return  {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate(args, data_loader, model, model_without_ddp, experts, phase):
    model.eval()
    experts = [t.eval() for t in experts]

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    target_dtype = None
    if model.bfloat16_enabled():
        target_dtype = torch.bfloat16
        
    with torch.no_grad():
        tgt_pres = []
        tgt_refs = []

        for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            if target_dtype != None:
                for key in src_input.keys():
                    if isinstance(src_input[key], torch.Tensor):
                        src_input[key] = src_input[key].to(target_dtype).cuda()
            
            # === expert forwards (no grad) ===
            expert_outputs_lists = []
            expert_input = {k: (v.to(torch.float32).cuda(non_blocking=True) if isinstance(v, torch.Tensor) else v)
                            for k, v in src_input.items()}
            for Tk in experts:
                expert_outputs = Tk(expert_input, tgt_input)
                for k in expert_outputs:
                    if isinstance(expert_outputs[k], torch.Tensor):
                        expert_outputs[k] = expert_outputs[k].to(target_dtype).cuda(non_blocking=True)
                expert_outputs_lists.append(expert_outputs)
                        
            gate_out = model(expert_outputs_lists, tgt_input, src_input['attention_mask'], task=args.task)
            total_loss = gate_out['loss']
                
            metric_logger.update(loss=total_loss.item())
        
        
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def kd_warmup_factor(step, warmup_steps):
    if step >= warmup_steps:
        return 1.0
    return step / float(warmup_steps)


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # 1. Get the current time
    now = datetime.datetime.now()

    # 2. Format it into a filename-safe string (e.g., "2025-09-29_15-22-12")
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
    parser = argparse.ArgumentParser('SIGNET scripts', parents=[utils.get_args_parser()])
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.dataset, 'gating-contrastive', timestamp_str)
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)