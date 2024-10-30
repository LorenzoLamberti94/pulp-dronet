#-----------------------------------------------------------------------------#
# Copyright(C) 2024 University of Bologna, Italy, ETH Zurich, Switzerland.    #
# All rights reserved.                                                        #
#                                                                             #
# Licensed under the Apache License, Version 2.0 (the "License");             #
# you may not use this file except in compliance with the License.            #
# See LICENSE in the top directory for details.                               #
# You may obtain a copy of the License at                                     #
#                                                                             #
#   http://www.apache.org/licenses/LICENSE-2.0                                #
#                                                                             #
# Unless required by applicable law or agreed to in writing, software         #
# distributed under the License is distributed on an "AS IS" BASIS,           #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
# See the License for the specific language governing permissions and         #
# limitations under the License.                                              #
#                                                                             #
# File:    training.py                                                        #
# Authors:                                                                    #
#          Lorenzo Lamberti <lorenzo.lamberti@unibo.it>                       #
#          Daniele Palossi  <dpalossi@iis.ee.ethz.ch>                         #
# Date:    01.03.2024                                                         #
#-----------------------------------------------------------------------------#

# Script Description:
# This script is used to train the weights of the PULP-DroNet CNN.
# You must specify the CNN architecture (--bypass, --depth_mul, --block_type ).
# Additional utils:
#       --early_stopping: When deactivated, this script will save the trained weights
#                     (".pth" files) for all the epochs (i.e., for 100 epochs the
#                     output will be a set of 100 weights).
#                     When activated, the script will save just one set of weights
#                     (the last one, which is not necessarily the best performing one).

import os
import sys
import argparse
from utility import str2bool
import numpy as np
import shutil
from os.path import join
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

# Import PULP-DroNet components
from model.dronet_v3 import ResBlock, Depthwise_Separable, Inverted_Linear_Bottleneck
from model.dronet_v3 import dronet
from utility import load_weights_into_network
from classes import Dataset
from utility import DronetDatasetV3
from utility import EarlyStopping, init_weights
from utility import custom_mse, custom_accuracy, custom_bce, custom_loss_v3
from utility import AverageMeter
from utility import write_log


# Global variables
alpha = 1.0
beta = 0.0


def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def create_parser(cfg):
    """Creates and returns argument parser for training configuration."""
    parser = argparse.ArgumentParser(description='PyTorch PULP-DroNet Training')
    parser.add_argument('-d', '--data_path', help='Path to training dataset',
                       default=cfg.data_path, metavar='DIRECTORY')
    parser.add_argument('--data_path_testing', help='Path to testing dataset',
                       metavar='DIRECTORY')
    parser.add_argument('--partial_training', default=None,
                       choices=[None, 'classification', 'regression'])
    parser.add_argument('-m', '--model_name', default=cfg.model_name)
    parser.add_argument('-w', '--model_weights_path', default=cfg.model_weights_path)
    parser.add_argument('--bypass', default=cfg.bypass, type=str2bool)
    parser.add_argument('--block_type', choices=["ResBlock", "Depthwise", "IRLB"],
                       default="ResBlock")
    parser.add_argument('--depth_mult', default=cfg.depth_mult, type=float)
    parser.add_argument('--epochs', default=cfg.epochs, type=int)
    parser.add_argument('-b', '--batch_size', default=cfg.training_batch_size, type=int)
    parser.add_argument('--gpu', default=cfg.gpu)
    parser.add_argument('-j', '--workers', default=cfg.workers, type=int)
    parser.add_argument('--resume_training', default=cfg.resume_training, type=str2bool)
    parser.add_argument('--hard_mining_train', default=cfg.hard_mining_train, type=str2bool)
    parser.add_argument('--early_stopping', default=cfg.early_stopping, type=str2bool)
    parser.add_argument('--patience', default=cfg.patience, type=int)
    parser.add_argument('--delta', default=cfg.delta, type=float)
    parser.add_argument('--lr', default=cfg.learning_rate, type=float)
    parser.add_argument('--lr_decay', default=cfg.lr_decay, type=float)
    parser.add_argument('-c', '--checkpoint_path', default=cfg.checkpoint_path)
    parser.add_argument('--logs_dir', default=cfg.logs_dir)
    parser.add_argument('--verbose', action='store_true')
    return parser

def validate(test_set, net, data_loader, tensorboard_writer, logs_dir, df_valid, epoch, device, args):
    """Validate model on validation or test set."""
    dataset_string = 'Validation' if test_set == 'valid' else 'Test'
    prefix = 'valid' if test_set == 'valid' else 'test'
    
    net.eval()
    acc_valid = AverageMeter('ACC', ':.3f')
    bce_valid = AverageMeter('BCE', ':.4f')
    mse_valid = AverageMeter('MSE', ':.4f')
    loss_valid = AverageMeter('Loss', ':.4f')
    
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            mse = custom_mse(labels, outputs, device)
            bce = custom_bce(labels, outputs, device)
            acc = custom_accuracy(labels, outputs, device)
            loss = custom_loss_v3(labels, outputs, device, partial_training=args.partial_training)
            
            mse_valid.update(mse.item())
            bce_valid.update(bce.item())
            acc_valid.update(acc.item())
            loss_valid.update(loss.item())
            
    print(f'{dataset_string} MSE: {float(mse_valid.avg*alpha):.4f} BCE: {float(bce_valid.avg*(1.0-beta)):.4f} Acc: {acc_valid.avg:.4f}')

    # Tensorboard logging
    tensorboard_writer.add_scalar(f'{dataset_string}/Acc', acc_valid.avg, epoch)
    tensorboard_writer.add_scalar(f'{dataset_string}/BCE', bce_valid.avg, epoch)
    tensorboard_writer.add_scalar(f'{dataset_string}/MSE', mse_valid.avg, epoch)
    tensorboard_writer.add_scalar(f'{dataset_string}/LossV3', loss_valid.avg, epoch)

    # Update DataFrame
    new_row = pd.DataFrame([[epoch, acc_valid.avg, bce_valid.avg, mse_valid.avg, loss_valid.avg]], 
                          columns=df_valid.columns)
    df_valid = pd.concat([df_valid, new_row], ignore_index=True)
    df_valid.to_csv(join(logs_dir, f'{prefix}.csv'), index=False, float_format="%.4f")

    # Write log
    log_str = (f'{dataset_string} [{epoch}][{batch_idx}/{len(data_loader)}]\t'
               f'Loss {loss_valid.val:.4f} ({loss_valid.avg:.4f})\t'
               f'MSE {mse_valid.val:.3f} ({mse_valid.avg:.3f})\t'
               f'BCE {bce_valid.val:.3f} ({bce_valid.avg:.3f})\t'
               f'ACC {acc_valid.val:.3f} ({acc_valid.avg:.3f})')
    write_log(logs_dir, log_str, prefix=prefix, should_print=False, mode='a', end='\n')
    
    return df_valid

def train(rank, world_size, args):
    micro_batch_size = args.batch_size // world_size
    print(f"Training process {rank} of {world_size}: micro_batch_size={micro_batch_size}")
    """Main training function for each GPU process."""
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    
    # Model initialization
    if args.block_type == "ResBlock":
        net = dronet(depth_mult=args.depth_mult, block_class=ResBlock, bypass=args.bypass)
    elif args.block_type == "Depthwise":
        net = dronet(depth_mult=args.depth_mult, block_class=Depthwise_Separable, bypass=args.bypass)
    elif args.block_type == "IRLB":
        net = dronet(depth_mult=args.depth_mult, block_class=Inverted_Linear_Bottleneck, bypass=args.bypass)

    if not args.resume_training:
        net.apply(init_weights)
    else:
        net = load_weights_into_network(args.model_weights_path, net, rank)

    net = net.to(rank)
    net = DDP(net, device_ids=[rank], output_device=rank)
    
    # Dataset initialization
    dataset = Dataset(args.data_path)
    dataset.initialize_from_filesystem()
    dataset_noaug = Dataset(args.data_path_testing or args.data_path)
    dataset_noaug.initialize_from_filesystem()
    
    transformations = transforms.Compose([transforms.CenterCrop(200), transforms.ToTensor()])
    
    # Create distributed dataloaders
    train_dataset = DronetDatasetV3(transform=transformations, dataset=dataset, 
                                   selected_partition='train')
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, 
                            batch_size=micro_batch_size,
                            sampler=train_sampler,
                            num_workers=args.workers,
                            pin_memory=True)
    
    validation_dataset = DronetDatasetV3(transform=transformations, dataset=dataset, 
                                        selected_partition='valid')
    validation_loader = DataLoader(validation_dataset,
                                 batch_size=micro_batch_size,
                                 shuffle=False,
                                 num_workers=args.workers,
                                 pin_memory=True)
    
    test_dataset = DronetDatasetV3(transform=transformations, dataset=dataset_noaug, 
                                  selected_partition='test')
    test_loader = DataLoader(test_dataset,
                           batch_size=micro_batch_size,
                           shuffle=False,
                           num_workers=args.workers,
                           pin_memory=True)

    # Initialize optimizer
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), 
                          weight_decay=args.lr_decay)
    
    # Setup directories and logging for rank 0
    if rank == 0:
        training_dir = join(os.path.dirname(__file__), 'training')
        training_model_dir = join(training_dir, args.model_name)
        logs_dir = join(training_model_dir, args.logs_dir)
        tensorboard_dir = join(training_model_dir, 
                             f'tensorboard_{datetime.now().strftime("%b%d_%H:%M:%S")}')
        checkpoint_dir = join(training_model_dir, 'checkpoint')
        
        os.makedirs(logs_dir, exist_ok=True)
        print("Logs directory: ", logs_dir)
        os.makedirs(tensorboard_dir, exist_ok=True)
        print("Tensorboard directory: ", tensorboard_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        print("Checkpoints directory: ", checkpoint_dir)
        
        tensorboard_writer = SummaryWriter(log_dir=tensorboard_dir)
        
        # Initialize DataFrames
        df_train = pd.DataFrame(columns=['Epoch','ACC','BCE','MSE','Loss'])
        df_valid = pd.DataFrame(columns=['Epoch','ACC','BCE','MSE','Loss'])
        df_test = pd.DataFrame(columns=['Epoch','ACC','BCE','MSE','Loss'])
        
        if args.early_stopping:
            early_stopping = EarlyStopping(patience=args.patience, delta=args.delta, 
                                         verbose=True,
                                         path=join(args.checkpoint_path, args.model_name, 
                                                 'checkpoint.pth'))
    
    # Training loop
    for epoch in range(args.epochs + 1):
        train_sampler.set_epoch(epoch)
        net.train()
        
        acc_train = AverageMeter('ACC', ':.3f')
        bce_train = AverageMeter('BCE', ':.4f')
        mse_train = AverageMeter('MSE', ':.4f')
        loss_train = AverageMeter('Loss', ':.4f')
        
        if rank == 0:
            print(f"Epoch: {epoch}/{args.epochs}")
            train_iterator = tqdm(train_loader, desc='Train')
        else:
            train_iterator = train_loader
        
        for batch_idx, data in enumerate(train_iterator):
            inputs, labels = data[0].to(rank), data[1].to(rank)
            optimizer.zero_grad()
            outputs = net(inputs)
            
            mse = custom_mse(labels, outputs, rank)
            bce = custom_bce(labels, outputs, rank)
            acc = custom_accuracy(labels, outputs, rank)
            loss = custom_loss_v3(labels, outputs, rank, partial_training=args.partial_training)
            
            loss.backward()
            optimizer.step()
            
            acc_train.update(acc.item())
            bce_train.update(bce.item())
            mse_train.update(mse.item())
            loss_train.update(loss.item())
            
            if rank == 0:
                train_iterator.set_postfix({'loss': loss_train.avg})
        
        if rank == 0:
            # Log training metrics
            tensorboard_writer.add_scalar('Training/Acc', acc_train.avg, epoch)
            tensorboard_writer.add_scalar('Training/BCE', bce_train.avg, epoch)
            tensorboard_writer.add_scalar('Training/MSE', mse_train.avg, epoch)
            tensorboard_writer.add_scalar('Training/LossV3', loss_train.avg, epoch)
            
            # Update training DataFrame and save CSV
            new_row = pd.DataFrame([[epoch, acc_train.avg, bce_train.avg, mse_train.avg, loss_train.avg]], 
                                 columns=df_train.columns)
            df_train = pd.concat([df_train, new_row], ignore_index=True)
            df_train.to_csv(join(logs_dir, 'train.csv'), index=False, float_format="%.4f")
            
            # Write training log
            log_str = (f'Train [{epoch}][{batch_idx}/{len(train_loader)}]\t'
                      f'Loss {loss_train.val:.4f} ({loss_train.avg:.4f})\t'
                      f'MSE {mse_train.val:.3f} ({mse_train.avg:.3f})\t'
                      f'BCE {bce_train.val:.3f} ({bce_train.avg:.3f})\t'
                      f'ACC {acc_train.val:.3f} ({acc_train.avg:.3f})')
            write_log(logs_dir, log_str, prefix='train', should_print=False, mode='a', end='\n')
            
            # Validation and testing
            df_valid = validate('valid', net.module, validation_loader, tensorboard_writer, 
                              logs_dir, df_valid, epoch, rank, args)
            df_test = validate('testing', net.module, test_loader, tensorboard_writer, 
                             logs_dir, df_test, epoch, rank, args)
            
            # Early stopping or checkpointing
            if args.early_stopping:
                val_mse = df_valid['MSE'].iloc[-1]
                val_bce = df_valid['BCE'].iloc[-1]
                early_stopping(val_mse * alpha + val_bce * (1.0 - beta), net.module)
                
                if early_stopping.early_stop:
                    print('Early stopping triggered')
                    break
            else:
                # Save checkpoint each epoch
                torch.save(net.module.state_dict(), 
                         join(checkpoint_dir, f'{args.model_name}_{epoch}.pth'))
                print('Parameters saved')
    
    # Training finished - cleanup and save final model
    if rank == 0:
        if args.early_stopping:
            # Copy the last checkpoint as final model
            shutil.copyfile(
                join(args.checkpoint_path, args.model_name, 'checkpoint.pth'),
                join('model', f'{args.model_name}_{epoch}.pth')
            )
            # Remove temporary checkpoints
            checkpoint_path = join(args.checkpoint_path, args.model_name)
            if os.path.exists(checkpoint_path):
                shutil.rmtree(checkpoint_path)
                print(f'Checkpoint folder {checkpoint_path} removed')
        else:
            torch.save(net.module.state_dict(), 
                      join(training_model_dir, f'{args.model_name}_{epoch}.pth'))
        print('Parameters saved')
        
        # Final test set evaluation
        from testing import testing
        test_mse, test_acc = testing(net.module, test_loader, rank)
        log_str = f'Testing set:\tMSE {test_mse:.3f}\tACC {test_acc:.3f}'
        write_log(logs_dir, log_str, prefix='valid', should_print=False, mode='a', end='\n')
    
    cleanup()

def main():
    """Main function - parses args and launches training processes."""
    from config import cfg
    parser = create_parser(cfg)
    args = parser.parse_args()
    
    # Print model configuration
    print(
        f'You defined PULP-Dronet architecture as follows:\n'
        f'Depth multiplier: {args.depth_mult}\n'
        f'Block type: {args.block_type}\n'
        f'Bypass: {args.bypass}'
    )
    
    # Select device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    n_gpus = torch.cuda.device_count()
    if n_gpus < 1:
        print("No CUDA devices available. Running on CPU.")
        return
        
    print(f"Training with {n_gpus} GPUs")
    print("pyTorch version:", torch.__version__)
    
    # Launch distributed training
    try:
        mp.spawn(train, args=(n_gpus, args,), nprocs=n_gpus, join=True)
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    # Set working directory
    working_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(working_dir)
    print('\nworking directory:', working_dir, "\n")
    
    # Run main
    main()