#!/usr/bin/env python3
import os
import numpy as np
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from datasets.dataset_reader_physics import read_data_train, read_data_val
from collections import namedtuple
from glob import glob
import time
import torch
from utils.deeplearningutilities.torch import Trainer, MyCheckpointManager
from evaluate_network import evaluate_ts as evaluate

# the train dir stores all checkpoints and summaries. The dir name is the name of this file without .py
train_dir = os.path.splitext(os.path.basename(__file__))[0]

dataset_path = os.path.join(os.path.dirname(__file__), '..', 'datasets',
                            'test_dataset')

val_files = sorted(glob(os.path.join(dataset_path, 'valid', '*.zst')))
train_files = sorted(glob(os.path.join(dataset_path, 'train', '*.zst')))

_k = 1000

TrainParams = namedtuple('TrainParams', ['max_iter', 'base_lr', 'batch_size'])
train_params = TrainParams(50 * _k, 0.001, 16)


def create_model():
    from models.TimeStepPreNetwork import TimestepPredictionNetwork
    """Returns an instance of the network for training and evaluation"""
    model = TimestepPredictionNetwork()
    return model


def main():

    device = torch.device('cpu')

    val_dataset = read_data_val(files=val_files, window=1, cache_data=True)

    dataset = read_data_train(files=train_files,
                              batch_size=train_params.batch_size,
                              window=2,
                              num_workers=2)
    data_iter = iter(dataset)

    trainer = Trainer(train_dir)

    model = create_model()
    model.to(device)

    boundaries = [
        25 * _k,
        30 * _k,
        35 * _k,
        40 * _k,
        45 * _k,
    ]
    lr_values = [
        1.0,
        0.5,
        0.25,
        0.125,
        0.5 * 0.125,
        0.25 * 0.125,
    ]

    def lrfactor_fn(x):
        factor = lr_values[0]
        for b, v in zip(boundaries, lr_values[1:]):
            if x > b:
                factor = v
            else:
                break
        return factor

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=train_params.base_lr,
                                 eps=1e-6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lrfactor_fn)

    step = torch.tensor(0)
    checkpoint_fn = lambda: {
        'step': step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }

    manager = MyCheckpointManager(checkpoint_fn,
                                  trainer.checkpoint_dir,
                                  keep_checkpoint_steps=list(
                                      range(1 * _k, train_params.max_iter + 1,
                                            1 * _k)))

    def loss_fn(pr_timestep, inp_t0, gt_t1):
        pr_t = pr_timestep + inp_t0[0]
        loss = torch.nn.MSELoss()(pr_t, gt_t1[0])
        return loss

    def train(model, batch):
        optimizer.zero_grad()
        losses = []

        batch_size = train_params.batch_size
        for batch_i in range(batch_size):
            inputs = ([
                batch['pos0'][batch_i], batch['vel0'][batch_i], None,
                batch['box'][batch_i], batch['box_normals'][batch_i]
            ])

            predict_timestep = model(inputs)

            l = loss_fn(predict_timestep, batch['time0'][batch_i], batch['time1'][batch_i])

            losses.append(l)

        total_loss = sum(losses) / batch_size
        total_loss.backward()
        optimizer.step()

        return total_loss

    if manager.latest_checkpoint:
        print('restoring from ', manager.latest_checkpoint)
        latest_checkpoint = torch.load(manager.latest_checkpoint)
        step = latest_checkpoint['step']
        model.load_state_dict(latest_checkpoint['model'])
        optimizer.load_state_dict(latest_checkpoint['optimizer'])
        scheduler.load_state_dict(latest_checkpoint['scheduler'])

    display_str_list = []
    while trainer.keep_training(step,
                                train_params.max_iter,
                                checkpoint_manager=manager,
                                display_str_list=display_str_list):

        data_fetch_start = time.time()
        batch = next(data_iter)
        batch_torch = {}
        for k in ('pos0', 'vel0', 'time0', 'time1', 'box', 'box_normals'):
            batch_torch[k] = [torch.from_numpy(x).to(device) for x in batch[k]]
        data_fetch_latency = time.time() - data_fetch_start
        trainer.log_scalar_every_n_minutes(5, 'DataLatency', data_fetch_latency)

        current_loss = train(model, batch_torch)
        scheduler.step()
        display_str_list = ['loss', float(current_loss)]

        if trainer.current_step % 10 == 0:
            trainer.summary_writer.add_scalar('TotalLoss', current_loss,
                                              trainer.current_step)
            trainer.summary_writer.add_scalar('LearningRate',
                                              scheduler.get_last_lr()[0],
                                              trainer.current_step)

        if trainer.current_step % (1 * _k) == 0:
            for k, v in evaluate(model,
                                 val_dataset,
                                 frame_skip=20,
                                 device=device).items():
                trainer.summary_writer.add_scalar('eval/' + k, v,
                                                  trainer.current_step)

    torch.save({'model': model.state_dict()}, 'model_weights.pt')
    if trainer.current_step == train_params.max_iter:
        return trainer.STATUS_TRAINING_FINISHED
    else:
        return trainer.STATUS_TRAINING_UNFINISHED


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn')
    sys.exit(main())
