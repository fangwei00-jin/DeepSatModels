from doctest import debug
import sys
import os
sys.path.insert(0, os.getcwd())
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utils.lr_scheduler import build_scheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from models import get_model
from utils.config_files_utils import read_yaml, copy_yaml, get_params_values
from utils.torch_utils import get_device, get_net_trainable_params, load_from_checkpoint
from data import get_dataloaders
from sklearn.metrics import confusion_matrix
from metrics.torch_metrics import get_mean_metrics
from metrics.numpy_metrics import get_classification_metrics, get_per_class_loss
from metrics.loss_functions import get_loss
from utils.summaries import write_mean_summaries, write_class_summaries
from data import get_loss_data_input

from loguru import logger, _defaults
import time

def save_netparam(net, path, devices=[0]):
    if len(devices) > 1:
        torch.save(net.module.state_dict(), path)
    else:
        torch.save(net.state_dict(), path)

def train_and_evaluate(net, dataloaders, config, device, lin_cls=False):

    def train_step(net, sample, loss_fn, optimizer, device, loss_input_fn):
        optimizer.zero_grad()
        # logger.info(sample['inputs'].shape)
        outputs = net(sample['inputs'].to(device))
        outputs = outputs.permute(0, 2, 3, 1)
        ground_truth = loss_input_fn(sample, device)
        loss = loss_fn['mean'](outputs, ground_truth)
        loss.backward()
        optimizer.step()
        return outputs, ground_truth, loss

    def evaluate(net, evalloader, loss_fn, config):
        num_classes = config['MODEL']['num_classes']
        predicted_all = []
        labels_all = []
        losses_all = []
        net.eval()
        with torch.no_grad():
            for step, sample in enumerate(evalloader):
                logits = net(sample['inputs'].to(device))
                logits = logits.permute(0, 2, 3, 1)
                _, predicted = torch.max(logits.data, -1)
                ground_truth = loss_input_fn(sample, device)
                loss = loss_fn['all'](logits, ground_truth)
                target, mask = ground_truth
                if mask is not None:
                    predicted_all.append(predicted.view(-1)[mask.view(-1)].cpu().numpy())
                    labels_all.append(target.view(-1)[mask.view(-1)].cpu().numpy())
                else:
                    predicted_all.append(predicted.view(-1).cpu().numpy())
                    labels_all.append(target.view(-1).cpu().numpy())
                losses_all.append(loss.view(-1).cpu().detach().numpy())

                # if step > 5:
                #    break

        logger.info("finished iterating over dataset after step %d" % step)
        logger.info("calculating metrics...")
        predicted_classes = np.concatenate(predicted_all)
        target_classes = np.concatenate(labels_all)
        losses = np.concatenate(losses_all)

        eval_metrics = get_classification_metrics(predicted=predicted_classes, labels=target_classes,
                                                  n_classes=num_classes, unk_masks=None)
        cm = confusion_matrix(target_classes, predicted_classes)
        logger.info(f'confusion matrix: shape:{cm.shape}\n{cm.astype(int)}')

        micro_acc, micro_precision, micro_recall, micro_F1, micro_IOU = eval_metrics['micro']
        macro_acc, macro_precision, macro_recall, macro_F1, macro_IOU = eval_metrics['macro']
        class_acc, class_precision, class_recall, class_F1, class_IOU = eval_metrics['class']

        un_labels, class_loss = get_per_class_loss(losses, target_classes, unk_masks=None)
        logger.info("Mean (micro) Evaluation metrics (micro/macro), loss: %.7f, iou: %.4f/%.4f, accuracy: %.4f/%.4f, "
              "precision: %.4f/%.4f, recall: %.4f/%.4f, F1: %.4f/%.4f, unique pred labels: %s" %
              (losses.mean(), micro_IOU, macro_IOU, micro_acc, macro_acc, micro_precision, macro_precision,
               micro_recall, macro_recall, micro_F1, macro_F1, np.unique(predicted_classes)))
        return (un_labels,
                {"macro": {"Loss": losses.mean(), "Accuracy": macro_acc, "Precision": macro_precision,
                           "Recall": macro_recall, "F1": macro_F1, "IOU": macro_IOU},
                 "micro": {"Loss": losses.mean(), "Accuracy": micro_acc, "Precision": micro_precision,
                           "Recall": micro_recall, "F1": micro_F1, "IOU": micro_IOU},
                 "class": {"Loss": class_loss, "Accuracy": class_acc, "Precision": class_precision,
                           "Recall": class_recall,
                           "F1": class_F1, "IOU": class_IOU}}
                )

    #------------------------------------------------------------------------------------------------------------------#
    num_classes = config['MODEL']['num_classes']
    num_epochs = config['SOLVER']['num_epochs']
    lr = float(config['SOLVER']['lr_base'])
    train_metrics_steps = config['CHECKPOINT']['train_metrics_steps']
    eval_steps = config['CHECKPOINT']['eval_steps']
    save_steps = config['CHECKPOINT']["save_steps"]
    save_path = config['CHECKPOINT']["save_path"]
    checkpoint = config['CHECKPOINT']["load_from_checkpoint"]
    num_steps_train = len(dataloaders['train'])
    local_device_ids = config['local_device_ids']
    weight_decay = get_params_values(config['SOLVER'], "weight_decay", 0)

    start_global = 1
    start_epoch = 1
    if checkpoint:
        load_from_checkpoint(net, checkpoint, partial_restore=False)

    logger.info("current learn rate: ", lr)

    if len(local_device_ids) > 1:
        net = nn.DataParallel(net, device_ids=local_device_ids)
    net.to(device)

    if save_path and (not os.path.exists(save_path)):
        os.makedirs(save_path)
    dt_info = time.strftime('%y%m%d_%H%M', time.localtime())
    save_path += f'/{dt_info}'
    config['CHECKPOINT']["save_path"] = save_path
    logger.info(f'savepath: {save_path}')
    if not config['debug']:
        writer = SummaryWriter(f'{save_path}/avg')
        writer_cls = SummaryWriter(f'{save_path}/cls')

    if not config['debug']:
        copy_yaml(config)

    loss_input_fn = get_loss_data_input(config)

    loss_fn = {'all': get_loss(config, device, reduction=None),
               'mean': get_loss(config, device, reduction="mean")}

    trainable_params = get_net_trainable_params(net)
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    optimizer.zero_grad()

    scheduler = build_scheduler(config, optimizer, num_steps_train)

    BEST_IOU = 0
    net.train()
    for epoch in range(start_epoch, start_epoch + num_epochs):  # loop over the dataset multiple times
        for step, sample in enumerate(dataloaders['train']):
            abs_step = start_global + (epoch - start_epoch) * num_steps_train + step
            train_start = time.time()
            logits, ground_truth, loss = train_step(net, sample, loss_fn, optimizer, device, loss_input_fn=loss_input_fn)
            if len(ground_truth) == 2:
                labels, unk_masks = ground_truth
            else:
                labels = ground_truth
                unk_masks = None
            # logger.info batch statistics ----------------------------------------------------------------------------------#
            if abs_step % train_metrics_steps == 0:
                logits = logits.permute(0, 3, 1, 2)
                batch_metrics = get_mean_metrics(
                    logits=logits, labels=labels, unk_masks=unk_masks, n_classes=num_classes, loss=loss, epoch=epoch,
                    step=step)
                if not config['debug']:
                    write_mean_summaries(writer, batch_metrics, abs_step, mode="train", optimizer=optimizer)
                logger.info(f"epoch: {epoch}, abs_step: {abs_step:>5}, step:{step + 1:>4}, step time:{time.time()-train_start:.2f} | "
                    f"loss:{loss:.2}, iou:{batch_metrics['IOU']:.4f}, accuracy: {batch_metrics['Accuracy']:.4f}, "
                    f"precision:{batch_metrics['Precision']:.4f}, recall: {batch_metrics['Recall']:.4f}, F1:{batch_metrics['F1']:.4f}")

            if abs_step % save_steps == 0 and not config['debug']:
                save_netparam(net, f'{save_path}/{epoch}epoch_{abs_step}step.pth', local_device_ids)

            # evaluate model ------------------------------------------------------------------------------------------#
            if abs_step % eval_steps == 0:
                eval_metrics = evaluate(net, dataloaders['eval'], loss_fn, config)
                if eval_metrics[1]['macro']['IOU'] > BEST_IOU:
                    if not config['debug']:
                        save_netparam(net, f'{save_path}/best.pth', local_device_ids)
                    BEST_IOU = eval_metrics[1]['macro']['IOU']
                logger.info(f'evaluation BEST_IOU: {BEST_IOU:.5f} \n{" "*20}current all_metrics {eval_metrics[1]["macro"]}')

                if not config['debug']:
                    write_mean_summaries(writer, eval_metrics[1]['micro'], abs_step, mode="eval_micro", optimizer=None)
                    write_mean_summaries(writer, eval_metrics[1]['macro'], abs_step, mode="eval_macro", optimizer=None)
                    write_class_summaries(writer_cls, [eval_metrics[0], eval_metrics[1]['class']], abs_step, mode="eval",
                                        optimizer=None)
                net.train()

        scheduler.step_update(abs_step)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--config_file', help='configuration (.yaml) file to use')
    parser.add_argument('--gpu_ids', default='0,1', type=str,
                         help='gpu ids to use')
    parser.add_argument('--lin', action='store_true',
                         help='train linear classifier only')
    parser.add_argument('--debug', action='store_true', help='while debuggind, do not save logs and params.')
    args = parser.parse_args()

    print(args)

    logger.remove() # remove the default log format
    if args.debug:
        fmt = "<green>{time:MM-DD HH:mm:ss}</green> | <cyan><b>{level: <5}</b></cyan>|<b>{line: >3}</b> {file} {function} | {message}"
    else:
        fmt = "<green>{time:MM-DD HH:mm:ss}</green> | <cyan><b>{level: <5}</b></cyan>| {message}"
    logfilter = lambda r : r['level'].no >= _defaults.LOGURU_DEBUG_NO if args.debug else _defaults.LOGURU_INFO_NO
    stdlogger = logger.add(sys.stdout, format=fmt)
    # logsink = logger.add('./runlog/{time}.log', format=fmt)

    config_file = args.config_file
    logger.debug(args.gpu_ids)
    device_ids = [int(d) for d in args.gpu_ids.split(',')]
    device = get_device(device_ids, allow_cpu=False)
    lin_cls = args.lin

    config = read_yaml(config_file)
    config['local_device_ids'] = device_ids
    config['debug'] = args.debug

    dataloaders = get_dataloaders(config)

    net = get_model(config, device)
    logger.info(f'net: {net}')

    train_and_evaluate(net, dataloaders, config, device)
