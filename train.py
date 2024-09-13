import numpy as np
import os
import pandas as pd
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from data.data_utils import *
from data.dataloader_detection import load_dataset_detection
from data.dataloader_classification import load_dataset_classification
from constants import *

# from args import get_args
from collections import OrderedDict
from json import dumps
from model.model import DCRNNModel_DoubleEncoder
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import config


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    utils.seed_torch()

    def ensure_folder_exists(folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    ensure_folder_exists(config.save_dir)

    log = utils.get_logger(config.save_dir, "train")
    tbx = SummaryWriter(config.save_dir)

    with open("config.py", "r") as file:
        file_content = file.read()

    log.info("Configs: {}".format(file_content))

    # Build dataset
    log.info("Building dataset...")
    if config.task == "detection":
        dataloaders, _, scaler = load_dataset_detection(
            train_batch_size=config.train_batch_size,
            test_batch_size=config.test_batch_size,
            time_step_size=config.time_step,
            max_seq_len=config.seq_len,
            standardize=False,
            num_workers=config.num_workers,
            adj_mat_dir=config.global_graph_mat,
            use_fft=True,
            preproc_dir=config.clip_dir,
        )
    elif config.task == "classification":
        dataloaders, _, scaler = load_dataset_classification(
            input_dir=config.input_dir,
            raw_data_dir=config.raw_data_dir,
            train_batch_size=config.train_batch_size,
            test_batch_size=config.test_batch_size,
            time_step_size=config.time_step_size,
            max_seq_len=config.max_seq_len,
            standardize=False,
            num_workers=config.num_workers,
            padding_val=0.0,
            augmentation=config.data_augment,
            adj_mat_dir=r"F:\Train_preprocess\adj_mx_3d_cor.pkl",
            graph_type=config.graph_type,
            top_k=config.top_k,
            filter_type=config.filter_type,
            use_fft=config.use_fft,
            preproc_dir=config.preproc_dir,
        )
    else:
        raise NotImplementedError

    # Build model
    log.info("Building model...")
    model = DCRNNModel_DoubleEncoder(device=device)

    if config.is_train:
        num_params = utils.count_parameters(model)
        log.info("Total number of trainable parameters: {}".format(num_params))

        model = model.to(device)

        # Train
        train(model, dataloaders, device, config.save_dir, log, tbx)

        # Load best model after training finished
        best_path = os.path.join(config.save_dir, "best.pth.tar")
        model = utils.load_model_checkpoint(best_path, model)
        model = model.to(device)

    # Evaluate on dev and test set
    log.info("Training DONE. Evaluating model...")
    dev_results = evaluate(
        model,
        dataloaders["dev"],
        device,
        is_test=True,
        nll_meter=None,
        eval_set="dev",
    )

    dev_results_str = ", ".join(
        "{}: {:.3f}".format(k, v) for k, v in dev_results.items()
    )
    log.info("DEV set prediction results: {}".format(dev_results_str))

    test_results = evaluate(
        model,
        dataloaders["test"],
        device,
        is_test=True,
        nll_meter=None,
        eval_set="test",
        best_thresh=dev_results["best_thresh"],
        save=True,
    )

    # Log to console
    test_results_str = ", ".join(
        "{}: {:.3f}".format(k, v) for k, v in test_results.items()
    )
    log.info("TEST set prediction results: {}".format(test_results_str))


def train(model, dataloaders, device, save_dir, log, tbx):
    """
    Perform training and evaluate on val set
    """

    # Define loss function
    if config.task == "detection":
        loss_fn = nn.BCEWithLogitsLoss().to(device)
    else:
        loss_fn = nn.CrossEntropyLoss().to(device)

    # Data loaders
    train_loader = dataloaders["train"]
    dev_loader = dataloaders["dev"]

    # Get saver
    saver = utils.CheckpointSaver(
        save_dir,
        metric_name=config.metric,
        maximize_metric=True if config.metric in ("f1", "acc", "auroc") else False,
        log=log,
    )

    # To train mode
    model.train()

    # Get optimizer and scheduler
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=config.init_lr,
        weight_decay=config.l2_weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    # average meter for validation loss
    nll_meter = utils.AverageMeter()

    # Train
    log.info("Training...")
    epoch = 0
    step = 0
    prev_val_loss = 1e10
    patience_count = 0
    early_stop = False
    while (epoch != config.num_epochs) and (not early_stop):
        epoch += 1
        log.info("Starting epoch {}...".format(epoch))
        total_samples = len(train_loader.dataset)
        with torch.enable_grad(), tqdm(total=total_samples) as progress_bar:
            for x, y, seq_lengths, supports1, supports2, _ in train_loader:
                batch_size = x.shape[0]

                # input seqs
                x = x.to(device)
                y = y.view(-1).to(device)  # (batch_size,)
                y = y.float()  # 当二分时BCE要求float
                seq_lengths = seq_lengths.view(-1).to(device)  # (batch_size,)
                for i in range(len(supports1)):
                    supports1[i] = supports1[i].to(device)
                for i in range(len(supports2)):
                    supports2[i] = supports2[i].to(device)

                # Zero out optimizer first
                optimizer.zero_grad()

                # Forward
                # (batch_size, num_classes)
                logits = model(x, seq_lengths, supports1, supports2)

                if logits.shape[-1] == 1:
                    logits = logits.view(-1)  # (batch_size,)
                loss = loss_fn(logits, y)
                loss_val = loss.item()

                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                step += batch_size

                # Log info
                progress_bar.update(batch_size)
                progress_bar.set_postfix(
                    epoch=epoch, loss=loss_val, lr=optimizer.param_groups[0]["lr"]
                )

                tbx.add_scalar("train/Loss", loss_val, step)
                tbx.add_scalar("train/LR", optimizer.param_groups[0]["lr"], step)

            if epoch % config.eval_interval == 0:
                # Evaluate and save checkpoint
                log.info("Evaluating at epoch {}...".format(epoch))
                eval_results = evaluate(
                    model,
                    dev_loader,
                    device,
                )
                best_path = saver.save(
                    epoch, model, optimizer, eval_results[config.metric]
                )

                # Accumulate patience for early stopping
                if eval_results["loss"] < prev_val_loss:
                    patience_count = 0
                else:
                    patience_count += 1
                prev_val_loss = eval_results["loss"]

                # Early stop
                if patience_count == config.early_stopping_patience:
                    early_stop = True

                # Back to train mode
                model.train()

                # Log to console
                results_str = ", ".join(
                    "{}: {:.3f}".format(k, v) for k, v in eval_results.items()
                )
                log.info("Dev {}".format(results_str))

                # Log to TensorBoard
                log.info("Visualizing in TensorBoard...")
                for k, v in eval_results.items():
                    tbx.add_scalar("eval/{}".format(k), v, step)

        # Step lr scheduler
        scheduler.step()


def evaluate(
    model,
    dataloader,
    device,
    is_test=False,
    nll_meter=None,
    eval_set="dev",
    best_thresh=0.5,
    save=False,
):
    # To evaluate mode
    model.eval()

    # Define loss function
    if config.task == "detection":
        loss_fn = nn.BCEWithLogitsLoss().to(device)
    else:
        loss_fn = nn.CrossEntropyLoss().to(device)

    y_pred_all = []
    y_true_all = []
    y_prob_all = []
    file_name_all = []
    with torch.no_grad(), tqdm(total=len(dataloader.dataset)) as progress_bar:
        for x, y, seq_lengths, supports1, supports2, file_name in dataloader:
            batch_size = x.shape[0]

            # Input seqs
            x = x.to(device)
            y = y.view(-1).to(device)  # (batch_size,)
            y = y.float()  # 当二分时BCE要求float
            seq_lengths = seq_lengths.view(-1).to(device)  # (batch_size,)
            for i in range(len(supports1)):
                supports1[i] = supports1[i].to(device)
            for i in range(len(supports2)):
                supports2[i] = supports2[i].to(device)

            # Forward
            # (batch_size, num_classes)
            logits = model(x, seq_lengths, supports1, supports2)

            if config.classes == 1:  # binary detection
                logits = logits.view(-1)  # (batch_size,)
                y_prob = torch.sigmoid(logits).cpu().numpy()  # (batch_size, )
                y_true = y.cpu().numpy().astype(int)
                y_pred = (y_prob > best_thresh).astype(int)  # (batch_size, )
            else:
                # (batch_size, num_classes)
                y_prob = F.softmax(logits, dim=1).cpu().numpy()
                y_pred = np.argmax(y_prob, axis=1).reshape(-1)  # (batch_size,)
                y_true = y.cpu().numpy().astype(int)

            # Update loss
            loss = loss_fn(logits, y)
            if nll_meter is not None:
                nll_meter.update(loss.item(), batch_size)

            y_pred_all.append(y_pred)
            y_true_all.append(y_true)
            y_prob_all.append(y_prob)
            file_name_all.extend(file_name)

            # Log info
            progress_bar.update(batch_size)

    y_pred_all = np.concatenate(y_pred_all, axis=0)
    y_true_all = np.concatenate(y_true_all, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)

    if save:
        # 将数据转换为 DataFrame
        data = {"y_pred": y_pred_all, "y_true": y_true_all, "y_prob": y_prob_all}
        df = pd.DataFrame(data)

        # 保存 DataFrame 到 CSV 文件
        df.to_csv(config.save_dir + "/y_all.csv", index=False)

    # Threshold search, for detection only
    if (config.task == "detection") and (eval_set == "dev") and is_test:
        best_thresh = utils.thresh_max_f1(y_true=y_true_all, y_prob=y_prob_all)
        # update dev set y_pred based on best_thresh
        y_pred_all = (y_prob_all > best_thresh).astype(int)  # (batch_size, )
    else:
        best_thresh = best_thresh

    scores_dict, _, _ = utils.eval_dict(
        y_pred=y_pred_all,
        y=y_true_all,
        y_prob=y_prob_all,
        file_names=file_name_all,
        average="binary" if config.task == "detection" else "weighted",
    )

    eval_loss = nll_meter.avg if (nll_meter is not None) else loss.item()
    results_list = [
        ("loss", eval_loss),
        ("acc", scores_dict["acc"]),
        ("F1", scores_dict["F1"]),
        ("recall", scores_dict["recall"]),
        ("precision", scores_dict["precision"]),
        ("best_thresh", best_thresh),
    ]
    if "auroc" in scores_dict.keys():
        results_list.append(("auroc", scores_dict["auroc"]))
    results = OrderedDict(results_list)

    return results


if __name__ == "__main__":
    main()
