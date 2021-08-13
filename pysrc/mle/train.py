from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import datetime
import argparse
import yaml
from shutil import copyfile

import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import sys
sys.path.append('../')
from common import trainer_mod
from common import make_datalist_mod
from common import data_transform_mod
from common import dataset_mod
from common import network_mod
import criterion_mod


if __name__ == '__main__':
    parser = argparse.ArgumentParser("./train.py")

    parser.add_argument(
        '--train_cfg', '-c',
        type=str,
        required=False,
        default="../../pyyaml/mle_train_config.yaml",
        help='Train hyperparameter config file',
    )

    FLAGS, unparsed = parser.parse_known_args()

    #Load yaml file
    try:
        print("Opening train config file %s", FLAGS.train_cfg)
        CFG = yaml.safe_load(open(FLAGS.train_cfg, 'r'))
    except Exception as e:
        print(e)
        print("Error opening train config file %s", FLAGS.train_cfg)
        quit()

    method_name = CFG["method_name"]
    save_top_path = CFG["save_top_path"]
    csv_name = CFG["csv_name"]
    multiGPU = int(CFG["multiGPU"])

    train_sequences = CFG["train"]
    valid_sequences = CFG["valid"]

    #get hyperparameter for learning
    original_size = CFG["hyperparameter"]["original_size"]
    resize = CFG["hyperparameter"]["resize"]
    mean_element = CFG["hyperparameter"]["mean_element"]
    std_element = CFG["hyperparameter"]["std_element"]
    hor_fov_deg = CFG["hyperparameter"]["hor_fov_deg"]
    optimizer_name = CFG["hyperparameter"]["optimizer_name"]
    lr_cnn = float(CFG["hyperparameter"]["lr_cnn"])
    lr_fc = float(CFG["hyperparameter"]["lr_fc"])
    batch_size = CFG["hyperparameter"]["batch_size"]
    num_epochs = CFG["hyperparameter"]["num_epochs"]
    dropout_rate = float(CFG["hyperparameter"]["dropout_rate"])
    dim_fc_out = int(CFG["hyperparameter"]["dim_fc_out"])

    try:
        print("Copy files to %s for further reference." % save_top_path)
        copyfile(FLAGS.train_cfg, save_top_path + "/train_config.yaml")
    except Exception as e:
        print(e)
        print("Error copying files, check permissions. Exiting....")
        quit()

    train_dataset = dataset_mod.Originaldataset(
        data_list = make_datalist_mod.makeMultiDataList(train_sequences, csv_name),
        transform = data_transform_mod.DataTransform(
            resize,
            mean_element,
            std_element,
        ),
        phase = "train"
    )

    valid_dataset = dataset_mod.Originaldataset(
        data_list = make_datalist_mod.makeMultiDataList(valid_sequences, csv_name),
        transform = data_transform_mod.DataTransform(
            resize,
            mean_element,
            std_element,
        ),
        phase = "valid"
    )

    net = network_mod.Network(resize, list_dim_fc_out=9, dropout_rate=0.1, use_pretrained_vgg=True)
    criterion = criterion_mod.Criterion()
    trainer = trainer_mod.Trainer(
        method_name,
        train_dataset,
        valid_dataset,
        net,
        criterion,
        optimizer_name,
        lr_cnn,
        lr_fc,
        batch_size,
        num_epochs,
        save_top_path,
        multiGPU
    )

    trainer.train()