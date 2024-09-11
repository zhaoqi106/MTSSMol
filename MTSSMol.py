import os
import shutil
import sys
import torch
import yaml
import numpy as np
from datetime import datetime

import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

# from utils.nt_xent import NTXentLoss
# from apex import amp

# apex_support = False
# try:
#     sys.path.append('./apex')
#     from apex import amp
#
#     apex_support = True
# except:
#     print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
#     apex_support = False


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))


class MTSSMol(object):
    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        
        dir_name = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join('ckpt', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.dataset = dataset
        # self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['loss'])

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def _step(self, model, data, data_mask, n_iter):
        x, pre_class_label1, pre_class_label2, pre_class_label3 = model(data)

        class_loss1 = self.criterion(pre_class_label1, data.y1)
        class_loss2 = self.criterion(pre_class_label2, data.y2)
        class_loss3 = self.criterion(pre_class_label3, data.y3)
        class_loss = class_loss1 + class_loss2 + class_loss3

        x1, _, _, _ = model(data_mask)
        mask_loss = (x - x1).pow(2).sum(axis=1).sqrt().mean()
        loss = class_loss + mask_loss
        # loss = self.criterion(pred, data.y.flatten())
        return loss

    def train(self):
        train_loader, valid_loader = self.dataset.get_data_loaders()

        if self.config['model_type'] == 'gin':
            from models.ginet_mtssmol import GINet
            model = GINet(**self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)

        else:
            raise ValueError('Undefined GNN model.')
        print(model)
        
        optimizer = torch.optim.Adam(
            model.parameters(), self.config['init_lr'], 
            weight_decay=eval(self.config['weight_decay'])
        )
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.config['epochs']-self.config['warm_up'], 
            eta_min=0, last_epoch=-1
        )

        # if apex_support and self.config['fp16_precision']:
        #     model, optimizer = amp.initialize(
        #         model, optimizer, opt_level='O2', keep_batchnorm_fp32=True
        #     )

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        for epoch_counter in range(self.config['epochs']):
            for bn, (data, data_mask )in enumerate(train_loader):
                optimizer.zero_grad()

                data = data.to(self.device)
                data_mask = data_mask.to(self.device)
                loss = self._step(model, data, data_mask, n_iter)

                # if n_iter % self.config['log_every_n_steps'] == 0:
                self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                self.writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr()[0], global_step=n_iter)
                print(epoch_counter, bn, loss.item())

                # if apex_support and self.config['fp16_precision']:
                #     with amp.scale_loss(loss, optimizer) as scaled_loss:
                #         scaled_loss.backward()
                # else:
                #     loss.backward()
                loss.backward()

                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader)
                print(epoch_counter, bn, valid_loss, '(validation)')
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
            
                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1
            
            if (epoch_counter+1) % self.config['save_every_n_epochs'] == 0:
                torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model_{}.pth'.format(str(epoch_counter))))

            # warmup for the first few epochs
            if epoch_counter >= self.config['warm_up']:
                scheduler.step()

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./ckpt', self.config['load_model'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):
        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0
            for (data, data_mask) in valid_loader:
                data = data.to(self.device)
                data_mask = data_mask.to(self.device)

                loss = self._step(model, data,data_mask, counter)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        
        model.train()
        return valid_loss


def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    print(config)

    if config['aug'] == 'node':
        from dataset.dataset import MoleculeDatasetWrapper
    elif config['aug'] == 'subgraph':
        from dataset.dataset_subgraph import MoleculeDatasetWrapper
    # elif config['aug'] == 'mix':
    #     from dataset.dataset_mix import MoleculeDatasetWrapper
    else:
        raise ValueError('Not defined molecule augmentation!')

    dataset = MoleculeDatasetWrapper(config['batch_size'], **config['dataset'])
    mtssmol = MTSSMol(dataset, config)
    mtssmol.train()


if __name__ == "__main__":
    main()



