import os
import json
from model.net import Net
from model.decoder import Decoder
from model.losses.loss_keypoints import LossKeypoints
from model.losses.loss_links import LossLinks
from builder import get_optimizer_from_arguments, get_lr_scheduler_from_arguments, get_accelerator_device_from_args
from dataset import get_dataloaders
from trainer import Trainer
from torch.utils.tensorboard import SummaryWriter

def load(ROOT_PATH = 'drive/Shareddrives/DLAV'):
    setup_file_name ='dlav_config.json'
    setup_file = open(setup_file_name)
    config = json.load(setup_file)
    setup_file.close()
    
    DATA_PATH = os.path.join(ROOT_PATH, config['dataset']['data_path'])
    model = Net(config)
    loss_keypoints = LossKeypoints(cost_class = config['training']['loss_keypoints']['cost_class'], cost_bbox = config['training']['loss_keypoints']['cost_bbox'])
    loss_links = LossLinks(cost_class = config['training']['loss_links']['cost_class'], cost_bbox = config['training']['loss_links']['cost_bbox'])
    optimizer = get_optimizer_from_arguments(config, model.parameters())
    lr_scheduler = get_lr_scheduler_from_arguments(config, optimizer)
    device = get_accelerator_device_from_args(config)
    train_loader, val_loader, _ = get_dataloaders(config, DATA_PATH)
    writer = SummaryWriter(os.path.join(ROOT_PATH, config['logging']['log_dir']))
    decoder = Decoder(config['decoder']['threshold'], config['decoder']['max_distance'])
    
    model.to(device)
    return model, decoder, loss_keypoints, loss_links, optimizer, lr_scheduler, config, device, train_loader, val_loader, writer

def train(model, decoder, loss_keypoints, loss_links, optimizer, lr_scheduler, config, device, train_loader, val_loader, writer, ROOT_PATH = 'drive/Shareddrives/DLAV'):
    trainer = Trainer(model, decoder, loss_keypoints, loss_links, optimizer, lr_scheduler, config['training']['clip_grad_value'], device)
    trainer.train(train_loader, val_loader, writer = writer, epoch = config['training']['max_epochs'], PATH = os.path.join(ROOT_PATH, config['logging']['weight_dir']))

def main(ROOT_PATH = 'drive/Shareddrives/DLAV'):
    model, decoder, loss_keypoints, loss_links, optimizer, lr_scheduler, config, device, train_loader, val_loader, writer = load(ROOT_PATH)
    train(model, decoder, loss_keypoints, loss_links, optimizer, lr_scheduler, config, device, train_loader, val_loader, writer, ROOT_PATH)
    writer.close()

if __name__ == '__main__' :
    main()  