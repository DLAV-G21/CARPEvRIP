import os
import sys
import json
from model.net import Net
from model.decoder import Decoder
from model.losses.loss_keypoints import LossKeypoints
from model.losses.loss_links import LossLinks
from builder import get_optimizer_from_arguments, get_lr_scheduler_from_arguments, get_accelerator_device_from_args
from dataset import get_dataloaders
from trainer import Trainer
from torch.utils.tensorboard import SummaryWriter

def load(ROOT_PATH = '/home/plumey'):
    setup_file_name ='dlav_config.json'
    setup_file = open(setup_file_name)
    config = json.load(setup_file)
    setup_file.close()

    config['model']['pretrained'] = os.path.join(ROOT_PATH, config['model']['pretrained'])
    config['logging']['log_dir'] = os.path.join(ROOT_PATH, config['logging']['log_dir'])
    config['logging']['weight_dir'] = os.path.join(ROOT_PATH, config['logging']['weight_dir'])
    
    DATA_PATH = os.path.join(ROOT_PATH, config['dataset']['data_path'])
    model = Net(config)
    loss_keypoints = LossKeypoints(cost_class = config['training']['loss_keypoints']['cost_class'], cost_bbox = config['training']['loss_keypoints']['cost_bbox'], max_distance = config['training']['loss_keypoints']['max_distance'])
    loss_links = LossLinks(cost_class = config['training']['loss_links']['cost_class'], cost_bbox = config['training']['loss_links']['cost_bbox'], max_distance = config['training']['loss_links']['max_distance'])
    optimizer = get_optimizer_from_arguments(config, model.parameters())
    lr_scheduler = get_lr_scheduler_from_arguments(config, optimizer)
    device = get_accelerator_device_from_args(config)
    train_loader, val_loader, _ = get_dataloaders(config, DATA_PATH)
    writer = SummaryWriter(config['logging']['log_dir'])
    decoder = Decoder(config['decoder']['threshold'], config['decoder']['max_distance'])
    
    model.to(device)
    return model, decoder, loss_keypoints, loss_links, optimizer, lr_scheduler, config, device, train_loader, val_loader, writer

def train(model, decoder, loss_keypoints, loss_links, optimizer, lr_scheduler, config, device, train_loader, val_loader, writer, ROOT_PATH = 'drive/Shareddrives/DLAV'):
    trainer = Trainer(model, decoder, loss_keypoints, loss_links, optimizer, lr_scheduler, config['training']['clip_grad_value'], device)
    trainer.train(train_loader, val_loader, writer = writer, epoch = config['training']['epochs'], PATH = config['logging']['weight_dir'])

def main(ROOT_PATH = '/home/plumey'):
    try:
        model, decoder, loss_keypoints, loss_links, optimizer, lr_scheduler, config, device, train_loader, val_loader, writer = load(ROOT_PATH)
        train(model, decoder, loss_keypoints, loss_links, optimizer, lr_scheduler, config, device, train_loader, val_loader, writer, ROOT_PATH)
        writer.close()
    except Exception as e:
        f = open(os.path.join(ROOT_PATH, 'error.log'), 'w')
        f.write('Failed :\n'+ str(sys.exc_info()))
        f.close()

if __name__ == '__main__' :
    main()