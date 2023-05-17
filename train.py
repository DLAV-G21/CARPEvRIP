import argparse
import os
import sys
import json
from datetime import datetime
from model.net import Net
from model.decoder import Decoder
from model.losses.loss_keypoints import LossKeypoints
from builder import get_optimizer_from_arguments, get_lr_scheduler_from_arguments, get_accelerator_device_from_args
from dataset import get_dataloaders
from trainer import Trainer
from torch.utils.tensorboard import SummaryWriter

def load(ROOT_PATH = '/home/plumey', setup_file_name ='dlav_config.json', override = False):
    if not os.path.isfile(setup_file_name):
        raise ValueError('config file does\'t exist :', setup_file_name)
    with open(setup_file_name) as setup_file:
        config = json.load(setup_file)

    save = os.path.join(ROOT_PATH, config['model']['model_saves'])
    if not os.path.isdir(save):
        os.makedirs(save)
    save = os.path.join(save, config['name'])

    if (not override) and os.path.isdir(save):
        timestamp = datetime.now().strftime('_%Y_%m_%d_%H_%M_%S')
        save += timestamp
        config['name'] += timestamp
    print(f'name :',config['name'])
    print(f'save :',save)

    if not os.path.isdir(save):
        os.makedirs(save)
    with open(os.path.join(save, 'config.json'), "w") as outfile:
        outfile.write(json.dumps(config))

    config['model']['backbone_save'] = os.path.join(ROOT_PATH, config['model']['backbone_save'])
    config['model']['model_saves'] = os.path.join(ROOT_PATH, config['model']['model_saves'])
    config['logging']['log_dir'] = os.path.join(ROOT_PATH, config['logging']['log_dir'])
    
    DATA_PATH = os.path.join(ROOT_PATH, config['dataset']['data_path'])
    model = Net(config)
    loss_keypoints = LossKeypoints(2, cost_class = config['training']['loss_keypoints']['cost_class'], cost_distance = config['training']['loss_keypoints']['cost_distance'], cost_OKS = config['training']['loss_keypoints']['cost_OKS'], scale_factor = config['training']['loss_keypoints']['scale_factor'], max_distance = config['decoder']['max_distance'], use_matcher = config['model']['use_matcher'])
    loss_links =     LossKeypoints(4, cost_class = config['training']['loss_links']['cost_class']    , cost_distance = config['training']['loss_links']['cost_distance']    , cost_OKS = config['training']['loss_links']['cost_OKS']    , scale_factor = config['training']['loss_links']['scale_factor']    , max_distance = config['decoder']['max_distance'], use_matcher = config['model']['use_matcher'])
    optimizer =      get_optimizer_from_arguments(config, model.parameters())
    lr_scheduler =   get_lr_scheduler_from_arguments(config, optimizer)
    device =         get_accelerator_device_from_args(config)
    train_loader, val_loader, _ = get_dataloaders(config, DATA_PATH)
    writer =  SummaryWriter(config['logging']['log_dir'])
    decoder = Decoder(threshold = config['decoder']['threshold'], max_distance = config['decoder']['max_distance'], nbr_max_car = config['dataset']['max_nb'], use_matcher = config['model']['use_matcher'], nb_keypoints = config['dataset']['nb_keypoints'])
    
    model.to(device)
    trainer = Trainer(save, model, decoder, loss_keypoints, loss_links, optimizer, lr_scheduler, config['training']['clip_grad_value'], device, train_loader, val_loader, writer = writer)

    return trainer, config

def train(trainer, config,eval_only):
    trainer.train(epoch = config['training']['epochs'],eval_only=eval_only)

def main(ROOT_PATH, setup_file_name, override, eval_only):
    """try:
        trainer, config = load(ROOT_PATH, setup_file_name, override)
        train(trainer, config)
        trainer.writer.close()
    except:
        f = open(os.path.join(ROOT_PATH, 'error.log'), 'w')
        f.write('Failed :\n'+ str(sys.exc_info()))
        f.close()
        """
    trainer, config = load(ROOT_PATH, setup_file_name, override)
    train(trainer, config, eval_only=eval_only)
    trainer.writer.close()

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="path to the root of the project")
    parser.add_argument("--config", help="path to the config file", default="dlav_config.json")
    parser.add_argument("-o", "--override", help="override or continue traning", action='store_true')
    parser.add_argument("-e","--eval_only",help="if we only need to perform the eval step",action="store_true")
    args = parser.parse_args()
    main(args.root, args.config, args.override,args.eval_only)