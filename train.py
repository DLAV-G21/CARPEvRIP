import os
from model.net import Net
from model.losses.loss_keypoints import LossKeypoints
from model.losses.loss_links import LossLinks
from builder import *
from dataset import get_dataloaders
from trainer import Trainer

def main():
    #setup_file_name = 'train.json'
    #setup_file = open()

    #config = json.load(())

    config = {
        'model' : {
            'pretrained' : True,
            'bn_momentum' : 0.1,
            },
        'loss_keypoints' : {
            'cost_class' : 1,
            'cost_bbox' : 1,
            },
        'loss_links' : {
            'cost_class' : 1,
            'cost_bbox' : 1,
            },
        'optimizer' : {
            'name' : 'adamw',
            'learning_rate' : 0.01,
            'betas' : (0.9, 0.999),
            'weight_decay' : 0,
            },
        'lr_scheduler' : {
            'name' : 'step',
            'nb_step' : 10,
            'decay_factor' : 0.1,
            },
        'hardware' : {
            'specific_gpu' : None,
            'num_gpu' : 1,
            },
        'training' : {
            'batch_size' : 16,
            'train_backbone' : False, 
            },
        'dataset' : {
            'data_path' : 16,
            'seed' : 16,
            'nb_keypoints' : 24,
            'nb_keypoints' : 49,
            'annotations_folder' : 16,
            'img_path' : 16,
            'segm_path' : 16,
            'input_size' : 16,
            'max_nb' : 16,
            },
        'data_augmentation' : {
            'use_occlusion_data_augm' : 16,
            'apply_augm' : 16,
            'normalize' : {
                'mean' : 0,
                'std' : 0,
                },
            },
            'prob_occlusion' : 16,
            'prob_blur' : 16,
            'nb_blur_source' : 16,
            'blur_radius' : 16,
        }
    
    params = None
    ROOT_PATH = ''
    DATA_PATH = os.path.join(ROOT_PATH, config['dataset']['data_path'])
    model = Net(config)
    loss_keypoints = LossKeypoints(cost_class = config['loss_keypoints']['cost_class'], cost_bbox = config['loss_keypoints']['cost_bbox'])
    loss_links = LossLinks(cost_class = config['loss_links']['cost_class'], cost_bbox = config['loss_links']['cost_bbox'])
    optimizer = get_optimizer_from_arguments(config, params)
    lr_scheduler = get_lr_scheduler_from_arguments(config, optimizer)
    device = get_accelerator_device_from_args(config)
    train_loader, val_loader, test_loader = get_dataloaders(config, DATA_PATH)

    model.to(device)

    trainer = Trainer(
        model,
        loss_keypoints,
        loss_links,
        optimizer,
        lr_scheduler,
        device
    )

    trainer.train(
        train_loader,
        val_loader,
        epoch = 10
    )

if __name__ == '__main__' :
    main()  