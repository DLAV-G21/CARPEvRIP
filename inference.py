import argparse
import os
import sys
import json
from model.net import Net
from model.decoder import Decoder
from builder import get_accelerator_device_from_args
from dataset import get_dataloader
from trainer import Trainer

def load(ROOT_PATH = '/home/plumey', setup_file_name ='dlav_config.json', images = 'images'):
    if not os.path.isfile(setup_file_name):
        raise ValueError('config file does\'t exist :', setup_file_name)
    setup_file = open(setup_file_name)
    config = json.load(setup_file)
    setup_file.close()
    config['model']['pretrained'] = os.path.join(ROOT_PATH, config['model']['pretrained'])

    model = Net(config)
    device = get_accelerator_device_from_args(config)
    val_loader = get_dataloader(config, images)
    decoder = Decoder(threshold = config['decoder']['threshold'], max_distance = config['decoder']['max_distance'], nbr_max_car = config['dataset']['max_nb'], use_matcher = config['model']['use_matcher'], nb_keypoints = config['dataset']['nb_keypoints'])

    model.to(device)
    trainer = Trainer(model, decoder, None, None, None, None, None, device, None, val_loader, None)

    return trainer, config

def to_json(results):
    out = {}
    for skelston in results:
        if(skelston['num_keypoints'] > 0):
            if(skelston['image_id'] in out):
                out[skelston['image_id']].append(skelston)
            else:
                out[skelston['image_id']] = [skelston]
        else:
            out[skelston['image_id']] = []
    return out

def save_json(results, json_out):
    

def save_img(results, img_out):

def main(ROOT_PATH, image, setup_file_name, json_out, img_out):
    try:
        trainer, config = load(ROOT_PATH, setup_file_name, image)
        results = trainer.eval()
        results = to_json(results)
        if(json_out is not None):
            save_json(results, json_out)
        if(img_out is not None):
            if not os.isdir(img_out):
                os.makedirs(img_out)
            save_img(results, img_out)
    except:
        f = open(os.path.join(ROOT_PATH, 'error.log'), 'w')
        f.write('Failed :\n'+ str(sys.exc_info()))
        f.close()




if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="path to the root of the project", required=True)
    parser.add_argument("image", help="path to images directory", required=True)
    parser.add_argument("--config", help="path to the config file", default="dlav_config.json", required=False)
    parser.add_argument("-j", "--json", help="path to the output file", default=None, required=False)
    parser.add_argument("-o", "--out", help="path to the output images directory", default=None, required=False)
    args = parser.parse_args()
    main(args.root, args.image, args.config, args.json, args.out)