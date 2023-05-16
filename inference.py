import argparse
import os
import sys
import json
from model.net import Net
from model.decoder import Decoder
from builder import get_accelerator_device_from_args
from dataset import load_dataloader_inference
from utils.visualizations import plot_and_save_keypoints_inference
from trainer import Trainer

def load(ROOT_PATH = '/home/plumey', setup_file_name ='dlav_config.json', image = 'images', anotation = None):
    if not os.path.isfile(setup_file_name):
        raise ValueError('config file does\'t exist :', setup_file_name)
    with open(setup_file_name) as setup_file:
        config = json.load(setup_file)
        
    config['model']['pretrained'] = os.path.join(ROOT_PATH, config['model']['pretrained'])
    config['model']['backbone_save'] = os.path.join(ROOT_PATH, config['model']['backbone_save'])
    config['model']['model_saves'] = os.path.join(ROOT_PATH, config['model']['model_saves'])
    config['logging']['log_dir'] = os.path.join(ROOT_PATH, config['logging']['log_dir'])
    config['logging']['weight_dir'] = os.path.join(ROOT_PATH, config['logging']['weight_dir'])

    model = Net(config)
    device = get_accelerator_device_from_args(config)
    val_loader = load_dataloader_inference(config, ROOT_PATH, image, anotation)
    decoder = Decoder(threshold = config['decoder']['threshold'], max_distance = config['decoder']['max_distance'], nbr_max_car = config['dataset']['max_nb'], use_matcher = config['model']['use_matcher'], nb_keypoints = config['dataset']['nb_keypoints'])

    model.to(device)
    trainer = Trainer(model, decoder, None, None, None, None, None, device, None, val_loader, None)

    return trainer, config

def to_json(results):
    out = {}
    for skelston in results:
        if(skelston['num_keypoints'] > 0):
            if(str(skelston['image_id']) in out):
                out[str(skelston['image_id'])].append(skelston)
            else:
                out[str(skelston['image_id'])] = [skelston]
        else:
            out[str(skelston['image_id'])] = []
    return out

def save_json(results, json_out):
    with open(json_out, "w") as outfile:
        outfile.write(json.dumps(results))

def save_img(results, ROOT_PATH, image, img_out, config):
    plot_and_save_keypoints_inference(
        os.path.join(ROOT_PATH, image),
        results,
        os.path.join(ROOT_PATH, img_out),
		[1, 1] if config['dataset']['normalize_position'] else config['dataset']['input_size']
    )

def main(ROOT_PATH, image, setup_file_name, json_out, img_out):
    try:
        trainer, config = load(ROOT_PATH, setup_file_name, image)
        results = trainer.eval()
        results = to_json(results)
        if(json_out is not None):
            save_json(results, json_out)
        if(img_out is not None):
            if not os.path.isdir(img_out):
                os.makedirs(img_out)
            save_img(results, ROOT_PATH, image, img_out, config)
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
    parser.add_argument("-a", "--anotation", help="path to the anotation file", default=None, required=False)
    args = parser.parse_args()
    main(args.root, args.image, args.config, args.json, args.out, args.anotation)