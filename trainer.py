import torch
import time
import os
from tqdm import tqdm
from utils.coco_evaluator import CocoEvaluator

class Trainer():
    epoch = None

    def __init__(
            self,
            model,
            decoder,
            loss_keypoints,
            loss_links,
            optimizer,
            lr_scheduler,
            clip_grad_value,
            device
            ):
        
        self.model = model
        self.decoder = decoder
        self.loss_keypoints = loss_keypoints
        self.loss_links = loss_links
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.clip_grad_value = clip_grad_value
        self.device = device
        
    def train(
            self,
            train_data: torch.utils.data.DataLoader,
            eval_data: torch.utils.data.DataLoader,
            writer,
            epoch=0,
            PATH='model'
            ):
        if not os.path.isdir(PATH):
            os.makedirs(PATH)
        # If the path does not exists, create it
        

        # If the learning rate scheduler is not empty
        if self.lr_scheduler is not None:
            for _ in range(self.model.epoch):
                # Step the learning rate scheduler
                self.lr_scheduler.step()

        # Iterate through the specified number of epochs
        for epoch_ in range(self.model.epoch, self.model.epoch + epoch):
            print('epoch :', epoch_)
            # Perform the training step
            self.train_step(train_data, writer, epoch_)
            # update the epoch value
            self.model.epoch = epoch_ + 1
            # Save the model
            self.model.save_weights(os.path.join(PATH, f'model_{self.model.epoch}.pth'))

            # Create an instance of the CocoEvaluator class to be used for evaluation
            coco_evaluator = CocoEvaluator(eval_data.dataset.coco, ["keypoints"])

            # If the coco evaluator is not empty
            if coco_evaluator is not None:
                # Run the evaluation step
                result = self.eval_step(eval_data, coco_evaluator, writer, epoch_)
                # If the result is better than the best result
                if(self.model.best_result <= result):
                    # Save the model
                    torch.save(self.model.state_dict(), os.path.join(PATH, 'best_result.pth'))

            # If the learning rate scheduler is not empty
            if self.lr_scheduler is not None:
                # Step the learning rate scheduler
                self.lr_scheduler.step()

    @torch.no_grad()
    def eval_step(self, eval_data, coco_evaluator, writer, epoch_):
        self.model.eval()
        epoch_len = len(eval_data)
        skeletons_found = 0
        for i, (image, image_id, _) in tqdm(enumerate(eval_data)):
            if(self.device != 'cpu'):
                image = image.to(self.device, non_blocking=True)
            start_time = time.time()

            predicted_keypoints, predicted_links = self.model(image)
            skeletons = self.decoder((predicted_keypoints, predicted_links), image_id)
            coco_evaluator.update_keypoints(skeletons)
            skeletons_found += len(skeletons)

            end_time = time.time()
            writer.add_scalar('eval iteration time', end_time - start_time, epoch_ * epoch_len + i)
        
        print('skeletons found :',skeletons_found)
        
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        return coco_evaluator.coco_eval['keypoints'].stats[0]

    def train_step(self, train_data, writer, epoch_):
        self.model.train()
        epoch_len = len(train_data)
        for i, (_, image, targeted_keypoints, scale, targeted_links, nb_cars) in tqdm(enumerate(train_data)):
            if(self.device != 'cpu'):
                image = image.to(self.device, non_blocking=True)
                targeted_keypoints = targeted_keypoints.to(self.device, non_blocking=True)
                targeted_links = targeted_links.to(self.device, non_blocking=True)
                scale = scale.to(self.device, non_blocking=True)
                nb_cars = nb_cars.to(self.device, non_blocking=True)
            
            start_time = time.time()
            self.optimizer.zero_grad()

            # predicte
            predicted_keypoints, predicted_links = self.model(image)

            # compute loss
            loss_keypoints = self.loss_keypoints(predicted_keypoints, targeted_keypoints, scale, nb_cars)
            loss_links = self.loss_links(predicted_links, targeted_links, scale, nb_cars)
            loss = loss_keypoints + loss_links
            loss.backward()
                
            # update model
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad_value)
            self.optimizer.step()
            end_time = time.time()

            writer.add_scalar('loss keypoints', loss_keypoints, epoch_ * epoch_len + i)
            writer.add_scalar('loss links', loss_links, epoch_ * epoch_len + i)
            writer.add_scalar('loss', loss, epoch_ * epoch_len + i)
            writer.add_scalar('train iteration time', end_time - start_time, epoch_ * epoch_len + i)
