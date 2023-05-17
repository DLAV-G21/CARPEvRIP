import torch
import time
import os
from tqdm import tqdm
from utils.coco_evaluator import CocoEvaluator

class Trainer():
    epoch = None

    def __init__(
            self,
            save,
            model,
            decoder,
            loss_keypoints,
            loss_links,
            optimizer,
            lr_scheduler,
            clip_grad_value,
            device,
            train_loader,
            val_loader,
            writer
            ):
        self.save = save
        self.model = model
        self.decoder = decoder
        self.loss_keypoints = loss_keypoints
        self.loss_links = loss_links
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.clip_grad_value = clip_grad_value
        self.device = device
        self.train_data = train_loader
        self.eval_data = val_loader
        self.writer = writer
        
    def train(
            self,
            epoch=0
            ):
        # If the path does not exists, create it
        if not os.path.isdir(self.save):
            os.makedirs(self.save)
        
        # If the learning rate scheduler is not empty
        if self.lr_scheduler is not None:
            for _ in range(self.model.epoch):
                # Step the learning rate scheduler
                self.lr_scheduler.step()

        # Iterate through the specified number of epochs
        for epoch_ in range(self.model.epoch, self.model.epoch + epoch):
            print('epoch :', epoch_)
            # Perform the training step
            self.train_step(self.train_data, self.writer, epoch_)
            # update the epoch value
            self.model.epoch = epoch_ + 1
            # Save the model
            self.model.save_weights(os.path.join(self.save, f'model_{self.model.epoch}.pth'))

            # Create an instance of the CocoEvaluator class to be used for evaluation
            coco_evaluator = CocoEvaluator(self.eval_data.dataset.coco, ["keypoints"])

            # If the coco evaluator is not empty
            if coco_evaluator is not None:
                # Run the evaluation step
                result = self.eval_step(self.eval_data, coco_evaluator, self.writer, epoch_)
                # If the result is better than the best result
                if(self.model.best_result <= result):
                    # Save the model
                    self.model.save_weights(os.path.join(self.save, 'best_result.pth'))

            # If the learning rate scheduler is not empty
            if self.lr_scheduler is not None:
                # Step the learning rate scheduler
                self.lr_scheduler.step()

    def eval(self):
        coco = self.eval_data.dataset.coco
        # Create an instance of the CocoEvaluator class to be used for evaluation
        coco_evaluator = CocoEvaluator(coco, ["keypoints"]) if coco is not None else None
        # eval_step is used to evaluate the model on a given dataset
        result = self.eval_step(self.eval_data, coco_evaluator, None, None, return_skeletons=True)
        return result
    
    @torch.no_grad()
    def eval_step(self, eval_data, coco_evaluator, writer, epoch_, return_skeletons = False):
        # Set the model to eval mode
        self.model.eval()
        # Get the length of the evaluation dataset
        epoch_len = len(eval_data)
        # Set the initial number of skeletons found to 0
        skeletons_found = 0
        # Create an empty list to store the skeletons
        skeletons_ = []
        # Iterate through each item in the evaluation dataset
        for i, (image, image_id) in tqdm(enumerate(eval_data)):
            # If the device is not a cpu, move the image to the device
            if(self.device != 'cpu'):
                image = image.to(self.device, non_blocking=True)
            # Start a timer to measure the iteration time
            start_time = time.time()

            # Get the predicted keypoints and links
            predicted_keypoints, predicted_links = self.model(image)
            # Decode the predicted keypoints and links
            skeletons = self.decoder((predicted_keypoints, predicted_links), image_id)
            # If the coco evaluator is not None, update the keypoints
            if coco_evaluator is not None:
                coco_evaluator.update_keypoints(skeletons)
            # if return_skeletons is true, add the skeletons to the list
            if return_skeletons:
                skeletons_.extend(skeletons)
            # Increase the number of skeletons found
            skeletons_found += len(skeletons)

            # End the timer and measure the iteration time
            end_time = time.time()
            # If a writer is provided, add the iteration time to the scalar
            if writer is not None:
                writer.add_scalar('eval iteration time', end_time - start_time, epoch_ * epoch_len + i)
        
        # Print the number of skeletons found
        print('skeletons found :',skeletons_found)
        # If the coco evaluator is not None
        if coco_evaluator is not None:
            # Synchronize between processes
            coco_evaluator.synchronize_between_processes()
            # Accumulate the results
            coco_evaluator.accumulate()
            # Summarize the results
            coco_evaluator.summarize()
            # If return_skeletons is true, return the coco_evaluator stats for keypoints
            return skeletons_ \
                if return_skeletons else \
                coco_evaluator.coco_eval['keypoints'].stats[0]
        # If return_skeletons is true, return the skeletons
        elif return_skeletons:
            return skeletons_
        # If neither conditions are true, return None
        return None

    def train_step(self, train_data, writer, epoch_):
        # Set the model to train mode
        self.model.train()
        # Get the length of the training data
        epoch_len = len(train_data)
        # Iterate through the training data
        for i, (_, image, targeted_keypoints, scale, targeted_links, nb_cars) in tqdm(enumerate(train_data)):
            # If device is not cpu, move data to device
            if(self.device != 'cpu'):
                image = image.to(self.device, non_blocking=True)
                targeted_keypoints = targeted_keypoints.to(self.device, non_blocking=True)
                targeted_links = targeted_links.to(self.device, non_blocking=True)
                scale = scale.to(self.device, non_blocking=True)
                nb_cars = nb_cars.to(self.device, non_blocking=True)
            
            # Record the start time
            start_time = time.time()
            # Zero out all gradients
            self.optimizer.zero_grad()

            # Feed the image into the model
            predicted_keypoints, predicted_links = self.model(image)

            # Calculate the losses
            loss_keypoints = self.loss_keypoints(predicted_keypoints, targeted_keypoints, scale, nb_cars)
            loss_links = self.loss_links(predicted_links, targeted_links, scale, nb_cars)
            # Sum the losses
            loss = loss_keypoints + loss_links
            # Backpropagate the loss
            loss.backward()
                
            # Clip gradients to a certain value
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad_value)
            # Take an optimizer step
            self.optimizer.step()
            # Record the end time
            end_time = time.time()
            
            # If a writer is present, log all losses and iteration time
            if writer is not None:
                writer.add_scalar('loss keypoints', loss_keypoints, epoch_ * epoch_len + i)
                writer.add_scalar('loss links', loss_links, epoch_ * epoch_len + i)
                writer.add_scalar('loss', loss, epoch_ * epoch_len + i)
                writer.add_scalar('train iteration time', end_time - start_time, epoch_ * epoch_len + i)
