import torch
import os
from tqdm import tqdm

class Trainer():
    epoch = None

    def __init__(self,
                 model,
                 loss_keypoints,
                 loss_links,
                 optimizer,
                 lr_scheduler,
                 clip_grad_value,
                 device
                 ):
        
        self.model = model
        self.loss_keypoints = loss_keypoints
        self.loss_links = loss_links
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.clip_grad_value = clip_grad_value
        self.device = device
        
    def train(self,
             train_data: torch.utils.data.DataLoader,
             eval_data: torch.utils.data.DataLoader,
             writer,
             epoch=0,
             PATH='model'
            ):
        
        if not os.path.isdir(PATH):
            os.makedirs(PATH)

        for epoch_ in range(self.model.epoch, self.model.epoch + epoch):
            self.train_step(train_data, writer, epoch_)
            self.model.epoch = epoch_
            torch.save(self.model.state_dict(), os.path.join(PATH, 'model_' + epoch + '.pth'))
            # result = self.eval_step(eval_data)
            # if(self.model.best_result < result):
            #     torch.save(self.model.state_dict(), PATH + '_best_result.pth')

            # update learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

    def eval_step(self, eval_data):
        self.model.eval()
        return None

    def train_step(self, train_data, writer, epoch_):
        self.model.train()
        epoch_len = len(train_data)
        for i, (name, image, targeted_keypoints, scale, targeted_links, nb_cars) in tqdm(enumerate(train_data)):
            if(self.device != 'cpu'):
                image = image.to(self.device, non_blocking=True)
                targeted_keypoints = targeted_keypoints.to(self.device, non_blocking=True)
                targeted_links = targeted_links.to(self.device, non_blocking=True)
                scale = scale.to(self.device, non_blocking=True)
                nb_cars = nb_cars.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()

            # predicte
            predicted_keypoints, predicted_links = self.model(image)

            try:
                # compute loss
                loss_keypoints = self.loss_keypoints(predicted_keypoints, targeted_keypoints, scale, nb_cars)
                loss_links = self.loss_links(predicted_links, targeted_links, scale, nb_cars)
                loss = loss_keypoints + loss_links 
                loss.backward()
            except :
                print()
                print('add to skip :', i-1)
                print()
                print(loss_keypoints)
                print(loss_links)
                print(loss)
                print()
                print(self.model.neck.state_dict())
                print(self.model.keypoints.state_dict())
                print(self.model.links.state_dict())
                raise True

            # update model
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad_value)
            self.optimizer.step()

            writer.add_scalar('loss_keypoints', loss_keypoints, epoch_ * epoch_len + i)
            writer.add_scalar('loss_links', loss_links, epoch_ * epoch_len + i)
            writer.add_scalar('loss', loss, epoch_ * epoch_len + i)
