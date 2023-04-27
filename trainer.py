import torch

class Trainer():
    epoch = None
    clip_grad_value = 0.01

    def __init__(
            self,
            model,
            loss_keypoints,
            loss_links,
            optimizer,
            lr_scheduler,
            device
            ):
        
        self.model = model
        self.loss_keypoints = loss_keypoints
        self.loss_links = loss_links
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        
    def train(self,
             train_data: torch.utils.data.DataLoader,
             eval_data: torch.utils.data.DataLoader,
             epoch=0,
             PATH='model'
            ):

        for epoch_ in range(self.model.epoch, self.model.epoch + epoch):
            self.train_step(train_data)
            self.model.epoch = epoch_
            torch.save(self.model.state_dict(), PATH + '_' + epoch + '.pth')
            # result = self.eval_step(eval_data)
            # if(self.model.best_result < result):
            #     torch.save(self.model.state_dict(), PATH + '_best_result.pth')

    def eval_step(self, eval_data):
        self.model.eval()
        return None

    def train_step(self, train_data):
        self.model.train()
        for image, targeted_keypoints, scale, targeted_links, nb_cars in train_data:
            if(self.device != 'cpu'):
                image = image.to(self.device, non_blocking=True)
                targeted_keypoints = targeted_keypoints.to(self.device, non_blocking=True)
                targeted_links = targeted_links.to(self.device, non_blocking=True)
                scale = scale.to(self.device, non_blocking=True)
                nb_cars = nb_cars.to(self.device, non_blocking=True)
            
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
            self.optimizer.zero_grad()

            # update learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
