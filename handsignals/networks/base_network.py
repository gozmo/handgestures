from handsignals import device
import time
import copy
import torch
import numpy as np
from handsignals.evaluate import evaluate_model, evaluate_io
from handsignals.dataset.image_dataset import ImageDataset

class BaseNetwork:

    def __init__(self, model):
        self.__model = model
        self.__model.double()
        self.__model.to(device)

    def train_model(self,
                    model_parameters,
                    training_dataset,
                    holdout_dataset,
                    criterion,
                    optimizer):
        start_time = time.time()

        best_model_wts = copy.deepcopy(self.__model.state_dict())
        best_f1 = 0.0
        dataloader_training_set = training_dataset.get_dataloader(model_parameters.batch_size, shuffle=True)

        for epoch in range(model_parameters.epochs):
            print('Epoch {}/{}'.format(epoch, model_parameters.epochs - 1))
            epoch_start_time = time.time()

            ###
            ### Training step
            ###
            self.__model.train()
            running_loss = 0.0

            for batch in dataloader_training_set:

                images = batch["image"]
                labels = batch["label"]

                image_input = torch.from_numpy(np.array(images))
                labels_input = torch.from_numpy(np.array(labels))

                image_input = image_input.to(device)
                labels_input = labels_input.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(True):

                    outputs = self.__model(image_input)
                    loss = criterion(outputs, labels_input)


                    # backward + optimize only if in training phase
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * images.size(0)

            ###
            ### Evaluation step
            ###
            self.__model.eval()

            _, _, holdout_f1_scores = evaluate_model.evaluate_model_on_dataset(self, holdout_dataset, "holdout")
            _, _, labeled_f1_scores = evaluate_model.evaluate_model_on_dataset(self, training_dataset, "training")

            epoch_f1 = labeled_f1_scores["f1"]
            evaluate_io.write_running_f1_score(epoch, epoch_f1, "holdout")
            evaluate_io.write_running_loss(epoch, running_loss)

            ###
            ### Clean up step
            ###

            if epoch_f1 > best_f1:
                #save best epoch for saving best network later on
                best_model_wts = copy.deepcopy(self.__model)

            ###
            ### Report step
            ###
            epoch_time_elapsed = time.time() - epoch_start_time
            time_elapsed = time.time() - start_time
            print(f"Epoch completed in: {epoch_time_elapsed}")
            print(f"Training time: {time_elapsed}")
            print(f"Loss: {running_loss}")
            print(f"Epoch f1: {epoch_f1}")
            print('-' * 10)

        time_elapsed = time.time() - start_time
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        self.__model.load_state_dict(best_model_wts)
        return self.__model, val_loss_history, train_loss_history

    def __classify(self, image):
        image = np.asarray([image])
        image_torch = torch.from_numpy(image)
        image_torch = image_torch.to(device)
        return self.__model(image_torch)

    def __classify_batch(self, batch):
        images = np.asarray(batch)
        images_torch = torch.from_numpy(images)
        images_torch = images_torch.to(device)
        return self.__model(images_torch)

    def save(self, path):
        torch.save(self.__model.state_dict(), path)

    def classify_image(self, image):

        prediction_distribution = self.__classify(image)
        prediction_result = PredictionResult(prediction_distribution)

        return prediction_result


    def classify_dataset(self, dataset: ImageDataset):
        predictions = []

        dataloader = dataset.get_dataloader()

        for batch in dataloader:
            #a = torch.cuda.memory_allocated(device=device)
            images = batch["image"]
            labels = batch["label"]

            prediction_distributions =  self.__classify_batch(images)

            for index in range(len(prediction_distributions)):
                prediction_distribution = prediction_distributions[index]
                label = labels[index]

                prediction_result = PredictionResult(prediction_distribution, label)

                predictions.append(prediction_result)

            del batch
            del prediction_distributions

        return predictions
