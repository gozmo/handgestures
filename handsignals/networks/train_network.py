import time
import copy
import torch
import numpy as np
from handsignals import device
from handsignals.evaluate import evaluate_model, evaluate_io


def train_model(
    model, model_parameters, training_dataset, holdout_dataset, criterion, optimizer
):

    start_time = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0
    dataloader_training_set = training_dataset.get_dataloader(
        model_parameters.batch_size, shuffle=True
    )

    for epoch in range(model_parameters.epochs):
        print("Epoch {}/{}".format(epoch, model_parameters.epochs - 1))
        epoch_start_time = time.time()

        ###
        ### Training step
        ###
        model.train()
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

                outputs = model(image_input)
                loss = criterion(outputs, labels_input)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * images.size(0)

        ###
        ### Evaluation step
        ###
        model.eval()

        model_to_eval = ConvNet(len(training_dataset.get_labels), model)
        _, _, holdout_f1_scores = evaluate_model.evaluate_model_on_dataset(
            model_to_eval, holdout_dataset, "holdout"
        )
        _, _, labeled_f1_scores = evaluate_model.evaluate_model_on_dataset(
            model_to_eval, training_dataset, "training"
        )

        epoch_f1 = labeled_f1_scores["f1"]
        evaluate_io.write_running_f1_score(epoch, epoch_f1, "holdout")
        evaluate_io.write_running_loss(epoch, running_loss)

        ###
        ### Clean up step
        ###

        if epoch_f1 > best_f1:
            pass

        ###
        ### Report step
        ###
        epoch_time_elapsed = time.time() - epoch_start_time
        time_elapsed = time.time() - start_time
        print(f"Epoch completed in: {epoch_time_elapsed}")
        print(f"Training time: {time_elapsed}")
        print(f"Loss: {running_loss}")
        print(f"Epoch f1: {epoch_f1}")
        print("-" * 10)

    time_elapsed = time.time() - start_time
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    model.load_state_dict(best_model_wts)
    # return model, val_loss_history, train_loss_history
