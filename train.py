import torch
from torch import optim, nn
import dataloaders as dl
from HookNet import HookNet
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

NUM_EPOCH = "CHOOSE NUMBER OF EPOCH"

# TRAIN MODULE
if __name__ == "__main__":

    # Loading subset
    train_loader, validation_loader, test_loader = dl.create_dataloaders()

    # Cnoice of model
    model = HookNet()

    # Choice of loss function
    loss_fn = nn.BCELoss()

    # Choice of optimizer
    optimizer = optim.SGD(model.parameters(), lr = 0.1)

    num_epoch = NUM_EPOCH

    # Initialization of vector where the loss values of train and validation are stored
    training_loss_values = []
    validation_loss_values = []
    epochs = []

    # Iterating epoch
    for epoch in range(num_epoch):

        # Initialization of training loss and number of segment within train subset
        loss_training = 0.0
        n_segments_training = 0.0

        # Number of current epoch
        now_epoch = epoch + 1 

        # Activation of train modality
        model.train()
        n_batch_training_elaborated = 0

        # Loading audio and labels
        for X, Y in train_loader:

            # Reset the gradients
            optimizer.zero_grad()

            # Prediction calculation
            prediction = model(X)

            # Saving of loss value per segment and number of segment elaborated
            loss = loss_fn(prediction , Y)
            loss_training += loss.item() * X.shape[0]
            n_segments_training += X.shape[0]

            # Computes the gradient of current tensor wrt graph leaves
            loss.backward()

            # Update model's parameters using the gradients stored in each parameter's
            optimizer.step()

            batch_training_elaborated += 1

        # Mean loss of single batch
        loss_training /= n_segments_training

        # Activation of validation modality
        model.eval()

        # Initialization of training loss and number of segment within train subset
        loss_validation = 0.0
        n_segments = 0

        # Initialization number of event
        n_real_positive = 0
        n_fake_positive = 0
        n_fake_negative = 0
        n_real_negative = 0

        # Soglia iniziale per decidere se una previsione indica un click.
        threshold = 0.5
        batch_validation = 0

        # Calculation of gradients
        with torch.no_grad():
            for X, Y in validation_loader:

                # Prediction elaboration by HookNet
                prediction = model(X)

                # Saving loss values per segment
                loss = loss_fn(prediction, Y)

                # Loss value and number of segment acquired saving
                loss_validation += loss.item() * X.shape[0]
                n_segments += X.shape[0]

                # Comparison of prediction and threshold.
                predicted_click = prediction >= threshold

                # Assignment of label as real
                real_click = Y == 1

                # Clicks located in the correct position
                real_positive = predicted_click & real_click

                # Clicks located where there is not click
                fake_positive = predicted_click & (~real_click)

                # Click present but not reported
                fake_negative = (~predicted_click) & real_click

                # Click not present confirmed
                real_negative = (~predicted_click) & (~real_click)

                # Number of events calculation
                n_real_positive += real_positive.sum().item()
                n_fake_positive += fake_positive.sum().item()
                n_fake_negative += fake_negative.sum().item()
                n_real_negative += real_negative.sum().item()

                # Incrementatiion of current batch
                batch_validation += 1

            if n_segments == 0:
                raise ValueError("Validation set empty")

            # Mean loss of single batch
            loss_validation /= n_segments

            # Creation of vectors where loss values from train and validation are stored
            training_loss_values.append(loss_training)
            validation_loss_values.append(loss_validation)

            # Creation of vector where number of epoch are stored
            epochs.append(now_epoch)

            # Saving loss per epoch information
            np.savez( Path(__file__).resolve().parent / "loss_per_epoch.npz", epochs=np.array(epochs), training_loss=np.array(training_loss_values), validation_loss=np.array(validation_loss_values))

            # Calcolation of precision metric
            if (n_real_positive + n_fake_positive) > 0:

                precision = n_real_positive / (n_real_positive + n_fake_positive)   

            else: precision = 0.0

            # Calcolation of recall metric    
            if (n_real_positive + n_fake_negative) > 0:

                recall = n_real_positive / (n_real_positive + n_fake_negative)

            else: recall = 0.0

            # Calcolation of F1 score metric
            if (precision + recall) > 0:

                f1 = 2 * precision * recall / (precision + recall)

            else: f1 = 0.0

            # Saving weight per epoch
            save_path = (Path(__file__).resolve().parent / f"Hooknet_epoch_{now_epoch}.pth")

            # Saving weigth as binary file 
            with save_path.open("xb") as file:
                torch.save(model.state_dict(), file)

        # Display of metrics
        print(f"Test loss: {loss_training:.4f}")
        print(f"Validation loss: {loss_validation:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")

        # Display and store of confusion matrix
        confusion_matrix = [[n_real_negative, n_fake_positive], [n_fake_negative, n_real_positive]]
        confusion_matrix_graph = ConfusionMatrixDisplay(confusion_matrix = np.array(confusion_matrix), display_labels = ["Nessun Click" , "Click"])
        confusion_matrix_graph.plot(cmap="viridis" , values_format="d")
        plt.title(f"Validation epoch: {now_epoch}")
        plt.tight_layout()
        plt.savefig( Path(__file__).resolve().parent / f"confusion_matrix_epoch_{now_epoch}.png", dpi=300)
        plt.close(confusion_matrix_graph.figure_)

    # Display of training and validation trend during every epoch
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, training_loss_values, label="Training loss")
    plt.plot(epochs, validation_loss_values, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE mean loss")
    plt.title("Loss training and validation")
    plt.xticks(epochs)
    plt.yscale("log")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig( Path(__file__).resolve().parent / "loss_per_epoch.png", dpi=300 )
    plt.show()
