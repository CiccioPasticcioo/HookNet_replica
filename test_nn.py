from pathlib import Path
import torch
from torch import nn
import dataloaders as dl
from HookNet import HookNet
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

best_weight_position = "INSERT POSITION OF WEIGHT TO TEST"

# TEST MODULE
if __name__ == "__main__":

    train_loader, validation_loader, test_loader = dl.create_dataloaders()

    # Loading of weight
    weight_path = ( Path(__file__).resolve().parent / str(best_weight_position))

    model = HookNet()

    weight = torch.load(weight_path, map_location="cpu", weights_only=True)
    model.load_state_dict(weight)

    # Activation of evaluation modality
    model.eval()

    # Choice of loss function
    loss_fn = nn.BCELoss()

    # Choice of threshold
    threshold = 0.5

    # Loading of test subset
    loader = test_loader

    # Initialization test validation, number of segments in track and batch
    loss_test = 0.0
    n_segments = 0
    batch_test = 0

    # Initialization number of event
    n_real_positive = 0
    n_fake_positive = 0
    n_fake_negative = 0
    n_real_negative = 0

    # Batch elaboration
    with torch.no_grad():
        for X, Y in loader:

            # Prediction elaboration by HookNet
            prediction = model(X)

            # Saving loss values per segment
            loss = loss_fn(prediction, Y)

            # Loss value and number of segment acquired saving
            loss_test += loss.item() * X.shape[0]
            n_segments += X.shape[0]

            # Comparison of prediction and threshold
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
            raise ValueError("Test set empty")

        # Mean loss of single batch
        loss_test /= n_segments

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

    # Display of metrics
    print(f"Test loss: {loss_test:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    # Display of confusion matrix
    confusion_matrix = [[n_real_negative, n_fake_positive], [n_fake_negative, n_real_positive]]
    confusion_matrix_graph = ConfusionMatrixDisplay(confusion_matrix = np.array(confusion_matrix), display_labels = ["No Click" , "Click"])
    confusion_matrix_graph.plot(cmap="viridis" , values_format="d")
    confusion_matrix_graph.ax_.set_xlabel("Predicted")
    confusion_matrix_graph.ax_.set_ylabel("Real")
    confusion_matrix_graph.ax_.set_title(f"Test set – threshold {threshold}")

    plt.tight_layout()
    plt.show()