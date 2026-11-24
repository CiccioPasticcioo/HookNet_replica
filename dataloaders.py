from data_extractor import click_dataset
from fma_data_estraction import datasets_creation
from torch.utils.data import DataLoader
import torch

BATCH_SIZE = 8

def create_dataloaders():
    
    train_data, test_data, validation_data = datasets_creation()

    train_dataloader = DataLoader(click_dataset(train_data), batch_size = BATCH_SIZE, shuffle = True)
    validation_dataloader = DataLoader(click_dataset(validation_data), batch_size = BATCH_SIZE, shuffle = False)
    test_dataloader = DataLoader(click_dataset(test_data), batch_size = BATCH_SIZE, shuffle = False)

    return train_dataloader, validation_dataloader, test_dataloader
