from pathlib import Path
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import sys

root_dir = Path(".").resolve()
sys.path.append(str(root_dir.absolute()))

from dataset.MovieLensDataset import MovieLensDataset
from tcn.models import TemporalConvNet

def test_model(model, test_loader):
    """
    Test the given model on the test loader.
    :param model: The given TCN model
    :param test_loader: Test loader
    """
    total_loss = 0
    total_utility = 0
    total_accuracy = 0
    for batch_index, (x_, y_) in enumerate(test_loader):
        x_ = Variable(x_)
        y_ = Variable(y_)
        if model.gpu:
            x_ = x_.cuda()
            y_ = y_.cuda()
        output = model(x_)
        loss = nn.NLLLoss()(output, y_)
        total_loss += loss
        # Utility is the average percentage of a cache hit
        utility = torch.mean(torch.exp(output[torch.arange(output.shape[0]), y_]))
        total_utility += utility
        total_accuracy += torch.mean(torch.argmax(output, axis=1) == y_, dtype=torch.float32)
    print("======================================================================================================")
    print("Test data")
    print(f"Loss: {total_loss / (batch_index + 1)}")
    print(f"utility: {total_utility / (batch_index + 1)}")
    print(f"accuracy: {total_accuracy / (batch_index + 1)}")

def train_test_tcn_movielens(library_size=None, request_limit=None):
    """
    Function that trains and test a TCN model for hardcoded hyperparameters using the MovieLens dataset
    :param library_size: Number of files in the library
    :param request_limit: Maximum number of requests allowed to be extracted.
    :return:
    """
    # Create train, validation, and test datasets
    float_tensor_transform = lambda x: torch.tensor(x).float()
    train_dataset = MovieLensDataset(root_dir / "ml-latest-small" / "ml-latest-small", split="train", library_limit=library_size,
                                     request_limit=request_limit, transform=float_tensor_transform)
    val_dataset = MovieLensDataset(root_dir / "ml-latest-small" / "ml-latest-small", split="validation",
                                   library_limit=library_size,
                                   request_limit=request_limit, transform=float_tensor_transform)
    test_dataset = MovieLensDataset(root_dir / "ml-latest-small" / "ml-latest-small", split="test", library_limit=library_size,
                                    request_limit=request_limit, transform=float_tensor_transform)

    # Create Data Loaders
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(train_dataset.__getitem__(0)[0].shape)
    print(train_dataset.__getitem__(100)[0].shape)
    print(train_dataset.__getitem__(200)[0].shape)
    print(f"Train size: {train_dataset.__len__()}, Validation size: {val_dataset.__len__()}, Test size: {test_dataset.__len__()}")

    # TCN Hyperparameters
    runs_folder = root_dir / "tcn" / "runs"
    models_folder = root_dir / "tcn" / "models" / "m"
    num_inputs = library_size   # 100 usually
    kernel_size = 6
    dropout = 0.2
    learning_rate = 0.1
    num_filters = library_size // 2     # 50 usually
    num_layers = 10
    loss_function = nn.NLLLoss()

    model = TemporalConvNet(
        num_inputs=num_inputs,
        num_channels=[num_filters] * num_layers,
        kernel_size=kernel_size,
        dropout=dropout,
        runs_folder=str(runs_folder),
        mode="classification",
        num_classes=library_size,
        gpu=True
    )
    model.cuda()
    model.fit(
        num_epoch=100,
        train_loader=train_loader,
        optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate),
        clip=-1,
        loss_function=loss_function,
        save_every_epoch=10,
        model_path=str(models_folder),
        valid_loader=val_loader,
        scheduler=None,
        print_every_epoch=10
    )
    model.load_state_dict(torch.load(str(models_folder) + "_best"))
    # Test the model on the test loader
    test_model(model, test_loader)

def grid_search_tcn(train_loader, val_loader, hyper_params: dict, key_idx: int, best_tuple: tuple=(None, None, float("inf"))):
    """
    This function performs grid search recursively on the given hyper-parameters possibilities to find the best hyper-parameters
    :param train_loader: Training data loader.
    :param val_loader: Valdiation data loader.
    :param hyper_params: Hyper parameters dictionary.
    :param key_idx: The key or hyper-parameter to assign a value to.
    :param best_tuple: The best hyper parameters, the best TCN model, and the best loss.
    :return: The best hyper parameters, the best TCN model, and the best loss.
    """
    if key_idx >= len(hyper_params):
        print(f"Trying: {hyper_params}")
        postfix = (f"i{hyper_params['num_inputs']}f{hyper_params['num_filters']}"
                   f"l{hyper_params['num_layers']}k{hyper_params['kernel_size']}"
                   f"d{hyper_params['dropout']}c{hyper_params['num_classes']}r{hyper_params['learning_rate']}"
                   f" {hyper_params['loss_function']}")
        runs_folder = root_dir / "tcn" / "grid" / f"runs_{postfix}"
        model_folder = root_dir / "tcn" / "grid" / "models" / f"{postfix}"
        # Train and evaluate model
        model = TemporalConvNet(
            num_inputs=hyper_params["num_inputs"],
            num_channels=[hyper_params["num_filters"]] * hyper_params["num_layers"],
            kernel_size=hyper_params["kernel_size"],
            dropout=hyper_params["dropout"],
            runs_folder=str(runs_folder),
            mode="classification",
            num_classes=hyper_params["num_classes"],
            gpu=True
        )
        model.cuda()
        print(model.parameters())
        best_val_loss = model.fit(
            num_epoch=100,
            train_loader=train_loader,
            optimizer=torch.optim.SGD(model.parameters(), lr=hyper_params["learning_rate"]),
            clip=-1,
            loss_function=hyper_params["loss_function"],
            save_every_epoch=10,
            model_path=str(model_folder),
            valid_loader=val_loader,
            scheduler=None,
            print_every_epoch=10
        )
        # Load weights and biases of model
        model.load_state_dict(torch.load(str(model_folder) + "_best"))
        # Return hyper parameters, model and loss
        if best_tuple[2] is None or best_val_loss < best_tuple[2]:
            return hyper_params.copy(), model, best_val_loss
        return best_tuple

    # Try a hyperparameter from the list of possible hyperparameters
    key = list(hyper_params.keys())[key_idx]
    hyper_param_list = hyper_params[key]
    best_params, best_model, best_loss = best_tuple
    for param in hyper_param_list:
        hyper_params[key] = param
        best_params, best_model, best_loss = grid_search_tcn(train_loader, val_loader, hyper_params, key_idx + 1, (best_params, best_model, best_loss))
    hyper_params[key] = hyper_param_list
    return best_params, best_model, best_loss

def find_tcn_grid_search_movielens(library_size=None, request_limit=None):
    """
    This function performs grid search for TCN on the MovieLens dataset
    :param library_size: Number of files in the library
    :param request_limit: Maximum number of requests allowed to be extracted.
    :return:
    """
    # Create train, validation, and test datasets
    float_tensor_transform = lambda x: torch.tensor(x).float()
    train_dataset = MovieLensDataset(root_dir / "ml-latest-small" / "ml-latest-small", split="train", library_limit=library_size,
                                     request_limit=request_limit, transform=float_tensor_transform)
    val_dataset = MovieLensDataset(root_dir / "ml-latest-small" / "ml-latest-small", split="validation",
                                   library_limit=library_size,
                                   request_limit=request_limit, transform=float_tensor_transform)
    test_dataset = MovieLensDataset(root_dir / "ml-latest-small" / "ml-latest-small", split="test", library_limit=library_size,
                                    request_limit=request_limit, transform=float_tensor_transform)

    # Create Data Loaders
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(train_dataset.__getitem__(0)[0].shape)
    print(train_dataset.__getitem__(100)[0].shape)
    print(train_dataset.__getitem__(200)[0].shape)
    print(f"Train size: {train_dataset.__len__()}, Validation size: {val_dataset.__len__()}, Test size: {test_dataset.__len__()}")

    # Specify range of hyper_parameters to try
    hyper_parameters = {
        "num_inputs": [library_size],
        "num_filters": [40, 50, 60],
        "num_layers": [6, 8, 10, 12],
        "kernel_size": [6, 8],
        "dropout": [0.2],
        "num_classes": [library_size],
        "learning_rate": [0.1, 0.01],
        "loss_function": [nn.NLLLoss()],
    }
    best_hyper_params, best_model, best_val_loss = grid_search_tcn(train_loader, val_loader, hyper_parameters, 0)

    print(best_hyper_params)
    test_model(best_model, test_loader)

def test_tcn_movielens():
    """
    Function that tests the pre-trained TCN model of the hardcoded hyper_params on the MovieLens dataset.
    :return:
    """
    # The best hyperparameters
    hyper_params = {
        'num_inputs': 100,
        'num_filters': 40,
        'num_layers': 8,
        'kernel_size': 6,
        'dropout': 0.2,
        'num_classes': 100,
        'learning_rate': 0.1
    }
    postfix = (f"i{hyper_params['num_inputs']}f{hyper_params['num_filters']}"
               f"l{hyper_params['num_layers']}k{hyper_params['kernel_size']}"
               f"d{hyper_params['dropout']}c{hyper_params['num_classes']}r{hyper_params['learning_rate']}")


    runs_folder = root_dir / "tcn" / "grid" / f"runs_{postfix}"
    model_folder = root_dir / "tcn" / "grid" / "models" / f"{postfix}"

    request_limit = None
    library_size = 100
    float_tensor_transform = lambda x: torch.tensor(x).float()
    train_dataset = MovieLensDataset(root_dir / "ml-latest-small" / "ml-latest-small", split="train", library_limit=library_size,
                                     request_limit=request_limit, transform=float_tensor_transform)
    val_dataset = MovieLensDataset(root_dir / "ml-latest-small" / "ml-latest-small", split="validation",
                                   library_limit=library_size,
                                   request_limit=request_limit, transform=float_tensor_transform)
    test_dataset = MovieLensDataset(root_dir / "ml-latest-small" / "ml-latest-small", split="test", library_limit=library_size,
                                    request_limit=request_limit, transform=float_tensor_transform)
    # Create Data Loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Train and evaluate model
    model = TemporalConvNet(
        num_inputs=hyper_params["num_inputs"],
        num_channels=[hyper_params["num_filters"]] * hyper_params["num_layers"],
        kernel_size=hyper_params["kernel_size"],
        dropout=hyper_params["dropout"],
        runs_folder=str(runs_folder),
        mode="classification",
        num_classes=hyper_params["num_classes"],
        gpu=True
    )
    model.cuda()
    # Load weights and biases of model
    model.load_state_dict(torch.load(str(model_folder) + "_best"))
    model.eval()
    # model.load_state_dict(torch.load("tcn/tcn_best"))
    print(model_folder)
    # test_model(model, train_loader)
    test_model(model, val_loader)
    test_model(model, test_loader)

if __name__ == "__main__":
    train_test_tcn_movielens(10, None)
    # find_tcn_grid_search_movielens(100, None)
    # test_tcn_movielens()
