import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset.MovieLensDataset import MovieLensDataset
from tcn.models import TemporalConvNet


def grid_search_tcn(train_loader, val_loader, hyper_params: dict, key_idx: int, best_tuple: tuple=(None, None, float("inf"))):
    if key_idx >= len(hyper_params):
        print(f"Trying: {hyper_params}")
        postfix = (f"i{hyper_params['num_inputs']}f{hyper_params['num_filters']}"
                   f"l{hyper_params['num_layers']}k{hyper_params['kernel_size']}"
                   f"d{hyper_params['dropout']}c{hyper_params['num_classes']}r{hyper_params['learning_rate']}")
        runs_folder = f"tcn/grid/runs_{postfix}/"
        model_folder = f"tcn/grid/models/{postfix}"
        # Train and evaluate model
        model = TemporalConvNet(
            num_inputs=hyper_params["num_inputs"],
            num_channels=[hyper_params["num_filters"]] * hyper_params["num_layers"],
            kernel_size=hyper_params["kernel_size"],
            dropout=hyper_params["dropout"],
            runs_folder=runs_folder,
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
            loss_function=nn.NLLLoss(),
            save_every_epoch=10,
            model_path=model_folder,
            valid_loader=val_loader,
            scheduler=None,
            print_every_epoch=10
        )
        # Load weights and biases of model
        model = model.load_state_dict(torch.load(model_folder + "_best"))
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

def test_model(model, test_loader):
    total_loss = 0
    total_utility = 0
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
    print("======================================================================================================")
    print("Test data")
    print(f"Loss: {total_loss / (batch_index + 1)}")
    print(f"utility: {total_utility / (batch_index + 1)}")


def test_tcn_movielens(library_size=None, request_limit=None):
    # Create train, validation, and test datasets
    float_tensor_transform = lambda x: torch.tensor(x).float()
    train_dataset = MovieLensDataset("../ml-latest-small/ml-latest-small/", split="train", library_limit=library_size,
                                     request_limit=request_limit, transform=float_tensor_transform)
    val_dataset = MovieLensDataset("../ml-latest-small/ml-latest-small/", split="validation",
                                   library_limit=library_size,
                                   request_limit=request_limit, transform=float_tensor_transform)
    test_dataset = MovieLensDataset("../ml-latest-small/ml-latest-small/", split="test", library_limit=library_size,
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
    runs_folder = "tcn/runs"
    models_folder = "tcn/models/m"
    num_inputs = 100
    kernel_size = 6
    dropout = 0.2
    learning_rate = 0.1
    num_filters = 50
    num_layers = 10

    model = TemporalConvNet(
        num_inputs=num_inputs,
        num_channels=[num_filters] * num_layers,
        kernel_size=kernel_size,
        dropout=dropout,
        runs_folder=runs_folder,
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
        loss_function=nn.NLLLoss(),
        save_every_epoch=10,
        model_path=models_folder,
        valid_loader=val_loader,
        scheduler=None,
        print_every_epoch=10
    )
    model = model.load_state_dict(torch.load(models_folder + "_best"))
    # Test the model on the test loader
    test_model(model, test_loader)


def find_tcn_grid_search_movielens(library_size=None, request_limit=None):
    # Create train, validation, and test datasets
    float_tensor_transform = lambda x: torch.tensor(x).float()
    train_dataset = MovieLensDataset("ml-latest-small/ml-latest-small/", split="train", library_limit=library_size,
                                     request_limit=request_limit, transform=float_tensor_transform)
    val_dataset = MovieLensDataset("ml-latest-small/ml-latest-small/", split="validation",
                                   library_limit=library_size,
                                   request_limit=request_limit, transform=float_tensor_transform)
    test_dataset = MovieLensDataset("ml-latest-small/ml-latest-small/", split="test", library_limit=library_size,
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
        "learning_rate": [0.1],
    }
    best_hyper_params, best_model, best_val_loss = grid_search_tcn(train_loader, val_loader, hyper_parameters, 0)

    print(best_hyper_params)
    test_model(best_model, test_loader)


if __name__ == "__main__":
    # test_tcn_movielens(100, None)
    # test_tcn_movielens(200, None)
    find_tcn_grid_search_movielens(100, None)
