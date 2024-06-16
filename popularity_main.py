import numpy as np

import utils
from forecasters.DESForecaster import DESForecaster
from forecasters.PopularityForecaster import PopularityForecaster


def grid_search_popularity(train: np.ndarray, validation: np.ndarray, hyper_param: dict, key_idx: int,
                           best_tuple: tuple = (None, 0), forecaster_type="popularity"):
    # If all parameters have been assigned a value we check performance
    if key_idx >= len(hyper_param):
        if forecaster_type == "popularity":
            forecaster = PopularityForecaster(train, horizon=hyper_param["horizon"], one_hot=hyper_param["one_hot"])
        elif forecaster_type == "des":
            forecaster = DESForecaster(train, horizon=hyper_param["horizon"], one_hot=hyper_param["one_hot"])
        else:
            raise ValueError(f"{forecaster_type} must be 'popularity' or 'des'")

        # Calculate total utility with validation requests
        total_utility = 0
        for req in validation:
            prediction = forecaster.predict()
            forecaster.update(req)
            total_utility += np.dot(prediction, req)

        total_utility /= len(validation)

        print(hyper_param.copy(), total_utility)
        # Only return this set of hyper params if it has a better result than the best known utility.
        if total_utility > best_tuple[1]:
            return hyper_param.copy(), total_utility
        return best_tuple

    # Iterates through all possible values for a particular parameter
    key = list(hyper_param.keys())[key_idx]
    hyper_param_list = hyper_param[key]
    best_params, best_utility = best_tuple
    for val in hyper_param_list:
        hyper_param[key] = val
        best_params, best_utility = grid_search_popularity(train, validation, hyper_param, key_idx + 1,
                                                           best_tuple=(best_params, best_utility), forecaster_type=forecaster_type)
    hyper_param[key] = hyper_param_list
    return best_params, best_utility

def find_popularity_grid_search_movielens(library_size=None, one_hot=False, forecaster_type="popularity"):
    print(f"One Hot: {one_hot}, Type: {forecaster_type}")
    print("=============================================================================")
    train, validation, test = utils.get_movie_lens_split("ml-latest-small/ml-latest-small", library_limit=library_size)
    train_vecs = utils.convert_to_vectors(train, library_size)
    val_vecs = utils.convert_to_vectors(validation, library_size)
    test_vecs = utils.convert_to_vectors(test, library_size)

    hyper_params = {
        "horizon": np.arange(100, len(validation), 100),
        "one_hot": [one_hot]
    }
    best_params, best_val_utility = grid_search_popularity(train_vecs, val_vecs, hyper_params, 0, forecaster_type=forecaster_type)

    # Test the performance on the test set
    if forecaster_type == "popularity":
        forecaster = PopularityForecaster(np.concatenate((train_vecs, val_vecs)), horizon=best_params["horizon"],
                                          one_hot=best_params["one_hot"])
    elif forecaster_type == "des":
        forecaster = DESForecaster(np.concatenate((train_vecs, val_vecs)), horizon=best_params["horizon"], one_hot=best_params["one_hot"])
    else:
        raise ValueError(f"{forecaster_type} must be 'popularity' or 'des'")

    total_utility = 0
    for req in test_vecs:
        prediction = forecaster.predict()
        # print(prediction)
        forecaster.update(req)
        total_utility += np.dot(prediction, req)

    total_utility /= len(test)

    print(best_params)
    print(f"Best Validation Utility {best_val_utility}")
    print(f"Test Utility: {total_utility}")

if __name__ == "__main__":
    # find_popularity_grid_search_movielens(100, False)
    # find_popularity_grid_search_movielens(100, True)

    find_popularity_grid_search_movielens(100, False, forecaster_type="des")
    find_popularity_grid_search_movielens(100, True, forecaster_type="des")
