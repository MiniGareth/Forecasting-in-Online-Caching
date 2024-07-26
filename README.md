# Forecasting in Online Caching

This repository contains the experiments done in the work [Forecasting in Online Caching]( http://resolver.tudelft.nl/uuid:89d1dd00-1a11-45b5-b800-9a892fa37c3b ) by Gareth Kit, for the course CSE 3000 Research Project.

Experiments are done to determine the effect of utilizing different forecasters on an optimistic caching policy based on the Online Follow the Regularized Leader[^1] using the static regret metric.

## Datasets
The dataset used for the experiments is from [MovieLens](https://grouplens.org/datasets/movielens/). The chosen dataset "ml-latest-small" that  was generated on September 26, 2018.

Requests were extracted from the dataset from the ```ratings.csv``` file and divided into train, validation, and test sets in a 8:1:1 split.

## Project Structure
- ```experiments``` contain 3 scripts that yields the results that is seen on the paper. 
  - ```main_forecasters.py``` calculates the "accuracy" and "prediction error" of forecasters on the MovieLens extracted requests and stores it in ```tables```.
  - ```main.py``` calculates the "regret" of OFTRL with different forecasters and stores it in ```tables```.
  - ```figure.py``` draws the results as plots that is seen on the paper. 
- ```dataset``` **does not** contain the MovieLens dataset rather a custom Pytorch Dataset class.
- ```forecasters``` contains all the implementations of the different forecasters.
- ```recommender``` and ```tcn``` contains key classes that are used by the recommender and TCN forecasters respectively.
- ```oftrl.py``` is the implementation of the OFTRL algorithm.
- ```utils``` contains useful utility functions.
- ```tests``` contains some tests of forecasters.
- ```tables``` and ```new_plots``` contains tables and plots of the results from some of the experiments.
- ```delftblue``` contains the scripts used when training TCN on the DelftBlue supercomputer cluster.

## Dependencies
```requirements.txt``` contains the extensive list of dependencies used in the creation of the repository, some are not necessary to the experiments. To summarize, the key packages required are:
- `cvxpy`
- `matplotlib`
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `torch`

It is recommended to install the packages as needed when running the python scripts.

## Running experiments

To run the `.py` executables in `experiments`, `tcn`, or `tests` one can run it from PyCharm or calling it from the terminal from the project root.

```bash
python experiments/main.py
```

## References
[^1]: Mhaisen, Naram, et al. "Optimistic no-regret algorithms for discrete caching." Proceedings of the ACM on Measurement and Analysis of Computing Systems 6.3 (2022): 1-28.