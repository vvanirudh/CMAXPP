# CMAX++ : Leveraging Experience in Planning and Execution using Inaccurate Models

## Dependencies

To install all the required dependencies, run the following command

``` shell
pip install -r requirements.txt
```

## Path issues

The project uses global path everywhere and requires that the project folder be placed at a specific path. Run the following commands from your home folder,

``` shell
mkdir -p workspaces/szeth_ws/src/
```

and then copy the project folder into the `szeth_ws/src' folder

## Reproducing Experiments

### 3D Mobile Robot Navigation

To reproduce the results for 3D mobile robot navigation experiment in the paper, run the following command inside the `src/` folder 

``` shell
python -m szeth.experiment.experiment_car_racing --agent <agent>
```
where `<agent>` should be one of `{'cmax', 'cmaxpp', 'adaptive_cmaxpp'}`. This saves the results in the `data/` folder

To reproduce the bar graph, run the following command in project root,

``` shell
python scripts/plot_car_racing_bar.py
```
The plot will be saved inside `plot/` folder.

To reproduce the sensitivity experiments reported in the appendix run the following script inside the `src/` folder

``` shell
./alpha_expts.sh
```

### 7D Pick-and-Place

To reproduce the results for 7D pick-and-place experiment in the paper, run the following command inside `src/` folder

``` shell
python -m szeth.experiment.experiment_pr2_7d_approximate --agent <agent>
```
where `<agent>` should be one of `{'cmax', 'cmaxpp', 'adaptive_cmaxpp', 'model', 'knn', 'qlearning'}` and `model` refers to the NN residual model learning approach. This saves the results in the `data/` folder

## Contributors

The repository is maintained and developed by [Anirudh Vemula](https://vvanirudh.github.io) from the Search based Planning Lab (SBPL) at CMU.
