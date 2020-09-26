# CMAX++ : Leveraging Experience in Planning and Execution using Inaccurate Models
## BibTeX Citation

```
@article{DBLP:journals/corr/abs-2009-09942,
  author    = {Anirudh Vemula and
               J. Andrew Bagnell and
               Maxim Likhachev},
  title     = {{CMAX++} : Leveraging Experience in Planning and Execution using Inaccurate
               Models},
  journal   = {CoRR},
  volume    = {abs/2009.09942},
  year      = {2020},
  url       = {https://arxiv.org/abs/2009.09942},
  archivePrefix = {arXiv},
  eprint    = {2009.09942},
  timestamp = {Wed, 23 Sep 2020 15:51:46 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2009-09942.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

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
