# EXP experiments
python -m szeth.experiment.experiment_car_racing --agent adaptive_cmaxpp --alpha_expt --alpha 10 --schedule exp --exp_step 0.9
python -m szeth.experiment.experiment_car_racing --agent adaptive_cmaxpp --alpha_expt --alpha 100 --schedule exp --exp_step 0.9
python -m szeth.experiment.experiment_car_racing --agent adaptive_cmaxpp --alpha_expt --alpha 1000 --schedule exp --exp_step 0.9
python -m szeth.experiment.experiment_car_racing --agent adaptive_cmaxpp --alpha_expt --alpha 10 --schedule exp --exp_step 0.7
python -m szeth.experiment.experiment_car_racing --agent adaptive_cmaxpp --alpha_expt --alpha 100 --schedule exp --exp_step 0.7
python -m szeth.experiment.experiment_car_racing --agent adaptive_cmaxpp --alpha_expt --alpha 1000 --schedule exp --exp_step 0.7
python -m szeth.experiment.experiment_car_racing --agent adaptive_cmaxpp --alpha_expt --alpha 10 --schedule exp --exp_step 0.5
python -m szeth.experiment.experiment_car_racing --agent adaptive_cmaxpp --alpha_expt --alpha 100 --schedule exp --exp_step 0.5
python -m szeth.experiment.experiment_car_racing --agent adaptive_cmaxpp --alpha_expt --alpha 1000 --schedule exp --exp_step 0.5

# LINEAR experiments
python -m szeth.experiment.experiment_car_racing --agent adaptive_cmaxpp --alpha_expt --alpha 10 --schedule linear
python -m szeth.experiment.experiment_car_racing --agent adaptive_cmaxpp --alpha_expt --alpha 100 --schedule linear
python -m szeth.experiment.experiment_car_racing --agent adaptive_cmaxpp --alpha_expt --alpha 200 --schedule linear

# TIME experiments
python -m szeth.experiment.experiment_car_racing --agent adaptive_cmaxpp --alpha_expt --alpha 10 --schedule time
python -m szeth.experiment.experiment_car_racing --agent adaptive_cmaxpp --alpha_expt --alpha 100 --schedule time
python -m szeth.experiment.experiment_car_racing --agent adaptive_cmaxpp --alpha_expt --alpha 1000 --schedule time

# STEP experiments
python -m szeth.experiment.experiment_car_racing --agent adaptive_cmaxpp --alpha_expt --alpha 10 --schedule step --step_freq 5
python -m szeth.experiment.experiment_car_racing --agent adaptive_cmaxpp --alpha_expt --alpha 100 --schedule step --step_freq 5
python -m szeth.experiment.experiment_car_racing --agent adaptive_cmaxpp --alpha_expt --alpha 200 --schedule step --step_freq 5
python -m szeth.experiment.experiment_car_racing --agent adaptive_cmaxpp --alpha_expt --alpha 10 --schedule step --step_freq 10
python -m szeth.experiment.experiment_car_racing --agent adaptive_cmaxpp --alpha_expt --alpha 100 --schedule step --step_freq 10
python -m szeth.experiment.experiment_car_racing --agent adaptive_cmaxpp --alpha_expt --alpha 200 --schedule step --step_freq 10
python -m szeth.experiment.experiment_car_racing --agent adaptive_cmaxpp --alpha_expt --alpha 10 --schedule step --step_freq 20
python -m szeth.experiment.experiment_car_racing --agent adaptive_cmaxpp --alpha_expt --alpha 100 --schedule step --step_freq 20
python -m szeth.experiment.experiment_car_racing --agent adaptive_cmaxpp --alpha_expt --alpha 200 --schedule step --step_freq 20
