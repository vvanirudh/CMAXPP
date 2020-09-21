from szeth.envs.car_racing.car_racing_env_revised import make_car_racing_env

env = make_car_racing_env(seed=None,
                          friction_params={'ice': 0.1})
env.reset()

env.render()
input()
