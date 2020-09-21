from szeth.envs.car_dynamics import Car
import Box2D


def step():
    car.step(1/50.0)
    world.Step(1.0/50, 6*30, 2*30)


def print_car_state():
    x, y = car.hull.position
    theta = car.hull.angle
    v = car.hull.linearVelocity


world = Box2D.b2World((0, 0))
car = Car(world, 0, 0, 0)

delta = 0.5
