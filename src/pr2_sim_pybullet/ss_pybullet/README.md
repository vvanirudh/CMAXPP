# ss-pybullet
STRIPStream + pybullet

<!--![Alt text](images/test.png?raw=true "Title")-->
<!--img src="images/pr2.png" height="300">&emsp;<img src="images/kuka.png" height="300"-->

## Installation

Install PyBullet on OS X or Linux using: 
```
$ pip install numpy pybullet
$ git clone --recurse-submodules https://github.com/caelan/ss-pybullet.git
$ cd ss-pybullet
$ git pull --recurse-submodules
```

## IKFast Compilation

### Franka Panda

```
$ cd ss-pybullet/pybullet_tools/ikfast/franka_panda
$ python2 setup.py build
```

### MOVO

```
$ cd ss-pybullet/pybullet_tools/ikfast/movo
$ python2 setup.py -a left
$ python2 setup.py -a arm
```

### PR2

```
$ cd ss-pybullet/pybullet_tools/ikfast/pr2
$ python2 setup.py build
```

## Tests

1) Test pybullet - ```python -c 'import pybullet'```

## Examples

Examples:
* Kuka IIWA pick planning - ```$ python -m examples.test_kuka_pick```
* PR2 motion planning - ```$ python -m examples.test_pr2_motion```
* PR2 teleoperation - ```$ python -m examples.teleop_pr2```
* PR2 visibility - ```$ python -m examples.test_visibility```
* PR2 copying - ```$ python -m examples.test_clone```
* TurtleBot collisions - ```$ python -m examples.test_turtlebot```
* Franka Panda workspace planning - ```$ python -m examples.test_franka```
* MOVO random workspace planning - ```$ python -m examples.test_movo```
* Dropping beads - ```$ python -m examples.test_water```

<img src="images/turtlebot.png" height="150">&emsp;<img src="images/movo.png" height="150">

## PDDLStream Examples

See the following examples: https://github.com/caelan/pddlstream/tree/master/examples/pybullet

[<img src="https://img.youtube.com/vi/3HJrkgIGK7c/0.jpg" height="200">](https://www.youtube.com/watch?v=3HJrkgIGK7c)
[<img src="https://img.youtube.com/vi/oWr6m12nXcM/0.jpg" height="200">](https://www.youtube.com/watch?v=oWr6m12nXcM)

## PyBullet Resources
* pip - https://pypi.python.org/pypi/pybullet
* Quickstart - https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/
* Forum - https://pybullet.org/Bullet/phpBB3/
* Wordpress - https://pybullet.org/wordpress/
* Examples - https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/examples
* Bindings - https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/pybullet.c
* PyBullet Planning - https://github.com/yijiangh/pybullet_planning
* PyPI - https://pypi.org/project/pybullet-planning/

## Bullet Resources
* Github - https://github.com/bulletphysics/bullet3

## Citation

Caelan Reed Garrett. PyBullet Planning. https://pypi.org/project/pybullet-planning/. 2018.
