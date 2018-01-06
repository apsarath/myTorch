# Blocksworld

Actions:

left arrow : move left

right arrow: move right

up arrow : pick up block

down arrow : drop block

numeric 0: To save the current state as jpeg (both current state and target)


Package dependencies:

[StackOverflow post on installing pygame and its dependencies](https://stackoverflow.com/questions/30743194/pygame-installation-mac-os-x)

## To Play the Game

```python Blocksworld.py```

#Parameters to change

NO_OF_BLOCKS : total number of blocks
MAX_LOCATIONS : Number of navigable locations
MAX_OBJECTS : Maximum object a location can hold (excluding the agent)

## To import the game module

```import blocksworldEnv```

```env = blocksworldEnv.Environment('None')```

env? To see other options while instantiating blocksworld

```init = env.reset('None')```

to set the blocks uncolored or for colored blocks use,

``` init = env.reset('colors')```

to select a target configuration from a file, make sure the target is described in the prescribed template and then,

``` init = env.reset('colors',<file_name.ext>, problem_index) ```

problem_index will be the line number in which the problem is described

init is returned with the initial observation [goal_achieved, current_config, target_config]

```O = env.step(int)```

O is returned with the observation list as mentioned for init

```O = env.random_step()```

for uniform sampling of action
