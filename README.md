# SC2Black
In order to run any of the code, you will need a lot of things installed and setup. Please follow the setup guides at https://github.com/deepmind/pysc2 and https://www.tensorflow.org/install (or https://www.tensorflow.org/install/gpu for GPU support). Use `pip install` to add any missing packages. We found the easiest tensorflow setup to be using anaconda.
## StarCraftII Minigames
To train the StarCraftII agent, use the command `py main.py`

Extra flags include:

 --continuation=True (run a pretrained saved model)
 
 --map=DefeatRoaches (set the starcraft minigame to run)
 
 --parallel=# (set the number of threads to train on)
 
 
The full list can be seen in main.py.
## OpenAI Gym

To train the OpenAI gym agent, cd `rl_agent/gym_test`, and run `py run_gym.py`

There are similar flags to the SC2 agent. Use --environment to set the environment (Pong-v0, Breakout-v0, etc.)
