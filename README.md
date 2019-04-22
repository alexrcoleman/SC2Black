# SC2Black
To train the StarCraftII agent, use the command `py main.py`

Extra flags include:

 --continuation=True (run a pretrained saved model)
 
 --map=DefeatRoaches (set the starcraft minigame to run)
 
 --parallel=# (set the number of threads to train on)
 
The full list can be seen in main.py

To train the OpenAI gym agent, cd `rl_agent/gym_test`, and run `py run_gym.py`

There are similar flags to the SC2 agent. Use --environment to set the environment (Pong-v0, Breakout-v0, etc.)
