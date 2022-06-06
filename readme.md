# BFW-PPO
The Battle of Wesnoth Proximal Policy Optimization (PPO) AI system implementaion

The [GYM environment](gym-bfw)  used to connect Wesnoth to the BFW-PPO script was created by [Dominik Stelter](https://github.com/DStelter94/ARLinBfW) as well as some wappers to initialize the environment in python. The method for how Wesnoth communicates with python was first introduced by this [discussion](https://forums.wesnoth.org/viewtopic.php?f=10\&t=51061) on the Wesnoth forms. The PythonAddon incorporated into Wesnoth was also developed by [Dominik Stelter](https://github.com/DStelter94/ARLinBfW) This addon adds functionality for the Python script to interact with the Wesnoth game.

Dominik implemented the communication between Python and Lua by having The Lua code return data (mostly the observation data of the environment) through the standard output on the console. This output is then sent to Python as a standard GYM environment. When actions are taken by the agent this is again written to a Lua file located in the Wesnoth Addon where the action is taken in the game.

##Development and Testing Environment Setup under Windows
Install Windows Services for Linux from Windows command prompt with elevated privileges

```wsl --install```

Start Ubuntu app from Windows start menu, create a user account. After logging into Linux, clone BFW-PPO project

```git clone https://github.com/ivanbalak/BFW-PPO.git```

Start Visual Studio Code in Linux environment from the project folder, then continue with section

``` cd bff-ppo ```
``` code .  ```

##Linux Environment Setup
Install Battle for Wesnoth

```sudo apt-get install wesnoth```

If the project is not yet cloned in previous steps, clone BFW-PPO project

```git clone https://github.com/ivanbalak/BFW-PPO.git```

Copy Bfw-gym add-on to wesnoth add-on folder

```cp PythonAddon ~/.config/wesnoth-1.14/data/add-ons```

Create a folder for input files for use by BFW gym

```mkdir ~/.config/wesnoth-1.14/data/input```

Configure Python environment and project dependencies. The project was tested with Python 3.8, but it can be compatible with later versions of Python. Below is the command sequence specific to Python 3.8: 

```python3.8 -m venv env ```
```source env/bin/activate```
```pip install -U pip wheel setuptools```
```pip3 install -e ./gym-bfw```
```pip3 install -r requirements.txt ```

# General Operating Considerations
The PPO AI for the BFW was designed and tested in the Windows Services for Linux environment with Ubuntu Linux distribution. The software should also function in a standalone Linux computer. The software can be launched from the Linux prompt and from the Visual Studio Code Integrated Development Environment.

In Linux prompt, change the current directory to BFW PPO project directory, for example

```cd ~/bfd-ppo```

Command to start training

```python PPO/train.py```

Display usage and optional training command arguments help screen

```train.py [-h]``

##Operation from VS Code IDE
In WSL Linux prompt, change the current directory to BFW PPO project directory, for example

```cd ~/bfd-ppo```

Start VS Code IDE

``` code .```

open Run Debug panel - type Ctrl-Shift-D or click Run & Debug button. to start training, select [PPO train single player] from the drop-down menu and click the
green arrow next to it.