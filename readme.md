# BFW-PPO
The Battle of Wesnoth Proximal Policy Optimization (PPO) AI system implementaion

The [GYM environment](gym-bfw)  used to connect Wesnoth to the BFW-PPO script was created by [Dominik Stelter](https://github.com/DStelter94/ARLinBfW) as well as some wappers to initialize the environment in python. The method for how Wesnoth communicates with python was first introduced by this [discussion](https://forums.wesnoth.org/viewtopic.php?f=10\&t=51061) on the Wesnoth forms. The PythonAddon incorporated into Wesnoth was also developed by [Dominik Stelter](https://github.com/DStelter94/ARLinBfW) This addon adds functionality for the Python script to interact with the Wesnoth game.

Dominik implemented the communication between Python and Lua by having the The Lua code return data (mostly the observation data of the environment) through the standard output on the console. This output is then sent to Python as a standard GYM environment. When actions are taken by the agent this is again written to a Lua file located in the Wesnoth Addon were the action is taken in the game.
