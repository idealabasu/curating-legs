# Curating Tunable, Compliant Legs for Specialized Tasks
This project proposes methods for tuning robot legs for specialized tasks. The [project website](https://idealab.asu.edu/curating-legs-web/) has more information. This project has two published papers. 

- F. Chen and D. M. Aukes, “Curating Tunable, Compliant Legs for Specialized Tasks,” International Journal of Robotics Research (IJRR), Accepted, 2025.
- F. Chen and D. M. Aukes, “Informed Repurposing of Quadruped Legs for New Tasks,” in 2025 IEEE International Conference on Robotics & Automation (ICRA), Accepted, 2025.

This repository contains the code for this project. 

## Setup
Please either clone or download the repository. 

We used [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) to manage the Python environment. Once Miniconda is installed and activated in the terminal, please navigate to the code's folder and run the following command. 

```
conda env create -f environment.yml
```
This will create a new environment called "curating-legs" and install all the dependencies. 


## Running the Scripts
Several scripts are runnable to realize various aspects of the project. All commands should be executed in the root folder. 

The data for the leg designs, locomoiton policies, and simulation evalutation results are already included in the data folder. Generating all of them with the code took several days with our workstation computer (AMD Ryzen Threadripper PRO 7975WX, NVIDIA GeForce RTX 4090). 

### Leg Deisgn
Generate a leg with the specificed leg offset (m), leg travel length (m), input range (rad), parallel stiffness (Nm/rad), and series stiffness (Nm/rad). 
```
python -m leg.opt 0.04 0.04 0.50 0.10 1.00
```

Visualize the design space. 
```
python -m leg.plot s
```

Span the design space. This took about 11 hours. 
```
python -m leg.search
```

### Locomotion Policy
Test the policy with the specified leg design (0-408), longitudinal speed command (m/s), and turning speed command (rad/s). 
```
python -m rl.exp 119 0.6 0.4
```

Analyze the performance metrics against the design parameters.  
```
python -m rl.analyze
```

Train the policy. This took about 21 hours. 
```
python -m rl.train
```

Evaluate the policy. This took about 11 hours. 
```
python -m rl.eval a
```

### Force-based Locomotion Policy
Test the policy with the specified leg combination (0-14), x and z offset (m) of the force application point, and x and z value (N) of the force. 
```
python -m fbrl.exp 1 -0.1 0.05 1 -2
```

Analyze the performance metrics against the leg combinations.  
```
python -m fbrl.analyze
```

Train the policy. This took about 110 hours. 
```
python -m fbrl.train
```

Evaluate the policy. This took several hours. 
```
python -m fbrl.eval
```

## Support
If you have any questions, please create an issue or contact Fuchen Chen at fchen65@asu.edu. 