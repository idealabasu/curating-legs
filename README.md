# Curating Tunable, Compliant Legs for Specialized Tasks
This repository contains the code for this project. A project overview is available on the [project website](https://iicfcii.github.io/curating-legs-web/). 

## Setup
Please either clone or download the repository. 

We used [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) to manage the Python environment. Once Miniconda is installed and activated in the terminal, please navigate to the code's folder and run the following command. 

```
conda env create -f environment.yml
```
This will create a new environment called "curating-legs" and install all the dependencies. 


## Running the Scripts
Several scripts are runnable to realize various aspects of the project. All commands should be executed in the root folder. 

The data for the leg designs, locomoiton policy, and simulation evalutation results are already included in the data folder. Generating them with the code took two days with our workstation computer (AMD Ryzen Threadripper PRO 7975WX, NVIDIA GeForce RTX 4090). 

### Design Legs
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
Test out the policy with the specified leg design (0-408), longitudinal speed (m/s), and turning speed (rad/s). 
```
python -m rl.exp 119 0.6 0.4
```

Analyze the performance trends against the design parameters.  
```
python -m rl.analyze
```

Train the policy. This took about 21 hours. 
```
python -m rl.trainer
```

Evaluate the policy. This took about 11 hours. 
```
python -m rl.eval a
```

## Support
If you have any questions, please create an issue or contact Fuchen Chen at fchen65@asu.edu. 