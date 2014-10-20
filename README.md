## ANYTIME ACTIVE LEARNING 

This repository contains the code used for experiments in Anytime active learning research published on the following paper: 

Ramirez-Loaiza, Maria E., Aron Culotta, and Mustafa Bilgic. 2014. “Anytime Active Learning.” In *AAAI Conference on Artificial Intelligence.*

## Content
In this file: 
- Repository
- Configuration
- How to run


## Repository

Structure of the repository 

- datautil -> various functions to load the data
- experiment
    - |_ anytime.py --> active learning loop for anytime AL
    - |_ experiment_utils.py --> printing resutls
- strategy
    - |_ baselearner.py  --> basic student class
    - |_ randomsampling.py --> AnytimeLearner 
    - |_ structured.py --> AnytimeLearner with structured reading classes
- expert
    - |_ baseexpert.py --> you need the class NeutralityExpert
- sentence
    - |_ baseexpert.py --> you need the class NeutralityExpert
- scripts --> contains all scripts used to run experiments for this research

## Configuration 

Coming soon. 

## How to run an experiment

Coming soon. 