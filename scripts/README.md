# Running scripts

## Joint formulation - Structured reading
To run an experiment of Structured reading with different parameters use the **sr_cali.sh** script file. The paramters and how to use are described in the next sections. Examples of how run the script are in the file **sr_joint.sh** (located in this same directory.

* *sr_cali.sh* runs the python code. This script should be run from the root directory of the code.  

### Example

To run IMDB data with uniform cost LR-Adapt, SR method, 1250 budget after bootstrap, stepsize 10 and maximum interations of 125

```bash
./sr_cali.sh     "imdb"       "[[1,1]]"  "uniform"  0     1      10     100   1250     125     "SR-MAX-LRL1-1-CALI-BT100"  "sr"  "NOCHEAT"  "sr-oracle-test/results-calibrated/" "lradaptv2" 2 "--calibrate" &
``` 

### Important parameters:

```
./sr_cali.sh    **name_train_data**   cost_values    cost_function   threshold   **classifier_penalty**  step_size   bootstrap   budget   max_iterations   prefix_file_name method_name cheating_method output_folder **classifier_name** cleaning_min **calibration**
```

* name\_train\_data: possible values: imdb, aviation
* classifier_penality: scalar
* classifier_name: possible values: lr, lrl2, lradapt, lradaptv2, mnb
* calibration: if we need SR to calibrate using z-scores use --calibrate flag
