# Running scripts

## Joint formulation

* *sr_cali.sh* runs the python code 

## Example

To run IMDB data with uniform cost LR-Adapt, SR method, 1250 budget after bootstrap, stepsize 10 and maximum interations of 125

```bash
./sr_cali.sh     "imdb"       ${costmovie}  "uniform"  0     1      10     100   1250     125     "SR-MAX-LRL1-1-CALI-BT100"  "sr"  "NOCHEAT"  "sr-oracle-test/results-calibrated/" "lradaptv2" 2 "--calibrate" &
``` 

Important parameters:
./sr_cali.sh    **name_train_data**   cost_values    cost_function   threshold   **classifier_penalty**  step_size   bootstrap   budget   max_iterations   prefix_file_name method_name cheating_method output_folder **classifier_name** cleaning_min **calibration**