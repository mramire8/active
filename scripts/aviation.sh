#!/bin/bash

#aviation random
##  vanilla active learning

python ./experiment/traintestLR.py --train "aviation" --cost-model "1,0" --cost-function "uniform" --trials 5 --budget 5000 --step-size 10 --bootstrap 50  --accu-model "1,0"  --expert "true"  --maxiter 500   > ../rnd-fix-trueoracle/aviation/AVI-RND-UNICOST-TRUEEXP-LR.TXT  2> ../rnd-fix-trueoracle/aviation/AVI-RND-UNICOST-TRUEEXP-LR.TXT
