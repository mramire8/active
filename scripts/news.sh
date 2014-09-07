#!/bin/bash
python ./experiment/traintest.py --train "20news" --cost-model "1,0" --cost-function "uniform" --trials 10 --budget 20000 --step-size 1 --bootstrap 10  --accu-model "1,0"  --expert "true"  --maxiter 550 > ../rnd-fix-trueoracle/news/NEWS-RND-UNICOST-TRUEEXP-A1.TXT  

