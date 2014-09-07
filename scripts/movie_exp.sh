#!/bin/bash
## MOVIE DATASET RND FIX K EXPERIMENTS 
# python ./experiment/traintest.py --train "imdb" --cost-model "5.387,-8.1752" --cost-function "log"     --fixk 10 --trials 10 --budget 20000 --step-size 1 --bootstrap 10  > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-FIX-10-LOGCOST-A1.TXT  &
# python ./experiment/traintest.py --train "imdb" --cost-model "5.387,-8.1752" --cost-function "log"     --fixk 20 --trials 10 --budget 20000 --step-size 1 --bootstrap 10  > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-FIX-20-LOGCOST-A1.TXT  &
# python ./experiment/traintest.py --train "imdb" --cost-model "5.387,-8.1752" --cost-function "log"     --fixk 30 --trials 10 --budget 20000 --step-size 1 --bootstrap 10  > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-FIX-30-LOGCOST-A1.TXT  &
# python ./experiment/traintest.py --train "imdb" --cost-model "5.387,-8.1752" --cost-function "log"     --fixk 40 --trials 10 --budget 20000 --step-size 1 --bootstrap 10  > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-FIX-40-LOGCOST-A1.TXT  &
# python ./experiment/traintest.py --train "imdb" --cost-model "5.387,-8.1752" --cost-function "log"     --fixk 50 --trials 10 --budget 20000 --step-size 1 --bootstrap 10  > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-FIX-50-LOGCOST-A1.TXT  &

#### with different bootstrap = quality and quantity
# python ./experiment/traintest.py --train "imdb" --cost-model "5.387,-8.1752" --cost-function "log"     --fixk 10 --trials 10 --budget 20000 --step-size 1 --bootstrap 500  --maxiter 2000 > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-FIX-10-LOGCOST-A1-BT500.TXT  &
# python ./experiment/traintest.py --train "imdb" --cost-model "5.387,-8.1752" --cost-function "log"     --fixk 50 --trials 10 --budget 20000 --step-size 1 --bootstrap 500  --maxiter 2000 > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-FIX-50-LOGCOST-A1-BT500.TXT  &
# python ./experiment/traintest.py --train "imdb" --cost-model "1,0"           --cost-function "uniform"           --trials 10 --budget 2000  --step-size 1 --bootstrap 10   --maxiter 2000 > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-RND-A1.TXT  


## revised cost model - version 2
# python ./experiment/traintest.py --train "imdb" --cost-model "0.0642,8.4204" --cost-function "linear"     --fixk 10 --trials 10 --budget 20000 --step-size 1 --bootstrap 10  > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-FIX-10-LOGCOST-A1.TXT  &
# python ./experiment/traintest.py --train "imdb" --cost-model "0.0642,8.4204" --cost-function "linear"     --fixk 20 --trials 10 --budget 20000 --step-size 1 --bootstrap 10  > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-FIX-20-LOGCOST-A1.TXT  &
# python ./experiment/traintest.py --train "imdb" --cost-model "0.0642,8.4204" --cost-function "linear"     --fixk 30 --trials 10 --budget 20000 --step-size 1 --bootstrap 10  > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-FIX-30-LOGCOST-A1.TXT  &
# python ./experiment/traintest.py --train "imdb" --cost-model "0.0642,8.4204" --cost-function "linear"     --fixk 40 --trials 10 --budget 20000 --step-size 1 --bootstrap 10  > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-FIX-40-LOGCOST-A1.TXT  &
# python ./experiment/traintest.py --train "imdb" --cost-model "0.0642,8.4204" --cost-function "linear"     --fixk 50 --trials 10 --budget 20000 --step-size 1 --bootstrap 10  > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-FIX-50-LOGCOST-A1.TXT  &

## revised cost model - version 2
# python ./experiment/traintest.py --train "imdb" --cost-model "0.0642,8.4204" --cost-function "linear"     --fixk 10 --trials 10 --budget 20000 --step-size 1 --bootstrap 10   --expert "log"  --maxiter 2000 > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-FIX-10-LINCOST-LOGEXP-A1.TXT  &
# python ./experiment/traintest.py --train "imdb" --cost-model "0.0642,8.4204" --cost-function "linear"     --fixk 20 --trials 10 --budget 20000 --step-size 1 --bootstrap 10   --expert "log"  --maxiter 2000 > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-FIX-20-LINCOST-LOGEXP-A1.TXT  &
# python ./experiment/traintest.py --train "imdb" --cost-model "0.0642,8.4204" --cost-function "linear"     --fixk 30 --trials 10 --budget 20000 --step-size 1 --bootstrap 10   --expert "log"  --maxiter 2000 > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-FIX-30-LINCOST-LOGEXP-A1.TXT  &
# python ./experiment/traintest.py --train "imdb" --cost-model "0.0642,8.4204" --cost-function "linear"     --fixk 40 --trials 10 --budget 20000 --step-size 1 --bootstrap 10   --expert "log"  --maxiter 2000 > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-FIX-40-LINCOST-LOGEXP-A1.TXT  &
# python ./experiment/traintest.py --train "imdb" --cost-model "0.0642,8.4204" --cost-function "linear"     --fixk 50 --trials 10 --budget 20000 --step-size 1 --bootstrap 10   --expert "log"  --maxiter 2000 > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-FIX-50-LINCOST-LOGEXP-A1.TXT  &
# python ./experiment/traintest.py --train "imdb" --cost-model "1,0" --cost-function "uniform"                        --trials 10 --budget 20000 --step-size 1 --bootstrap 10   --expert "log"  --maxiter 2000 > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-RND-UNICOST-LOGEXP-A1.TXT  


# #### with fitted parameters version 3, log expert and power cost? 
python ./experiment/traintest.py --train "imdb" --cost-model "0.0642,8.4204" --cost-function "linear"     --fixk 10 --trials 10 --budget 20000 --step-size 1 --bootstrap 10  --accu-model "0.134690536616,0.320550112092"  --expert "log"  --maxiter 2000 > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-FIX-10-LINCOST-LOGEXP-A1.TXT  &
python ./experiment/traintest.py --train "imdb" --cost-model "0.0642,8.4204" --cost-function "linear"     --fixk 20 --trials 10 --budget 20000 --step-size 1 --bootstrap 10  --accu-model "0.134690536616,0.320550112092"  --expert "log"  --maxiter 2000 > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-FIX-20-LINCOST-LOGEXP-A1.TXT  &
python ./experiment/traintest.py --train "imdb" --cost-model "0.0642,8.4204" --cost-function "linear"     --fixk 30 --trials 10 --budget 20000 --step-size 1 --bootstrap 10  --accu-model "0.134690536616,0.320550112092"  --expert "log"  --maxiter 2000 > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-FIX-30-LINCOST-LOGEXP-A1.TXT  &
python ./experiment/traintest.py --train "imdb" --cost-model "0.0642,8.4204" --cost-function "linear"     --fixk 40 --trials 10 --budget 20000 --step-size 1 --bootstrap 10  --accu-model "0.134690536616,0.320550112092"  --expert "log"  --maxiter 2000 > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-FIX-40-LINCOST-LOGEXP-A1.TXT  &
python ./experiment/traintest.py --train "imdb" --cost-model "0.0642,8.4204" --cost-function "linear"     --fixk 50 --trials 10 --budget 20000 --step-size 1 --bootstrap 10  --accu-model "0.134690536616,0.320550112092"  --expert "log"  --maxiter 2000 > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-FIX-50-LINCOST-LOGEXP-A1.TXT  
# python ./experiment/traintest.py --train "imdb" --cost-model "1,0" --cost-function "uniform"                        --trials 10 --budget 20000 --step-size 1 --bootstrap 10  --accu-model "0.134690536616,0.320550112092"  --expert "log"  --maxiter 2000 > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-RND-UNICOST-LOGEXP-A1.TXT  

#### with fitted parameters version 3, log expert and power cost? 
# python ./experiment/traintest.py --train "imdb" --cost-model "0.0642,8.4204" --cost-function "linear"     --fixk 60  --trials 10 --budget 20000 --step-size 1 --bootstrap 10  --accu-model "0.134690536616,0.320550112092"  --expert "log"  --maxiter 2000 > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-FIX-60-LINCOST-LOGEXP-A1.TXT  &
# python ./experiment/traintest.py --train "imdb" --cost-model "0.0642,8.4204" --cost-function "linear"     --fixk 70  --trials 10 --budget 20000 --step-size 1 --bootstrap 10  --accu-model "0.134690536616,0.320550112092"  --expert "log"  --maxiter 2000 > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-FIX-70-LINCOST-LOGEXP-A1.TXT  &
# python ./experiment/traintest.py --train "imdb" --cost-model "0.0642,8.4204" --cost-function "linear"     --fixk 80  --trials 10 --budget 20000 --step-size 1 --bootstrap 10  --accu-model "0.134690536616,0.320550112092"  --expert "log"  --maxiter 2000 > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-FIX-80-LINCOST-LOGEXP-A1.TXT  &
# python ./experiment/traintest.py --train "imdb" --cost-model "0.0642,8.4204" --cost-function "linear"     --fixk 90  --trials 10 --budget 20000 --step-size 1 --bootstrap 10  --accu-model "0.134690536616,0.320550112092"  --expert "log"  --maxiter 2000 > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-FIX-90-LINCOST-LOGEXP-A1.TXT  &
# python ./experiment/traintest.py --train "imdb" --cost-model "0.0642,8.4204" --cost-function "linear"     --fixk 100 --trials 10 --budget 20000 --step-size 1 --bootstrap 10  --accu-model "0.134690536616,0.320550112092"  --expert "log"  --maxiter 2000 > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-FIX-100-LINCOST-LOGEXP-A1.TXT  
# python ./experiment/traintest.py --train "imdb" --cost-model "1,0" --cost-function "uniform"                         --trials 10 --budget 20000 --step-size 1 --bootstrap 10  --accu-model "0.134690536616,0.320550112092"  --expert "log"  --maxiter 2000 > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-RND-UNICOST-LOGEXP-A1.TXT  





## revised cost model - version 2
# python ./experiment/traintest.py --train "imdb" --cost-model "1,0" --cost-function "uniform" --trials 10 --budget 20000 --step-size 1 --bootstrap 10  --accu-model "0.5,0" --expert "fixed"  --maxiter 2000 > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-RND-UNICOST-EXP50-A1.TXT  &
# python ./experiment/traintest.py --train "imdb" --cost-model "1,0" --cost-function "uniform" --trials 10 --budget 20000 --step-size 1 --bootstrap 10  --accu-model "0.6,0" --expert "fixed"  --maxiter 2000 > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-RND-UNICOST-EXP60-A1.TXT  &
# python ./experiment/traintest.py --train "imdb" --cost-model "1,0" --cost-function "uniform" --trials 10 --budget 20000 --step-size 1 --bootstrap 10  --accu-model "0.7,0" --expert "fixed"  --maxiter 2000 > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-RND-UNICOST-EXP70-A1.TXT  &
# python ./experiment/traintest.py --train "imdb" --cost-model "1,0" --cost-function "uniform" --trials 10 --budget 20000 --step-size 1 --bootstrap 10  --accu-model "0.8,0" --expert "fixed"  --maxiter 2000 > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-RND-UNICOST-EXP80-A1.TXT  &
# python ./experiment/traintest.py --train "imdb" --cost-model "1,0" --cost-function "uniform" --trials 10 --budget 20000 --step-size 1 --bootstrap 10  --accu-model "0.9,0" --expert "fixed"  --maxiter 2000 > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-RND-UNICOST-EXP90-A1.TXT  
