#!/bin/bash
# Scritp to run the direct expert experiments on the movie dataset

# #### with fitted parameters version 3, log expert and power cost? 
python ./experiment/traintest.py --train "imdb" --cost-model "1,0" --cost-function "uniform"  --fixk 10 --trials 10 --budget 20000 --step-size 1 --bootstrap 10  --accu-model "[[10, 0.598], [25, 0.842], [50, 0.794], [75, 0.9], [101, 0.905], [124, 1.0], [147, 1.0]]"  --expert "direct"  --maxiter 2000 > ../rnd-fix-trueoracle/movie/uni-cost/MOVIE-FIX-10-LINCOST-DIREXP-A1.TXT  & 
python ./experiment/traintest.py --train "imdb" --cost-model "1,0" --cost-function "uniform"  --fixk 20 --trials 10 --budget 20000 --step-size 1 --bootstrap 10  --accu-model "[[10, 0.598], [25, 0.842], [50, 0.794], [75, 0.9], [101, 0.905], [124, 1.0], [147, 1.0]]"  --expert "direct"  --maxiter 2000 > ../rnd-fix-trueoracle/movie/uni-cost/MOVIE-FIX-20-LINCOST-DIREXP-A1.TXT  & 
python ./experiment/traintest.py --train "imdb" --cost-model "1,0" --cost-function "uniform"  --fixk 30 --trials 10 --budget 20000 --step-size 1 --bootstrap 10  --accu-model "[[10, 0.598], [25, 0.842], [50, 0.794], [75, 0.9], [101, 0.905], [124, 1.0], [147, 1.0]]"  --expert "direct"  --maxiter 2000 > ../rnd-fix-trueoracle/movie/uni-cost/MOVIE-FIX-30-LINCOST-DIREXP-A1.TXT  & 
python ./experiment/traintest.py --train "imdb" --cost-model "1,0" --cost-function "uniform"  --fixk 40 --trials 10 --budget 20000 --step-size 1 --bootstrap 10  --accu-model "[[10, 0.598], [25, 0.842], [50, 0.794], [75, 0.9], [101, 0.905], [124, 1.0], [147, 1.0]]"  --expert "direct"  --maxiter 2000 > ../rnd-fix-trueoracle/movie/uni-cost/MOVIE-FIX-40-LINCOST-DIREXP-A1.TXT  & 
python ./experiment/traintest.py --train "imdb" --cost-model "1,0" --cost-function "uniform"  --fixk 50 --trials 10 --budget 20000 --step-size 1 --bootstrap 10  --accu-model "[[10, 0.598], [25, 0.842], [50, 0.794], [75, 0.9], [101, 0.905], [124, 1.0], [147, 1.0]]"  --expert "direct"  --maxiter 2000 > ../rnd-fix-trueoracle/movie/uni-cost/MOVIE-FIX-50-LINCOST-DIREXP-A1.TXT  &
python ./experiment/traintest.py --train "imdb" --cost-model "1,0" --cost-function "uniform"  --fixk 60 --trials 10 --budget 20000 --step-size 1 --bootstrap 10  --accu-model "[[10, 0.598], [25, 0.842], [50, 0.794], [75, 0.9], [101, 0.905], [124, 1.0], [147, 1.0]]"  --expert "direct"  --maxiter 2000 > ../rnd-fix-trueoracle/movie/uni-cost/MOVIE-FIX-60-LINCOST-DIREXP-A1.TXT  &
python ./experiment/traintest.py --train "imdb" --cost-model "1,0" --cost-function "uniform"            --trials 10 --budget 20000 --step-size 1 --bootstrap 10  --accu-model "[[10, 0.598], [25, 0.842], [50, 0.794], [75, 0.9], [101, 0.905], [124, 1.0], [147, 1.0]]"  --expert "direct"  --maxiter 2000 > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-RND-UNICOST-DIREXP-A1.TXT  


## revised cost model - version 2
# python ./experiment/traintest.py --train "imdb" --cost-model "1,0" --cost-function "uniform" --trials 10 --budget 20000 --step-size 1 --bootstrap 10  --accu-model "0.5,0" --expert "fixed"  --maxiter 2000 > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-RND-UNICOST-EXP50-A1.TXT  &
# python ./experiment/traintest.py --train "imdb" --cost-model "1,0" --cost-function "uniform" --trials 10 --budget 20000 --step-size 1 --bootstrap 10  --accu-model "0.6,0" --expert "fixed"  --maxiter 2000 > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-RND-UNICOST-EXP60-A1.TXT  &
# python ./experiment/traintest.py --train "imdb" --cost-model "1,0" --cost-function "uniform" --trials 10 --budget 20000 --step-size 1 --bootstrap 10  --accu-model "0.7,0" --expert "fixed"  --maxiter 2000 > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-RND-UNICOST-EXP70-A1.TXT  &
# python ./experiment/traintest.py --train "imdb" --cost-model "1,0" --cost-function "uniform" --trials 10 --budget 20000 --step-size 1 --bootstrap 10  --accu-model "0.8,0" --expert "fixed"  --maxiter 2000 > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-RND-UNICOST-EXP80-A1.TXT  &
# python ./experiment/traintest.py --train "imdb" --cost-model "1,0" --cost-function "uniform" --trials 10 --budget 20000 --step-size 1 --bootstrap 10  --accu-model "0.9,0" --expert "fixed"  --maxiter 2000 > ../rnd-fix-trueoracle/movie/log-cost/MOVIE-RND-UNICOST-EXP90-A1.TXT  
