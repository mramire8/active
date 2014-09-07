SAVEIFS=$IFS
IFS=$(echo -en "\n\b")

cost_model="[[10, 5.2], [25, 6.5], [50, 7.6], [75, 9.1], [100, 10.3], [125, 13.6]]" 
echo "Uncertainty full "
python ./experiment/unc_fixk.py --train "aviation"  --fixk -1 --neutral-threshold 0.3 --trials 5 --budget 15000 --step-size 10 --bootstrap 2 --maxiter 150 --cost-function "direct" --cost-model ${cost_model} --expert-penalty 1 > ../results/fixk-neutral/MOVIE-UNC-NEUEXP-LR.TXT  2> ../results/fixk-neutral/MOVIE-UNC-NEUEXP-LR.TXT  &

INVESTIGATION 
echo "anytime with no utility"
./anytime.sh "aviation"   0.3   ${cost_model}  1 10  2   "Zero" "ZERO2" "direct"  &

## random
# echo "random"
python ./experiment/traintestLR.py --train "aviation" --cost-model ${cost_model} --cost-function "direct" --trials 5 --budget 15000 --step-size 10 --bootstrap 2  --accu-model "[1,0]"  --expert "neutral"  --maxiter 150 --fixk -1  > ../results/fixk-neutral/MOVIE-RND-UNICOST-NUEEXP-LR.TXT  2> ../results/fixk-neutral/MOVIE-RND-UNICOST-NUEEXP-LR.TXT  &

echo " random fix 10"
python ./experiment/traintestLR.py --train "aviation" --cost-model ${cost_model} --cost-function "direct" --trials 5 --budget 15000 --step-size 10 --bootstrap 2  --accu-model "[1,0]"  --expert "neutral"  --maxiter 150 --fixk 10  > ../results/fixk-neutral/MOVIE-RND10-NEUEXP-LR.TXT   2> ../results/fixk-neutral/MOVIE-RND10-NEUEXP-LR.TXT  


# python ./experiment/unc_fixk.py --train "imdb"  --fixk -1 --neutral-threshold 0.4 --trials 5 --budget 20000 --step-size 10 --bootstrap 50 --maxiter 200 --cost-function "direct" --cost-model "[[10.0,5.7], [25.0,8.2], [50.1,10.9], [75,15.9], [100,16.7], [125,17.8]]" --expert-penalty 0.3 > ../results/fixk-neutral/MOVIE-UNC-UNICOST-TRUEXP-LR.TXT  2> ../results/fixk-neutral/MOVIE-UNC-UNICOST-TRUEXP-LR.TXT  

# echo "anytime with no utility"
# ./anytime.sh "imdb"   0.4   "[1,0]"  0.3  10  50   "Zero" "VANILLA-ZERO" "uniform"  &


# echo "uncertainty"
# python ./experiment/traintestLR.py --train "imdb" --cost-model "[1,0]" --cost-function "uniform" --trials 5 --budget 2000 --step-size 10 --bootstrap 50  --accu-model "[1,0]"  --expert "true"  --maxiter 200 --fixk -1 --student "unc" > ../results/fixk-neutral/MOVIE-UNC-UNICOST-VANILLA-LR.TXT  2> ../results/fixk-neutral/MOVIE-UNC-VANILLA-NUEEXP-LR.TXT  &

# echo "random"
# python ./experiment/traintestLR.py --train "imdb" --cost-model "[1,0]" --cost-function "uniform" --trials 5 --budget 2000 --step-size 10 --bootstrap 50  --accu-model "[1,0]"  --expert "true"  --maxiter 200 --fixk -1  > ../results/fixk-neutral/MOVIE-RND-UNICOST-VANILLA-LR.TXT  2> ../results/fixk-neutral/MOVIE-RND-VANILLA-NUEEXP-LR.TXT  &

# echo " random fix 10"
# python ./experiment/traintestLR.py --train "imdb" --cost-model "[1,0]" --cost-function "uniform" --trials 5 --budget 2000 --step-size 10 --bootstrap 50  --accu-model "[1,0]"  --expert "true"  --maxiter 200 --fixk 10  > ../results/fixk-neutral/MOVIE-RND10-UNICOST-VANILLA-LR.TXT  2> ../results/fixk-neutral/MOVIE-RND10-VANILLA-NUEEXP-LR.TXT  

echo "All experiments are done"


IFS=$SAVEIFS
