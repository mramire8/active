SAVEIFS=$IFS
IFS=$(echo -en "\n\b")

# echo "Uncertainty full "
# python ./experiment/unc_fixk.py --train "imdb"  --fixk -1 --neutral-threshold 0.4 --trials 5 --budget 15000 --step-size 10 --bootstrap 50 --maxiter 150 --cost-function "direct" --cost-model "[[10.0,5.7], [25.0,8.2], [50.1,10.9], [75,15.9], [100,16.7], [125,17.8]]" --expert-penalty 0.3 > ../results/fixk-neutral/MOVIE-UNC-NEUEXP-LR.TXT  2> ../results/fixk-neutral/MOVIE-UNC-NEUEXP-LR.TXT  &
## random
# echo "random"
# python ./experiment/traintestLR.py --train "imdb" --cost-model "[[10.0,5.7], [25.0,8.2], [50.1,10.9], [75,15.9], [100,16.7], [125,17.8]]" --cost-function "direct" --trials 5 --budget 15000 --step-size 10 --bootstrap 50  --accu-model "[1,0]"  --expert "neutral"  --maxiter 150 --fixk -1  > ../results/fixk-neutral/MOVIE-RND-UNICOST-NUEEXP-LR.TXT  2> ../results/fixk-neutral/MOVIE-RND-UNICOST-NUEEXP-LR.TXT  &


# INVESTIGATION 
# echo "anytime with no utility"
# ./anytime.sh "imdb"   0.4   "[[10.0,5.7], [25.0,8.2], [50.1,10.9], [75,15.9], [100,16.7], [125,17.8], [150,22.7], [175,19.9], [200,17.4]]"  0.3  10  50   "Zero" "ZERO2" "direct"  &

echo " random fix 10"
python ./experiment/traintestLR.py --train "imdb" --cost-model "[[10.0,5.7], [25.0,8.2], [50.1,10.9], [75,15.9], [100,16.7], [125,17.8]]" --cost-function "direct" --trials 5 --budget 15000 --step-size 10 --bootstrap 50  --accu-model "[1,0]"  --expert "neutral"  --maxiter 150 --fixk 10  > ../results/fixk-neutral/MOVIE-RND10-NEUEXP-LR.TXT   2> ../results/fixk-neutral/MOVIE-RND10-NEUEXP-LR.TXT  

python ./experiment/anytime.py --train ${dataset} --lambda-value ${lambda} --fixk 25 --neutral-threshold ${neuthr} --trials 5 --budget 20000 --step-size ${step} --bootstrap ${bt}  --maxiter 200 --cost-function ${costfn} --cost-model ${cost}  > ../results/fixk-neutral/${dataset}-ANYTIMEBIN-TH${i}-${prefix}.TXT  --expert-penalty ${penalty} 2> ../results/fixk-neutral/${dataset}-ANYTIMEBIN-TH${i}-${prefix}.TXT  




#############################################################
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
