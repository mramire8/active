SAVEIFS=$IFS
IFS=$(echo -en "\n\b")

dataset=$1
#fixk=$2

cost=$2
costfn=$3
neuthr=$4
penalty=$5

step=$6
bt=$7
budget=$8
maxit=${9}
prefix=${10}
student=${11}

#lambda=$7
# UNCERTAINTY CHEAT - IF 100 IS NOT ENOUGH, IGNORE LABEL AND COST

# #### MOVIE COST MODEL
python ./experiment/anytime.py --train ${dataset} --fixk 25 --neutral-threshold ${neuthr} --trials 10 --budget ${budget} --step-size ${step} --bootstrap ${bt}  --maxiter ${maxit} --cost-function ${costfn} --cost-model ${cost} --student ${student} --expert-penalty ${penalty} > "C:/Users/mramire8/Google Drive/AAL-Experiments/aal_python/anytime-fixk/${dataset}-ANYTIMEBIN-${student}-TH${i}-${prefix}.TXT" 2> "C:/Users/mramire8/Google Drive/AAL-Experiments/aal_python/anytime-fixk/${dataset}-ANYTIMEBIN-${student}-TH${i}-${prefix}.TXT"

# #### MOVIE COST MODEL
# python ./experiment/anytime.py --train ${dataset}  --fixk 25 --neutral_threshold ${i} --trials 5 --budget 20000 --step-size 10 --bootstrap 50  --maxiter 200 --cost-function "direct" --cost-model "[[10.0,5.7], [25.0,8.2], [50.1,10.9], [75,15.9], [100,16.7], [125,17.8], [150,22.7], [175,19.9], [200,17.4]]"  > ../results/fixk-neutral/${dataset}-ANYTIME2-TH${i}.TXT  2> ../results/fixk-neutral/${dataset}-ANYTIME2-TH${i}.TXT  

#### AVIATION COST MODEL
# python ./experiment/anytime.py --train ${dataset}  --fixk 25 --neutral_threshold ${i} --trials 5 --budget 35000 --step-size 10 --bootstrap 50  --maxiter 400 --cost-function "direct" --cost-model "[[10, 5.2], [25, 6.5], [50, 7.6], [75, 9.1], [100, 10.3], [125, 13.6]]" --expert-penalty 1 > ../results/fixk-neutral/${dataset}-ANYTIME-TH${i}.TXT  2> ../results/fixk-neutral/${dataset}-ANYTIME-TH${i}.TXT  


IFS=$SAVEIFS

