
SAVEIFS=$IFS
IFS=$(echo -en "\n\b")

dataset=$1
fixk=$2

cost=$3
costfn=$4
neuthr=$5
penalty=$6

step=$7
bt=$8
budget=$9
maxit=${10}
prefix=${11}

python ./experiment/unc_fixk.py --train ${dataset}  --fixk ${fixk} --neutral-threshold ${neuthr} --trials 10 --budget ${budget} --step-size ${step} --bootstrap ${bt}  --maxiter ${maxit} --cost-function ${costfn} --cost-model ${cost} --expert-penalty ${penalty} > "C:/Users/mramire8/Google Drive/AAL-Experiments/aal_python/anytime-fixk/${dataset}-UNCFIXBIN-${fixk}-TH${neuthr}-NEU-${prefix}.TXT"  2> "C:/Users/mramire8/Google Drive/AAL-Experiments/aal_python/anytime-fixk/${dataset}-UNCFIXBIN-${fixk}-TH${neuthr}-NEU-${prefix}.TXT"

#> ../results/fixk-neutral/${dataset}-UNCFIXBIN-${fixk}-TH${neuthr}-NEU-${prefix}.TXT  2> ../results/fixk-neutral/${dataset}-UNCFIXBIN-${fixk}-TH${neuthr}-NEU-${prefix}.TXT  




###AVIATION
# python ./experiment/unc_fixk.py --train ${dataset}  --fixk ${fixk} --neutral-threshold ${i} --trials 5 --budget 35000 --step-size 10 --bootstrap 50  --maxiter 400 --cost-function "direct" --cost-model "[[10, 5.2], [25, 6.5], [50, 7.6], [75, 9.1], [100, 10.3], [125, 13.6]]" --expert-penalty 1 > ../results/fixk-neutral/${dataset}-UNCFIX-${fixk}-TH${i}-NEUTRAL.TXT  2> ../results/fixk-neutral/${dataset}-UNCFIX-${fixk}-TH${i}-NEUTRAL.TXT  


IFS=$SAVEIFS

