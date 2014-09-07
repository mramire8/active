#random fix k
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



echo " random fix ${fixk}"
python ./experiment/traintestLR.py --train ${dataset} --cost-model ${cost} --cost-function ${costfn} --trials 10 --budget ${budget} --step-size ${step} --bootstrap ${bt}  --accu-model "[1,0]"  --expert "neutral"  --maxiter ${maxit} --fixk ${fixk} --neutral-threshold ${neuthr} --expert-penalty ${penalty} > "C:/Users/mramire8/Google Drive/AAL-Experiments/aal_python/anytime-fixk/${dataset}-RND${fixk}-NEUEXP-LR-${prefix}.TXT" 2> "C:/Users/mramire8/Google Drive/AAL-Experiments/aal_python/anytime-fixk/${dataset}-RND${fixk}-NEUEXP-LR-${prefix}.TXT"

#../results/fixk-neutral/${dataset}-RND${fixk}-NEUEXP-LR-${prefix}.TXT   2> ../results/fixk-neutral/${dataset}-RND${fixk}-NEUEXP-LR-${prefix}.TXT  


IFS=$SAVEIFS
