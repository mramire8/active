SAVEIFS=$IFS
IFS=$(echo -en "\n\b")

dataset=$1
#fixk=$2

cost=$2
costfn=$3
thres=$4
penalty=$5

step=$6
bt=$7
budget=$8
maxit=${9}
prefix=${10}
student=${11}
cheat=${12}  ## IGNORE THIS ARGUMENT FOR NOW
folder=${13}
clf=${14}
limit=${15}
cali=${16}

if [ "${cheat}" = "NOCHEAT" ] 
then
	# #### MOVIE CHEATING MODEL
# #### MOVIE CHEATING MODEL
	echo "C:/Users/mramire8/Google Drive/AAL-Experiments/aal_python/${folder}"
	python ./sentences/sent_cheat.py --train ${dataset} --trials 5 --budget ${budget} --step-size ${step} --bootstrap ${bt}  --maxiter ${maxit} --cost-function ${costfn} --cost-model ${cost} --student ${student} --expert-penalty ${penalty}  --classifier ${clf}  --prefix ${prefix}  --limit ${limit} ${cali}> "C:/Users/mramire8/Google Drive/AAL-Experiments/aal_python/${folder}${dataset}-${clf}-${cheat}-${student}-${prefix}.TXT" 2> "C:/Users/mramire8/Google Drive/AAL-Experiments/aal_python/${folder}${dataset}-${clf}-${cheat}-${student}-${prefix}err.TXT"
else
	# #### MOVIE CHEATING MODEL
# #### MOVIE CHEATING MODEL
	# python ./sentences/sent_cheat.py --train ${dataset} --trials 5 --budget ${budget} --step-size ${step} --bootstrap ${bt}  --maxiter ${maxit} --cost-function ${costfn} --cost-model ${cost} --student ${student} --expert-penalty ${penalty} --neutral-threshold ${thres} --cheating > "C:/Users/mramire8/Google Drive/AAL-Experiments/aal_python/${13}${dataset}-SRCHEAT-${cheat}-${student}-${prefix}.TXT" 2> "C:/Users/mramire8/Google Drive/AAL-Experiments/aal_python/${13}${dataset}-SRCHEAT-${cheat}-${student}-${prefix}err.TXT"
	echo "We cannot do cheating experiment right now"
fi


IFS=$SAVEIFS

