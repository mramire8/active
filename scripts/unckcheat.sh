dataset=$2
fixk=$1
i=0.4

# UNCERTAINTY CHEAT - IF 100 IS NOT ENOUGH, USE COST, IGNORE LABEL
# python ./experiment/unck_cheat.py --train ${dataset}  --fixk ${fixk} --neutral_threshold ${i} --trials 5 --budget 30000 --step-size 10 --bootstrap 50  --maxiter 300 --cost-function "direct" --cost-model "[[10.0,5.7], [25.0,8.2], [50.1,10.9], [75,15.9], [100,16.7], [125,17.8], [150,22.7], [175,19.9], [200,17.4]]"  > ../results/fixk-neutral/${dataset}-UNCCHEAT2-${fixk}-TH${i}-NEUTRAL.TXT  2> ../results/fixk-neutral/${dataset}-UNCCHEAT2-${fixk}-TH${i}-NEUTRAL.TXT  


## UNCERTAINTY CHEAT - IF 100 IS NOT ENOUGH, IGNORE LABEL AND COST
python ./experiment/unckcheatv2.py --train ${dataset}  --fixk ${fixk} --neutral_threshold ${i} --trials 5 --budget 20000 --step-size 10 --bootstrap 50  --maxiter 300 --cost-function "direct" --cost-model "[[10.0,5.7], [25.0,8.2], [50.1,10.9], [75,15.9], [100,16.7], [125,17.8], [150,22.7], [175,19.9], [200,17.4]]"  > ../results/fixk-neutral/${dataset}-UNCCHEATv2-${fixk}-TH${i}-NEUTRAL.TXT  2> ../results/fixk-neutral/${dataset}-UNCCHEATv2-${fixk}-TH${i}-NEUTRAL.TXT  

