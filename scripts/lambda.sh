dataset=$1
lambda=$2
# fixk=$1
i=0.4


python ./experiment/anytime.py --train ${dataset}  --fixk 25 --neutral_threshold ${i} --trials 5 --budget 20000 --step-size 10 --bootstrap 50  --maxiter 200 --cost-function "direct" --cost-model "[[10.0,5.7], [25.0,8.2], [50.1,10.9], [75,15.9], [100,16.7], [125,17.8], [150,22.7], [175,19.9], [200,17.4]]" --lambda-value ${lambda} > ../results/fixk-neutral/${dataset}-ANYTIME-TH${i}-LAMBDA${lambda}.TXT  2> ../results/fixk-neutral/${dataset}-ANYTIME-TH${i}-LAMBDA${lambda}.TXT  

