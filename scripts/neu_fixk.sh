
fixk=$1

for i in  0.3 0.35 0.4 0.45 
do
	python ./experiment/traintestfixk.py --train "aviation"  --fixk ${fixk} --neutral_threshold ${i} --trials 5 --budget 3000 --step-size 10 --bootstrap 50  --maxiter 300 > ../results/fixk-neutral/AVIATION-FIX-${fixk}-TH${i}-NEUTRAL.TXT  2> ../results/fixk-neutral/AVIATION-FIX-${fixk}-TH${i}-NEUTRAL.TXT  
done
