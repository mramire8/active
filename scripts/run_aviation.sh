# run_aviation.sh

nohup ./random_fixk.sh > rnd.log 2> rnd.log & 
nohup ./aal-unc.sh > al.log 2> al.log & 
nohup ./uncertainty.sh > un.log 2> un.log &


