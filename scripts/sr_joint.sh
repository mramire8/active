SAVEIFS=$IFS
IFS=$(echo -en "\n\b")

echo "ANYTIME TFE"

costmovie="[[10.0,5.7], [25.0,8.2], [50.1,10.9], [75,15.9], [100,16.7], [125,17.8]]"
costaviation="[[10, 5.2], [25, 6.5], [50, 7.6], [75, 9.1], [100, 10.3], [125, 13.6]]" 

# INVESTIGATION 
#------------------------------------------------------------------------------------------
#########NOT CHEATING                                   
#------------------------------------------------------------------------------------------

## FK-TFE-MAX                                                                             
# ./sr_cali.sh     "imdb"       ${costmovie}  "uniform"  0     1      10     100   1250     125     "SR-MAX-LRA2-CALIBRATE-BT100"  "sr"  "NOCHEAT"  "sr-oracle-test/results-calibrated/" "lradaptv2" 2 "--calibrate" &
./sr_cali.sh     "imdb"       ${costmovie}  "uniform"  0     1      10     100   1250     125     "SR-MAX-LRL1-1-CALI-BT100"  "sr"  "NOCHEAT"  "sr-oracle-test/results-calibrated/" "lradaptv2" 2 "--calibrate" &
./sr_cali.sh     "imdb"       ${costmovie}  "uniform"  0     10     10     100   1250     125     "SR-MAX-LRL1-10-CALI-BT100"  "sr"  "NOCHEAT"  "sr-oracle-test/results-calibrated/" "lradaptv2" 2 "--calibrate" &
./sr_cali.sh     "imdb"       ${costmovie}  "uniform"  0     100    10     100   1250     125     "SR-MAX-LRL1-100-CALI-BT100"  "sr"  "NOCHEAT"  "sr-oracle-test/results-calibrated/" "lradaptv2" 2 "--calibrate" &
./sr_cali.sh     "imdb"       ${costmovie}  "uniform"  0     1      10     100   1250     125     "SR-MAX-LRL2-1-CALI-BT100"  "sr"  "NOCHEAT"  "sr-oracle-test/results-calibrated/" "lrl2" 2 "--calibrate" &
./sr_cali.sh     "imdb"       ${costmovie}  "uniform"  0     10     10     100   1250     125     "SR-MAX-LRL2-10-CALI-BT100"  "sr"  "NOCHEAT"  "sr-oracle-test/results-calibrated/" "lrl2" 2 "--calibrate" &
./sr_cali.sh     "imdb"       ${costmovie}  "uniform"  0     100    10     100   1250     125     "SR-MAX-LRL2-100-CALI-BT100"  "sr"  "NOCHEAT"  "sr-oracle-test/results-calibrated/" "lrl2" 2 "--calibrate" &
# ./sr_rnd.sh     "imdb"       ${costmovie}  "uniform"  0     1      10     100  1250     125     "RN-LRAD-LIM2_BT100"     "rnd_first1"  "NOCHEAT"  "sr-oracle-test/results-sent-clean/"  "lradapt"  2 &

# ./sr_cali.sh     "aviation"   ${costmovie}  "uniform"  0     1      10     50   1250     125     "SR-MAX-LRA2-CALIBRATE-BT50"  "sr"   "NOCHEAT"  "sr-oracle-test/results-calibrated/" "lradaptv2" 2 "--calibrate" 


#------------------------------------------------------------------------------------------
### not run yet
#------------------------------------------------------------------------------------------

echo "All experiments are done"
IFS=$SAVEIFS

