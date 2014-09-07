SAVEIFS=$IFS
IFS=$(echo -en "\n\b")
# dataset=$1
# fixk=$2

# cost=$3
# costfn=$4
# neuthr=$5
# penalty=$6

# step=$7
# bt=$8
# budget=$9
# maxit=${10}
# prefix=${11}
costmovie="[[10.0,5.7], [25.0,8.2], [50.1,10.9], [75,15.9], [100,16.7], [125,17.8]]"
costaviation="[[10, 5.2], [25, 6.5], [50, 7.6], [75, 9.1], [100, 10.3], [125, 13.6]]" 

# ##                      dataset  fixk   cost           costfn   neuthr   penalty  step  bt budget   maxit   prefix
echo "Fix k = 10"                                                              
./uncertainty_fixk.sh   "aviation"  10  ${costaviation} "direct"   0.3    0.01       10   2   3000    300  "UNC-EXP3"  &
echo "Fix k = 25"                                                        
./uncertainty_fixk.sh   "aviation"  25  ${costaviation} "direct"   0.3    0.01       10   2   3000    300  "UNC-EXP3"  
echo "Fix k = 50"                                                        
./uncertainty_fixk.sh   "aviation"  50  ${costaviation} "direct"   0.3    0.01       10   2   3000    300  "UNC-EXP3"  &

echo "Fix k = 10"                                                                                            
./uncertainty_fixk.sh   "imdb"      10   ${costmovie}   "direct"   0.4     0.3       10   2  20000   300    "UNC-EXP3"  
echo "Fix k = 25" 
./uncertainty_fixk.sh   "imdb"      25   ${costmovie}   "direct"   0.4     0.3       10   2  20000   300    "UNC-EXP3"  &
echo "Fix k = 50" 
./uncertainty_fixk.sh   "imdb"      50   ${costmovie}   "direct"   0.4     0.3       10   2  20000   300    "UNC-EXP3"  



echo "Fix k = 75"                                                        
./uncertainty_fixk.sh   "aviation"  75  ${costaviation} "direct"   0.3    0.01       10   2   3000    300  "UNC-EXP3"  &
echo "Fix k = 100"                                                       
./uncertainty_fixk.sh   "aviation"  100 ${costaviation} "direct"   0.3    0.01       10   2   3000    300  "UNC-EXP3"  
                                                                   
echo "Fix k = 75" 
./uncertainty_fixk.sh   "imdb"      75   ${costmovie}   "direct"   0.4     0.3       10   2  20000   300    "UNC-EXP3"  &
echo "Fix k = 100"
./uncertainty_fixk.sh   "imdb"      100  ${costmovie}   "direct"   0.4     0.3       10   2  20000   300    "UNC-EXP3"  

    

## fixk, dataset, neutralThreshold
# echo "Fix k = 75"
# ./uncertainty_fixk.sh 75   "aviation"  0.3  "[[10, 5.2], [25, 6.5], [50, 7.6], [75, 9.1], [100, 10.3], [125, 13.6]]" 1  10  2  &
# echo "Fix k = 100"
# ./uncertainty_fixk.sh 100  "aviation"  0.3  "[[10, 5.2], [25, 6.5], [50, 7.6], [75, 9.1], [100, 10.3], [125, 13.6]]" 1  10  2  &
# echo "Fix k = 50"
# ./uncertainty_fixk.sh 50   "aviation"  0.3  "[[10, 5.2], [25, 6.5], [50, 7.6], [75, 9.1], [100, 10.3], [125, 13.6]]" 1  10  2  & 
# echo "Fix k = 10"
# ./uncertainty_fixk.sh 10   "aviation"  0.3  "[[10, 5.2], [25, 6.5], [50, 7.6], [75, 9.1], [100, 10.3], [125, 13.6]]" 1  10  2  
# echo "Fix k = 25"
# ./uncertainty_fixk.sh 25   "aviation"  0.3  "[[10, 5.2], [25, 6.5], [50, 7.6], [75, 9.1], [100, 10.3], [125, 13.6]]" 1  10  2  


echo "All experiments are done"


IFS=$SAVEIFS
