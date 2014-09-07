SAVEIFS=$IFS
IFS=$(echo -en "\n\b")

costmovie="[[10.0,5.7], [25.0,8.2], [50.1,10.9], [75,15.9], [100,16.7], [125,17.8]]"
costaviation="[[10, 5.2], [25, 6.5], [50, 7.6], [75, 9.1], [100, 10.3], [125, 13.6]]" 

#### Order of the parameters 
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
###           dataset      fixk cost              costfn    neuthr   penalty   step  bt   budget   maxit   prefix

./rndfixk.sh  "aviation"   10   ${costaviation}   "direct"   0.3     0.01       10    2    3000    300   "EXP3" &

./rndfixk.sh  "aviation"   25   ${costaviation}   "direct"   0.3     0.01       10    2    3000    300   "EXP3" 

./rndfixk.sh  "aviation"   50   ${costaviation}   "direct"   0.3     0.01       10    2    3000    300   "EXP3"  &

./rndfixk.sh  "imdb"       25   ${costmovie}      "direct"   0.4     0.3        10    2    20000   300   "EXP3"  

./rndfixk.sh  "imdb"       10   ${costmovie}      "direct"   0.4     0.3        10    2    20000   300   "EXP3"  &

./rndfixk.sh  "imdb"       50   ${costmovie}      "direct"   0.4     0.3        10    2    20000   300   "EXP3"

./rndfixk.sh  "imdb"       75   ${costmovie}      "direct"   0.4     0.3        10    2    20000   300   "EXP3"  &

./rndfixk.sh  "imdb"       100  ${costmovie}      "direct"   0.4     0.3        10    2    20000   300   "EXP3"  

./rndfixk.sh  "aviation"   75   ${costaviation}   "direct"   0.3     0.01       10    2    3000    300   "EXP3"  &

./rndfixk.sh  "aviation"   100  ${costaviation}   "direct"   0.3     0.01       10    2    3000    300   "EXP3"  



IFS=$SAVEIFS
