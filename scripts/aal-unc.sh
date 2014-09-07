SAVEIFS=$IFS
IFS=$(echo -en "\n\b")

echo "ANYTIME"

costmovie="[[10.0,5.7], [25.0,8.2], [50.1,10.9], [75,15.9], [100,16.7], [125,17.8]]"
costaviation="[[10, 5.2], [25, 6.5], [50, 7.6], [75, 9.1], [100, 10.3], [125, 13.6]]" 

# dataset=$1
# #fixk=$2

# cost=$2
# costfn=$3
# neuthr=$4
# penalty=$5

# step=$6
# bt=$7
# budget=$8
# maxit=${9}
# prefix=${10}
# student=${11}

# INVESTIGATION 
##           dataset       cost             costfn   neuthr   penalty   step   bt   budget   maxit   prefix      student
./anytime.sh "aviation"    ${costaviation}  "direct"  0.3     0.01      10     2     3000    300     "ANY-UNC3"   "anyunc"   &
                                                       
./anytime.sh "aviation"    ${costaviation}  "direct"  0.3     0.01      10     2     3000    300     "ANY-ZERO3"  "anyzero"  &

./anytime.sh "imdb"        ${costmovie}     "direct"  0.4     0.3       10     2    20000   300     "ANY-UNC3"    "anyunc"   &

./anytime.sh "imdb"        ${costmovie}     "direct"  0.4     0.3       10     2    20000   300     "ANY-ZERO3"   "anyzero" 


# ## dataset, threshold, cost model, penalty
# ./anytime.sh  "aviation"  0.3  "[[10, 5.2], [25, 6.5], [50, 7.6], [75, 9.1], [100, 10.3], [125, 13.6]]"  1  10  2 "ANY"


## LAMBDA lambda

# ./lambda.sh "imdb" 0.01  &

# ./lambda.sh "imdb" 0.02  &

# ./lambda.sh "imdb" 0.03  &

# ./lambda.sh "imdb" 0.04  &

# ./lambda.sh "imdb" 0.05  &

# ./lambda.sh "imdb" 0.06  &

# ./lambda.sh "imdb" 0.0075  &

# ./lambda.sh "imdb" 0.005  &

# ./lambda.sh "imdb" 0.0025  &

# ./lambda.sh "imdb" 0.001  


echo "All experiments are done"
IFS=$SAVEIFS

