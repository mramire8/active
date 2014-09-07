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
# ./anytime.sh "aviation"    ${costaviation}  "direct"  0.3     0.01       5     2     3000    200     "ANY-UNC3"   "anyunc"   &
                                                       
# ./anytime.sh "aviation"    ${costaviation}  "direct"  0.3     0.01       5     2     3000    200     "ANY-ZERO3"  "anyzero"  

./anytime.sh "imdb"        ${costmovie}     "direct"  0.4     0.3       10     50    20000   200     "ANY-UNC3"    "anyunc"  &

./anytime.sh "imdb"        ${costmovie}     "direct"  0.4     0.3       10     50    20000   200     "ANY-ZERO3"   "anyzero" 


echo "All experiments are done"
IFS=$SAVEIFS

