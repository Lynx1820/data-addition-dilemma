#! /usr/bin/env bash
#$ -S /bin/bash  # run job as a Bash shell [IMPORTANT]
#$ -cwd          # run job in the current working directory

xargs -L1 -P16 -I__ sh -c __ <<EOF
python run_sequential.py --data mimic --model lr -n 1000
python run_sequential.py --data mimic --model lr -n 2000
python run_sequential.py --data mimic --model lr -n 3000
python run_sequential.py --data mimic --model lr -n 4000
python run_sequential.py --data mimic --model lr -n 5000
python run_sequential.py --data mimic --model lr -n 6000
python run_sequential.py --data mimic --model lr -n 7000
python run_sequential.py --data mimic --model lr -n 8000
python run_sequential.py --data mimic --model lr -n 9000
python run_sequential.py --data mimic --model lr -n 10000
python run_sequential.py --data mimic --model lr -n 11000
python run_sequential.py --data mimic --model lr -n 12000
python run_sequential.py --data mimic --model lr -n 13000
python run_sequential.py --data mimic --model lr -n 14000
python run_sequential.py --data mimic --model lr -n 15000
python run_sequential.py --data mimic --model lr -n 16000
python run_sequential.py --data mimic --model lr -n 17000
python run_sequential.py --data mimic --model lr -n 18000
python run_sequential.py --data mimic --model lr -n 19000
python run_sequential.py --data mimic --model lr -n 20000
python run_sequential.py --data mimic --model lr -n 21000
python run_sequential.py --data mimic --model lr -n 22000
python run_sequential.py --data mimic --model lr -n 23000
python run_sequential.py --data mimic --model lr -n 24000
python run_sequential.py --data mimic --model lr -n 25000
python run_sequential.py --data mimic --model lr -n 26000
python run_sequential.py --data mimic --model lr -n 27000
python run_sequential.py --data mimic --model lr -n 28000
python run_sequential.py --data mimic --model lr -n 29000
python run_sequential.py --data mimic --model lr -n 30000

python run_sequential.py --data mimic --model svm -n 1000
python run_sequential.py --data mimic --model svm -n 2000
python run_sequential.py --data mimic --model svm -n 3000
python run_sequential.py --data mimic --model svm -n 4000
python run_sequential.py --data mimic --model svm -n 5000
python run_sequential.py --data mimic --model svm -n 6000
python run_sequential.py --data mimic --model svm -n 7000
python run_sequential.py --data mimic --model svm -n 8000
python run_sequential.py --data mimic --model svm -n 9000
python run_sequential.py --data mimic --model svm -n 10000
python run_sequential.py --data mimic --model svm -n 11000
python run_sequential.py --data mimic --model svm -n 12000
python run_sequential.py --data mimic --model svm -n 13000
python run_sequential.py --data mimic --model svm -n 14000
python run_sequential.py --data mimic --model svm -n 15000
python run_sequential.py --data mimic --model svm -n 16000
python run_sequential.py --data mimic --model svm -n 17000
python run_sequential.py --data mimic --model svm -n 18000
python run_sequential.py --data mimic --model svm -n 19000
python run_sequential.py --data mimic --model svm -n 20000
python run_sequential.py --data mimic --model svm -n 21000
python run_sequential.py --data mimic --model svm -n 22000
python run_sequential.py --data mimic --model svm -n 23000
python run_sequential.py --data mimic --model svm -n 24000
python run_sequential.py --data mimic --model svm -n 25000
python run_sequential.py --data mimic --model svm -n 26000
python run_sequential.py --data mimic --model svm -n 27000
python run_sequential.py --data mimic --model svm -n 28000
python run_sequential.py --data mimic --model svm -n 29000
python run_sequential.py --data mimic --model svm -n 30000

python run_sequential.py --data mimic --model nn -n 1000
python run_sequential.py --data mimic --model nn -n 2000
python run_sequential.py --data mimic --model nn -n 3000
python run_sequential.py --data mimic --model nn -n 4000
python run_sequential.py --data mimic --model nn -n 5000
python run_sequential.py --data mimic --model nn -n 6000
python run_sequential.py --data mimic --model nn -n 7000
python run_sequential.py --data mimic --model nn -n 8000
python run_sequential.py --data mimic --model nn -n 9000
python run_sequential.py --data mimic --model nn -n 10000
python run_sequential.py --data mimic --model nn -n 11000
python run_sequential.py --data mimic --model nn -n 12000
python run_sequential.py --data mimic --model nn -n 13000
python run_sequential.py --data mimic --model nn -n 14000
python run_sequential.py --data mimic --model nn -n 15000
python run_sequential.py --data mimic --model nn -n 16000
python run_sequential.py --data mimic --model nn -n 17000
python run_sequential.py --data mimic --model nn -n 18000
python run_sequential.py --data mimic --model nn -n 19000
python run_sequential.py --data mimic --model nn -n 20000
python run_sequential.py --data mimic --model nn -n 21000
python run_sequential.py --data mimic --model nn -n 22000
python run_sequential.py --data mimic --model nn -n 23000
python run_sequential.py --data mimic --model nn -n 24000
python run_sequential.py --data mimic --model nn -n 25000
python run_sequential.py --data mimic --model nn -n 26000
python run_sequential.py --data mimic --model nn -n 27000
python run_sequential.py --data mimic --model nn -n 28000
python run_sequential.py --data mimic --model nn -n 29000
python run_sequential.py --data mimic --model nn -n 30000

python run_sequential.py --data mimic --model xgb -n 1000
python run_sequential.py --data mimic --model xgb -n 2000
python run_sequential.py --data mimic --model xgb -n 3000
python run_sequential.py --data mimic --model xgb -n 4000
python run_sequential.py --data mimic --model xgb -n 5000
python run_sequential.py --data mimic --model xgb -n 6000
python run_sequential.py --data mimic --model xgb -n 7000
python run_sequential.py --data mimic --model xgb -n 8000
python run_sequential.py --data mimic --model xgb -n 9000
python run_sequential.py --data mimic --model xgb -n 10000
python run_sequential.py --data mimic --model xgb -n 11000
python run_sequential.py --data mimic --model xgb -n 12000
python run_sequential.py --data mimic --model xgb -n 13000
python run_sequential.py --data mimic --model xgb -n 14000
python run_sequential.py --data mimic --model xgb -n 15000
python run_sequential.py --data mimic --model xgb -n 16000
python run_sequential.py --data mimic --model xgb -n 17000
python run_sequential.py --data mimic --model xgb -n 18000
python run_sequential.py --data mimic --model xgb -n 19000
python run_sequential.py --data mimic --model xgb -n 20000
python run_sequential.py --data mimic --model xgb -n 21000
python run_sequential.py --data mimic --model xgb -n 22000
python run_sequential.py --data mimic --model xgb -n 23000
python run_sequential.py --data mimic --model xgb -n 24000
python run_sequential.py --data mimic --model xgb -n 25000
python run_sequential.py --data mimic --model xgb -n 26000
python run_sequential.py --data mimic --model xgb -n 27000
python run_sequential.py --data mimic --model xgb -n 28000
python run_sequential.py --data mimic --model xgb -n 29000
python run_sequential.py --data mimic --model xgb -n 30000
EOF