#!/usr/bin/env bash

for method in dnn rf extra
do
  arg1="--save_file=210106/"$method""_""
  arg3="--method=$method"
  python tune-main.py "$arg1" "$arg3" --max_evals=1000 # normalize:True in defult
done