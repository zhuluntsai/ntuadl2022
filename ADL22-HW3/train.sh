# ./train.sh data/train.jsonl

python convert_data.py --test_path $1 --task train

python train.py --seed 111 --train_file data/train_xsum_re.csv --model_name_or_path mt5-small-finetune-re