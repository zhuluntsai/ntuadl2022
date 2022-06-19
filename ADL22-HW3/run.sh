# ./run.sh data/sample_test.jsonl output.jsonl

python convert_data.py --test_path $1 --task test

python run.py --output_path $2