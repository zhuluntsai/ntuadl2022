# ./run.sh data/context.json data/test.json data/output_qa.csv

python convert_data.py --context_path $1 --test_path $2

# python p1_test.py --model_name_or_path bert-base-chinese-finetuned-swag2  --do_predict
# python p2_test.py --model_name_or_path bert-base-chinese-finetuned-squad  --do_predict

python p1_test.py --model_name_or_path chinese-macbert-base-swag  --do_predict
python p2_test.py --model_name_or_path chinese-macbert-large-squad  --do_predict

python convert_data.py --output_path $3