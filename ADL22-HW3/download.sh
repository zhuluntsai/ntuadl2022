
wget 'https://www.dropbox.com/s/mno359jjdewo91g/config.json?dl=1' -O mt5-small-finetune/config.json
wget 'https://www.dropbox.com/s/9owcyxnt99wmsb3/pytorch_model.bin?dl=1' -O mt5-small-finetune/pytorch_model.bin
wget 'https://www.dropbox.com/s/tav56ap2oh8d9jq/special_tokens_map.json?dl=1' -O mt5-small-finetune/special_tokens_map.json
wget 'https://www.dropbox.com/s/tdudgwfpn2mlizv/spiece.model?dl=1' -O mt5-small-finetune/spiece.model
wget 'https://www.dropbox.com/s/p5otnrtwcv4totg/tokenizer_config.json?dl=1' -O mt5-small-finetune/tokenizer_config.json
wget 'https://www.dropbox.com/s/1ajt0r2f8sfe8r2/tokenizer.json?dl=1' -O mt5-small-finetune/tokenizer.json

wget 'https://www.dropbox.com/s/rqa3wnzlqt2ofkc/train_xsum.csv?dl=1' -O data/train_xsum.csv
wget 'https://www.dropbox.com/s/a48yiim9eaoya10/valid_xsum.csv?dl=1' -O data/valid_xsum.csv