#!/usr/bin/env python
# coding=utf-8

import json, csv, argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_path",
        type=str,
        default='data/train.jsonl',
    )
    parser.add_argument(
        "--valid_path",
        type=str,
        default='data/public.jsonl',
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default='data/sample_test.jsonl',
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    return args

def convert_jsonl_to_xsum(task, path):
    data = list(open(path))    
    
    f = open(f'data/{task}_xsum.csv', 'w')
    writer = csv.writer(f)
    row = ['document', 'summary', 'id']
    writer.writerow(row)

    for i, d in enumerate(data):
        d = json.loads(d)
        row = [d['maintext'].replace('\n', ''), d['title'], d['id']]
        writer.writerow(row)

    f.close()

def convert_jsonl_to_xsum_test(task, path):
    data = list(open(path))    
    
    f = open(f'data/{task}_xsum_re.csv', 'w')
    writer = csv.writer(f)
    row = ['document', 'summary', 'id']
    writer.writerow(row)

    for i, d in enumerate(data):
        d = json.loads(d)
        row = [d['maintext'].replace('\n', ''), ' ', d['id']]
        writer.writerow(row)

    f.close()

def main():
    args = parse_args()

    train_path = args.train_path
    valid_path = args.valid_path
    test_path = args.test_path
    output_path = args.output_path
    task = args.task

    # convert_jsonl_to_xsum('train', train_path)
    # convert_jsonl_to_xsum('valid', valid_path)
    convert_jsonl_to_xsum_test(task, test_path)

if __name__ == '__main__':
    main()