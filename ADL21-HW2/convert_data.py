#!/usr/bin/env python
# coding=utf-8

import json, csv, argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--context_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    return args


def convert_swag(task, context):
    option_path = f'data/{task}.json'
    option = json.load(open(option_path))
    
    f = open(f'data/{task}_swag.csv', 'w')
    writer = csv.writer(f)
    row = ['sent1', 'option0', 'option1', 'option2', 'option3', 'label']
    writer.writerow(row)

    for i, o in enumerate(option):
        context_list = [context[p] for p in o['paragraphs']]
        relevant = o['paragraphs'].index(o['relevant'])

        row = [o['question'], context_list[0], context_list[1], context_list[2], context_list[3], relevant,]
        writer.writerow(row)

    f.close()

def convert_swag_test(test_path, context):
    option_path = test_path
    option = json.load(open(option_path))
    
    f = open(f'data/test_swag.csv', 'w')
    writer = csv.writer(f)
    row = ['id', 'sent1', 'option0', 'option1', 'option2', 'option3']
    writer.writerow(row)

    for i, o in enumerate(option):
        context_list = [context[p] for p in o['paragraphs']]

        row = [o['id'], o['question'], context_list[0], context_list[1], context_list[2], context_list[3]]
        writer.writerow(row)

    f.close()

def convert_squad(task, context):
    option_path = f'data/{task}.json'
    option = json.load(open(option_path))
    
    f = open(f'data/{task}_squad.csv', 'w')
    writer = csv.writer(f)
    row = ['answers', 'context', 'id', 'question']
    writer.writerow(row)

    for i, o in enumerate(option):
        context_list = [context[p] for p in o['paragraphs']]
        relevant = o['paragraphs'].index(o['relevant'])

        row = [o['answer'], context[o['relevant']], o['id'], o['question']]
        writer.writerow(row)

    f.close()

def convert_json(task, context):
    option_path = f'data/{task}.json'
    option = json.load(open(option_path))

    dataset = {'data': []}
    for i, o in enumerate(option):
        context_list = [context[p] for p in o['paragraphs']]
        relevant = o['paragraphs'].index(o['relevant'])

        answer = o['answer']
        answer = {'answer_start': [answer['start']], 'text': [answer['text']]}

        s = {
            'answers': answer,
            'context': context[o['relevant']],
            'id': o['id'],
            'question': o['question'],
        }
        
        # dataset['answers'].append(o['answer'])
        # dataset['context'].append(context[o['relevant']])
        # dataset['id'].append(o['id'])
        # dataset['question'].append(o['question'])

        dataset['data'].append(s)


    with open(f'data/{task}_squad.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False)


def qa_output_to_csv(output_qa_path):
    output_path = 'output/eval_predictions.json'
    output = json.load(open(output_path))

    f = open(output_qa_path, 'w')
    writer = csv.writer(f)
    row = ['id', 'answer']
    writer.writerow(row)

    for k in output:
        writer.writerow([k, output[k]])

    f.close()


def main():
    args = parse_args()

    context_path = args.context_path
    test_path = args.test_path
    output_path = args.output_path
    
    if context_path != None and test_path != None:
        context = json.load(open(context_path))
        convert_swag_test(test_path, context)

    if output_path != None:
        qa_output_to_csv(output_path)

if __name__ == '__main__':
    main()