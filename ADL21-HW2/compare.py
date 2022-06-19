import json


j1 = json.load(open('data/ts_output.json'))['data']
j2 = json.load(open('multi_choice.json'))['data']

correct = []

for f1, f2 in zip(j1, j2):
    same = f1['context'] == f2['context']
    correct.append(same)

acc = sum(correct) / len(correct)
miss = len(correct) - sum(correct)
print(miss, len(correct))