import csv, os
import numpy as np
import matplotlib.pyplot as plt

# for i in range(10):
#     command = f'python plot.py --model_name_or_path chinese-macbert-large-squad/epoch_{i} --num_train_epochs 1'
#     os.system(command)
    

f = open('plot.csv', 'r')
reader = csv.reader(f)
label = []
loss = []
em = []

for i, item in enumerate(reader):
    label.append(f'epoch_{i}')
    loss.append(round(float(item[1]), 5))
    em.append(round(float(item[2]), 5))


x = np.arange(len(label))

plt.figure()
plt.plot(x, loss, '-o')
plt.xticks(x, label, rotation = 45)
plt.ylabel('loss')
plt.subplots_adjust(bottom=0.15)
plt.savefig('loss.png')

plt.figure()
plt.plot(x, em, '-o')
plt.xticks(x, label, rotation = 45)
plt.yticks(np.arange(round(min(em), 1), round(max(em), 1), 0.5))
plt.ylabel('EM')
plt.subplots_adjust(bottom=0.15)
plt.savefig('em.png')