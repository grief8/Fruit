# -*- coding: utf-8 -*-
import os
import random

path = 'E:\\Dataset\\Watermelon221\\'
mydir = os.listdir(path)

i = 0
fn = ''
with open(path + 'summary.txt', 'w') as summary, open(path + 'test.txt', 'w') as test:
    for dir_name in mydir:
        dn = dir_name.split('-')[1]
        # if fn != dn:
        #     fn = dn
        #     i = i + 1
        s = dir_name + ' ' + str(int(dn) + 1) + '\n'
        if random.random() < 0.1:
            test.write(s)
            summary.write(s)
        elif random.random() < 0.3:
        #     summary.write(s)
            test.write(s)
        else:
            summary.write(s)
print('successful')