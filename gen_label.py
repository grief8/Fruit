# -*- coding: utf-8 -*-
import os
import random


def gen_common(path):
    mydir = os.listdir(path)
    with open('summary.txt', 'w') as summary, open('test.txt', 'w') as test:
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


def gen_classifier(path):
    with open('classifier_summary.txt', 'w', encoding='utf-8') as summary, open('classifier_test.txt', 'w', encoding='utf-8') as test:
        i = 0
        for pth in path:
            for fname in os.listdir(pth):
                s = fname + ' ' + str(int(i)) + '\n'
                # if random.random() < 0.1:
                #     test.write(os.path.join(pth, s))
                #     summary.write(os.path.join(pth, s))
                if random.random() < 0.3:
                    #     summary.write(s)
                    test.write(os.path.join(pth, s))
                else:
                    summary.write(os.path.join(pth, s))
            i = i + 1
        print('finish ' + pth)


if __name__ == '__main__':
    # path = 'E:\\Dataset\\Watermelon221\\'
    # gen_common(path)
    path = {'E:\\Dataset\\上传\\watermelon\\', 'E:\\Dataset\\上传\\Mango\\', 'E:\\Dataset\\上传\\Apple\\'}
    gen_classifier(path)