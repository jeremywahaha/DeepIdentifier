import os
import sys
import time
import math
import numpy as np
import random
import torch

def reshape_data(input, sws=500):
    input = np.array(input).reshape((-1, 1, 6, sws)).transpose((0, 1, 3, 2))
    return input

def read_file(fn):
    infile = []
    f = open(fn, "r")
    for line in f:
        line = line.split(' ')
        line.pop()
        line = [float(n) for n in line]
        infile.append(line)
    infile.pop()
    f.close()
    return np.array(infile)

def read_UIR_dataset(folder_name):
    ori_train_data, ori_test_data = [], []
    sub_name = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    
    for sn in sub_name:
        sub_path = folder_name + '/' + sn
        for file_name in os.listdir(sub_path):
            file_path = sub_path + '/' + file_name
            if "train_data" in file_name:
                ori_train_data.append(read_file(file_path))
            elif "test_data" in file_name:
                ori_test_data.append(read_file(file_path))

    return np.array(ori_train_data), np.array(ori_test_data)

def produce_data_label(data_t, data_f, down):
    new_data_f = []
    for i in data_f:
        new_data_f.append(i[np.random.randint(i.shape[0], size=int(data_t.shape[0]/down))])
    new_data_f = np.array(new_data_f).reshape((-1, data_t.shape[1]))
    label_t = np.repeat([1.0], data_t.shape[0], axis=0)
    label_f = np.repeat([0.0], new_data_f.shape[0], axis=0)
    data = np.concatenate((data_t, new_data_f), axis=0)
    label = np.concatenate((label_t, label_f), axis=0)
    return np.array(data), np.array(label)

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
