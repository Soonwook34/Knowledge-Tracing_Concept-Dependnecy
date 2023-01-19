import json
import random
import argparse
import numpy as np
from scipy.stats import multinomial
from tqdm import tqdm
from copy import deepcopy

class DivideDataArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(DivideDataArgParser, self).__init__()
        self.add_argument('--min_log', type=int, default=0,
                          help='Minimum length of logs')
        self.add_argument('--train_ratio', type=float, default=0.8,
                          help='Train set ratio')
        self.add_argument('--valid_ratio', type=float, default=0.1,
                          help='Validation set ratio')
        self.add_argument('--name', type=str, default='KSC',
                          help='Dataset name')


def reverse(arr):
    index = [-1] * 4
    arg_min = np.argmin(arr)
    arg_max = np.argmax(arr)
    if arg_min == arg_max:
        return arr
    else:
        index[arg_min] = arg_max
        index[arg_max] = arg_min
        mid = []
        for i in range(4):
            if i not in [arg_min, arg_max]:
                mid.append(i)
        index[mid[0]] = mid[1]
        index[mid[1]] = mid[0]
    return arr[index]

# ##############################
# MIN_LOG = 0  # 15
# TRAIN_RATIO = 0.8
# VALID_RATIO = 0.1
# DIR_PATH = "../data/poly/"
# SHUFFLE = True
# ##############################
def divide_data(args):
    # 로그 가져오기
    print(f"loading log_data_{args.name}.json...")
    with open(f"log_data_{args.name}.json", encoding='utf8') as i_f:
        stus = json.load(i_f)

    # 모든 concept에 같은 수의 exercise가 있다는 가정
    exercise_size = stus[0]['logs'][-1]['exer_id'] + 1
    concept_size = stus[0]['logs'][-1]['knowledge_code'][0]
    # 각 concept마다 일정 부분을 train/valid/test set으로 split
    train_size = int(exercise_size / concept_size * args.train_ratio)
    valid_size = int(exercise_size / concept_size * args.valid_ratio)
    test_size = int(exercise_size / concept_size) - train_size - valid_size
    # 데이터셋 나누기
    train_set, valid_set, test_set = [], [], []
    train_set_0, valid_set_0, test_set_0 = [], [] ,[]
    valid_set_size, test_set_size = 0, 0
    with tqdm(stus, unit="stu") as stus_bar:
        for stu in stus_bar:
            user_id = stu['user_id']
            stu_train = {'user_id': user_id}
            stu_valid = {'user_id': user_id}
            stu_test = {'user_id': user_id}
            valid_set_size += int(stu['log_num'] * args.valid_ratio)
            test_set_size += stu['log_num'] - int(stu['log_num'] * args.train_ratio) - int(stu['log_num'] * args.valid_ratio)

            logs_by_concept = [[] for c in range(concept_size)]
            for log in stu['logs']:
                logs_by_concept[log['knowledge_code'][0] - 1].append(log)

            stu_train['log_num'] = 0
            stu_train['logs'] = []
            stu_valid['log_num'] = 0
            stu_valid['logs'] = []
            stu_test['log_num'] = 0
            stu_test['logs'] = []
            for logs in logs_by_concept:
                random.shuffle(logs)
                # 로그 나누기
                stu_train['log_num'] += train_size
                stu_train['logs'] += logs[:train_size]
                stu_valid['log_num'] += valid_size
                stu_valid['logs'] += logs[train_size:train_size+valid_size]
                stu_test['log_num'] += test_size
                stu_test['logs'] += logs[train_size+valid_size:]

            # 나눈 로그 저장
            for log in stu_train['logs']:
                train_set.append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'],
                                  'knowledge_code': log['knowledge_code']})
            valid_set.append(deepcopy(stu_valid))
            test_set.append(deepcopy(stu_test))

            # create concept_n=0 dataset
            for log in stu_train['logs']:
                train_set_0.append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'],
                                  'knowledge_code': [1]})
            for log in stu_valid['logs']:
                log['knowledge_code'] = [1]
                del log['p_e']
            for log in stu_test['logs']:
                log['knowledge_code'] = [1]
                del log['p_e']
            valid_set_0.append(stu_valid)
            test_set_0.append(stu_test)
    # train set 학생 순서 섞기 (시간 포함)
    random.shuffle(train_set)

    # 파일에 저장
    print(f"Train {len(train_set)}, Valid {valid_set_size}({len(valid_set)}), Test {test_set_size}({len(test_set)})")
    with open(f"train_set_{args.name}.json", 'w', encoding='utf8') as output_file:
        json.dump(train_set, output_file, indent=4, ensure_ascii=False)
    for user in valid_set:
        for log in user['logs']:
            del log['p_e']
    for user in test_set:
        for log in user['logs']:
            del log['p_e']
    with open(f"valid_set_{args.name}.json", 'w', encoding='utf8') as output_file:
        json.dump(valid_set, output_file, indent=4, ensure_ascii=False)
    with open(f"test_set_{args.name}.json", 'w', encoding='utf8') as output_file:
        json.dump(test_set, output_file, indent=4, ensure_ascii=False)
    with open(f"train_set_{args.name}_0.json", 'w', encoding='utf8') as output_file:
        json.dump(train_set_0, output_file, indent=4, ensure_ascii=False)
    with open(f"valid_set_{args.name}_0.json", 'w', encoding='utf8') as output_file:
        json.dump(valid_set_0, output_file, indent=4, ensure_ascii=False)
    with open(f"test_set_{args.name}_0.json", 'w', encoding='utf8') as output_file:
        json.dump(test_set_0, output_file, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    args = DivideDataArgParser().parse_args()
    print(args)
    divide_data(args)
