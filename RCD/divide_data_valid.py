import json
import random
import argparse

class DivideDataArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(DivideDataArgParser, self).__init__()
        self.add_argument('--min_log', type=int, default=15,
                          help='Minimum length of logs')
        self.add_argument('--shuffle', action='store_true', default=False,
                          help='Shuffle log data')
        self.add_argument('--train_ratio', type=float, default=0.8,
                          help='Train set ratio')
        self.add_argument('--valid_ratio', type=float, default=0.1,
                          help='Validation set ratio')
        self.add_argument('--dir', type=str, default="../data/poly/",
                          help='Directory path where log_data.json in')


# ##############################
# MIN_LOG = 0  # 15
# TRAIN_RATIO = 0.8
# VALID_RATIO = 0.1
# dir = "../data/poly/"
# SHUFFLE = True
# ##############################
def divide_data(args):
    '''
    1. delete students who have fewer than min_log response logs
    2. divide dataset into train_set and test_set (TRAIN_RATIO:1-TRAIN_RATIO) -> Valid 포함까지로 코드 변경
    :return:
    '''
    # 로그 가져오기
    with open(args.dir + 'log_data.json', encoding='utf8') as i_f:
        stus = json.load(i_f)
    # 1. delete students who have fewer than min_log response logs
    stu_i = 0
    l_log = 0
    while stu_i < len(stus):
        if stus[stu_i]['log_num'] < args.min_log:
            del stus[stu_i]
            stu_i -= 1
        else:
            l_log += stus[stu_i]['log_num']
        stu_i += 1
    # 2. divide dataset into train_set and test_set + valid set
    train_set, valid_set, test_set = [], [], []
    train_tmp = []
    valid_set_size, test_set_size = 0, 0
    for stu in stus:
        user_id = stu['user_id']
        stu_train = {'user_id': user_id}
        stu_valid = {'user_id': user_id}
        stu_test = {'user_id': user_id}
        train_size = int(stu['log_num'] * args.train_ratio)
        valid_size = int(stu['log_num'] * args.valid_ratio)
        test_size = stu['log_num'] - train_size - valid_size
        valid_set_size += valid_size
        test_set_size += test_size

        logs = []
        for log in stu['logs']:
            logs.append(log)
        if args.shuffle:
            random.shuffle(logs)

        # 로그 나누기
        stu_train['log_num'] = train_size
        stu_train['logs'] = logs[:train_size]
        stu_valid['log_num'] = valid_size
        stu_valid['logs'] = logs[train_size:train_size+valid_size]
        stu_test['log_num'] = test_size
        stu_test['logs'] = logs[train_size+valid_size:]

        # 나눈 로그 저장
        valid_set.append(stu_valid)
        test_set.append(stu_test)
        for log in stu_train['logs']:
            train_set.append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'],
                                   'knowledge_code': log['knowledge_code']})
            if 'option' in log:
                train_set[-1]['option'] = log['option']
    # train set 학생 순서 섞기 (시간 포함)
    random.shuffle(train_set)

    #     # 나눈 로그 저장
    #     valid_set.append(stu_valid)
    #     test_set.append(stu_test)
    #     stu_train_logs = []
    #     for log in stu_train['logs']:
    #         stu_train_logs.append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'],
    #                           'option': log['option'], 'knowledge_code': log['knowledge_code']})
    #     train_tmp.append(stu_train_logs)
    # # train set 학생 순서 섞기 (시간 X)
    # random.shuffle(train_tmp)
    # for train in train_tmp:
    #     train_set += train

    # 파일에 저장
    print(f"Train {len(train_set)}, Valid {valid_set_size}({len(valid_set)}), Test {test_set_size}({len(test_set)})")
    with open(args.dir + 'train_set.json', 'w', encoding='utf8') as output_file:
        json.dump(train_set, output_file, indent=4, ensure_ascii=False)
    with open(args.dir + 'valid_set.json', 'w', encoding='utf8') as output_file:
        json.dump(valid_set, output_file, indent=4, ensure_ascii=False)
    with open(args.dir + 'test_set.json', 'w', encoding='utf8') as output_file:
        json.dump(test_set, output_file, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    args = DivideDataArgParser().parse_args()
    print(args)
    divide_data(args)
