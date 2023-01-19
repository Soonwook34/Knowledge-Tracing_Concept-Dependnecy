import json
import random
import argparse
from scipy.stats import bernoulli

class DivideDataArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(DivideDataArgParser, self).__init__()
        self.add_argument('--min_log', type=int, default=0,
                          help='Minimum length of logs')
        self.add_argument('--shuffle', action='store_true',
                          help='Shuffle log data')
        self.add_argument('--train_ratio', type=float, default=0.8,
                          help='Train set ratio')
        self.add_argument('--valid_ratio', type=float, default=0.1,
                          help='Validation set ratio')
        self.add_argument('--name', type=str, default='test',
                          help='Dataset name')


# ##############################
# MIN_LOG = 0  # 15
# TRAIN_RATIO = 0.8
# VALID_RATIO = 0.1
# DIR_PATH = "../data/poly/"
# SHUFFLE = True
# ##############################
def divide_data(args):
    # 로그 가져오기
    with open(f"log_data_{args.name}.json", encoding='utf8') as i_f:
        stus = json.load(i_f)
    # 데이터셋 나누기
    train_set, valid_set, test_set = [], [], []
    valid_set_reverse, test_set_reverse = [], []
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
        # 데이터셋 전체 셔플 여부
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
        for log in stu_train['logs']:
            train_set.append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'],
                              'knowledge_code': log['knowledge_code']})
        valid_set.append(stu_valid)
        test_set.append(stu_test)
        # reverse valid/test 데이터
        for log in stu_valid['logs']:
            a_se = bernoulli.rvs(1 - log["p_e"])
            log['score'] = a_se
        for log in stu_test['logs']:
            a_se = bernoulli.rvs(1 - log["p_e"])
            log['score'] = a_se
        valid_set_reverse.append(stu_valid)
        test_set_reverse.append(stu_test)
    # train set 학생 순서 섞기 (시간 포함)
    random.shuffle(train_set)

    # 파일에 저장
    print(f"Train {len(train_set)}, Valid {valid_set_size}({len(valid_set)}), Test {test_set_size}({len(test_set)})")
    with open(f"train_set_{args.name}.json", 'w', encoding='utf8') as output_file:
        json.dump(train_set, output_file, indent=4, ensure_ascii=False)
    with open(f"valid_set_{args.name}.json", 'w', encoding='utf8') as output_file:
        json.dump(valid_set, output_file, indent=4, ensure_ascii=False)
    with open(f"test_set_{args.name}.json", 'w', encoding='utf8') as output_file:
        json.dump(test_set, output_file, indent=4, ensure_ascii=False)
    with open(f"valid_set_{args.name}_reverse.json", 'w', encoding='utf8') as output_file:
        json.dump(valid_set_reverse, output_file, indent=4, ensure_ascii=False)
    with open(f"test_set_{args.name}_reverse.json", 'w', encoding='utf8') as output_file:
        json.dump(test_set_reverse, output_file, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    args = DivideDataArgParser().parse_args()
    print(args)
    divide_data(args)
