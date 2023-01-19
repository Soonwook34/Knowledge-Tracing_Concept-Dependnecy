import json
import argparse
from tqdm import tqdm

class KEGraphArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(KEGraphArgParser, self).__init__()
        self.add_argument('--dir', type=str, default="../data/poly/",
                          help='Directory Path')

def build_local_map(args):
    data_file = args.dir + 'log_data.json'
    with open('config.txt') as i_f:
        i_f.readline()
        student_n, exer_n, knowledge_n = list(map(eval, i_f.readline().split(',')))

    temp_list = []
    with open(data_file, encoding='utf8') as i_f:
        data = json.load(i_f)
    k_from_e = ''  # e(src) to k(dst)
    e_from_k = ''  # k(src) to k(dst)
    with tqdm(data, unit="stu") as data_bar:
        for line in data_bar:
            for log in line['logs']:
                exer_id = log['exer_id'] - 1
                for k in log['knowledge_code']:
                    if (str(exer_id) + '\t' + str(k - 1 + exer_n)) not in temp_list or (
                            str(k - 1 + exer_n) + '\t' + str(exer_id)) not in temp_list:
                        k_from_e += str(exer_id) + '\t' + str(k - 1 + exer_n) + '\n'
                        e_from_k += str(k - 1 + exer_n) + '\t' + str(exer_id) + '\n'
                        temp_list.append((str(exer_id) + '\t' + str(k - 1 + exer_n)))
                        temp_list.append((str(k - 1 + exer_n) + '\t' + str(exer_id)))
    with open(args.dir + 'graph/k_from_e.txt', 'w') as f:
        f.write(k_from_e)
    with open(args.dir + 'graph/e_from_k.txt', 'w') as f:
        f.write(e_from_k)


if __name__ == '__main__':
    args = KEGraphArgParser().parse_args()
    build_local_map(args)
