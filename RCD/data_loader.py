import json
import torch


class TrainDataLoader(object):
    '''
    data loader for training
    '''
    def __init__(self, dir_path):
        self.batch_size = 256
        self.ptr = 0
        self.data = []

        data_file = dir_path + 'train_set.json'
        config_file = 'config.txt'
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        with open(config_file) as i_f:
            i_f.readline()
            student_n, exercise_n, knowledge_n = i_f.readline().split(',')
        self.knowledge_dim = int(knowledge_n)
        self.student_dim = int(student_n)
        self.exercise_dim = int(exercise_n)

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        input_stu_ids, input_exer_ids, input_knowedge_embs, ys, options = [], [], [], [], []
        for count in range(self.batch_size):
            log = self.data[self.ptr + count]
            # one-hot
            knowledge_emb = [0.] * self.knowledge_dim
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code - 1] = 1.0
            y = log['score']
            if 'option' in log:
                option = log['option']
            else:
                option = 0
            input_stu_ids.append(log['user_id'] - 1)
            input_exer_ids.append(log['exer_id'] - 1)
            input_knowedge_embs.append(knowledge_emb)
            ys.append(y)
            options.append(option)

        self.ptr += self.batch_size
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowedge_embs), torch.LongTensor(ys), torch.LongTensor(options)

    def is_end(self):
        if self.ptr + self.batch_size > len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0


class ValTestDataLoader(object):
    def __init__(self, dir_path, d_type='predict', reverse=False):
        self.ptr = 0
        self.data = []
        self.d_type = d_type

        if d_type == 'predict':
            data_file = dir_path + 'test_set.json'
        elif d_type == 'validation':
            data_file = dir_path + 'valid_set.json'
        elif d_type == 'test':
            data_file = dir_path + 'test_set.json'
            if reverse:
                data_file = data_file[:-5] + "_reverse.json"
        config_file = 'config.txt'
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        with open(config_file) as i_f:
            i_f.readline()
            _, _, knowledge_n = i_f.readline().split(',')
            self.knowledge_dim = int(knowledge_n)

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        logs = self.data[self.ptr]['logs']
        user_id = self.data[self.ptr]['user_id']
        input_stu_ids, input_exer_ids, input_knowledge_embs, ys, options = [], [], [], [], []
        for log in logs:
            input_stu_ids.append(user_id - 1)
            input_exer_ids.append(log['exer_id'] - 1)
            # one-hot
            knowledge_emb = [0.] * self.knowledge_dim
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code - 1] = 1.0
            input_knowledge_embs.append(knowledge_emb)
            y = log['score']
            # OT
            if 'option' in log:
                option = log['option']
            else:
                option = 0
            ys.append(y)
            options.append(option)
        self.ptr += 1
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowledge_embs), torch.LongTensor(ys), torch.LongTensor(options)

    def is_end(self):
        if self.ptr >= len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0
