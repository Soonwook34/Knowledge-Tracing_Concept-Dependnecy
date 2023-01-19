import json
import os

if __name__ == '__main__':
    with open('log_data_ori.json', encoding='utf8') as i_f:
        stus = json.load(i_f)

    for stu in stus:
        for log in stu['logs']:
            log['knowledge_code'] = [1]

    with open(f"./log_data_0.json", 'w', encoding='utf8') as output_file:
        json.dump(stus, output_file, indent=4, ensure_ascii=False)
