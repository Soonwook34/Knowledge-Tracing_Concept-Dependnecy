import json

def check_stats(file_name):

    with open(file_name, encoding="utf8") as i_f:
        data = json.load(i_f)

    log_num = []
    score = {0: 0, 1: 0}

    max_concept = 0
    for line in data:
        for log in line["logs"]:
            max_concept = max(log["knowledge_code"][0], max_concept)
    score_by_concept = []
    for i in range(max_concept):
        score_by_concept.append({0: 0, 1: 0})

    for line in data:
        log_num.append(line["log_num"])
        for log in line["logs"]:
            score[log["score"]] += 1
            score_by_concept[log["knowledge_code"][0] - 1][log["score"]] += 1

    avg = sum(log_num, 0.0) / len(log_num)
    min_num = min(log_num)
    min_index = log_num.index(min_num)
    max_num = max(log_num)
    max_index = log_num.index(max_num)
    correct_rate = score[1] / (score[0] + score[1]) * 100

    print(f"평균 로그: {avg}\n최소 길이 로그: {min_num} (user_id {min_index + 1})\n최대 길이 로그: {max_num} (user_id {max_index + 1})\n정답 비율: {correct_rate:.2f}% ({score[1]}:{score[0]})")
    for c, sco in enumerate(score_by_concept):
        print(f"concept {c}: {sco[1] / (sco[0] + sco[1]) * 100:.3f}% ({sco[1]}:{sco[0]})")
    print(score_by_concept)

if __name__ == '__main__':
    check_stats("./log_data_KSC.json")
