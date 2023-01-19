'''
--gpu 0 --mode 1 --log test --dir ../data/poly
gpu는 기존 모델의 gpu와 같아야 함
'''

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from utils import CommonArgParser
from data_loader import ValTestDataLoader

def test(args):
    data_loader = ValTestDataLoader(args.dir, 'test')
    device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
    net = torch.load(f"model/RCD_{args.log}_best.pt")

    data_loader.reset()
    net.eval()

    test_data = []
    while not data_loader.is_end():
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels, options = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels, options = input_stu_ids.to(
            device), input_exer_ids.to(device), input_knowledge_embs.to(device), labels.to(device), options.to(device)
        test_data.append([input_stu_ids, input_exer_ids, input_knowledge_embs, labels, options])

    # KT / KT+OT / OT
    if args.mode == 0:
        loss_function = nn.NLLLoss()
    elif args.mode == 1:
        loss_function_kt = nn.NLLLoss()
        loss_function_ot = nn.CrossEntropyLoss()
    else:
        loss_function = nn.CrossEntropyLoss()

    correct_count, exer_count = 0, 0
    pred_all, label_all = [], []
    running_test_loss = 0.0
    with tqdm(test_data, unit="it") as test_bar:
        test_bar.set_description(f"Test {args.log} model")
        if args.mode == 1:
            pred_all, label_all = [[], []], [[], []]
            correct_count = [0, 0]
        for data in test_bar:
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels, options = data
            # KT / KT+OT / OT
            if args.mode == 0:
                # forward
                output_1 = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
                output_pred = output_1.view(-1)
                # loss (KT)
                output_0 = torch.ones(output_1.size()).to(device) - output_1
                output = torch.cat((output_0, output_1), 1)
                loss = loss_function(torch.log(output + 1e-10), labels)
                # count hit
                for i in range(len(labels)):
                    if (labels[i] == 1 and output_pred[i] > 0.5) or (labels[i] == 0 and output_pred[i] < 0.5):
                        correct_count += 1
                # store pred
                pred_all += output_pred.to(torch.device('cpu')).tolist()
                label_all += labels.to(torch.device('cpu')).tolist()
            elif args.mode == 1:
                # forward
                output_kt, output_ot = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
                output_pred_kt = output_kt.view(-1)
                output_pred_ot = torch.argmax(output_ot, dim=1)
                # loss (KT+OT)
                output_0 = torch.ones(output_kt.size()).to(device) - output_kt
                output_kt = torch.cat((output_0, output_kt), 1)
                loss_kt = loss_function_kt(torch.log(output_kt + 1e-10), labels)
                loss_ot = loss_function_ot(output_ot, options)
                loss = args.lamb * loss_kt + (1 - args.lamb) * loss_ot
                # count hit (KT+OT 다 맞아야 정답)
                for i in range(len(labels)):
                    if (labels[i] == 1 and output_pred_kt[i] > 0.5) or (labels[i] == 0 and output_pred_kt[i] < 0.5):
                        correct_count[0] += 1
                    if options[i] == output_pred_ot[i]:
                        correct_count[1] += 1
                # store pred
                pred_all[0] += output_pred_kt.to(torch.device('cpu')).tolist()
                label_all[0] += labels.to(torch.device('cpu')).tolist()
                pred = output_ot.flatten()
                option = nn.functional.one_hot(options, num_classes=4).flatten()
                pred_all[1] += pred.to(torch.device('cpu')).tolist()
                label_all[1] += option.to(torch.device('cpu')).tolist()
            else:
                # forward
                output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
                output_pred = torch.argmax(output, dim=1)
                # count hit
                for i in range(len(labels)):
                    if options[i] == output_pred[i]:
                        correct_count += 1
                # loss OT
                loss = loss_function(output, options)
                # store pred
                pred = output.flatten()
                label = nn.functional.one_hot(options, num_classes=4).flatten()
                pred_all += pred.to(torch.device('cpu')).tolist()
                label_all += label.to(torch.device('cpu')).tolist()
            exer_count += len(labels)
            running_test_loss += loss.item() * len(labels)
            if args.mode == 1:
                test_bar.set_postfix(acc_KT=f"{correct_count[0] / exer_count:.6f}",
                                     acc_OT=f"{correct_count[1] / exer_count:.6f}",
                                     test_loss=f"{running_test_loss / exer_count:.6f}")
            else:
                test_bar.set_postfix(acc=f"{correct_count / exer_count:.6f}",
                                     test_loss=f"{running_test_loss / exer_count:.6f}")

    # compute accuracy, RMSE, AUC, test loss
    if args.mode == 1:
        accuracy, rmse, auc = [0, 0], [0, 0], [0, 0]
        for i in range(2):
            pred_all[i] = np.array(pred_all[i])
            label_all[i] = np.array(label_all[i])
        pred_all = np.array(pred_all, dtype=type(pred_all[0]))
        label_all = np.array(label_all, dtype=type(label_all[0]))
        for i in range(2):
            accuracy[i] = correct_count[i] / exer_count
            rmse[i] = np.sqrt(np.mean((label_all[i] - pred_all[i]) ** 2))
            auc[i] = roc_auc_score(label_all[i], pred_all[i])
        test_loss = running_test_loss / exer_count
        print_str = f"model={args.log}\n" \
                    f"accuracy_KT={accuracy[0]:.6f}, rmse_KT={rmse[0]:.6f}, auc_KT={auc[0]:.6f}\n" \
                    f"accuracy_OT={accuracy[1]:.6f}, rmse_OT={rmse[1]:.6f}, auc_OT={auc[1]:.6f}, test_loss={test_loss:.6f}"
    else:
        pred_all = np.array(pred_all)
        label_all = np.array(label_all)
        accuracy = correct_count / exer_count
        rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
        auc = roc_auc_score(label_all, pred_all)
        test_loss = running_test_loss / exer_count
        print_str = f"model={args.log}\n" \
                    f"accuracy={accuracy:.6f}, rmse={rmse:.6f}, auc={auc:.6f}, test_loss={test_loss:.6f}"
    print(print_str)

    with open(f'result/RCD_test_result.txt', 'a', encoding='utf8') as f:
        f.write(print_str + '\n')


if __name__ == '__main__':
    args = CommonArgParser().parse_args()

    test(args)

