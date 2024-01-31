import glob
import json
import natsort # python3 get_best_accuracy.py 로 실행
import argparse

logs = natsort.natsorted(glob.glob('./results/240125/*nbclss_*/log.txt'))
print(logs)
for log in logs:
    model = log.split('/')[-2]
    with open(log) as f:
        lines = [i.strip() for i in f.readlines()]
        dict_collection = [json.loads(line) for line in lines]
        best_acc = 0
        best_train_loss = 1.0
        best_valid_loss = 1.0
        best_acc_epoch = 0
        best_train_loss_epoch = 0
        best_valid_loss_epoch = 0
        for epoch in dict_collection:
            if best_acc < epoch.get('test_acc1'):
                best_acc = epoch.get('test_acc1')
                best_acc_epoch = epoch.get('epoch')

            if epoch.get('train_loss') < best_train_loss:
                best_train_loss = epoch.get('train_loss')
                best_train_loss_epoch = epoch.get('epoch')

            if epoch.get('test_loss') < best_valid_loss:
                best_valid_loss = epoch.get('test_loss')
                best_valid_loss_epoch = epoch.get('epoch')

        print(f"model: {model}\n\tbest_acc_epoch: {best_acc_epoch},  best_accuracy: {best_acc:.2f}, best_train_loss_epoch: {best_train_loss_epoch},  best_train_loss: {best_train_loss:.2f}, best_valid_loss_epoch: {best_valid_loss_epoch},  best_valid_loss: {best_valid_loss:.2f}\n")
