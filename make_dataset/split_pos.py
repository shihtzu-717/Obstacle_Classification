train_list = []
with open('../results/230710_set1-12/train.txt', 'r') as f:
    train_list = f.readlines()
train_pos_list = [i for i in train_list if not "positive" in i]
print(len(train_pos_list))

val_list = []
with open('../results/230710_set1-12/valid.txt', 'r') as f:
    val_list = f.readlines()
val_pos_list = [i for i in val_list if not "positive" in i]
print(len(val_pos_list))

with open('../results/230710_set1-12/train_except_pos.txt', 'w') as f:
    f.writelines(train_pos_list)

with open('../results/230710_set1-12/valid_except_pos.txt', 'w') as f:
    f.writelines(val_pos_list)