train_list = []
with open('../results/230710_set1-12/critical_train.txt', 'r') as f:
    train_list = [i.strip() for i in f.readlines()]

valid_list = []
with open('../results/230710_set1-12/critical_valid.txt', 'r') as f:
    valid_list = [i.strip() for i in f.readlines()]


train_set = set(train_list)
valid_set = set(valid_list)
and_set = set()
for i in train_set & valid_set:
    and_set.add(i)

print(and_set)
print(len(train_list), len(train_set))
print(len(valid_list), len(valid_set))

# with open('../results/230710_set1-12/new_train.txt', 'w') as f:
#     for i in train_set:
#         f.write(i+'\n')

# with open('../results/230710_set1-12/new_valid.txt', 'w') as f:
#     for i in valid_set:
#         f.write(i+'\n')