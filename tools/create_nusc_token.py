# nusc all data
import pickle

pkl = ['train', 'val', 'test']
for p in pkl:
    work_path = '../data/nuscenes/nuscenes_infos_{}.pkl'.format(p)
    with open(work_path, "rb") as f:
        data = pickle.load(f)
    for i in range(len(data['infos'])):
        token = data['infos'][i]['token']
        with open('datasets/nusc/depth_{}.txt'.format(p), 'a') as f:
            f.writelines(token + '\n')

print(len(open('../datasets/nusc/depth_train.txt').readlines()))

# Filter out tokens that do not have prev or next attributes in depth_train.txt
# save the result to depth_train_prevnext.txt
import json

dic = {}
with open('../data/nuscenes/v1.0-trainval/sample.json', 'r') as f:
    json_ = json.load(f)
for i in range(len(json_)):
    dic[json_[i]["token"]] = i

with open('/mnt/cfs/algorithm/wenjie.yuan/SurroundDepth/datasets/nusc/depth_train.txt', 'r') as f:
    content = f.readlines()
with open('/mnt/cfs/algorithm/wenjie.yuan/SurroundDepth/datasets/nusc/depth_train_prenext.txt', 'a') as f:
    for con in content:
        if con[-1:] == '\n':
            con = con[:-1]
        tmp = json_[dic[con]]
        if tmp["prev"] != "" and tmp["next"] != "":
            f.writelines(con + '\n')

