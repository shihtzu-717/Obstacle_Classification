import os
from argparse import ArgumentParser
import glob
from tqdm import tqdm

def calculate_stat(path, cnt):
    info = dict()
    info['NS'] = 0
    total = 0
    for i in range(cnt+1):
        info[str(i)]=0

    # annot_path = path+'/*/annotations/*.txt'
    annot_path = path+'/*.txt'
    annots_file = glob.glob(annot_path, recursive=True)

    for i in tqdm(annots_file):
        with open(i, 'r') as f:
            total+=1
            lines = f.readlines()
            if len(lines)==0:
                info['NS']+=1
            for line in lines:
                l = line.split(' ')
                try:
                    info[str(l[0])]+=1
                except KeyError:
                    pass

    for key in info.keys():
        if key == 'NS':
            print("Negative Sample: ",info[key])
        else:
            print("Class-",int(key)," : ",info[key])
    return total, info

def main():
    parser = ArgumentParser()
    parser.add_argument('--path', '-p', type=str, default='./annotation', help='annotation folder path')
    parser.add_argument('--cnt', '-c', type=int, help='class count number', default=12)
    args = parser.parse_args()

    calculate_stat(args.path, args.cnt)

if __name__ == "__main__":
    main()



