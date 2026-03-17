import csv,math
from collections import Counter

data=list(csv.DictReader(open("tennis.csv")))
target=list(data[0].keys())[-1]
features=list(data[0].keys())[:-1]

def entropy(vals):
    c=Counter(vals)
    total=len(vals)
    e=0
    for v in c.values():
        p=v/total
        e-=p*math.log2(p)
    return e

def id3(data,features):
    y=[r[target] for r in data]
    if len(set(y))==1:
        return y[0]

    gains={}
    for f in features:
        vals=set(r[f] for r in data)
        we=0
        for v in vals:
            subset=[r for r in data if r[f]==v]
            we+=(len(subset)/len(data))*entropy([r[target] for r in subset])
        gains[f]=entropy(y)-we

    best=max(gains,key=gains.get)
    tree={best:{}}

    for v in set(r[best] for r in data):
        subset=[r for r in data if r[best]==v]
        tree[best][v]=id3(subset,[f for f in features if f!=best])

    return tree

print(id3(data,features))