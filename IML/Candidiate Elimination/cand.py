import csv

with open('data.csv')as f:
  data=list(csv.reader(f))
data=data[1:]
attribute=[row[:-1]for row in data]
target=[row[-1]for row in data]


def phase(attribute,target):
  s=attribute[0].copy()
  g=[["?"for _ in s]for _ in s]
  for i ,h in enumerate(attribute):
    if target[i].lower()=="yes":
      for j in range(len(s)):
        if h[j]!=s[j]:
          h[j]="?"
          g[j][j]="?"
    else:
      for j in range(len(s)):
        if h[j]!=s[j]:
          g[j][j]=s[j]
        else:
          g[j][j]="?"
  g=[row for row in g if not all(val=="?" for val in row)]
  return s,g
s_final,g_final=phase(attribute,target)
print("Specefic:",s_final)
print("Generic:",g_final)