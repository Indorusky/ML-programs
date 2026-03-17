import csv

def finds(file_path):
 
 with open(file_path,'r') as f:
  data=list(csv.reader(f))
  
  hypothesis=['']*(len(data[0])-1)
  for rows in data:
   attributes=rows[:-1]
   target=rows[-1]
  
   if target.lower()=='yes':
    if hypothesis[0]=='':
      hypothesis=attributes.copy()	
    else:
      for i in range(len(attributes)):
        if hypothesis[i]!=attributes[i]:
          hypothesis[i]='?'
 return hypothesis
print("The Final Hypothesis:",finds("data.csv"))