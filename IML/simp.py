import numpy as np
import matplotlib.pyplot as plt

x=np.array([1,2,3,4,5])
y=np.array([30,35,40,50,55])

a=np.sum((x-x.mean())*(y-y.mean()))/np.sum((x-x.mean())**2)
b=y.mean()-a*x.mean()

print("Slope(a):",a)
print("Intercept(b):",b)
print("Regression Value:(y=a*x+b)")

x_user=(float(input("Enter the No.Of Study Hours(x):")))
y_pred=a*x_user+b

print("Predicted Value:",y_pred)

plt.scatter(x,y)
plt.plot(x,a*x+b)
plt.xlabel("Study Hours")
plt.ylabel;("Marks")
plt.title("Simple Linear Regression")
plt.show()