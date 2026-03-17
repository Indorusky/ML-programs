from sklearn.datasets import make_regression
from sklearn.linear_model import *
from sklearn.model_selection import *
from sklearn.metrics import *

# create dataset
X, y = make_regression(n_samples=300, n_features=1, noise=15)

# split dataset
Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2)

model = SGDRegressor(max_iter=1, tol=None)

max_epochs = 100
stop_epochs = 5
best_loss = 1e9
count = 0

for epoch in range(max_epochs):
    model.partial_fit(Xtr, ytr)

    loss = mean_squared_error(yv, model.predict(Xv))
    print(f"Epoch {epoch+1}, loss={loss:.2f}")

    if loss < best_loss:
        best_loss = loss
        count = 0
    else:
        count += 1

    if count >= stop_epochs:
        print("Stopped after 5 non-improving epochs")
        break