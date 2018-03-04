import numpy as np
import pandas as pd
from four_layer_neural_net import FourLayerNet

df = pd.read_csv("train-dammy.csv")
del df["PassengerId"]
del df["Name"]
del df["Ticket"]
del df["Cabin"]
df["Sex"].replace("male", 0, inplace=True)
df["Sex"].replace("female", 1, inplace=True)
df["Embarked"].replace("Q", 0, inplace=True)
df["Embarked"].replace("S", 1, inplace=True)
df["Embarked"].replace("C", 2, inplace=True)
df["Age"].fillna(29.7, inplace=True)
df["Embarked"].fillna(1, inplace=True)

x_train = np.array(df.drop("Survived", axis=1))

array_train_label = np.empty((0,2), int)
for i in np.array(df["Survived"]):
    if i == 0:
        array_train_label = np.append(array_train_label, np.array([[1,0]]), axis=0)
    elif i == 1:
        array_train_label = np.append(array_train_label, np.array([[0,1]]), axis=0) 
t_train = array_train_label

#ハイパーパラメタ
iters_num = 1500
train_size = x_train.shape[0]
batch_size = 50
learning_rate = 0.01

train_loss_list = []
train_acc_list = []
iter_per_epoch = max(round(train_size / batch_size), 1)


network = FourLayerNet(input_size = 7, hidden_size_1 = 70, hidden_size_2 = 70, output_size = 2)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    grad = network.numerical_gradient(x_batch, t_batch)
    
    for key in ("W1", "b1", "W2", "b2", "W3", "b3"):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        train_acc_list.append(train_acc)
        print("train acc | " + str(train_acc))
