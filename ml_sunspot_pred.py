import numpy as np 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

### read in the data ######

arr = np.loadtxt('SN_ms_tot_V2.0_20240223.txt', usecols = (0,3))


tt = arr[np.where(arr[:,1]>=0),0]
WW = arr[np.where(arr[:,1]>=0),1]

# averaging the montly values to get the yearly values

t = np.cumsum(np.ones(np.max(tt).astype(int) - np.min(tt).astype(int))) + np.min(tt)-1
Ww = WW[0]

num = 0
W = np.array(np.zeros(len(t)))

for i in t:
    W[num] = np.mean(WW[np.where(tt==i)])
    num += 1

plt.plot(t,W)
plt.show()

p = 10 # number of data points
k = 2 #number of outputs

def data_split(W, p, k):
    n = len(W)
    X = np.zeros([n-p-k+1, p])
    y = np.zeros([n-p-k+1, k])

    i = p

    while i < n-k+1:
        j = 0
        while j < p:
            X[i-p,j] = W[i-j-1]/100
            j += 1

        j = 0
        while j < k:
            y[i-p,j] = W[i+j]/100
            j += 1
    
        i += 1

    return X, y


X, y = data_split(W, p, k)


print(np.shape(y))



# preparing the data for machine learning nalysis


I = np.arange(np.shape(X)[0])
np.random.shuffle(I)

x = X[I,:]
Y = y[I,:]

x_train = np.float32(x[0:199,:])
x_val = np.float32(x[200:229,:])
x_test = np.float32(x[230:-1,:])

y_train = np.float32(Y[0:199,:])
y_val = np.float32(Y[200:229,:])
y_test = np.float32(Y[230:-1,:])



# model architecture

def build_model(p, k):

    l1=nn.Linear(p, 5)
    r1=nn.ReLU()
    l2=nn.Linear(5, k)
    r2=nn.ReLU()

    model = nn.Sequential(\
        l1,
        r1,
        l2,
        r2, \
    )

    return model

model = build_model(p, k)

print('Model: ', model)

### model hyperparameters

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

batch_num = 10
epoch_num = 100

# training

def model_training(model, batch_num, epoch_num, x_val, x_train, y_val, y_train):
    
    batch_train = np.round(np.shape(y_train)[0]/batch_num)
    batch_val = np.round(np.shape(y_val)[0]/batch_num)

    loss_train = np.zeros(epoch_num)
    loss_val = np.zeros(epoch_num)

    for epoch in np.arange(epoch_num):
        for j in np.arange(batch_num):
         
            x_train_batch = x_train[j*batch_train.astype(int):(j+1)*batch_train.astype(int)-1, :]
            x_val_batch = x_val[j*batch_val.astype(int):(j+1)*batch_val.astype(int)-1, :]
            y_train_batch = y_train[j*batch_train.astype(int):(j+1)*batch_train.astype(int)-1, :]
            y_val_batch = y_val[j*batch_val.astype(int):(j+1)*batch_val.astype(int)-1, :]

            x_train_tensor = torch.from_numpy(x_train_batch)
            x_val_tensor = torch.from_numpy(x_val_batch)
            y_train_tensor = torch.from_numpy(y_train_batch)
            y_val_tensor = torch.from_numpy(y_val_batch)
        

            y_val_pred = model(x_val_tensor)
            loss_val[epoch] += loss(y_val_pred, y_val_tensor)
        
            y_train_pred = model(x_train_tensor)
            loss_train_step = loss(y_train_pred, y_train_tensor)
            loss_train[epoch] += loss_train_step

            optimizer.zero_grad()
            loss_train_step.backward()
            optimizer.step()
    
    return model, loss_train, loss_val


model, loss_train, loss_val = model_training(model, batch_num, epoch_num, x_val, x_train, y_val, y_train)

print(loss_val)

plt.semilogy(loss_train[1:-1], label='train loss')
plt.semilogy(loss_val[1:-1], label='val loss')
plt.legend()
plt.show()

# testing using mean square error

x_test_tensor = torch.from_numpy(x_test)
y_test_pred = model(x_test_tensor)

print(y_test_pred)
print(y_test)

print(np.sqrt(np.mean((y_test-y_test_pred.detach().numpy())**2)))



# forecast

f_step = 12 # define forecast steps


t_pred = np.arange(f_step) + t[-1] + 1
W_input = W[-1*p:]/100

def forecast(W_input, f_step):

    W_pred = np.zeros(f_step)

    for i in np.arange(f_step):
        W_input = np.float32(W_input)
        W_input_tensor = torch.from_numpy(W_input)
        W_pred_tensor = model(W_input_tensor)

        W_pred[i] = W_pred_tensor.detach().numpy()[0]
        W_input[0:-1] = W_input[1:]
        W_input[-1] = W_pred[i]
     
    return W_pred 

W_pred = forecast(W_input, f_step)

print(W_pred*100)

plt.plot(t,W)
plt.plot(t_pred,W_pred*100)
plt.show()








