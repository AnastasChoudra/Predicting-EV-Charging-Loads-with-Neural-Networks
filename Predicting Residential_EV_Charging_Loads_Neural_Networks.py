import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import torch
from torch import nn
from torch import optim

traffic_reports = pd.read_csv("datasets/Local traffic 		distribution.csv")
traffic_reports.head()

ev_charging_reports = pd.read_csv("datasets/EV charging 	reports.csv")
ev_charging_reports.head()

ev_charging_traffic = ev_charging_reports.merge(traffic_reports, 
                                left_on='Start_plugin_hour', 
                                right_on='Date_from')

ev_charging_traffic.head()
ev_charging_traffic.info()

drop_columns = ['session_ID', 'Garage_ID', 'User_ID', 
                'Shared_ID',
                'Plugin_category','Duration_category', 
                'Start_plugin', 'Start_plugin_hour', 					'End_plugout', 'End_plugout_hour', 
                'Date_from', 'Date_to']

ev_charging_traffic = 	ev_charging_traffic.drop(columns=drop_columns, axis=1)
ev_charging_traffic.head()

for column in ev_charging_traffic.columns:
    if ev_charging_traffic[column].dtype == 'object':
        ev_charging_traffic[column] = 							ev_charging_traffic[column].str.replace(',', '.')
    
ev_charging_traffic.head()

for column in ev_charging_traffic.columns:
ev_charging_traffic[column] = 		ev_charging_traffic[column].astype(float)

ev_charging_traffic.head()

numerical_features = ev_charging_traffic.drop(['El_kWh'], 	axis=1).columns
X = ev_charging_traffic[numerical_features]

y = ev_charging_traffic['El_kWh']

X_train, X_test, y_train, y_test = train_test_split(X, y,
       train_size=0.80,
       test_size=0.20,
       random_state=2) # set a random seed - do not modify

print("Training size:", X_train.shape)
print("Testing size:", X_test.shape)
#Training size: (5466, 26)
#Testing size: (1367, 26)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_test_predictions = linear_model.predict(X_test)
test_mse = mean_squared_error(y_test, linear_test_predictions)
print("Linear Regression - Test Set MSE:", test_mse)
# Linear Regression - Test Set MSE: 131.4188163356643

# Convert training set
X_train_tensor = torch.tensor(X_train.values, 	dtype=torch.float)
y_train_tensor = torch.tensor(y_train.values, 	dtype=torch.float).view(-1,1)

# Convert testing set
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float)
y_test_tensor = torch.tensor(y_test.values, 	dtype=torch.float).view(-1,1)

torch.manual_seed(42)

model = nn.Sequential(
    nn.Linear(26, 56),
    nn.ReLU(),
    nn.Linear(56, 26),
    nn.ReLU(),
    nn.Linear(26, 1)
)
loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0007)

num_epochs = 3000 # number of training iterations
for epoch in range(num_epochs):
    outputs = model(X_train_tensor) # forward pass 
    mse = loss(outputs, y_train_tensor) # calculate the loss 
    mse.backward() # backward pass
    optimizer.step() # update the weights and biases
optimizer.zero_grad() # reset the gradients to zero

    # keep track of the loss during training
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], MSE Loss: 			{mse.item()}')
#Epoch [500/3000], MSE Loss: 146.75697326660156
#Epoch [1000/3000], MSE Loss: 139.6661834716797
#Epoch [1500/3000], MSE Loss: 118.94923400878906
#Epoch [2000/3000], MSE Loss: 116.21517181396484
#Epoch [2500/3000], MSE Loss: 110.67818450927734
#Epoch [3000/3000], MSE Loss: 107.02542877197266    

# save the neural network
torch.save(model, 'models/model.pth')  
# using the loaded neural network `loaded_model`
model.eval() # set the model to evaluation mode
with torch.no_grad(): # disable gradient calculations
    predictions = model(X_test_tensor) # generate apartment rent predictions
    test_loss = loss(predictions, y_test_tensor) # calculate testing set MSE loss
    
print('Neural Network - Test Set MSE:', test_loss.item()) # print testing set MSE
# Neural Network - Test Set MSE: 122.73201751708984

# load the model
model4500 = torch.load('models/model4500.pth')

# using the loaded neural network `loaded_model`
model4500.eval() # set the model to evaluation mode
with torch.no_grad(): # disable gradient calculations
    predictions = model4500(X_test_tensor) # generate apartment rent predictions
    test_loss = loss(predictions, y_test_tensor) # calculate testing set MSE loss
    
print('Neural Network - Test Set MSE:', test_loss.item()) # print testing set MSE
# Neural Network - Test Set MSE: 115.21600341796875
# Pretty cool! The increased training improved our test loss to about 115.2, a full 12% improvement on our linear regression baseline. 
# So the nonlinearity introduced by the neural network actually helped us out.
