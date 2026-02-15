

import reader_with_players_new_advanced as reader
import models
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch.nn as nn

import ili
from ili.inference import InferenceRunner
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, Subset
from ili.dataloaders import TorchLoader


from torch.distributions.transforms import AffineTransform



import numpy as np
import pandas as pd


import multiprocessing as mp


#r_polars = reader.Reader_Polars('../Data/Complete_data/merged_nba_games_updating.csv', '/home/jsarrato/PersonalProjects/NBBA/players_data.csv', latest_matches=30, latest_vs_matches = 30, latest_other_matches = 0, latest_player_dates=0)

r_polars = reader.Reader_Polars('/home/jsarrato/PersonalProjects/NBBA/Data/DataNew_advanced/processed_team_boxscores_ML.csv', '/home/jsarrato/PersonalProjects/NBBA/Data/DataNew_advanced/processed_player_boxscores.csv', latest_matches=10, latest_vs_matches = 2, latest_other_matches = 0, latest_player_dates=0)


dataset_current_game, dataset_old_games_home, dataset_old_games_away, dataset_old_games_vs, dataset_players_home, dataset_players_away, labels_spread, labels_total, bet_pred, bet_pred_total, dates, home_teams, away_teams = r_polars.make_dataset()








"""s
# Example input arrays (replace with your actual data)
home_moneylines = bet_pred[:,0]  # Example moneylines for the home team
away_moneylines = bet_pred[:,1]  # Example moneylines for the away team
results = labels  # Example results (home points - away points)

# Convert dates to a more readable format
dates2 = pd.to_datetime(dates, format='%Y%m%d')


# Function to determine the predicted winner based on moneylines
def predict_winner(home_ml, away_ml):
    if np.isnan(home_ml) or np.isnan(away_ml):
        return None  # Cannot make a prediction if data is missing
    if home_ml < away_ml:
        return 'home'
    elif away_ml < home_ml:
        return 'away'
    else:
        return 'draw'  # In case the moneylines are equal

# Function to determine the actual winner based on result
def actual_winner(result):
    if np.isnan(result):
        return None  # No result if data is missing
    if result > 0:
        return 'home'
    elif result < 0:
        return 'away'
    else:
        return 'draw'

# Evaluate predictions
predictions = [predict_winner(home_ml, away_ml) for home_ml, away_ml in zip(home_moneylines, away_moneylines)]
actuals = [actual_winner(result) for result in results]

# Create a DataFrame and filter out rows with missing data
data = pd.DataFrame({
    'Date': dates2,
    'Prediction': predictions,
    'Actual': actuals
})
data['Correct'] = data['Prediction'] == data['Actual']
data = data.dropna(subset=['Prediction', 'Actual']).reset_index(drop = True)  # Drop rows where predictions or actuals are None

# Calculate accuracy over time
data['Cumulative Accuracy'] = data['Correct'].cumsum() / (data.index + 1)

# Calculate centered rolling mean accuracy with a window of size 50
data['Centered Accuracy'] = data['Correct'].rolling(window=500, min_periods=1, center=True).mean()


# Plot cumulative accuracy evolution
plt.figure(figsize=(12, 6))
plt.plot(data['Date'], data['Cumulative Accuracy'], label='Cumulative Accuracy', color='blue')

# Plot centered rolling mean accuracy
plt.plot(data['Date'], data['Centered Accuracy'], label='Centered Mean Accuracy (Window=50)', color='green')

# Add labels, legend, and grid
plt.xlabel('Date')
plt.ylabel('Accuracy')
plt.title('Accuracy Over Time')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""



"""
# Example training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
"""
#    Train and validate a PyTorch model.
#    
#    Args:
#        model: PyTorch model to train.
#        train_loader: DataLoader for the training data.
#        val_loader: DataLoader for the validation data.
#        criterion: Loss function.
#        optimizer: Optimizer for training.
#        device: Device to use for training (e.g., 'cuda' or 'cpu').
#        num_epochs: Number of epochs to train for.
"""
    model.to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 20)

        # Training phase
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data)
            total_train += labels.size(0)

        train_loss /= total_train
        train_accuracy = correct_train.double() / total_train

        print(f'Training Loss: {train_loss:.4f} Accuracy: {train_accuracy:.4f}')

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data)
                total_val += labels.size(0)

        val_loss /= total_val
        val_accuracy = correct_val.double() / total_val

        print(f'Validation Loss: {val_loss:.4f} Accuracy: {val_accuracy:.4f}')

    print('Training complete.')
"""








"""
mask = np.all(~np.isnan(inputs), axis = 1)

inputs, labels = torch.Tensor(inputs[mask,:]), torch.Tensor(labels[mask]).reshape(-1,1)

bet_pred = bet_pred[mask]
"""



class LSTM_3designs(nn.Module):
    def __init__(self, input_size=7, hidden_size=32, output_size=2, input_size2=7, hidden_size2=32, num_layers=1, num_layers2=1, input_size3=7, hidden_size3=32, num_layers3=1, glob_var_size = 10):
        super(LSTM_3designs, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout = 0.15)

        self.rnn2 = nn.LSTM(input_size2, hidden_size2, num_layers2, batch_first=True, dropout = 0.15)
        
        self.rnn3 = nn.LSTM(input_size3, hidden_size3, num_layers3, batch_first=True, dropout = 0.15)

        self.fc = nn.Linear(2 * hidden_size + 1 * hidden_size2 + 1 * glob_var_size + 0 * r_polars.N_players_per_team * hidden_size3, output_size)
        
        self.drop = nn.Dropout(0.25)

    def forward(self, x):
        x1 = x['current_game']
        x2 = x['games_home']
        x3 = x['games_away']
        x4 = x['games_vs']
        x5 = x['players_home']
        x6 = x['players_away']
        # x is a list of 20 inputs for the LSTM
        hidden_states = []  # To collect final hidden states from each LSTM

        rnn_out, _ = self.rnn(x2)  # Process each input through the LSTM
        hidden_states.append(rnn_out[:, -1, :])  # Collect the last hidden state of the sequence

        rnn_out, _ = self.rnn(x3)  # Process each input through the LSTM
        hidden_states.append(rnn_out[:, -1, :])  # Collect the last hidden state of the sequence
  
        rnn_out2, _ = self.rnn2(x4)
        hidden_states.append(rnn_out2[:, -1, :])  # Collect the last hidden state of the sequence
        """
        for jj in range(x5.shape[1]):
            rnn_out3, _ = self.rnn3(x5[:,jj,:,:])
            hidden_states.append(rnn_out3[:, -1, :]) 
            
        for jj in range(x6.shape[1]):
            rnn_out3, _ = self.rnn3(x6[:,jj,:,:])
            hidden_states.append(rnn_out3[:, -1, :])
        """
        
        
        

            
        # Concatenate all hidden states along the feature dimension
        concatenated_hidden = torch.cat(hidden_states, dim=1)
        concatenated_hidden2 = torch.cat([concatenated_hidden, x1], dim=1)

        # Fully connected layer to output predictions
        out = self.fc(concatenated_hidden2)
        
        out = self.drop(out)

        return out




class TensorDict(dict):
    def to(self, device):
        """
        Move all tensors in the dictionary to the specified device.
        """
        for key, tensor in self.items():
            self[key] = tensor.to(device)
        return self  # Allow method chaining   
        
    def unsqueeze(self, axis):
        """
        Move all tensors in the dictionary to the specified device.
        """
        for key, tensor in self.items():
            self[key] = torch.unsqueeze(tensor, axis)
        return self  # Allow method chaining   
        




class MultiInputDataset(Dataset):
    def __init__(self, input_data, labels):
        """
        Args:
            input_data: List of tuples, where each tuple contains multiple tensors (inputs).
            labels: List of tensors, where each tensor corresponds to a label.
        """
        self.input_data = input_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return inputs as a dictionary and labels as a tensor
        inputs = TensorDict({
            'current_game': self.input_data[idx][0],
            'games_home': self.input_data[idx][1],
            'games_away': self.input_data[idx][2],
            'games_vs': self.input_data[idx][3],
            'players_home': self.input_data[idx][4],
            'players_away': self.input_data[idx][5]
        })
        label = self.labels[idx]
        return inputs, label


# Convert all datasets to float32

dataset_current_game = torch.tensor(dataset_current_game, dtype=torch.float32)
dataset_old_games_home = torch.tensor(dataset_old_games_home, dtype=torch.float32)
dataset_old_games_away = torch.tensor(dataset_old_games_away, dtype=torch.float32)
dataset_old_games_vs = torch.tensor(dataset_old_games_vs, dtype=torch.float32)
dataset_players_home = torch.tensor(dataset_players_home, dtype=torch.float32)
dataset_players_away = torch.tensor(dataset_players_away, dtype=torch.float32)

dataset_players_away = torch.nan_to_num(dataset_players_away)
dataset_players_home = torch.nan_to_num(dataset_players_home)


def normalize_tensor(tensor):
    reduce_dims = tuple(range(tensor.ndim - 1))

    x_mean = tensor.mean(dim=reduce_dims)
    x_std = tensor.std(dim=reduce_dims)

    # avoid division by zero
    x_std = torch.clamp(x_std, min=1e-16)

    # z-normalize x
    x_transform = AffineTransform(
    loc=x_mean, scale=x_std, event_dim=1)

    return x_transform.inv(tensor)
    
    
def normalize_tensor_masked(tensor):
    # Create a mask for values that are not equal to -9999
    mask = tensor != -9999
    
    # Compute the mean and std while ignoring -9999 values using the mask
    reduce_dims = tuple(range(tensor.ndim - 1))
    x_mean = (tensor * mask).sum(dim=reduce_dims) / mask.sum(dim=reduce_dims)
    x_std = ((tensor - x_mean) ** 2 * mask).sum(dim=reduce_dims) / mask.sum(dim=reduce_dims)
    
    x_std = torch.sqrt(x_std)  # Calculate standard deviation
    
    # Move mean and std to the appropriate device
    x_mean = x_mean
    x_std = x_std

    # Avoid division by zero
    x_std = torch.clamp(x_std, min=1e-16)

    # Normalize tensor by applying the transformation
    x_transform = AffineTransform(loc=x_mean, scale=x_std, event_dim=1)

    # Apply the transformation only to the values that are not -9999
    transformed_tensor = x_transform.inv(tensor)

    # Use the mask to set the -9999 values back to their original form
    transformed_tensor = torch.where(mask, transformed_tensor, tensor)

    return transformed_tensor
    
    
dataset_current_game = normalize_tensor_masked(dataset_current_game)    
dataset_old_games_home = normalize_tensor_masked(dataset_old_games_home)    
dataset_old_games_away = normalize_tensor_masked(dataset_old_games_away)    
dataset_old_games_vs = normalize_tensor_masked(dataset_old_games_vs)    
dataset_players_home = normalize_tensor_masked(dataset_players_home)    
dataset_players_away = normalize_tensor_masked(dataset_players_away)    



labels = torch.tensor(labels_total.reshape((-1,1)) , dtype=torch.float32)



input_data = [
    (dataset_current_game[ii], dataset_old_games_home[ii], dataset_old_games_away[ii], dataset_old_games_vs[ii], dataset_players_home[ii], dataset_players_away[ii]) for ii in range(len(labels))
]

# Create the dataset
dataset = MultiInputDataset(input_data, labels)

"""
save_path = "dataset_30games_10vsgames_30playergames_assumeknowninactive.pt"
torch.save({
    'input_data': dataset.input_data,
    'labels': dataset.labels,
}, save_path)
"""
# Split indices
test_size = int(0.1 * len(dataset))  # 5% for testing
val_size = int(0.2 * (len(dataset) - test_size))  # 10% of the remaining 95% for validation
train_size = len(dataset) - test_size - val_size  # Rest for training

# Explicitly select last elements for the test set
test_indices = list(range(len(dataset) - test_size, len(dataset)))

# Remaining indices for train and validation
remaining_indices = list(range(len(dataset) - test_size))

# Shuffle the remaining indices for training and validation
torch.manual_seed(42)  # For reproducibility
shuffled_indices = torch.randperm(len(remaining_indices)).tolist()

# Split shuffled indices into train and validation
val_indices = shuffled_indices[:val_size]
train_indices = shuffled_indices[val_size:]

# Create Subsets
train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)















model = LSTM_3designs(input_size=(2 + r_polars.N_other_game_fields), hidden_size=32, output_size=128, input_size2=(2 + r_polars.N_other_game_fields), hidden_size2=32, num_layers=2, num_layers2=2, input_size3=(r_polars.N_player_fields), hidden_size3 = 16, num_layers3=2,  glob_var_size = r_polars.N_current_game_fields)

"""
model = models.MLP(inputs.shape[1], [512,256,128], 64, 0.2)
batch_size = 64
"""
N_samples = 1000




loader = TorchLoader(train_loader, val_loader)




accelerator = Accelerator()
device = accelerator.device


# define training arguments
train_args = {
    'training_batch_size': batch_size,
    'learning_rate': 5e-5,
    'stop_after_epochs': 5,
    'max_epochs': 500
}

nets = [ili.utils.load_nde_lampe(model='maf', hidden_features=64, num_transforms=8, x_normalize=False, theta_normalize = True, device=device, embedding_net = model)]

prior = ili.utils.Uniform(low=[-100], high=[500], device=device)


# initialize the trainer
# initialize the trainer
runner = InferenceRunner.load(
    backend='lampe',
    engine='NPE',
    prior=prior,
    nets=nets,
    device=device,
    train_args=train_args,
    proposal=None,
    out_dir='.'
)


posterior_ensemble, summaries = runner(loader=loader)

#posterior_ensemble = pickle.load(open('posterior.pkl', 'rb'))

N_samples = 1000

samples_train = np.zeros((N_samples,train_size,1))
labels_train = np.zeros((train_size))

for ii in range(len(train_dataset)):
	print(ii,train_size)
	samples_train[:,ii,:] = posterior_ensemble.sample((N_samples,), train_dataset[ii][0].unsqueeze(0).to(device))
	labels_train[ii] = float(train_dataset[ii][1][0])


samples_val = np.zeros((N_samples,val_size,1))
labels_val = np.zeros((val_size))

for ii in range(len(val_dataset)):
	print(ii,val_size)
	samples_val[:,ii,:] = posterior_ensemble.sample((N_samples,), val_dataset[ii][0].unsqueeze(0).to(device))
	labels_val[ii] = float(val_dataset[ii][1][0])

  
samples_test = np.zeros((N_samples,test_size,1))
labels_test = np.zeros((test_size))

for ii in range(len(test_dataset)):
	print(ii,test_size)
	samples_test[:,ii,:] = posterior_ensemble.sample((N_samples,), test_dataset[ii][0].unsqueeze(0).to(device))
	labels_test[ii] = float(test_dataset[ii][1][0])


test_loss = torch.tensor([0])
for ii in range(len(test_dataset)):
	test_loss = test_loss + posterior_ensemble.log_prob(test_dataset[ii][1].to(device), test_dataset[ii][0].unsqueeze(0).to(device))
test_loss = test_loss/torch.Tensor([test_size])
print(f'Test loss: {test_loss}')


"""
val_loss = torch.tensor([0])
for ii in range(len(val_dataset)):
	val_loss = val_loss + posterior_ensemble.log_prob(val_dataset[ii][1].to(device), val_dataset[ii][0].unsqueeze(0).to(device))
val_loss = val_loss/torch.Tensor([val_size])
print(f'val loss: {val_loss}')



train_loss = torch.tensor([0])
for ii in range(len(train_dataset)):
	train_loss = train_loss + posterior_ensemble.log_prob(train_dataset[ii][1].to(device), train_dataset[ii][0].unsqueeze(0).to(device))
train_loss = train_loss/torch.Tensor([train_size])
print(f'train loss: {train_loss}')
"""



        
predictions = np.median(samples_train, axis = 0).reshape((-1))
predictions_val = np.median(samples_val, axis = 0).reshape((-1))
predictions_test = np.median(samples_test, axis = 0).reshape((-1))

def winlossaccuracy(pred, real):
	return np.sum( np.array(np.sign(pred) == np.sign(real), dtype = int) )/len(pred)
	
print(winlossaccuracy(predictions, labels_train))
print(winlossaccuracy(predictions_val, labels_val))
print(winlossaccuracy(predictions_test, labels_test))

def overunderaccuracy(pred, bookie, real):
	mask = (bookie != 0) & (~np.isnan(bookie))
	
	pred = pred[mask]
	bookie = bookie[mask]
	real = real[mask]
	
	print(len(pred))
	
	return len(pred[ ((pred > bookie) & (real > bookie)) | ((pred < bookie) & (real < bookie)) ])/len(pred)
	
	
print(overunderaccuracy(predictions, bet_pred_total[train_indices], labels_train))
print(overunderaccuracy(predictions_val, bet_pred_total[val_indices], labels_val))
print(overunderaccuracy(predictions_test, bet_pred_total[-test_size:], labels_test))

"""
print( len(predictions_test[(predictions_test > bet_pred_total[-2000:,0]) & (np.array(labels_test).reshape(-1) > bet_pred_total[-2000:,0])])/ len( predictions_test[ (predictions_test > bet_pred_total[-2000:,0]) ] ))

print( len(predictions_test[(predictions_test < bet_pred_total[-2000:,0]) & (np.array(labels_test).reshape(-1) < bet_pred_total[-2000:,0])])/ len( predictions_test[ (predictions_test < bet_pred_total[-2000:,0]) ] ))"""



def simulate_kelly_start_WL(bet_pred, dates, samps, real):

	odds_home = bet_pred[:,0]
	odds_away = bet_pred[:,1]

	# Step 1: Get the unique dates and their corresponding indices
	unique_dates = np.unique(dates)
	grouped_indices = {test_date: np.where(dates == test_date)[0] for test_date in unique_dates}

	net_evolution_test = np.zeros((100, len(unique_dates)))

	game_ratio_return = []
	game_kelly = []

	for kk in range(100):
			start_net = 1000
			net = 1000
			debt = 0
			net_evolution_test[kk,0] = start_net
			max_loss = 0
			nbets = 0
			nlosses=0
			nconsecutivelosses = 0
			nmaxconsecutivelosses = 0
			# Step 2: Shuffle the unique dates
			shuffled_dates = np.random.permutation(unique_dates)

			# Step 3: Concatenate indices for each date in the shuffled order
			indices = np.concatenate([grouped_indices[test_date] for test_date in shuffled_dates])
			
			current_date = dates[indices[0]]
			daily_revenue = 0
			date_count = 0
			daydebt = 0


			for aindex,ii in enumerate(indices):
				if dates[ii] != current_date:
					current_date = dates[ii]
					debt+= daydebt
					net+=daydebt
					net += daily_revenue
					date_count+=1
					net_evolution_test[kk,date_count] = net
					daily_revenue = 0
					daydebt = 0
				max_bet = net/10
				bet_pred_ii = bet_pred[ii,0]
				if np.isnan(bet_pred_ii):
					continue
				ratio_home = (np.abs(bet_pred[ii,0])/100)**np.sign(bet_pred[ii,0])
				ratio_away = (np.abs(bet_pred[ii,1])/100)**np.sign(bet_pred[ii,1])
				
				if np.median(samps[:, ii, 0]) > 0:
					ratio = ratio_home
					win_prob = len(samps[:, ii, 0][samps[:, ii, 0] > 0])/N_samples
					loss_prob = 1 - win_prob
					kelly_fraction = win_prob - loss_prob / ratio 
					game_kelly.append(kelly_fraction)
					if kelly_fraction > 1:
						#kelly_fraction = 1
						daydebt = max_bet*(kelly_fraction-1)
					if kelly_fraction > 0:
						if np.sign(real[ii]) == 1:
							daily_revenue+=max_bet*ratio_home*kelly_fraction
							game_ratio_return.append(1+0.1*ratio_home*kelly_fraction)
							nconsecutivelosses = 0
						else:
							daily_revenue-=max_bet*kelly_fraction
							game_ratio_return.append(1-0.1*kelly_fraction)
							nlosses +=1
							nconsecutivelosses+=1
						nbets+=1	
					
				elif np.median(samps[:, ii, 0]) < 0:
					ratio = ratio_away
					win_prob = len(samps[:, ii, 0][samps[:, ii, 0] < 0])/N_samples
					loss_prob = 1 - win_prob
					kelly_fraction = win_prob - loss_prob / ratio 
					game_kelly.append(kelly_fraction)
					if kelly_fraction > 1:
						#kelly_fraction = 1
						daydebt = max_bet*(kelly_fraction-1)
					if kelly_fraction > 0:
						if np.sign(real[ii]) == -1:
							daily_revenue+=max_bet*ratio_away*kelly_fraction
							game_ratio_return.append(1+0.1*ratio_home*kelly_fraction)
							nconsecutivelosses = 0
						else:
							daily_revenue-=max_bet*kelly_fraction
							game_ratio_return.append(1-0.1*kelly_fraction)
							nlosses +=1
							nconsecutivelosses+=1
						nbets+=1
						
				if nconsecutivelosses > nmaxconsecutivelosses:
					nmaxconsecutivelosses = nconsecutivelosses
				if net-start_net < max_loss:
					max_loss = net-start_net
			print(kk,net,max_loss, nbets, nlosses, nmaxconsecutivelosses,1-nlosses/max([nbets,0.01]), debt)





	plt.figure()

	plt.plot(np.arange(0,len(unique_dates),1), np.median(net_evolution_test, axis = 0))
	plt.fill_between(np.arange(0,len(unique_dates),1), np.percentile(net_evolution_test, 16,axis = 0), np.percentile(net_evolution_test, 84,axis = 0), alpha = 0.3)
	plt.fill_between(np.arange(0,len(unique_dates),1), np.percentile(net_evolution_test, 2.5,axis = 0), np.percentile(net_evolution_test, 97.5,axis = 0), alpha = 0.2)
	plt.yscale('log')
	plt.show()
					
			
	day_ratios = net_evolution_test[-1, 1:]/net_evolution_test[-1, :-1]

	plt.figure()
	plt.hist(day_ratios, bins = 30)
	plt.show()
		
		
		
	
	
simulate_kelly_start_WL(bet_pred[-test_size:], dates[-test_size:], samples_test, labels_test)

simulate_kelly_start_WL(bet_pred[train_indices], dates[train_indices], samples_train, labels_train)

simulate_kelly_start_WL(bet_pred[val_indices], dates[val_indices], samples_val, labels_val)
	
				
		
		
		
		
		
		
		
			
def simulate_kelly_start_OU_or_SPREAD(threshold, dates, samps, real): # WE ASSUME -110 odds for both sides


	# Step 1: Get the unique dates and their corresponding indices
	unique_dates = np.unique(dates)
	grouped_indices = {test_date: np.where(dates == test_date)[0] for test_date in unique_dates}

	net_evolution_test = np.zeros((100, len(unique_dates)))

	game_ratio_return = []
	game_kelly = []

	for kk in range(100):
			start_net = 1000
			net = 1000
			debt = 0
			net_evolution_test[kk,0] = start_net
			max_loss = 0
			nbets = 0
			nlosses=0
			nconsecutivelosses = 0
			nmaxconsecutivelosses = 0
			# Step 2: Shuffle the unique dates
			shuffled_dates = np.random.permutation(unique_dates)

			# Step 3: Concatenate indices for each date in the shuffled order
			indices = np.concatenate([grouped_indices[test_date] for test_date in shuffled_dates])
			
			current_date = dates[indices[0]]
			daily_revenue = 0
			date_count = 0
			daydebt = 0


			for aindex,ii in enumerate(indices):
				if dates[ii] != current_date:
					current_date = dates[ii]
					debt+= daydebt
					net+=daydebt
					net += daily_revenue
					date_count+=1
					net_evolution_test[kk,date_count] = net
					daily_revenue = 0
					daydebt = 0
				max_bet = net/10
				if np.isnan(threshold[ii]):
					continue
				ratio_home = 100/110
				ratio_away = 100/110
				
				if np.median(samps[:, ii, 0])  > threshold[ii]:
					ratio = ratio_home
					win_prob = len(samps[:, ii, 0][samps[:, ii, 0] > threshold[ii]])/N_samples
					loss_prob = 1 - win_prob
					kelly_fraction = win_prob - loss_prob / ratio 
					game_kelly.append(kelly_fraction)
					if kelly_fraction > 1:
						#kelly_fraction = 1
						daydebt = max_bet*(kelly_fraction-1)
					if kelly_fraction > 0:
						if real[ii] > threshold[ii]:
							daily_revenue+=max_bet*ratio_home*kelly_fraction
							game_ratio_return.append(1+0.1*ratio_home*kelly_fraction)
							nconsecutivelosses = 0
						else:
							daily_revenue-=max_bet*kelly_fraction
							game_ratio_return.append(1-0.1*kelly_fraction)
							nlosses +=1
							nconsecutivelosses+=1
						nbets+=1	
					
				elif np.median(samps[:, ii, 0]) < threshold[ii]:
					ratio = ratio_away
					win_prob = len(samps[:, ii, 0][samps[:, ii, 0] < threshold[ii]])/N_samples
					loss_prob = 1 - win_prob
					kelly_fraction = win_prob - loss_prob / ratio 
					game_kelly.append(kelly_fraction)
					if kelly_fraction > 1:
						#kelly_fraction = 1
						daydebt = max_bet*(kelly_fraction-1)
					if kelly_fraction > 0:
						if real[ii] < threshold[ii]:
							daily_revenue+=max_bet*ratio_away*kelly_fraction
							game_ratio_return.append(1+0.1*ratio_home*kelly_fraction)
							nconsecutivelosses = 0
						else:
							daily_revenue-=max_bet*kelly_fraction
							game_ratio_return.append(1-0.1*kelly_fraction)
							nlosses +=1
							nconsecutivelosses+=1
						nbets+=1
						
				if nconsecutivelosses > nmaxconsecutivelosses:
					nmaxconsecutivelosses = nconsecutivelosses
				if net-start_net < max_loss:
					max_loss = net-start_net
			print(kk,net,max_loss, nbets, nlosses, nmaxconsecutivelosses,1-nlosses/max([nbets,0.01]), debt)





	plt.figure()

	plt.plot(np.arange(0,len(unique_dates),1), np.median(net_evolution_test, axis = 0))
	plt.fill_between(np.arange(0,len(unique_dates),1), np.percentile(net_evolution_test, 16,axis = 0), np.percentile(net_evolution_test, 84,axis = 0), alpha = 0.3)
	plt.fill_between(np.arange(0,len(unique_dates),1), np.percentile(net_evolution_test, 2.5,axis = 0), np.percentile(net_evolution_test, 97.5,axis = 0), alpha = 0.2)
	plt.yscale('log')
	plt.show()
					
			
	day_ratios = net_evolution_test[-1, 1:]/net_evolution_test[-1, :-1]

	plt.figure()
	plt.hist(day_ratios, bins = 30)
	plt.show()
		
		
			
	
simulate_kelly_start_OU_or_SPREAD(bet_pred_total[-test_size:], dates[-test_size:], samples_test, labels_test)	
	
	
	
	
	
	
	
	
		
				
				
				
				
				
				
				
				
				







def plot_TARP_element(tarp_obj, c, label):
    plt.plot(tarp_obj[1], np.median(tarp_obj[0], axis = 0), color = c, label = label)
    plt.fill_between(tarp_obj[1], np.percentile(tarp_obj[0], 2.5, axis = 0), np.percentile(tarp_obj[0], 97.5, axis = 0), color = c, alpha = 0.5)
    return


tarp_test = tarp.get_tarp_coverage(samples_test, labels_test.reshape((-1,1)), bootstrap = True)

tarp_train = tarp.get_tarp_coverage(samples_train, labels_train.reshape((-1,1)), bootstrap = True)

tarp_val = tarp.get_tarp_coverage(samples_val, labels_val.reshape((-1,1)), bootstrap = True)

plt.figure(figsize = (5,5))
plt.grid()
# now plot both limits against eachother
plt.plot([0,1], [0,1], 'k--')
plot_TARP_element(tarp_train, 'gray', 'train')
plot_TARP_element(tarp_val, list(mpl.rcParams['axes.prop_cycle'])[0]['color'], 'val')
plot_TARP_element(tarp_test, list(mpl.rcParams['axes.prop_cycle'])[1]['color'], 'test')
plt.title('Validation')
plt.xlabel('Credibility Level')
plt.ylabel('Coverage')
plt.legend()
plt.show()
















percentiles = np.percentile(samples_train[:,:,0], np.arange(0.5, 100.5), axis=0).T



 
input_dim = 100 + 2  # Nsamples + odds_over + odds_under
max_bet = 100  # Example max bet value
 
mask = ~np.any(np.isnan(bet_pred[train_indices]), axis = 1) & ~np.isnan(labels_train.flatten())

# Convert data to PyTorch tensors and split into training and validation sets
X_train = np.hstack((percentiles[mask,:], bet_pred[train_indices][mask,:]))  # Concatenate samples and odds
y_train = np.hstack((bet_pred[train_indices][mask,:], labels[train_indices][mask].reshape(-1, 1)))  # Combine odds/thresholds and actual scores




percentiles = np.percentile(samples_val[:,:,0], np.arange(0.5, 100.5), axis=0).T


 
input_dim = 100 + 2  # Nsamples + odds_over + odds_under
max_bet = 100  # Example max bet value
 
mask = ~np.any(np.isnan(bet_pred[val_indices]), axis = 1) & ~np.isnan(labels_val.flatten())

# Convert data to PyTorch tensors and split into valing and validation sets
X_val = np.hstack((percentiles[mask,:], bet_pred[val_indices][mask,:]))  # Concatenate samples and odds
y_val = np.hstack((bet_pred[val_indices][mask,:], labels[val_indices][mask].reshape(-1, 1)))  # Combine odds/thresholds and actual scores










# Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)



percentiles = np.percentile(samples_test[:,:,0], np.arange(0.5, 100.5), axis=0).T

mask = ~np.any(np.isnan(bet_pred_test), axis = 1) & ~np.isnan(labels_test.flatten())


# Convert data to PyTorch tensors and split into valing and validation sets
X_test = np.hstack((percentiles[mask,:], bet_pred_test[mask]))  # Concatenate samples and odds
y_test = np.hstack((bet_pred_test[mask,:], labels_test[mask].reshape(-1, 1)))  # Combine odds/thresholds and actual scores
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


betmodel = models.Bet_Strat_spread(102, [32])

betmodel.train_model(X_train, y_train, validation_data=(X_val, y_val), print_every = 1, lr_patience=5, stop_patience=10, lr_factor=0.01)

betmodel.test_model(X_test, y_test)


torch.save({
    'model_state_dict': betmodel.state_dict(),
}, 'bet_model_checkpoint.pth')



odds_home = bet_pred_test[:,0]
odds_away = bet_pred_test[:,1]


net_evolution_test_MLBetStrat = np.zeros((100, len(unique_dates)))

fracs = []

for kk in range(100):
		start_net = 1000
		net = 1000
		net_evolution_test_MLBetStrat[kk,0] = start_net
		max_loss = 0
		nbets = 0
		nlosses=0
		nconsecutivelosses = 0
		nmaxconsecutivelosses = 0
		# Step 2: Shuffle the unique dates
		shuffled_dates = np.random.permutation(unique_dates)

		# Step 3: Concatenate indices for each date in the shuffled order
		indices = np.concatenate([grouped_indices[test_date] for test_date in shuffled_dates])
		
		current_date = test_dates[indices[0]]
		daily_revenue = 0
		date_count = 0
		


		for aindex,ii in enumerate(indices):
			res = ''
			if test_dates[ii] != current_date:
				current_date = test_dates[ii]
				net += daily_revenue
				date_count+=1
				net_evolution_test_MLBetStrat[kk,date_count] = net
				daily_revenue = 0
			max_bet = net/10
			bet_pred_ii = bet_pred_test[ii,0]
			if np.isnan(bet_pred_ii):
				continue
			ratio_home = (np.abs(bet_pred_test[ii,0])/100)**np.sign(bet_pred_test[ii,0])
			ratio_away = (np.abs(bet_pred_test[ii,1])/100)**np.sign(bet_pred_test[ii,1])
			
			betfracs = betmodel( torch.Tensor( np.concatenate( (percentiles[ii], bet_pred_test[ii]) ) ) ).detach().numpy()
			
			if (betfracs[0] > betfracs[1]) and betfracs[0] > 10**-2:
				betfracbet = betfracs[0]
				if np.sign(labels_test[ii]) == 1:
					fracs.append( ratio_home*betfracs[0] )
					daily_revenue+=max_bet*ratio_home*betfracs[0]
					nconsecutivelosses = 0
					res = 'win'
				else:
					daily_revenue-=max_bet
					nlosses +=1
					nconsecutivelosses+=1
					fracs.append( -betfracs[0] )
					res = 'loss'
				nbets+=1	
				
			elif (betfracs[1] > betfracs[0]) and betfracs[1] > 10**-2:
				betfracbet = betfracs[1]
				ratio = ratio_away
				if np.sign(labels_test[ii]) == -1:
					fracs.append( ratio_home*betfracs[1] )
					daily_revenue+=max_bet*ratio_away*betfracs[1]
					nconsecutivelosses = 0
					res = 'win'
				else:
					fracs.append( -betfracs[1] )
					daily_revenue-=max_bet
					nlosses +=1
					nconsecutivelosses+=1
					res = 'loss'
				nbets+=1
					
			if nconsecutivelosses > nmaxconsecutivelosses:
				nmaxconsecutivelosses = nconsecutivelosses
			if net-start_net < max_loss:
				max_loss = net-start_net
		print(kk,net,max_loss, nbets, nlosses, nmaxconsecutivelosses,1-nlosses/nbets)





plt.figure()

plt.plot(np.arange(0,len(unique_dates),1), np.median(net_evolution_test, axis = 0), color = 'orange', label = 'Kelly Strat')
plt.fill_between(np.arange(0,len(unique_dates),1), np.percentile(net_evolution_test, 16,axis = 0), np.percentile(net_evolution_test, 84,axis = 0), alpha = 0.3, color = 'orange')
plt.fill_between(np.arange(0,len(unique_dates),1), np.percentile(net_evolution_test, 2.5,axis = 0), np.percentile(net_evolution_test, 97.5,axis = 0), alpha = 0.2, color = 'orange')

"""
plt.plot(np.arange(0,len(unique_dates),1), np.median(net_evolution_test_greedy, axis = 0), color = 'green', label = 'Greedy Kelly Strat')
plt.fill_between(np.arange(0,len(unique_dates),1), np.percentile(net_evolution_test_greedy, 16,axis = 0), np.percentile(net_evolution_test_greedy, 84,axis = 0), alpha = 0.3, color = 'green')
plt.fill_between(np.arange(0,len(unique_dates),1), np.percentile(net_evolution_test_greedy, 2.5,axis = 0), np.percentile(net_evolution_test_greedy, 97.5,axis = 0), alpha = 0.2, color = 'green')
"""

plt.plot(np.arange(0,len(unique_dates),1), np.median(net_evolution_test_MLBetStrat, axis = 0), color = 'blue', label = 'ML Strat')
plt.fill_between(np.arange(0,len(unique_dates),1), np.percentile(net_evolution_test_MLBetStrat, 16,axis = 0), np.percentile(net_evolution_test_MLBetStrat, 84,axis = 0), alpha = 0.3, color = 'blue')
plt.fill_between(np.arange(0,len(unique_dates),1), np.percentile(net_evolution_test_MLBetStrat, 2.5,axis = 0), np.percentile(net_evolution_test_MLBetStrat, 97.5,axis = 0), alpha = 0.2, color = 'blue')
plt.yscale('log')
plt.legend()
plt.show()
				
				
			
			
day_ratios = net_evolution_test_MLBetStrat[:, 1:]/net_evolution_test_MLBetStrat[:, :-1]

print(np.median(day_ratios), np.mean(day_ratios))

plt.figure()
plt.hist(day_ratios.flatten(), bins = 30)
plt.show()				
		
				
				
	
	
	
	
	
np.save('net_evo_Kelly.npy', net_evolution_test)
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
import torch.optim as optim

def train_model(model, train_loader, val_loader, test_loader, num_epochs=10, learning_rate=0.001, device='cuda'):
    # Define the loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        train_loss = 0.0

        for batch in train_loader:
            x, y = batch  # Assume each batch contains inputs 'x' and labels 'y'
            x = {key: value.to(device) for key, value in x.items()}
            y = y.to(device)


            # Forward pass
            predictions = model(x)
            loss = criterion(predictions, y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()* y.size(0)  # Accumulate loss (scaled by batch size)

        train_loss /= len(train_loader.dataset)  # Average loss over all training samples

        # Validation phase
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x = {key: value.to(device) for key, value in x.items()}
                y = y.to(device)

                predictions = model(x)
                loss = criterion(predictions, y)
                val_loss += loss.item()* y.size(0)

        val_loss /= len(val_loader.dataset)

        

        # Testing phase
        model.eval()  # Set the model to evaluation mode
        test_loss = 0.0

        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x = {key: value.to(device) for key, value in x.items()}
                y = y.to(device)

                predictions = model(x)
                loss = criterion(predictions, y)
                test_loss += loss.item()* y.size(0)

        test_loss /= len(test_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}")
	

	
	
train_model(model, train_loader, val_loader, test_loader, num_epochs=50, learning_rate=0.001, device='cpu')






class LSTM_3designs(nn.Module):
    def __init__(self, input_size=7, hidden_size=32, output_size=2, input_size2=7, hidden_size2=32, num_layers=1, num_layers2=1, input_size3=7, hidden_size3=32, num_layers3=1, glob_var_size = 10):
        super(LSTM_3designs, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.rnn2 = nn.LSTM(input_size2, hidden_size2, num_layers2, batch_first=True)
        
        self.rnn3 = nn.LSTM(input_size3, hidden_size3, num_layers3, batch_first=True)

        self.fc = nn.Linear(2 * hidden_size + 1 * hidden_size2 + 1 * glob_var_size + 2 * dataset[0][0]['players_home'].shape[0] * hidden_size3, output_size)

    def forward(self, x):
        x1 = x['current_game']
        x2 = x['games_home']
        x3 = x['games_away']
        x4 = x['games_vs']
        x5 = x['players_home']
        x6 = x['players_away']
        # x is a list of 20 inputs for the LSTM
        hidden_states = []  # To collect final hidden states from each LSTM

        rnn_out, _ = self.rnn(x2)  # Process each input through the LSTM
        hidden_states.append(rnn_out[:, -1, :])  # Collect the last hidden state of the sequence

        rnn_out, _ = self.rnn(x3)  # Process each input through the LSTM
        hidden_states.append(rnn_out[:, -1, :])  # Collect the last hidden state of the sequence
  
        rnn_out2, _ = self.rnn2(x4)
        hidden_states.append(rnn_out2[:, -1, :])  # Collect the last hidden state of the sequence
        
        for jj in range(x5.shape[1]):
            rnn_out3, _ = self.rnn3(x5[:,jj,:,:])
            hidden_states.append(rnn_out3[:, -1, :]) 
            
        for jj in range(x6.shape[1]):
            rnn_out3, _ = self.rnn3(x6[:,jj,:,:])
            hidden_states.append(rnn_out3[:, -1, :])
        
        
        

            
        # Concatenate all hidden states along the feature dimension
        concatenated_hidden = torch.cat(hidden_states, dim=1)
        concatenated_hidden2 = torch.cat([concatenated_hidden, x1], dim=1)

        # Fully connected layer to output predictions
        out = self.fc(concatenated_hidden2)

        return out








model = LSTM_3designs(input_size=dataset[0][0]['games_home'].shape[-1], hidden_size=32, output_size=1, input_size2=dataset[0][0]['games_vs'].shape[-1], hidden_size2=32, num_layers=1, num_layers2=1, input_size3=dataset[0][0]['players_home'].shape[-1], hidden_size3 = 32, num_layers3=1,  glob_var_size = dataset[0][0]['current_game'].shape[-1])






preds = []
labs = []
for ii in tqdm(range(len(train_dataset))):
    preds.append(float(model(train_dataset[ii][0].unsqueeze(0))))
    labs.append(float(train_dataset[ii][1]))
    
plt.plot(labs, preds, 'o', label = 'Train')



preds_val = []
labs_val = []
for ii in tqdm(range(len(val_dataset))):
    preds_val.append(float(model(val_dataset[ii][0].unsqueeze(0))))
    labs_val.append(float(val_dataset[ii][1]))
    
plt.plot(labs_val, preds_val, 'o', label = 'Validation')




preds_test = []
labs_test = []
for ii in tqdm(range(len(test_dataset))):
    preds_test.append(float(model(test_dataset[ii][0].unsqueeze(0))))
    labs_test.append(float(test_dataset[ii][1]))
    
plt.plot(labs_test, preds_test, 'o', label = 'Test')

plt.legend()
plt.show()




print(overunderaccuracy(np.array(preds), bet_pred_total[train_indices], np.array(labs)))
print(overunderaccuracy(np.array(preds_val), bet_pred_total[val_indices], np.array(labs_val)))
print(overunderaccuracy(np.array(preds_test), bet_pred_total[-test_size:], np.array(labs_test)))



def winlossaccuracy(pred, real):
	return np.sum( np.array(np.sign(pred) == np.sign(real), dtype = int) )/len(pred)
	
print(winlossaccuracy(preds, labs))
print(winlossaccuracy(preds_val, labs_val))
print(winlossaccuracy(preds_test, labs_test))

