"""import reader
import models
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

r = reader.reader('../Data/nba_games.csv', latest_matches=20, latest_vs_matches = 20)

inputs, labels = r.make_dataset()

mask = labels>0

inputs, labels = torch.Tensor(inputs[mask,:]), torch.Tensor(labels[mask].reshape(-1,1))

X_train, X_val, y_train, y_val = train_test_split(inputs, labels, test_size=0.2, random_state=42)

model = models.MLP(inputs.shape[1], [256]*8, 1)

model.train_model(X_train, y_train, validation_data=(X_val, y_val), learning_rate=0.001, num_epochs=1000, print_every=1, lr_patience=20, stop_patience=30, lr_factor=0.3)


# Generate predictions on the training data
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient calculation for evaluation
    predictions = model(X_train)
    predictions_val = model(X_val)

# Plot predicted values vs. true labels
plt.figure(figsize=(8, 6))
plt.scatter(y_train.numpy(), predictions.numpy(), alpha=0.7, color='b')
plt.scatter(y_val.numpy(), predictions_val.numpy(), alpha=0.7, color='g')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)  # Diagonal line for reference
plt.xlabel('True Labels (Sum of Inputs)')
plt.ylabel('Predicted Values')
plt.title('Predicted Values vs. True Labels')
plt.show()
"""






import reader
import models
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch.nn as nn

import ili
from ili.inference import InferenceRunner
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset
from ili.dataloaders import TorchLoader


import pandas as pd



r = reader.reader('../Data/Complete_data/merged_nba_games_updating.csv', latest_matches=20, latest_vs_matches = 20, latest_other_matches = 20)


inputs, labels, bet_pred, dates = r.make_dataset()














# Example input arrays (replace with your actual data)
home_moneylines = bet_pred[:,0]  # Example moneylines for the home team
away_moneylines = bet_pred[:,1]  # Example moneylines for the away team
results = labels  # Example results (home points - away points)

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
    'Prediction': predictions,
    'Actual': actuals
})
data['Correct'] = data['Prediction'] == data['Actual']
data = data.dropna(subset=['Prediction', 'Actual']).reset_index(drop = True)  # Drop rows where predictions or actuals are None

# Calculate accuracy over time
data['Cumulative Accuracy'] = data['Correct'].cumsum() / (data.index + 1)


# Plot cumulative accuracy evolution
data['Cumulative Accuracy'].plot(title='Cumulative Accuracy Over Time', figsize=(10, 6), color='blue', label='Cumulative Accuracy')

# Calculate centered rolling mean accuracy with a window of size 50
data['Centered Accuracy'] = data['Correct'].rolling(window=500, min_periods=1, center=True).mean()

# Plot centered rolling mean accuracy
data['Centered Accuracy'].plot(color='green', label='Centered Mean Accuracy (Window=500)')


# Add labels and legend
plt.xlabel('Number of Games')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()


# Print cumulative accuracy by game number
print(data[['Prediction', 'Actual', 'Correct', 'Cumulative Accuracy']])











mask = np.all(~np.isnan(inputs), axis = 1)

inputs, labels = torch.Tensor(inputs[mask,:]), torch.Tensor(labels[mask].reshape(-1,1))

bet_pred = bet_pred[mask]







inputs_current_game = inputs[:, :2 + r.N_current_game_fields]

inputs_prev_home_games = inputs[:, 2 + r.N_current_game_fields : 2 + r.N_current_game_fields + (3 + r.N_other_game_fields)*r.latest_matches].reshape((-1, (3 + r.N_other_game_fields), r.latest_matches)).transpose(1,2)

inputs_prev_away_games = inputs[:, 2 + r.N_current_game_fields + (3 + r.N_other_game_fields)*r.latest_matches : 2 + r.N_current_game_fields + 2*(3 + r.N_other_game_fields)*r.latest_matches].reshape((-1, (3 + r.N_other_game_fields), r.latest_matches)).transpose(1,2)

inputs_prev_vs_games = inputs[:, 2 + r.N_current_game_fields + 2*(3 + r.N_other_game_fields)*r.latest_matches : 2 + r.N_current_game_fields + 2*(3 + r.N_other_game_fields)*r.latest_matches + (2 + r.N_other_game_fields)*r.latest_vs_matches ].reshape((-1, (2 + r.N_other_game_fields), r.latest_vs_matches)).transpose(1,2)

inputs_prev_other_games = inputs[:, 2 + r.N_current_game_fields + 2*(3 + r.N_other_game_fields)*r.latest_matches + (2 + r.N_other_game_fields)*r.latest_vs_matches :  2 + r.N_current_game_fields + 2*(3 + r.N_other_game_fields)*r.latest_matches + (2 + r.N_other_game_fields)*r.latest_vs_matches + (3 + r.N_other_game_fields)*r.latest_other_matches ].reshape((-1, (3 + r.N_other_game_fields), r.latest_other_matches)).transpose(1,2)







class LSTM_3designs(nn.Module):
    def __init__(self, input_size=7, hidden_size=32, output_size=2, input_size2=7, hidden_size2=32, input_size3=7, hidden_size3=32, num_layers=1, num_layers2=1, num_layers3=1, glob_var_size = 10):
        super(LSTM_3designs, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.rnn2 = nn.LSTM(input_size2, hidden_size2, num_layers2, batch_first=True)
        
        self.rnn3 = nn.LSTM(input_size3, hidden_size3, num_layers3, batch_first=True)

        self.fc = nn.Linear(2 * hidden_size + 1 * hidden_size2 + 1 *  hidden_size3 + glob_var_size, output_size)


    def forward(self, x):
        if len(x.shape) < 2:
            x = x.reshape(1,-1)
        x1 = x[:, :2 + r.N_current_game_fields]
        x2 = x[:, 2 + r.N_current_game_fields : 2 + r.N_current_game_fields + (3 + r.N_other_game_fields)*r.latest_matches].reshape((-1, (3 + r.N_other_game_fields), r.latest_matches)).transpose(1,2)
        x3 = x[:, 2 + r.N_current_game_fields + (3 + r.N_other_game_fields)*r.latest_matches : 2 + r.N_current_game_fields + 2*(3 + r.N_other_game_fields)*r.latest_matches].reshape((-1, (3 + r.N_other_game_fields), r.latest_matches)).transpose(1,2)
        x4 = x[:, 2 + r.N_current_game_fields + 2*(3 + r.N_other_game_fields)*r.latest_matches : 2 + r.N_current_game_fields + 2*(3 + r.N_other_game_fields)*r.latest_matches + (2 + r.N_other_game_fields)*r.latest_vs_matches ].reshape((-1, (2 + r.N_other_game_fields), r.latest_vs_matches)).transpose(1,2)
        # x is a list of 20 inputs for the LSTM
        x5 = x[:,  2 + r.N_current_game_fields + 2*(3 + r.N_other_game_fields)*r.latest_matches + (2 + r.N_other_game_fields)*r.latest_vs_matches :  2 + r.N_current_game_fields + 2*(3 + r.N_other_game_fields)*r.latest_matches + (2 + r.N_other_game_fields)*r.latest_vs_matches + (3 + r.N_other_game_fields)*r.latest_other_matches ].reshape((-1, (3 + r.N_other_game_fields), r.latest_other_matches)).transpose(1,2)
        # x is a list of 20 inputs for the LSTM
        hidden_states = []  # To collect final hidden states from each LSTM


        rnn_out, _ = self.rnn(x2)  # Process each input through the LSTM
        if len(rnn_out.shape) > 2:
            hidden_states.append(rnn_out[:, -1, :])  # Collect the last hidden state of the sequence
        else:
            hidden_states.append(rnn_out[:, :])
            
        rnn_out, _ = self.rnn(x3)  # Process each input through the LSTM
        if len(rnn_out.shape) > 2:
            hidden_states.append(rnn_out[:, -1, :])  # Collect the last hidden state of the sequence
        else:
            hidden_states.append(rnn_out[:, :])
                
        rnn_out2, _ = self.rnn2(x4)

        if len(rnn_out2.shape) > 2:
            hidden_states.append(rnn_out2[:, -1, :])  # Collect the last hidden state of the sequence
        else:
            hidden_states.append(rnn_out2[:, :])
            
        rnn_out3, _ = self.rnn3(x5)

        if len(rnn_out3.shape) > 2:
            hidden_states.append(rnn_out3[:, -1, :])  # Collect the last hidden state of the sequence
        else:
            hidden_states.append(rnn_out3[:, :])



        # Concatenate all hidden states along the feature dimension
        concatenated_hidden = torch.cat(hidden_states, dim=1)
        concatenated_hidden2 = torch.cat([concatenated_hidden, x1], dim=1)

        # Fully connected layer to output predictions
        out = self.fc(concatenated_hidden2)
        return out

model = LSTM_3designs(input_size=(3 + r.N_other_game_fields), hidden_size=32, output_size=1, input_size2=(2 + r.N_other_game_fields), hidden_size2=32, input_size3=(3 + r.N_other_game_fields), hidden_size3=32 , num_layers=3, num_layers2=3, num_layers3=3, glob_var_size = 10)


batch_size = 64





inputs_test = inputs[-2000:, :]
labels_test = labels[-2000:, :]
bet_pred_test = bet_pred[-2000:, :]


X_train, X_val, y_train, y_val, indices_train, indices_val = train_test_split(inputs[:-2000], labels[:-2000], np.arange(len(labels[:-2000])), test_size=0.2, random_state=60)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
loader = TorchLoader(train_loader, val_loader)




accelerator = Accelerator()
device = accelerator.device


# define training arguments
train_args = {
    'training_batch_size': batch_size,
    'learning_rate': 3e-5,
    'stop_after_epochs': 5,
    'max_epochs': 500
}

nets = [ili.utils.load_nde_lampe(model='maf', hidden_features=50, num_transforms=5, embedding_net=model, x_normalize=False, device=device)]

prior = ili.utils.Uniform(low=[-100], high=[100], device=device)


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


loader = TorchLoader(train_loader, val_loader)

posterior_ensemble, summaries = runner(loader=loader)


samples_train = np.zeros((N_samples,len(train_dataset),1))

for ii in range(len(train_dataset)):
     print(ii,len(train_dataset))
     samples_train[:,ii,:] = posterior_ensemble.sample((N_samples,), torch.Tensor(train_dataset[ii][0]).to(device))
     

samples_val = np.zeros((N_samples,len(val_dataset),1))
truths_val = np.zeros((len(val_dataset),))

for ii in range(len(val_dataset)):
     print(ii,len(val_dataset))
     samples_val[:,ii,:] = posterior_ensemble.sample((N_samples,), torch.Tensor(val_dataset[ii][0]).to(device))
     truths_val[ii] = float(val_dataset[ii][1])
  
  
samples_test = np.zeros((N_samples,len(labels_test),1))

for ii in range(len(labels_test)):
     print(ii,len(labels_test))
     samples_test[:,ii,:] = posterior_ensemble.sample((N_samples,), torch.Tensor(inputs_test[ii]).to(device))
     
  
        
predictions = np.median(samples_train, axis = 0).reshape((-1))
predictions_val = np.median(samples_val, axis = 0).reshape((-1))
predictions_test = np.median(samples_test, axis = 0).reshape((-1))

print(np.sum( np.array(np.sign(predictions) == np.sign(y_train.reshape(-1)), dtype = int) )/len(predictions))
print(np.sum( np.array(np.sign(predictions_val) == np.sign(y_val.reshape(-1)), dtype = int) )/len(predictions_val))
print(np.sum( np.array(np.sign(predictions_test) == np.sign(labels_test.reshape(-1)), dtype = int) )/len(predictions_test))



odds_home = bet_pred_test[:,0]
odds_away = bet_pred_test[:,1]


net_evolution_test = np.zeros((100, len(predictions_test)))


for kk in range(100):
		start_net = 1000
		net = 1000
		max_loss = 0
		nbets = 0
		nlosses=0
		nconsecutivelosses = 0
		nmaxconsecutivelosses = 0
		indeces_orig = np.arange(0,len(predictions_test),1)
		indeces =np.random.choice(indeces_orig, size = 2000,replace = False)

		for aindex,ii in enumerate(indeces):
			max_bet = net/10
			bet_pred_ii = bet_pred_test[ii,0]
			if np.isnan(bet_pred_ii):
				net_evolution_test[kk, aindex] = net
				continue
			ratio_home = (np.abs(bet_pred_test[ii,0])/100)**np.sign(bet_pred_test[ii,0])
			ratio_away = (np.abs(bet_pred_test[ii,1])/100)**np.sign(bet_pred_test[ii,1])
			
			if np.median(samples_test[:, ii, 0]) > 0:
				ratio = ratio_home
				win_prob = len(samples_test[:, ii, 0][samples_test[:, ii, 0] > 0])/N_samples
				loss_prob = 1 - win_prob
				kelly_fraction = win_prob - loss_prob / ratio 
				if kelly_fraction > 1:
					kelly_fraction = 0.5
				if kelly_fraction > 0:
					if np.sign(labels_test[ii]) == 1:
						net+=max_bet*ratio_home*kelly_fraction
						nconsecutivelosses = 0
					else:
						net-=max_bet*kelly_fraction
						nlosses +=1
						nconsecutivelosses+=1
					nbets+=1	
				
			elif np.median(samples_test[:, ii, 0]) < 0:
				ratio = ratio_away
				win_prob = len(samples_test[:, ii, 0][samples_test[:, ii, 0] > 0])/N_samples
				loss_prob = 1 - win_prob
				kelly_fraction = win_prob - loss_prob / ratio 
				if kelly_fraction > 1:
					kelly_fraction = 0.5
				if kelly_fraction > 0:
					if np.sign(labels_test[ii]) == 1:
						net+=max_bet*ratio_home*kelly_fraction
						nconsecutivelosses = 0
					else:
						net-=max_bet*kelly_fraction
						nlosses +=1
						nconsecutivelosses+=1
					nbets+=1
			if nconsecutivelosses > nmaxconsecutivelosses:
				nmaxconsecutivelosses = nconsecutivelosses
			if net-start_net < max_loss:
				max_loss = net-start_net
			net_evolution_test[kk , aindex] = net
		print(kk,net,max_loss, nbets, nlosses, nmaxconsecutivelosses,1-nlosses/nbets)





plt.figure()

plt.plot(np.arange(0,len(predictions_test),1), np.median(net_evolution_test, axis = 0))
plt.fill_between(np.arange(0,len(predictions_test),1), np.percentile(net_evolution_test, 16,axis = 0), np.percentile(net_evolution_test, 84,axis = 0), alpha = 0.3)
plt.fill_between(np.arange(0,len(predictions_test),1), np.percentile(net_evolution_test, 2.5,axis = 0), np.percentile(net_evolution_test, 97.5,axis = 0), alpha = 0.2)
plt.show()
				
		
	
		
from ili.validation.metrics import PosteriorCoverage, PlotSinglePosterior

# Uncertainties 
metric1 = PosteriorCoverage(
     num_samples=1000, sample_method='direct',
     labels=dimension_labels,
     plot_list = ["coverage", "histogram", "predictions", "tarp"],
     out_dir='./metrics_train'
 )
 

 
fig = metric1(
     posterior=posterior_ensemble, # NeuralPosteriorEnsemble instance from sbi package
     x=X_train, theta=y_train
 )
 
metric1 = PosteriorCoverage(
     num_samples=1000, sample_method='direct',
     labels=dimension_labels,
     plot_list = ["coverage", "histogram", "predictions", "tarp"],
     out_dir='./metrics_test'
 )
 
fig = metric1(
     posterior=posterior_ensemble, # NeuralPosteriorEnsemble instance from sbi package
     x=inputs_test, theta=labels_test
 )


metric1 = PosteriorCoverage(
     num_samples=1000, sample_method='direct',
     labels=dimension_labels,
     plot_list = ["coverage", "histogram", "predictions", "tarp"],
     out_dir='./metrics_val'
 )

 
fig = metric1(
     posterior=posterior_ensemble, # NeuralPosteriorEnsemble instance from sbi package
     x=X_val, theta=y_val
 )
 








 
 
samples_train_temp = samples_train.copy()


percentiles = np.zeros( (len(indices_train), 100) )
for ii in range(100):
    percentiles[:,ii] = np.percentile(samples_train, ii+0.5, axis = 0).reshape((-1))

 
input_dim = 100 + 2  # Nsamples + odds_over + odds_under
max_bet = 100  # Example max bet value
 
mask = ~np.any(np.isnan(bet_pred[indices_train]), axis = 1) & ~np.isnan(labels[indices_train].detach().numpy().flatten())

# Convert data to PyTorch tensors and split into training and validation sets
X_train = np.hstack((percentiles[mask,:], bet_pred[indices_train][mask,:]))  # Concatenate samples and odds
y_train = np.hstack((bet_pred[indices_train][mask,:], labels[indices_train][mask].reshape(-1, 1)))  # Combine odds/thresholds and actual scores







samples_val_temp = samples_val.copy()


percentiles = np.zeros( (len(indices_val), 100) )
for ii in range(100):
    percentiles[:,ii] = np.percentile(samples_val, ii+0.5, axis = 0).reshape((-1))

 
input_dim = 100 + 2  # Nsamples + odds_over + odds_under
max_bet = 100  # Example max bet value
 
mask = ~np.any(np.isnan(bet_pred[indices_val]), axis = 1) & ~np.isnan(labels[indices_val].detach().numpy().flatten())

# Convert data to PyTorch tensors and split into valing and validation sets
X_val = np.hstack((percentiles[mask,:], bet_pred[indices_val][mask,:]))  # Concatenate samples and odds
y_val = np.hstack((bet_pred[indices_val][mask,:], labels[indices_val][mask].reshape(-1, 1)))  # Combine odds/thresholds and actual scores










# Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

betmodel = models.Bet_Strat_spread(102, [32])

betmodel.train_model(X_train, y_train, validation_data=(X_val, y_val))


torch.save({
    'model_state_dict': betmodel.state_dict(),
}, 'bet_model_checkpoint_3channels.pth')


percentiles = np.zeros( (len(inputs_test), 100) )
for ii in range(100):
    percentiles[:,ii] = np.percentile(samples_test, ii+0.5, axis = 0).reshape((-1))




odds_home = bet_pred_test[:,0]
odds_away = bet_pred_test[:,1]


net_evolution_test = np.zeros((100, len(predictions_test)))

fracs = []

for kk in range(100):
		start_net = 1000
		net = 1000
		max_loss = 0
		nbets = 0
		nlosses=0
		nconsecutivelosses = 0
		nmaxconsecutivelosses = 0
		indeces_orig = np.arange(0,len(predictions_test),1)
		indeces =np.random.choice(indeces_orig, size = 1000,replace = False)

		for aindex,ii in enumerate(indeces):
			max_bet = net/10
			bet_pred_ii = bet_pred_test[ii,0]
			if np.isnan(bet_pred_ii):
				net_evolution_test[kk, aindex] = net
				continue
			ratio_home = (np.abs(bet_pred_test[ii,0])/100)**np.sign(bet_pred_test[ii,0])
			ratio_away = (np.abs(bet_pred_test[ii,1])/100)**np.sign(bet_pred_test[ii,1])
			
			betfracs = betmodel(torch.Tensor( np.concatenate( (percentiles[ii,:],  bet_pred_test[ii])) )).detach().numpy()
			
			if (betfracs[0] > betfracs[1]):
				ratio = ratio_home

				if np.sign(labels_test[ii]) == 1:
					net+=max_bet*ratio_home*betfracs[0]
					nconsecutivelosses = 0
				else:
					net-=max_bet*betfracs[0]
					nlosses +=1
					nconsecutivelosses+=1
				nbets+=1	
			
			if (betfracs[0] > betfracs[1]):
				ratio = ratio_away

				if np.sign(labels_test[ii]) == 1:
					net+=max_bet*ratio_home*betfracs[1]
					nconsecutivelosses = 0
				else:
					net-=max_bet*betfracs[1]
					nlosses +=1
					nconsecutivelosses+=1
				nbets+=1
				
			
					
			if nconsecutivelosses > nmaxconsecutivelosses:
				nmaxconsecutivelosses = nconsecutivelosses
			if net-start_net < max_loss:
				max_loss = net-start_net
			net_evolution_test[kk , aindex] = net
		print(kk,net,max_loss, nbets, nlosses, nmaxconsecutivelosses,1-nlosses/nbets)





plt.figure()

plt.plot(np.arange(0,len(predictions_test),1), np.median(net_evolution_test, axis = 0))
plt.fill_between(np.arange(0,len(predictions_test),1), np.percentile(net_evolution_test, 16,axis = 0), np.percentile(net_evolution_test, 84,axis = 0), alpha = 0.3)
plt.fill_between(np.arange(0,len(predictions_test),1), np.percentile(net_evolution_test, 2.5,axis = 0), np.percentile(net_evolution_test, 97.5,axis = 0), alpha = 0.2)
plt.show()
				
				
				
				
				
			
				
				
				
				
				
