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
import numpy as np

batch_size = 64
N_samples = 1000

r = reader.reader('../Data/Complete_data/merged_nba_games_updating.csv', latest_matches=10, latest_vs_matches = 10, latest_other_matches = 0, predict_path = '../Data/Complete_data/nba_games_predict.csv')


inputs, labels, bet_pred = r.make_dataset()

mask = np.all(~np.isnan(inputs), axis = 1)

inputs_new = torch.Tensor(inputs[~mask,:])
inputs, labels = torch.Tensor(inputs[mask,:]), torch.Tensor(labels[mask].reshape(-1,1))
bet_pred_new = torch.Tensor(bet_pred[~mask])

indeces = np.arange(len(labels))

bet_pred = bet_pred[mask]

samples_test = np.zeros((N_samples,2000,1))



inputs_test = inputs[-2000:, :]
labels_test = labels[-2000:, :]
bet_pred_test = bet_pred[-2000:, :]


X_train, X_val, y_train, y_val = train_test_split(inputs[:-2000], labels[:-2000], test_size=0.2, random_state=60)

model = models.MLP(inputs.shape[1], [64] + [32], 16, 0.05)

import ili
from ili.inference import InferenceRunner
from accelerate import Accelerator

accelerator = Accelerator()
device = accelerator.device


# define training arguments
train_args = {
    'training_batch_size': 64,
    'learning_rate': 2e-5,
    'stop_after_epochs': 10,
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

from torch.utils.data import DataLoader, TensorDataset
from ili.dataloaders import TorchLoader
import numpy as np

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
loader = TorchLoader(train_loader, val_loader)

posterior_ensemble, summaries = runner(loader=loader)

for ii in range(len(labels_test)):
     print(ii,len(labels_test))
     samples_test[:,ii + test_range*15,:] = posterior_ensemble.sample((N_samples,), torch.Tensor(inputs_test[ii]).to(device))
     
  



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
     
  
N_samples = 100000
samples_new = np.zeros((N_samples,inputs_new.shape[0],1))

for ii in range(inputs_new.shape[0]):
     print(ii,inputs_new.shape[0])
     samples_new[:,ii,:] = posterior_ensemble.sample((N_samples,), torch.Tensor(inputs_new[ii]).to(device))
     plt.figure()
     plt.title(r.teams[int(inputs_new[ii,0])] + ' vs. ' + r.teams[int(inputs_new[ii,1])])
     plt.hist(samples_new[:,ii,:], bins = 50)
     plt.gca().axvline(x = bet_pred_new[ii,0], ls = '--', color = 'r')
     plt.gca().text(0.025,0.025,str(np.round( 100*len(samples_new[:,ii,:][samples_new[:,ii,:] > float(bet_pred_new[ii,0])])/N_samples ,2)) + "% over", horizontalalignment='left',verticalalignment='center', transform=plt.gca().transAxes)
     plt.gca().text(0.025,0.075,str(np.round( 100*len(samples_new[:,ii,:][samples_new[:,ii,:] < float(bet_pred_new[ii,0])])/N_samples ,2)) + "% under", horizontalalignment='left',verticalalignment='center', transform=plt.gca().transAxes)
     
plt.show()
        
predictions = np.median(samples_train, axis = 0).reshape((-1))
predictions_val = np.median(samples_val, axis = 0).reshape((-1))
predictions_test = np.median(samples_test, axis = 0).reshape((-1))

print(np.sum( np.array(np.sign(predictions) == np.sign(y_train.reshape(-1)), dtype = int) )/len(predictions))
print(np.sum( np.array(np.sign(predictions_val) == np.sign(y_val.reshape(-1)), dtype = int) )/len(predictions_val))
print(np.sum( np.array(np.sign(predictions_test) == np.sign(labels_test.reshape(-1)), dtype = int) )/len(predictions_test))


# Plot predicted values vs. true labels
plt.figure(figsize=(8, 6))
plt.scatter(y_train, predictions, alpha=0.7, color='b')
plt.scatter(y_val, predictions_val, alpha=0.7, color='g')
#plt.scatter(labels, bet_pred[:,0], alpha=0.7, color='k')
plt.scatter(labels_test, predictions_test, alpha=0.7, color='r')

plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)  # Diagonal line for reference
plt.xlabel('True Labels (Sum of Inputs)')
plt.ylabel('Predicted Values')
plt.title('Predicted Values vs. True Labels')
plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(labels_test, predictions_test, alpha=0.7, color='b')
plt.scatter(labels_test, bet_pred_test[:,0], alpha=0.7, color='k')
plt.plot([labels_test.min(), labels_test.max()], [labels_test.min(), labels_test.max()], 'r--', lw=2)  # Diagonal line for reference
plt.xlabel('True Labels (Sum of Inputs)')
plt.ylabel('Predicted Values')
plt.title('Predicted Values vs. True Labels')
plt.show()










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
		indeces =np.random.choice(indeces_orig, size = 1500,replace = False)

		for aindex,ii in enumerate(indeces):
			max_bet = net/10
			bet_pred_ii = bet_pred_test[ii,0]
			if np.isnan(bet_pred_ii):
				net_evolution_test[kk, aindex] = net
				continue
			ratio_home = (np.abs(bet_pred_test[ii,0])/100)**np.sign(bet_pred_test[ii,0])
			ratio_away = (np.abs(bet_pred_test[ii,1])/100)**np.sign(bet_pred_test[ii,1])
			
			if np.median(samples_test[:, ii, 0]) > 0 and len(samples_test[:, ii, 0][samples_test[:, ii, 0] > 0])/N_samples > 0.8:
				ratio = ratio_home
				
				win_prob = len(samples_test[:, ii, 0][samples_test[:, ii, 0] > 0])/N_samples
				loss_prob = 1 - win_prob
				
				kelly_fraction = win_prob - loss_prob / ratio 
				
				if kelly_fraction > 1:
					kelly_fraction = 0.8
				
				if kelly_fraction > 0:
				
					if np.sign(labels_test[ii]) == 1:
						net+=max_bet*ratio_home*kelly_fraction
						nconsecutivelosses = 0
					else:
						net-=max_bet*kelly_fraction
						nlosses +=1
						nconsecutivelosses+=1
						
					nbets+=1
				

				
				
			elif np.median(samples_test[:, ii, 0]) < 0 and len(samples_test[:, ii, 0][samples_test[:, ii, 0] < 0])/N_samples > 0.8:
				ratio = ratio_home
				
				win_prob = len(samples_test[:, ii, 0][samples_test[:, ii, 0] > 0])/N_samples
				loss_prob = 1 - win_prob
				
				kelly_fraction = win_prob - loss_prob / ratio 
				
				if kelly_fraction > 1:
					kelly_fraction = 0.8
				
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
		print(kk, frac,net,max_loss, nbets, nlosses, nmaxconsecutivelosses,1-nlosses/nbets)





plt.figure()

plt.plot(np.arange(0,len(predictions_test),1), np.median(net_evolution_test, axis = 0))
plt.fill_between(np.arange(0,len(predictions_test),1), np.percentile(net_evolution_test, 16,axis = 0), np.percentile(net_evolution_test, 84,axis = 0), alpha = 0.3)
plt.fill_between(np.arange(0,len(predictions_test),1), np.percentile(net_evolution_test, 2.5,axis = 0), np.percentile(net_evolution_test, 97.5,axis = 0), alpha = 0.2)
plt.show()
				
		









odds_over = bet_pred_test[:,1]
odds_under = bet_pred_test[:,2]


net_evolution_test = np.zeros((100, len(np.arange(0.5,0.75,0.01)), len(predictions_test)))


for kk in range(100):
	for jj, frac in enumerate(np.arange(0.5,0.75,0.01)):
		start_net = 1000
		net = 1000
		max_loss = 0
		nbets = 0
		nlosses=0
		nconsecutivelosses = 0
		nmaxconsecutivelosses = 0
		indeces_orig = np.arange(0,len(predictions_test),1)
		indeces =np.random.choice(indeces_orig, size = 1500,replace = False)

		for aindex,ii in enumerate(indeces):
			max_bet = net/10
			bet_pred_ii = bet_pred_test[ii,0]
			if np.isnan(bet_pred_ii):
				net_evolution_test[kk, jj , aindex] = net
				continue
			ratio_over = (np.abs(bet_pred_test[ii,1])/100)**np.sign(bet_pred_test[ii,1])
			ratio_under = (np.abs(bet_pred_test[ii,2])/100)**np.sign(bet_pred_test[ii,2])
			if np.percentile(samples_test[:, ii, 0], (1-frac)*100) > bet_pred_ii:
				if labels_test[ii] > bet_pred_ii:
					net+=max_bet*ratio_over
					nconsecutivelosses = 0

				else:
					net-=max_bet
					nlosses +=1
					nconsecutivelosses+=1

				nbets+=1

			if np.percentile(samples_test[:, ii, 0], frac*100) < bet_pred_ii:
				if labels_test[ii] < bet_pred_ii:
					net+=max_bet*ratio_under
					nconsecutivelosses = 0

				else:
					net-=max_bet
					nlosses+=1
					nconsecutivelosses+=1

				nbets+=1
				if nconsecutivelosses > nmaxconsecutivelosses:
					nmaxconsecutivelosses = nconsecutivelosses
			if net-start_net < max_loss:
				max_loss = net-start_net
			net_evolution_test[kk, jj , aindex] = net
		print(kk, frac,net,max_loss, nbets, nlosses, nmaxconsecutivelosses,1-nlosses/nbets)

# Choose a colormap (e.g., 'viridis', 'plasma', 'inferno', 'cividis')
cmap = plt.get_cmap('hsv')

# Generate a list of colors by sampling from the colormap
colors = [cmap(i / (len(np.arange(0.5,0.75,0.01)) - 1)) for i in range(len(np.arange(0.5,0.82,0.01)))]
plt.figure()
for ii,a in enumerate(np.arange(0.5,0.75,0.01)):
	if a > 0.68 and a < 0.80:
		plt.figure()
	
		plt.title(a)
		plt.plot(np.arange(0,len(predictions_test),1), np.median(net_evolution_test[:,ii,:], axis = 0), color = colors[ii])
		plt.fill_between(np.arange(0,len(predictions_test),1), np.percentile(net_evolution_test[:,ii,:], 16,axis = 0), np.percentile(net_evolution_test[:,ii,:], 84,axis = 0), alpha = 0.3, color = colors[ii])
		plt.fill_between(np.arange(0,len(predictions_test),1), np.percentile(net_evolution_test[:,ii,:], 2.5,axis = 0), np.percentile(net_evolution_test[:,ii,:], 97.5,axis = 0), alpha = 0.2, color = colors[ii])
plt.show()
						
		
	
for frac in np.arange(0.1,1,0.1):
	win = 0
	loss = 0
	total = 0
	for ii in range(len(predictions_test)):
		bet = bet_pred_test[ii,0]
		if np.isnan(bet):
			continue
		if len(samples_test[:, ii, 0][samples_test[:, ii, 0]>bet])/1000 > frac:
			if float(labels_test[ii]) > bet:
				win+=1
			else:
				loss+=1
			total+=1
		if len(samples_test[:, ii, 0][samples_test[:, ii, 0]<bet])/1000 > frac:
			if float(labels_test[ii]) < bet:
				win+=1
			else:
				loss+=1
			total+=1
	print(frac, win/total, loss/total, total)
	
	
	
	
a
	
	
	
	
for frac in np.arange(0.5,1,0.1):
	over = 0
	under = 0
	for ii in range(len(predictions_test)):
		if labels_test[ii] > np.percentile(samples_test[:, ii, 0], frac*100):
			over+=1
		else:
			under+=1
	print(frac, over/len(predictions_test), under/len(predictions_test))
	
		
	
	
	
for frac in np.arange(0.1,1,0.1):
	inside = 0
	outside = 0
	for ii in range(len(predictions_val)):
		if (truths_val[ii] < np.percentile(samples_val[:, ii, 0], 50 + frac*100/2)) and (truths_val[ii] > np.percentile(samples_val[:, ii, 0], 50 - frac*100/2)):
			inside+=1
		else:
			outside+=1
	print(frac, inside/len(predictions_val), outside/len(predictions_val))	
	
	
for frac in np.arange(0.1,1,0.1):
	inside = 0
	outside = 0
	for ii in range(len(predictions_val)):
		if (truths_val[ii] > np.percentile(samples_val[:, ii, 0], (1-frac)*100)) or (truths_val[ii] < np.percentile(samples_val[:, ii, 0], frac*100)):
			inside+=1
		else:
			outside+=1
	print(frac, inside/len(predictions_val), outside/len(predictions_val))	
			
truths_test = np.array(labels_test).flatten()
for frac in np.arange(0.1,1,0.1):
	inside = 0
	outside = 0
	for ii in range(len(predictions_test)):
		if (truths_test[ii] > np.percentile(samples_test[:, ii, 0], (1-frac)*100)) or (truths_test[ii] < np.percentile(samples_test[:, ii, 0], frac*100)):
			inside+=1
		else:
			outside+=1
	print(frac, inside/len(predictions_test), outside/len(predictions_test))	
		
		

truths_test = np.array(labels_test).flatten()
for frac in np.arange(0.1,1,0.1):
	wins = 0
	losses = 0
	for ii in range(len(predictions_test)):
		if (truths_test[ii] > np.percentile(samples_test[:, ii, 0], (1-frac)*100)) or (truths_test[ii] < np.percentile(samples_test[:, ii, 0], frac*100)):
			inside+=1
		else:
			outside+=1
	print(frac, inside/len(predictions_test), outside/len(predictions_test))	
		
	
	
	
		
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
 
 
 
 
samples_test_temp = samples_test.copy()

indices = np.arange(len(inputs_test))

percentiles = np.zeros( (len(inputs_test), 100) )
for ii in range(100):
    percentiles[:,ii] = np.percentile(samples_test, ii+0.5, axis = 0).reshape((-1))

 
input_dim = 100 + 2  # Nsamples + odds_over + odds_under
max_bet = 100  # Example max bet value
 
mask = ~np.any(np.isnan(bet_pred_test), axis = 1) & ~np.isnan(labels_test.detach().numpy().flatten())

# Convert data to PyTorch tensors and split into training and validation sets
X = np.hstack((percentiles[mask,:], bet_pred_test[mask,:]))  # Concatenate samples and odds
y = np.hstack((bet_pred_test[mask,:], labels_test[mask].reshape(-1, 1)))  # Combine odds/thresholds and actual scores
indices = indices[mask]

X_train, X_val, y_train, y_val, ii_train, ii_val = train_test_split(X, y, indices, test_size=0.2, random_state=42)

# Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

betmodel = Bet_Strat_spread(102, [32])

betmodel.train_model(X_train, y_train, validation_data=(X_val, y_val))


