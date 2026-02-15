import reader
import scrappers
import models
import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import torch.nn as nn
import pickle
import pandas as pd
import os
from accelerate import Accelerator

accelerator = Accelerator()
device = accelerator.device


csv_file = "../Data/Complete_data/merged_nba_games_updating.csv"

scraper = scrappers.NBAGameScraper(csv_file, fetch_bet_data=True)
scraper.update(delta = 0)


csv_file_predict = "../Data/Complete_data/nba_games_predict.csv"

scraper = scrappers.NBAGameScraper(csv_file_predict, fetch_bet_data=True)
scraper.scrape(datetime.now(), datetime.now()+timedelta(days = 1), append=False)



r = reader.Reader_Polars(csv_file, latest_matches=30, latest_vs_matches = 30, latest_other_matches = 0, predict_path = csv_file_predict)

inputs, labels, bet_pred, dates, hteams, ateams = r.make_dataset(only_new_games=True)

mask = ~np.any(np.isnan(bet_pred), axis = 1) & (~np.all(inputs == 0, axis =1)) 
inputs = inputs[mask]
labels = labels[mask]
bet_pred = bet_pred[mask]
dates = dates[mask]
hteams = hteams[mask]
ateams = ateams[mask]



class LSTM_2designs(nn.Module):
    def __init__(self, input_size=7, hidden_size=32, output_size=2, input_size2=7, hidden_size2=32, num_layers=1, num_layers2=1, glob_var_size = 10):
        super(LSTM_2designs, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.rnn2 = nn.LSTM(input_size2, hidden_size2, num_layers2, batch_first=True)

        self.fc = nn.Linear(2 * hidden_size + 1 * hidden_size2 + glob_var_size, output_size)


    def forward(self, x):
        if len(x.shape) < 2:
            x = x.reshape(1,-1)
        x1 = x[:, :2 + r.N_current_game_fields]
        x2 = x[:, 2 + r.N_current_game_fields : 2 + r.N_current_game_fields + (3 + r.N_other_game_fields)*r.latest_matches].reshape((-1, (3 + r.N_other_game_fields), r.latest_matches)).transpose(1,2)
        x3 = x[:, 2 + r.N_current_game_fields + (3 + r.N_other_game_fields)*r.latest_matches : 2 + r.N_current_game_fields + 2*(3 + r.N_other_game_fields)*r.latest_matches].reshape((-1, (3 + r.N_other_game_fields), r.latest_matches)).transpose(1,2)
        x4 = x[:, 2 + r.N_current_game_fields + 2*(3 + r.N_other_game_fields)*r.latest_matches : 2 + r.N_current_game_fields + 2*(3 + r.N_other_game_fields)*r.latest_matches + (2 + r.N_other_game_fields)*r.latest_vs_matches ].reshape((-1, (2 + r.N_other_game_fields), r.latest_vs_matches)).transpose(1,2)
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



        # Concatenate all hidden states along the feature dimension
        concatenated_hidden = torch.cat(hidden_states, dim=1)
        concatenated_hidden2 = torch.cat([concatenated_hidden, x1], dim=1)

        # Fully connected layer to output predictions
        out = self.fc(concatenated_hidden2)
        return out



posterior_ensemble = pickle.load(open('Model_30_30_correctedrecords_homeaway_separated_20241222/posterior_30_30_correctedrecords_homeaway_separated_20241222.pkl', 'rb'))

betmodel = models.Bet_Strat_spread(102, [32])

checkpoint = torch.load('Model_30_30_correctedrecords_homeaway_separated_20241222/bet_model_checkpoint_median.pth')
betmodel.load_state_dict(checkpoint['model_state_dict'])
  
N_samples = 25000
samples = np.zeros((N_samples,inputs.shape[0],1))




for ii in range(inputs.shape[0]):
     print(ii,inputs.shape[0])
     samples[:,ii,:] = posterior_ensemble.sample((N_samples,), torch.Tensor(inputs[ii]).to(device))
     
     
percentiles = np.zeros( (len(inputs), 100) )
for ii in range(100):
    percentiles[:,ii] = np.percentile(samples, ii+0.5, axis = 0).reshape((-1))

# Function to convert American odds to decimal odds
def american_to_decimal(odds):
    decimal_odds = (odds / 100) if (odds > 0) else (100 / np.abs(odds))
    return decimal_odds


headers = ['date','home', 'away', 'homeML', 'awayML', 'Pred', 'Pred_p16', 'Pred_p84', 'Kelly', 'ML']
df = pd.DataFrame(columns=headers)

budget = 200+50+100

for ii in range(inputs.shape[0]):
     print("In Today's game " + str(ii))
     print("Scraped odds for " + hteams[ii] + " are " + str(bet_pred[ii][0]) + " or " + str(round(american_to_decimal(bet_pred[ii][0])+1, 2)))
     print("Scraped odds for " + ateams[ii] + " are " + str(bet_pred[ii][1]) + " or " + str(round(american_to_decimal(bet_pred[ii][1])+1, 2)))
     print("Expected outcome: " + str(round(np.median(samples[:,ii,:]) , 2)))
     betfracs = betmodel( torch.Tensor( np.concatenate( (percentiles[ii,:],  bet_pred[ii])) ) ).cpu().detach().numpy()
     bet_team = 'home' if np.argmax(betfracs) == 0 else 'away'
     bet_team_name = hteams[ii] if np.argmax(betfracs) == 0 else ateams[ii]
     print("ML_Bet_Strat: Bet " + str(round(np.max(betfracs), 5)) + " on " + bet_team + " team: " + bet_team_name)
     print(str(round(np.max(betfracs) * budget/10, 5)) + " €")
     
     win_index = 0 if np.median( samples[:,ii,:] ) > 0 else 1
     win_team = 'home' if np.median( samples[:,ii,:] ) > 0 else 'away'
     win_prob = len(samples[:,ii,:][samples[:,ii,:] > 0])/N_samples if win_team == 'home' else len(samples[:,ii,:][samples[:,ii,:] < 0])/N_samples
     loss_prob = 1-win_prob
     ratio = (np.abs(bet_pred[ii,win_index])/100)**np.sign(bet_pred[ii,win_index])
     
     kelly = win_prob - loss_prob / ratio
     
     print("Kelly: Bet " + str(round(kelly, 5)) + " on " + win_team + " team ")
     print(betfracs)
     print("")
     
     row = {
        'date': dates[ii],
        'home': hteams[ii],
        'away': ateams[ii],
        'homeML': bet_pred[ii][0],
        'awayML': bet_pred[ii][1],
        'Pred': np.median(samples[:,ii,:]),
        'Pred_p16': np.percentile(samples[:,ii,:], 16),
        'Pred_p84': np.percentile(samples[:,ii,:], 84),
        'Kelly': kelly,
        'ML': np.max(betfracs)
     }
     df.loc[len(df)] = row
     
# Filepath to save the CSV
filename = "Model_30_30_correctedrecords_homeaway_separated_20241222/Predictions_model_30_30-20241612.csv"

# Check if the file exists
if os.path.exists(filename):
    df_old = pd.read_csv(filename)
    
    """
    
    net_evolution_test = np.load('Model_30_30_correctedrecords_homeaway_separated_20241222/net_evo_Kelly.npy')
    net_evolution_test_MLBetStrat = np.load('Model_30_30_correctedrecords_homeaway_separated_20241222/net_evo_ML.npy')
    
    plt.figure()
    
    plt.plot(np.arange(0,net_evolution_test.shape[1],1), np.median(net_evolution_test, axis = 0), color = 'orange', label = 'Kelly Strat')
    plt.fill_between(np.arange(0,net_evolution_test.shape[1],1), np.percentile(net_evolution_test, 16,axis = 0), np.percentile(net_evolution_test, 84,axis = 0), alpha = 0.3, color = 'orange')
    plt.fill_between(np.arange(0,net_evolution_test.shape[1],1), np.percentile(net_evolution_test, 2.5,axis = 0), np.percentile(net_evolution_test, 97.5,axis = 0), alpha = 0.2, color = 'orange')
    
    plt.plot(np.arange(0,net_evolution_test_MLBetStrat.shape[1],1), np.median(net_evolution_test_MLBetStrat, axis = 0), color = 'blue', label = 'ML Strat')
    plt.fill_between(np.arange(0,net_evolution_test_MLBetStrat.shape[1],1), np.percentile(net_evolution_test_MLBetStrat, 16,axis = 0), np.percentile(net_evolution_test_MLBetStrat, 84,axis = 0), alpha = 0.3, color = 'blue')
    plt.fill_between(np.arange(0,net_evolution_test_MLBetStrat.shape[1],1), np.percentile(net_evolution_test_MLBetStrat, 2.5,axis = 0), np.percentile(net_evolution_test_MLBetStrat, 97.5,axis = 0), alpha = 0.2, color = 'blue')
    
    day_ratios = net_evolution_test_MLBetStrat[:, 1:]/net_evolution_test_MLBetStrat[:, :-1]
    day_ratios_K = net_evolution_test[:, 1:]/net_evolution_test[:, :-1]

    plt.figure()
    plt.hist(day_ratios.flatten(), bins = 30)
    plt.hist(day_ratios_K.flatten(), bins = 30)
    plt.show()		
    """		
		
		

    
    if df['date'][0] not in df_old['date']:
        # Append to the existing file
        df.to_csv(filename, mode='a', header=False, index=False)
        print(f"Data saved to {filename}")
    else:
        print(f"Data already existed in to {filename}")
else:
    # Create a new file
    df.to_csv(filename, mode='w', header=True, index=False)

    print(f"Data saved to {filename}")

