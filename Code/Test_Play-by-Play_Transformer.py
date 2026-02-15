import fireducks.pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import glob


import multiprocessing as mp
from functools import partial
import pandas as pd
import numpy as np
from tqdm import tqdm

class NBAPlayByPlayDataset(Dataset):
    def __init__(self, csv_paths, encode = True, num_workers=10):
        """
        Args:
            csv_paths: List of paths to game CSV files
            num_workers: Number of parallel processes to use
        """
        self.sequences = []
        self.labels = []
        self.lengths = []
        self.max_seq_len = 0
        self.max_time = 0
        self.encode = encode
        
        # First pass: collect all data needed for encoders (parallel)
        print("Collecting all data for encoders (parallel)...")
        with mp.Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(self._collect_encoder_data, csv_paths), 
                          total=len(csv_paths)))
        
        # Combine results from parallel processing
        all_player_ids = set()
        all_team_ids = set()
        all_events = set()
        all_actions = set()
        all_data_for_encoding = []
        
        for player_ids, team_ids, events, actions, seq_len, max_t, sample in results:
            all_player_ids.update(player_ids)
            all_team_ids.update(team_ids)
            all_events.update(events)
            all_actions.update(actions)
            if seq_len > self.max_seq_len:
                self.max_seq_len = seq_len
            if max_t > self.max_time:
                self.max_time = max_t
            all_data_for_encoding.append(sample)
        
        # Add unknown IDs
        all_team_ids.add(-999)
        all_player_ids.add(-999)
        all_events.add(-999)
        all_actions.add(-999)


        self.N_teams = len(all_team_ids)
        self.N_players = len(all_player_ids)
        self.N_events = len(all_events)
        self.N_actions = len(all_actions)
        
        # Create and fit encoders
        self.player_encoder = LabelEncoder().fit(list(all_player_ids))
        self.team_encoder = LabelEncoder().fit(list(all_team_ids))
        self.event_encoder = LabelEncoder().fit(list(all_events))
        self.action_encoder = LabelEncoder().fit(list(all_actions))
        
        # Fit OneHotEncoder on all possible values
        combined_for_encoding = pd.concat(all_data_for_encoding)
        categorical_features = [
            'EVENTMSGTYPE', 'EVENTMSGACTIONTYPE',
            'PERSON1TYPE', 'PERSON2TYPE', 'PERSON3TYPE',
            'PLAYER1_ID', 'PLAYER2_ID', 'PLAYER3_ID',
            'PLAYER1_TEAM_ID', 'PLAYER2_TEAM_ID', 'PLAYER3_TEAM_ID'
        ]
        self.onehot_encoder = OneHotEncoder(handle_unknown='ignore')
        self.onehot_encoder.fit(combined_for_encoding[categorical_features])
        
        # Second pass: process all games with fitted encoders (parallel)
        print("Processing game data with fitted encoders (parallel)...")
        process_fn = partial(self._process_single_file, 
                           player_encoder=self.player_encoder,
                           team_encoder=self.team_encoder,
                            event_encoder=self.event_encoder,
                            action_encoder=self.action_encoder,
                           onehot_encoder=self.onehot_encoder,
                           max_time=self.max_time,
                           max_seq_len=self.max_seq_len)
        
        with mp.Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(process_fn, csv_paths), 
                             total=len(csv_paths)))
        
        # Combine results
        for seq, label, length in results:
            if seq is not None:
                self.sequences.append(seq)
                self.labels.append(label)
                self.lengths.append(length)
    
    def _collect_encoder_data(self, path):
        """Helper function for parallel data collection"""
        try:
            df = pd.read_csv(path)

            df.fillna({
            'PLAYER1_ID': -999, 'PLAYER2_ID': -999, 'PLAYER3_ID': -999,
            'PLAYER1_TEAM_ID': -999, 'PLAYER2_TEAM_ID': -999, 'PLAYER3_TEAM_ID': -999,
            'PERSON1TYPE': -999, 'PERSON2TYPE': -999, 'PERSON3TYPE': -999
            }, inplace=True)
            
            # Collect player and team IDs
            player_ids = set()
            team_ids = set()
            events = set(df['EVENTMSGTYPE'].dropna().unique().astype(int))
            actions = set(df['EVENTMSGACTIONTYPE'].dropna().unique().astype(int))

            for col in ['PLAYER1_ID', 'PLAYER2_ID', 'PLAYER3_ID']:
                unique_ids = df[col].dropna().unique().astype(int)
                player_ids.update(unique_ids)
            for col in ['PLAYER1_TEAM_ID', 'PLAYER2_TEAM_ID', 'PLAYER3_TEAM_ID']:
                unique_ids = df[col].dropna().unique().astype(int)
                team_ids.update(unique_ids)
            
            # Calculate time features
            df['TIME_ELAPSED'] = pd.to_datetime(df['WCTIMESTRING'], format='%I:%M %p').astype('int64') // 10**9
            df['TIME_ELAPSED'] -= df['TIME_ELAPSED'][0]
            max_t = df['TIME_ELAPSED'].max()
            
            # Prepare sample for encoder fitting
            sample = df[[
                'EVENTMSGTYPE', 'EVENTMSGACTIONTYPE',
                'PERSON1TYPE', 'PERSON2TYPE', 'PERSON3TYPE',
                'PLAYER1_ID', 'PLAYER2_ID', 'PLAYER3_ID',
                'PLAYER1_TEAM_ID', 'PLAYER2_TEAM_ID', 'PLAYER3_TEAM_ID'
            ]].copy()
            
            return (player_ids, team_ids, events, actions, len(df), max_t, sample)
        except Exception as e:
            print(f"Error collecting data from {path}: {str(e)}")
            return (set(), set(), 0, 0, None)
    
    def _process_single_file(self, path, player_encoder, team_encoder, event_encoder, action_encoder, onehot_encoder, max_time, max_seq_len):
        """Helper function for parallel file processing"""
        df = pd.read_csv(path)

        length = len(df)

        df['SCOREMARGIN'] = df['SCOREMARGIN'].replace('TIE', '0').fillna(method='ffill').fillna(0).astype(float)
        
        # Get final scores added
        #label = float(df['SCOREMARGIN'].iloc[-1])
        final_score = str(df['SCORE'].iloc[-1])
        if final_score == 'nan':
            print(f"Final score is NaN for {path}")
            return (None, None)
        score1,score2 = final_score.split(' - ')
        total_final_score = float(score1) + float(score2)
        label = total_final_score


        # Preprocess data
        df.fillna({
        'PLAYER1_ID': -999, 'PLAYER2_ID': -999, 'PLAYER3_ID': -999,
        'PLAYER1_TEAM_ID': -999, 'PLAYER2_TEAM_ID': -999, 'PLAYER3_TEAM_ID': -999,
        'PERSON1TYPE': -999, 'PERSON2TYPE': -999, 'PERSON3TYPE': -999
        }, inplace=True)
        df.fillna(-999, inplace=True)
        
        # Feature engineering
        df['TIME_ELAPSED'] = pd.to_datetime(df['WCTIMESTRING'], format='%I:%M %p').astype('int64') // 10**9
        df['TIME_ELAPSED'] -= df['TIME_ELAPSED'][0]
        df['TIME_ELAPSED'] = df['TIME_ELAPSED'] / max_time

        if self.encode:
            
            # Encode IDs
            df['PLAYER1_ID'] = player_encoder.transform(df['PLAYER1_ID'].astype(int))
            df['PLAYER2_ID'] = player_encoder.transform(df['PLAYER2_ID'].astype(int))
            df['PLAYER3_ID'] = player_encoder.transform(df['PLAYER3_ID'].astype(int))
            df['PLAYER1_TEAM_ID'] = team_encoder.transform(df['PLAYER1_TEAM_ID'].astype(int))
            df['PLAYER2_TEAM_ID'] = team_encoder.transform(df['PLAYER2_TEAM_ID'].astype(int))
            df['PLAYER3_TEAM_ID'] = team_encoder.transform(df['PLAYER3_TEAM_ID'].astype(int))

            df['EVENTMSGTYPE'] = event_encoder.transform(df['EVENTMSGTYPE'].astype(int))
            df['EVENTMSGACTIONTYPE'] = action_encoder.transform(df['EVENTMSGACTIONTYPE'].astype(int))
            
        # Prepare features
        features = df[[
            'EVENTMSGTYPE', 'EVENTMSGACTIONTYPE',
            'PERSON1TYPE', 'PERSON2TYPE', 'PERSON3TYPE',
            'PLAYER1_ID', 'PLAYER2_ID', 'PLAYER3_ID',
            'PLAYER1_TEAM_ID', 'PLAYER2_TEAM_ID', 'PLAYER3_TEAM_ID'
        ]]
        
        # Transform features
        #categorical_data = onehot_encoder.transform(features).toarray()
        categorical_data = features.copy()
        numerical_data = df[['TIME_ELAPSED','SCOREMARGIN']].values
        processed_data = np.concatenate([numerical_data, categorical_data], axis=1)
        
        # Pad sequence
        if processed_data is not None and processed_data.shape[0] < max_seq_len:
            padding = np.zeros((max_seq_len - processed_data.shape[0], processed_data.shape[1]))
            processed_data = np.vstack([processed_data, padding])

        
        return (processed_data, label, length)

    
    # Keep the rest of the methods (_pad_sequence, __len__, __getitem__) the same
    # ...
    def _pad_sequence(self, sequence):
        """Pad sequence to max length with zeros"""
        if sequence.shape[0] < self.max_seq_len:
            padding = np.zeros((self.max_seq_len - sequence.shape[0], sequence.shape[1]))
            return np.vstack([sequence, padding])
        return sequence
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        label = torch.FloatTensor([self.labels[idx]])
        length = torch.LongTensor([self.lengths[idx]])
        return sequence, label, length
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)



class BasketballScorePredictor(nn.Module):
    def __init__(self, input_size=100, hidden_size=16):
        super(BasketballScorePredictor, self).__init__()
        self.encoder = nn.LSTM(input_size=1, hidden_size=hidden_size, 
                              num_layers=2, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x, lengths):

        x = x[:, :, 1]  # TIME_ELAPSED, SCOREMARGIN
        # x shape: (batch_size, seq_len)
        x = x.unsqueeze(-1)  # (batch_size, seq_len, 1)
        
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # LSTM encoding
        packed_out, _ = self.encoder(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        
        # Attention mechanism
        attn_weights = torch.softmax(self.attention(out), dim=1)
        context = torch.sum(attn_weights * out, dim=1)
        
        # Regression
        return self.regressor(context).squeeze(-1)  # (batch_size,)
    



## 2. Intermediate Transformer Model
class BasketballTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=3, max_length=30):
        super().__init__()
        self.d_model = d_model
        
        # Dual embedding paths
        self.value_embedding = nn.Sequential(
            nn.Linear(1, d_model//2),
            nn.GELU(),
            nn.Linear(d_model//2, d_model)
        )
        
        self.delta_embedding = nn.Sequential(
            nn.Linear(1, d_model//2),
            nn.GELU(), 
            nn.Linear(d_model//2, d_model)
        )
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(max_length, d_model))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )
        
    def forward(self, x, deltas, mask):
        # Embed inputs (batch_size, seq_len, d_model)
        x_embed = self.value_embedding(x.unsqueeze(-1)) 
        delta_embed = self.delta_embedding(deltas.unsqueeze(-1))
        
        # Combine embeddings with positional info
        x_combined = x_embed + delta_embed + self.positional_encoding[:x.size(1)]
        
        # Transformer processing
        x_out = self.transformer(x_combined, src_key_padding_mask=mask)
        
        # Adaptive pooling
        if mask is not None:
            x_out = x_out.masked_fill(mask.unsqueeze(-1), 0)
            pooled = x_out.sum(dim=1) / (~mask).sum(dim=1, keepdim=True).clamp(min=1)
        else:
            pooled = x_out.mean(dim=1)
            
        return self.head(pooled).squeeze(-1)

## 3. Training Utilities
def pad_collate(batch):
    inputs = [item[0] for item in batch]
    deltas = [item[1] for item in batch]
    targets = torch.stack([item[2] for item in batch])
    
    lengths = torch.tensor([len(x) for x in inputs])
    padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    padded_deltas = torch.nn.utils.rnn.pad_sequence(deltas, batch_first=True, padding_value=0)
    mask = torch.arange(padded_inputs.size(1))[None, :] >= lengths[:, None]
    
    return padded_inputs, padded_deltas, targets, mask










## 1. Enhanced Dataset with Delta Features
class BasketballDataset(Dataset):
    def __init__(self, num_samples=10000, max_length=30, max_points_per_step=5):
        self.num_samples = num_samples
        self.max_length = max_length
        self.max_points_per_step = max_points_per_step
        self.data = []
        self.labels = []
        
        self._generate_data()
    
    def _generate_data(self):
        for _ in range(self.num_samples):
            seq_length = np.random.randint(10, self.max_length + 1)
            score_diff = [0]
            total_points = 0
            
            for _ in range(seq_length - 1):
                if np.random.rand() < 0.4 and len(score_diff) > 1:
                    score_diff.append(score_diff[-1])
                else:
                    change = np.random.randint(-self.max_points_per_step, self.max_points_per_step + 1)
                    while score_diff[-1] + change == score_diff[-1]:
                        change = np.random.randint(-self.max_points_per_step, self.max_points_per_step + 1)
                    
                    new_diff = score_diff[-1] + change
                    score_diff.append(new_diff)
                    total_points += abs(change)
            
            self.data.append(np.array(score_diff, dtype=np.float32))
            self.labels.append(total_points)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        deltas = np.diff(sequence, prepend=0).astype(np.float32)
        return (torch.tensor(sequence, dtype=torch.float32),
                torch.tensor(deltas, dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.float32))









# Hyperparameters
batch_size = 32
epochs = 50

# Dataset and DataLoader
game_files = glob.glob('/home/jsarrato/PersonalProjects/NBBA/Data/Play-By-Play/2024-25/*v2.csv') + glob.glob('/home/jsarrato/PersonalProjects/NBBA/Data/Play-By-Play/2023-24/*v2.csv') + glob.glob('/home/jsarrato/PersonalProjects/NBBA/Data/Play-By-Play/2021-22/*v2.csv') + glob.glob('/home/jsarrato/PersonalProjects/NBBA/Data/Play-By-Play/2022-23/*v2.csv')
dataset = NBAPlayByPlayDataset(game_files, encode = False)

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Determine input dimension from first sample
input_dim = dataset[0][0].shape[1]

# Model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = LastValueLSTM().to(device)
model = BasketballScorePredictor(input_size=input_dim).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

best_val_loss = float('inf')
early_stop_counter = 0

for epoch in range(50):  # Max epochs
    # Training phase
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
        x, y, z  = batch
        x, y, z = x.to(device), y.squeeze(-1).to(device), z.squeeze(-1).to(device)
        
    
        optimizer.zero_grad()
        pred = model(x,z)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
            x, y, z  = batch
            x, y, z = x.to(device), y.squeeze(-1).to(device), z.squeeze(-1).to(device)
            pred = model(x,z)
            val_loss += criterion(pred, y).item()
    
    # Calculate metrics
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        early_stop_counter += 1
        if early_stop_counter >= 5:
            print("Early stopping triggered")
            break


def get_result(data, field_goal_events, threept_actions, free_throw_events, free_trow_made_actions):
    res = 0
    res+=len(data[:,1][np.isin(data[:,1], np.array(field_goal_events))])*2
    res+=len(data[:,1][np.isin(data[:,1], np.array(field_goal_events)) & np.isin(data[:,2], np.array(threept_actions))])
    res+=len(data[:,1][np.isin(data[:,1], np.array(free_throw_events)) & np.isin(data[:,2], np.array(threept_actions))])
    return res
















        final_score = str(df['SCOREMARGIN'].iloc[-1])
        if final_score == 'nan':
            print(f"Final score is NaN for {path}")
            return (None, None)
        score1,score2 = final_score.split(' - ')
        total_final_score = float(score1) + float(score2)
        label = total_final_score
    


    def score_from_margins(margins, len):
        return np.sum(np.abs(np.diff(margins[:len])))