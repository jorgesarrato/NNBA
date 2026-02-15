import torch
import torch.nn as nn
import torch.optim as optim

def compute_accuracy(predictions, labels):
    predicted_classes = torch.argmax(predictions, dim=1)  # Get class predictions
    correct = (predicted_classes == labels).sum().item()  # Count correct predictions
    accuracy = correct / labels.size(0)  # Compute accuracy
    return accuracy
    
class MLP(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size, drop):
        super(MLP, self).__init__()
        layers = []
        in_size = input_size
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            in_size = hidden_size
        layers.append(nn.Linear(in_size, output_size))
        #layers.append(nn.Softmax(dim=1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def train_model(self, X_train, y_train, validation_data=None, learning_rate=0.001, num_epochs=100,
                    print_every=10, lr_patience=5, stop_patience=10, lr_factor=0.1):
        """
        Trains the MLP model with learning rate reduction and early stopping based on validation loss.

        Parameters:
        - X_train (torch.Tensor): Training input features
        - y_train (torch.Tensor): Training target values
        - validation_data (tuple): Tuple of (X_val, y_val) for validation data
        - learning_rate (float): Initial learning rate
        - num_epochs (int): Total number of epochs
        - print_every (int): Print losses every N epochs
        - lr_patience (int): Number of epochs with no improvement to wait before reducing LR
        - stop_patience (int): Number of epochs with no improvement to wait before stopping
        - lr_factor (float): Factor by which to reduce the LR
        """
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Initialize variables for early stopping and LR scheduling
        best_val_loss = float('inf')
        epochs_no_improve = 0
        X_val, y_val = validation_data if validation_data is not None else (None, None)

        for epoch in range(num_epochs):
            # Training phase
            self.train()
            predictions = self.forward(X_train)
            train_loss = criterion(predictions, y_train.squeeze())
            train_acc = compute_accuracy(predictions, y_train.squeeze())
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # Validation phase
            if validation_data is not None:
                self.eval()
                with torch.no_grad():
                    val_predictions = self.forward(X_val)
                    val_loss = criterion(val_predictions, y_val.squeeze())
                    val_acc = compute_accuracy(val_predictions, y_val.squeeze())
                
                # Check if validation loss improved
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0  # Reset counter
                else:
                    epochs_no_improve += 1

                # Learning rate reduction
                """if epochs_no_improve >= lr_patience:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= lr_factor
                    print(f"Reducing learning rate to {optimizer.param_groups[0]['lr']}")
                    epochs_no_improve = 0  # Reset after reducing LR"""

                # Early stopping
                if epochs_no_improve >= stop_patience:
                    print("Early stopping triggered")
                    break

                # Print training and validation loss
                if (epoch + 1) % print_every == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], '
                          f'Train Loss: {train_loss.item():.4f}, '
                          f'Train Acc: {train_acc:.4f}, '
                          f'Validation Loss: {val_loss.item():.4f}'
                          f'Train Val: {val_acc:.4f}, ')
            else:
                # Print only training loss if no validation data
                if (epoch + 1) % print_every == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss.item():.4f}')













class LSTM_2designs(nn.Module):
    def __init__(self, input_size=7, hidden_size=32, output_size=2, input_size2=7, hidden_size2=32, num_layers=1, num_layers2=1, glob_var_size = 10):
        super(RNNWithTimeDeltasAndNoise, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.rnn2 = nn.LSTM(input_size2, hidden_size2, num_layers2, batch_first=True)

        self.fc = nn.Linear(2 * hidden_size + 1 * hidden_size2 + glob_var_size, output_size)


    def forward(self, x, x2, x3, x4):
        # x is a list of 20 inputs for the LSTM
        hidden_states = []  # To collect final hidden states from each LSTM


        rnn_out, _ = self.rnn(x)  # Process each input through the LSTM
        if len(rnn_out.shape) > 2:
            hidden_states.append(rnn_out[:, -1, :])  # Collect the last hidden state of the sequence
        else:
            hidden_states.append(rnn_out[:, :])
            
        rnn_out, _ = self.rnn(x2)  # Process each input through the LSTM
        if len(rnn_out.shape) > 2:
            hidden_states.append(rnn_out[:, -1, :])  # Collect the last hidden state of the sequence
        else:
            hidden_states.append(rnn_out[:, :])
                
        rnn_out2, _ = self.rnn2(x3.squeeze())

        if len(rnn_out2.shape) > 2:
            hidden_states.append(rnn_out2[:, -1, :])  # Collect the last hidden state of the sequence
        else:
            hidden_states.append(rnn_out2[:, :])



        # Concatenate all hidden states along the feature dimension
        concatenated_hidden = torch.cat(hidden_states, dim=1)
        concatenated_hidden2 = torch.cat([concatenated_hidden, x4], dim=1)

        # Fully connected layer to output predictions
        out = self.fc(concatenated_hidden)
        return out



# Function to convert American odds to decimal odds
def american_to_decimal(odds):
    decimal_odds = torch.where(odds > 0, (odds / 100), (100 / torch.abs(odds)))
    return decimal_odds

# Custom profit loss function
class ProfitLoss(nn.Module):
    def __init__(self, max_bet):
        super(ProfitLoss, self).__init__()
        self.max_bet = max_bet

    def forward(self, y_pred, y_true):
        thresholds = y_true[:, 0]       # Over/under line (threshold)
        odds_over = y_true[:, 1]        # American odds for "over"
        odds_under = y_true[:, 2]       # American odds for "under"
        actual_scores = y_true[:, 3]    # Real game scores

        bet_over = y_pred[:, 0]         # Model output for "over" bet fraction
        bet_under = y_pred[:, 1]        # Model output for "under" bet fraction

        # Convert American odds to decimal odds
        decimal_odds_over = american_to_decimal(odds_over)
        decimal_odds_under = american_to_decimal(odds_under)

        # Define outcome indicators
        outcome_over = (actual_scores > thresholds).float()   # 1 if over, else 0
        outcome_under = 1 - outcome_over                      # Complementary outcome

        # Calculate profit for over and under bets
        profit_over = bet_over   * self.max_bet * (decimal_odds_over  * outcome_over  - (1 - outcome_over ))
        profit_under = bet_under * self.max_bet * (decimal_odds_under * outcome_under - (1 - outcome_under))

        # Total profit (negative for minimization)
        total_profit = profit_over + profit_under
        return -torch.sum(total_profit)  # Negative profit for loss minimization



class Bet_Strat(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes):
        super(Bet_Strat, self).__init__()
        layers = []
        in_size = input_size
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        layers.append(nn.Linear(in_size, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return nn.Sigmoid()(self.network(x))

    def train_model(self, X_train, y_train, validation_data=None, learning_rate=0.001, num_epochs=100,
                    print_every=10, lr_patience=5, stop_patience=10, lr_factor=0.1):
        """
        Trains the MLP model with learning rate reduction and early stopping based on validation loss.

        Parameters:
        - X_train (torch.Tensor): Training input features
        - y_train (torch.Tensor): Training target values
        - validation_data (tuple): Tuple of (X_val, y_val) for validation data
        - learning_rate (float): Initial learning rate
        - num_epochs (int): Total number of epochs
        - print_every (int): Print losses every N epochs
        - lr_patience (int): Number of epochs with no improvement to wait before reducing LR
        - stop_patience (int): Number of epochs with no improvement to wait before stopping
        - lr_factor (float): Factor by which to reduce the LR
        """
        
        criterion = ProfitLoss(max_bet = 100)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Initialize variables for early stopping and LR scheduling
        best_val_loss = float('inf')
        epochs_no_improve = 0
        X_val, y_val = validation_data if validation_data is not None else (None, None)

        for epoch in range(num_epochs):
            # Training phase
            self.train()
            predictions = self.forward(X_train)
            train_loss = criterion(predictions, y_train)
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # Validation phase
            if validation_data is not None:
                self.eval()
                with torch.no_grad():
                    val_predictions = self.forward(X_val)
                    val_loss = criterion(val_predictions, y_val)
                
                # Check if validation loss improved
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0  # Reset counter
                else:
                    epochs_no_improve += 1

                # Learning rate reduction
                """if epochs_no_improve >= lr_patience:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= lr_factor
                    print(f"Reducing learning rate to {optimizer.param_groups[0]['lr']}")
                    epochs_no_improve = 0  # Reset after reducing LR"""

                # Early stopping
                if epochs_no_improve >= stop_patience:
                    print("Early stopping triggered")
                    break

                # Print training and validation loss
                if (epoch + 1) % print_every == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], '
                          f'Train Loss: {train_loss.item():.4f}, '
                          f'Validation Loss: {val_loss.item():.4f}')
            else:
                # Print only training loss if no validation data
                if (epoch + 1) % print_every == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss.item():.4f}')



















# Custom profit loss function
class ProfitLoss_Spread(nn.Module):
    def __init__(self, max_bet):
        super(ProfitLoss_Spread, self).__init__()
        self.max_bet = max_bet

    def forward(self, y_pred, y_true):
        odds_home = y_true[:, 0]        # American odds for "home"
        odds_away = y_true[:, 1]       # American odds for "away"
        actual_scores = y_true[:, 2]    # Real game scores

        bet_home = y_pred[:, 0]         # Model output for "home" bet fraction
        bet_away = y_pred[:, 1]        # Model output for "away" bet fraction

        # Convert American odds to decimal odds
        decimal_odds_home = american_to_decimal(odds_home)
        decimal_odds_away = american_to_decimal(odds_away)

        # Define outcome indicators
        outcome_home = (actual_scores > 0).float()   # 1 if home, else 0
        outcome_away = 1 - outcome_home                      # Complementary outcome

        profit_home = bet_home * self.max_bet * (
        decimal_odds_home * outcome_home - (1 - outcome_home)
        )
        

        profit_away = bet_away * self.max_bet * (
        decimal_odds_away * outcome_away - (1 - outcome_away)
        )

        # Total profit (negative for minimization)
        total_profit = profit_home + profit_away
        
        """print(decimal_odds_home)
        print(decimal_odds_away)
        print(outcome_home)
        print(outcome_away)
        print(bet_home)
        print(bet_away)
        print(profit_home)
        print(profit_away)
        print(total_profit / (bet_home* self.max_bet + bet_away* self.max_bet))"""

        return -torch.mean(total_profit / (bet_home* self.max_bet + bet_away* self.max_bet))
        
        #-torch.median(total_profit / (bet_home* self.max_bet + bet_away* self.max_bet)) # Negative profit for loss minimization



"""
        profit_home = bet_home   * self.max_bet * (decimal_odds_home  * outcome_home  - (1 - outcome_home ))
        profit_away = bet_away * self.max_bet * (decimal_odds_away * outcome_away - (1 - outcome_away))
"""



class Bet_Strat_spread(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes):
        super(Bet_Strat_spread, self).__init__()
        layers = []
        in_size = input_size
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        layers.append(nn.Linear(in_size, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return nn.Sigmoid()(self.network(x))
        
        
    def test_model(self, X_val, y_val):
        criterion = ProfitLoss_Spread(max_bet = 100)
        self.eval()# Validation phase
        with torch.no_grad():
            val_predictions = self.forward(X_val)
            val_loss = criterion(val_predictions, y_val)
        print(val_loss)

    def train_model(self, X_train, y_train, validation_data=None, learning_rate=0.001, num_epochs=100,
                    print_every=10, lr_patience=5, stop_patience=10, lr_factor=0.1):
        """
        Trains the MLP model with learning rate reduction and early stopping based on validation loss.

        Parameters:
        - X_train (torch.Tensor): Training input features
        - y_train (torch.Tensor): Training target values
        - validation_data (tuple): Tuple of (X_val, y_val) for validation data
        - learning_rate (float): Initial learning rate
        - num_epochs (int): Total number of epochs
        - print_every (int): Print losses every N epochs
        - lr_patience (int): Number of epochs with no improvement to wait before reducing LR
        - stop_patience (int): Number of epochs with no improvement to wait before stopping
        - lr_factor (float): Factor by which to reduce the LR
        """
        
        criterion = ProfitLoss_Spread(max_bet = 100)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Initialize variables for early stopping and LR scheduling
        best_val_loss = float('inf')
        epochs_no_improve = 0
        X_val, y_val = validation_data if validation_data is not None else (None, None)

        for epoch in range(num_epochs):
            # Training phase
            self.train()
            predictions = self.forward(X_train)
            train_loss = criterion(predictions, y_train)
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # Validation phase
            if validation_data is not None:
                self.eval()
                with torch.no_grad():
                    val_predictions = self.forward(X_val)
                    val_loss = criterion(val_predictions, y_val)
                
                # Check if validation loss improved
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_no_improve = 0  # Reset counter
                else:
                    epochs_no_improve += 1

                # Learning rate reduction
                """if epochs_no_improve >= lr_patience:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= lr_factor
                    print(f"Reducing learning rate to {optimizer.param_groups[0]['lr']}")
                    epochs_no_improve = 0  # Reset after reducing LR"""

                # Early stopping
                if epochs_no_improve >= stop_patience:
                    print("Early stopping triggered")
                    break

                # Print training and validation loss
                if (epoch + 1) % print_every == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], '
                          f'Train Loss: {train_loss.item():.4f}, '
                          f'Validation Loss: {val_loss.item():.4f}')
            else:
                # Print only training loss if no validation data
                if (epoch + 1) % print_every == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss.item():.4f}')




