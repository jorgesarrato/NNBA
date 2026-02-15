import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests
import time
from nba_api.stats.static import teams

# Set headers
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9',
    'Origin': 'https://www.nba.com',
    'Referer': 'https://www.nba.com',
    'Connection': 'keep-alive'
}


# Retrieve all games for a season
def get_all_games_for_season(season='2022-23', league_id='00'):
    gamefinder = leaguegamefinder.LeagueGameFinder(season_nullable=season, league_id_nullable=league_id, season_type_nullable='Regular Season', headers=headers)
    games_df = gamefinder.get_data_frames()[0]
    
    # Remove duplicates based on GAME_ID
    games_df = games_df.drop_duplicates(subset='GAME_ID')
    
    return games_df[['GAME_ID', 'GAME_DATE', 'MATCHUP']]

# Fetch player and team boxscores with timeout
def fetch_boxscores(game_id, games_df):
    """
    Fetch player and team boxscores for a single game ID with timeout handling.
    """
    try:
        box_score = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id, headers=headers, timeout = 120).get_data_frames()
        if not box_score:
            return pd.DataFrame(), pd.DataFrame()
        
        # Player and team boxscores
        game_info_players = box_score[0]
        game_info_teams = box_score[1]
        
        # Add game date to both
        game_date = games_df[games_df['GAME_ID'] == game_id]['GAME_DATE'].values[0]
        game_info_players['GAME_DATE'] = pd.to_datetime(game_date).date()
        game_info_teams['GAME_DATE'] = pd.to_datetime(game_date).date()
        
        # Determine home/away and add to team boxscores
        matchup = games_df[games_df['GAME_ID'] == game_id]['MATCHUP'].values[0]
        team_abbreviation = game_info_teams['TEAM_ABBREVIATION']
        game_info_teams['HOME_AWAY'] = team_abbreviation.apply(
            lambda team: 'Home' if (f'vs. {team}' in matchup or f'{team} @' in matchup) else 'Away' if (f'@ {team}' in matchup or f'{team} vs.' in matchup) else 'Unknown'
        )
        
        return game_info_players, game_info_teams
    except Exception as e:
        print(f"Error fetching boxscore for GAME_ID {game_id}: {e}")
        return pd.DataFrame(), pd.DataFrame()

# Process MIN column
def process_minutes_column(df):
    """
    Clean the MIN column, converting it to minutes as a float.
    """
    if 'MIN' in df.columns:
        def convert_min(min_str):
            try:
                # Handle cases like '24.00000:23'
                min_str = min_str.split(':')  # Split into minutes and seconds
                minutes = int(float(min_str[0]))
                seconds = int(min_str[1]) if len(min_str) > 1 else 0
                return minutes + seconds / 60
            except:
                return 0  # Handle NaN or improperly formatted strings

        df['MIN'] = df['MIN'].astype(str).apply(convert_min)
    return df

# Fetch all player and team boxscores in parallel
def fetch_all_boxscores_parallel(game_ids, games_df, max_workers=8):
    """
    Fetch player and team boxscores for all game IDs in parallel with timeout handling.
    """
    all_player_boxscores = []
    all_team_boxscores = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_boxscores, game_id, games_df): game_id for game_id in game_ids}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching Boxscores"):
            player_result, team_result = future.result()
            if not player_result.empty:
                player_result = process_minutes_column(player_result)
                all_player_boxscores.append(player_result)
            if not team_result.empty:
                team_result = process_minutes_column(team_result)
                all_team_boxscores.append(team_result)
    
    player_df = pd.concat(all_player_boxscores, ignore_index=True) if all_player_boxscores else pd.DataFrame()
    team_df = pd.concat(all_team_boxscores, ignore_index=True) if all_team_boxscores else pd.DataFrame()
    return player_df, team_df

# Main script for multiple seasons
if __name__ == '__main__':
    # Define seasons
    start_year = 1983
    end_year = 2024
    seasons = [f"{year}-{str(year+1)[-2:]}" for year in range(start_year, end_year + 1)]

    for season in seasons:
        print(f"Processing season: {season}")

        games_df = get_all_games_for_season(season)
        print(f"Retrieved {len(games_df)} games for the {season} regular season.")
        
        # Retrieve player and team boxscores
        player_boxscores_df, team_boxscores_df = fetch_all_boxscores_parallel(games_df['GAME_ID'].tolist(), games_df)
        
        # Save to CSV
        player_boxscores_df.to_csv(f'player_boxscores_{season}.csv', index=False)
        team_boxscores_df.to_csv(f'team_boxscores_{season}.csv', index=False)





def calculate_records(season_data):
    """
    Add cumulative home and away records for a team up to each game in the season.
    """
    # Create a dataframe to store cumulative records
    season_data['HOME_WINS'] = 0
    season_data['AWAY_WINS'] = 0
    season_data['HOME_LOSSES'] = 0
    season_data['AWAY_LOSSES'] = 0
    
    # Process each team separately
    for team_id in season_data['TEAM_ID'].unique():
        team_games = season_data[season_data['TEAM_ID'] == team_id].copy()
        team_games = team_games.sort_values('GAME_DATE')
        
        # Initialize cumulative stats
        home_wins = away_wins = home_losses = away_losses = 0
        
        records = []
        for idx, row in team_games.iterrows():
            # Append current record before this game
            if row['HOME_AWAY'] == 'Home':
                records.append((home_wins, away_wins, home_losses, away_losses))
                if row['PTS'] > team_games[team_games['GAME_ID'] == row['GAME_ID']]['PTS'].iloc[1]:
                    home_wins += 1
                else:
                    home_losses += 1
            else:  # Away game
                records.append((home_wins, away_wins, home_losses, away_losses))
                if row['PTS'] > team_games[team_games['GAME_ID'] == row['GAME_ID']]['PTS'].iloc[0]:
                    away_wins += 1
                else:
                    away_losses += 1
        
        # Add the calculated records back to the season data
        team_games[['HOME_WINS', 'AWAY_WINS', 'HOME_LOSSES', 'AWAY_LOSSES']] = pd.DataFrame(records, index=team_games.index)
        season_data.update(team_games)

    return season_data


def combine_all_seasons_data_with_records(file_prefix, output_file):
    """
    Combine all season data from multiple CSVs into a single DataFrame, adding home/away records.
    """
    all_data = []
    for season in seasons:
        try:
            file_path = f"{file_prefix}_{season}.csv"
            season_data = pd.read_csv(file_path)
            season_data = season_data.sort_values('GAME_DATE')
            
            # Calculate records for the current season
            season_data = calculate_records(season_data)
            
            all_data.append(season_data)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    
    combined_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    combined_df['GAME_DATE'] = pd.to_datetime(combined_df['GAME_DATE'])
    combined_df = combined_df.sort_values('GAME_DATE')
    combined_df.to_csv(output_file, index=False)
    print(f"Saved combined data to {output_file}")


# Combine all season data at the end
def combine_all_seasons_data(file_prefix, output_file):
    """
    Combine all season data from multiple CSVs into a single DataFrame.
    """
    all_data = []
    for season in seasons:
        try:
            file_path = f"{file_prefix}_{season}.csv"
            season_data = pd.read_csv(file_path)
            season_data = season_data.sort_values('GAME_DATE')
            all_data.append(season_data)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    
    combined_df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
    combined_df['GAME_DATE'] = pd.to_datetime(combined_df['GAME_DATE'])
    combined_df = combined_df.sort_values('GAME_DATE')
    combined_df.to_csv(output_file, index=False)
    print(f"Saved combined data to {output_file}")

# Combine all data at the end
if __name__ == '__main__':
    # Combine all player boxscores
    combine_all_seasons_data('player_boxscores', 'combined_player_boxscores.csv')
    
    # Combine all team boxscores
    combine_all_seasons_data_with_records('team_boxscores', 'combined_team_boxscores.csv')




















def combine_team_boxscores(team_boxscores_df):
    """
    Combine two team rows for each game into a single row with duplicated columns.
    """
    
    # Separate home and away rows into different DataFrames
    home_rows = team_boxscores_df[team_boxscores_df['HOME_AWAY'] == 'Away'] # I invert this because it looks my previous code is mislabelling them
    away_rows = team_boxscores_df[team_boxscores_df['HOME_AWAY'] == 'Home']

    # Merge home and away rows on GAME_ID and GAME_DATE
    combined_df = pd.merge(
        home_rows,
        away_rows,
        on=['GAME_ID', 'GAME_DATE', 'MIN'],
        suffixes=('_HOME', '_AWAY'),
        how='inner'
    )
    
    # Keep only GAME_ID and GAME_DATE without suffixes
    common_columns = ['GAME_ID', 'GAME_DATE', 'MIN']
    for col in common_columns:
        if col + '_HOME' in combined_df.columns:
            combined_df.drop(columns=[col + '_HOME', col + '_AWAY'], inplace=True)
    
    return combined_df
    
    



def replace_nan_based_on_comment(row, columns):
    replacement_value = None
    if 'did not play' in str(row['COMMENT']).lower() or str(row['COMMENT']).lower().startswith('dnp'):
        if 'coach' in str(row['COMMENT']).lower():
                replacement_value = -1
        else:
                replacement_value = -2
        
    elif 'did not dress' in str(row['COMMENT']).lower() or str(row['COMMENT']).lower().startswith('dnd'):
            replacement_value = -3
    elif 'did not travel' in str(row['COMMENT']).lower() or str(row['COMMENT']).lower().startswith('dnt'):
            replacement_value = -4
    elif 'not with team' in str(row['COMMENT']).lower() or str(row['COMMENT']).lower().startswith('nwt'):
            replacement_value = -5
    else:
            replacement_value = -6
    
    # Replace NaN values for the specified columns
    for col in columns:
        row[col] = replacement_value
    return row


def map_start_position(position):
    if pd.isna(position):  # If NaN
        return 0
    elif position == 'G':  # If 'G'
        return 1
    elif position == 'C':  # If 'C'
        return 2
    elif position == 'F':  # If 'F'
        return 3
    return position  # For any other value, keep it as is


# Combine team boxscores after loading
if __name__ == '__main__':
    # Load combined team boxscores
    combined_team_boxscores = pd.read_csv('combined_team_boxscores.csv')
    
    # Combine team rows into single rows per game
    combined_team_boxscores_processed = combine_team_boxscores(combined_team_boxscores)
    
    
    # Assuming df is your original dataframe

    # Mapping of original columns to the new column names
    column_mapping = {
    "PTS_HOME": "pts_home",
    "FGM_HOME": "fg_home",
    "FGA_HOME": "fg_att_home",
    "FG3M_HOME": "3pt_home",
    "FG3A_HOME": "3pt_att_home",
    "FTM_HOME": "ft_home",
    "FTA_HOME": "ft_att_home",
    "OREB_HOME": "oreb_home",
    "DREB_HOME": "dreb_home",
    "AST_HOME": "ast_home",
    "STL_HOME": "stl_home",
    "BLK_HOME": "blk_home",
    "TO_HOME": "to_home",
    "PF_HOME": "pf_home",
    "FGM_AWAY": "fg_away",
    "FGA_AWAY": "fg_att_away",
    "FG3M_AWAY": "3pt_away",
    "FG3A_AWAY": "3pt_att_away",
    "FTM_AWAY": "ft_away",
    "FTA_AWAY": "ft_att_away",
    "OREB_AWAY": "oreb_away",
    "DREB_AWAY": "dreb_away",
    "AST_AWAY": "ast_away",
    "STL_AWAY": "stl_away",
    "BLK_AWAY": "blk_away",
    "TO_AWAY": "to_away",
    "PF_AWAY": "pf_away",
    "PTS_AWAY": "pts_away",
    "GAME_DATE": "Date"
    }

    # Rename the columns based on the mapping
    combined_team_boxscores_processed.rename(columns=column_mapping, inplace=True)
    
    
    
    # Add columns for year, month, and day
    combined_team_boxscores_processed['Year'] = combined_team_boxscores_processed['Date'].apply(lambda x: int(x[:4]))  # Extracting the year
    combined_team_boxscores_processed['Month'] = combined_team_boxscores_processed['Date'].apply(lambda x: int(x[5:7]))  # Extracting the month
    combined_team_boxscores_processed['Day'] = combined_team_boxscores_processed['Date'].apply(lambda x: int(x[8:10]))  # Extracting the day

    # Now we need to map the TEAM_ID_HOME and TEAM_ID_AWAY to the full team names
    # Get all NBA teams and create a mapping from team ID to team name
    nba_teams = {team['id']: team['full_name'] for team in teams.get_teams()}

    # Map TEAM_ID_HOME and TEAM_ID_AWAY to the corresponding team names
    combined_team_boxscores_processed['home'] = combined_team_boxscores_processed['TEAM_ID_HOME'].map(nba_teams)
    combined_team_boxscores_processed['away'] = combined_team_boxscores_processed['TEAM_ID_AWAY'].map(nba_teams)
    
    # Save the processed team boxscores
    combined_team_boxscores_processed.to_csv('processed_team_boxscores.csv', index=False)
    print(f"Processed team boxscores saved to 'processed_team_boxscores.csv'")
    
    
    
    
    
    
    
    
    
    combined_player_boxscores = pd.read_csv('combined_player_boxscores.csv')
    
    column_mapping = {
    "PTS": "pts",
    "FGM": "fg",
    "FGA": "fg_att",
    "FG3M": "3pt",
    "FG3A": "3pt_att",
    "FTM": "ft",
    "FTA": "ft_att",
    "OREB": "oreb",
    "DREB": "dreb",
    "AST": "ast",
    "STL": "stl",
    "BLK": "blk",
    "TO": "to",
    "PF": "pf",
    "PLUS_MINUS": "+/-",
    "MIN": "mp",
    "GAME_DATE": "date",
    "PLAYER_NAME": "Player"
    }
    
    # Rename the columns based on the mapping
    combined_player_boxscores.rename(columns=column_mapping, inplace=True)
    
    
    # Now we need to map the TEAM_ID_HOME and TEAM_ID_AWAY to the full team names
    # Get all NBA teams and create a mapping from team ID to team name
    nba_teams = {team['id']: team['full_name'] for team in teams.get_teams()}

    # Map TEAM_ID_HOME and TEAM_ID_AWAY to the corresponding team names
    combined_player_boxscores['team'] = combined_player_boxscores['TEAM_ID'].map(nba_teams) 

    # Specify the columns where NaNs should be replaced
    columns_to_check = ['fg', 'fg_att', 'FG_PCT', '3pt', '3pt_att', 'FG3_PCT',
       'ft', 'ft_att', 'FT_PCT', 'oreb', 'dreb', 'REB', 'ast', 'stl', 'blk',
       'to', 'pf', 'pts', '+/-']

    # Apply the function to rows where 'MIN' == 0
    combined_player_boxscores.loc[combined_player_boxscores['mp'] == 0] = combined_player_boxscores[combined_player_boxscores['mp'] == 0].apply(replace_nan_based_on_comment, axis=1, columns=columns_to_check)
    
    # Apply the mapping to the 'START_POSITION' column
    combined_player_boxscores['START_POSITION'] = combined_player_boxscores['START_POSITION'].apply(map_start_position)
    
    columns_to_drop = ['GAME_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_CITY', 'PLAYER_ID',
       'NICKNAME', 'COMMENT']
    combined_player_boxscores = combined_player_boxscores.drop(columns=columns_to_drop)
    
    combined_player_boxscores = combined_player_boxscores[combined_player_boxscores['pts'] >= 0]
    #combined_player_boxscores = combined_player_boxscores[combined_player_boxscores['START_POSITION'] > 0]
    

    combined_player_boxscores.to_csv('processed_player_boxscores.csv', index=False)
    print(f"Processed team boxscores saved to 'processed_team_boxscores.csv'")
