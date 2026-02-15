import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, playbyplayv3, playbyplayv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os



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
        playbyplay2 = playbyplayv2.PlayByPlayV2(game_id, headers = headers, timeout = 120).get_data_frames()[0]
        playbyplay3 = playbyplayv3.PlayByPlayV3(game_id, headers = headers, timeout = 120).get_data_frames()[0]

        playbyplay2.to_csv(folder + '/' + game_id + '-v2.csv', index = False)
        playbyplay3.to_csv(folder + '/' + game_id + '-v3.csv', index = False)
        

        
        return playbyplay2
    except Exception as e:
        print(f"Error fetching boxscore for GAME_ID {game_id}: {e}")
        return pd.DataFrame()



# Fetch all player and team boxscores in parallel
def fetch_all_boxscores_parallel(game_ids, games_df, max_workers=10):
    """
    Fetch player and team boxscores for all game IDs in parallel with timeout handling.
    """

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_boxscores, game_id, games_df): game_id for game_id in game_ids}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching Boxscores"):
            _ = future.result()

    return

# Main script for multiple seasons
if __name__ == '__main__':
    # Define seasons
    start_year = 2010
    end_year = 2020
    seasons = [f"{year}-{str(year+1)[-2:]}" for year in range(start_year, end_year + 1)]

    for season in seasons:

        folder = '/home/jsarrato/PersonalProjects/NBBA/Data/Play-By-Play/'+season

        if not os.path.exists(folder):
            os.makedirs(folder)


        print(f"Processing season: {season}")
        try:
            games_df = get_all_games_for_season(season)
            print(f"Retrieved {len(games_df)} games for the {season} regular season.")
            
            fetch_all_boxscores_parallel(games_df['GAME_ID'].tolist(), games_df)
            
            print(f"Saved boxscores for all games in the {season} season.")
        except Exception as e:
            print(f"Error processing season {season}: {e}")







