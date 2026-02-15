from nba_api.stats.endpoints import boxscoreadvancedv3, scoreboardv2
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time
from nba_api.stats.endpoints import leaguegamefinder

from nba_api.stats.static import teams

headers_url = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9',
    'Origin': 'https://www.nba.com',
    'Referer': 'https://www.nba.com',
    'Connection': 'keep-alive'
}

nba_teams = teams.get_teams()
team_abbr_to_id = {team['abbreviation']: team['id'] for team in nba_teams}
all_games = pd.DataFrame()

for team in nba_teams:
    team_id = team['id']
    gamefinder = leaguegamefinder.LeagueGameFinder(team_id_nullable=team_id, headers = headers_url)
    games = gamefinder.get_data_frames()[0]
    all_games = pd.concat([all_games, games], ignore_index=True)


all_games = all_games.sort_values("GAME_DATE").drop_duplicates("GAME_ID").reset_index(drop = True)
all_games
all_game_IDs = all_games['GAME_ID']
all_game_DATEs = all_games['GAME_DATE']









# Initialize headers and data
headers = []
data = pd.DataFrame()

def fetch_game_data(game_DATE):
    try:
        # Fetch scoreboard data for the game date
        scoreboard_info = scoreboardv2.ScoreboardV2(game_date=game_DATE, headers=headers_url).get_data_frames()
        game_data = []
        
        # Iterate over games on the date
        for ii, game_ID in enumerate(scoreboard_info[0]["GAME_ID"]):
            try:
                # Fetch boxscore data for the game
                boxscore_info = boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=game_ID, headers=headers_url).get_data_frames()
                
                # Extract team and player data
                home_team = scoreboard_info[0].iloc[ii]["HOME_TEAM_ID"]
                visitor_team = scoreboard_info[0].iloc[ii]["VISITOR_TEAM_ID"]

                score_info_row_home = scoreboard_info[1][(scoreboard_info[1]["GAME_ID"] == game_ID) & (scoreboard_info[1]["TEAM_ID"] == home_team)]
                score_info_row_visitor = scoreboard_info[1][(scoreboard_info[1]["GAME_ID"] == game_ID) & (scoreboard_info[1]["TEAM_ID"] == visitor_team)]

                boxscore_info_row_home = boxscore_info[1][(boxscore_info[1]["gameId"] == game_ID) & (boxscore_info[1]["teamId"] == home_team)]
                boxscore_info_row_visitor = boxscore_info[1][(boxscore_info[1]["gameId"] == game_ID) & (boxscore_info[1]["teamId"] == visitor_team)]

                home_players = list(boxscore_info[0][boxscore_info[0]['teamId'] == home_team]["personId"])
                visitor_players = list(boxscore_info[0][boxscore_info[0]['teamId'] == visitor_team]["personId"])

                # Pad players list to 20 for consistency
                home_players = home_players + ['N/A'] * (20 - len(home_players))
                visitor_players = visitor_players + ['N/A'] * (20 - len(visitor_players))

                # Create combined data row
                all_info = [*score_info_row_home.iloc[0], *boxscore_info_row_home.iloc[0], *home_players,
                            *score_info_row_visitor.iloc[0], *boxscore_info_row_visitor.iloc[0], *visitor_players]
                
                game_data.append(all_info)
            except:
                continue

        return game_data
    except:
        return []

 # Replace with your list of game dates
all_data = []
with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(tqdm(executor.map(fetch_game_data, all_game_DATEs), total=len(all_game_DATEs)))

# Combine results into a single DataFrame
for result in results:
    all_data.extend(result)

# Create the DataFrame from the combined data
if all_data:
    data = pd.DataFrame(all_data, columns=data_columns)

# Save or process the data as needed
print(data.head())
