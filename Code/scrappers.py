from datetime import datetime, timedelta
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import csv
import os
from time import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

class NBAGameScraper:
    def __init__(self, csv_file, fetch_bet_data=True, max_workers=5):
        self.csv_file = csv_file
        self.fetch_bet_data = fetch_bet_data
        self.fields = ["Date", "Year", "Month", "Day", "away", "home",
                       "q1_home", "q2_home", "q3_home", "q4_home", "OT_home",
                       "q1_away", "q2_away", "q3_away", "q4_away", "OT_away",
                       "fg_home", "fg_att_home", "3pt_home", "3pt_att_home",
                       "ft_home", "ft_att_home", "oreb_home", "dreb_home",
                       "ast_home", "stl_home", "blk_home", "to_home", "pf_home",
                       "fg_away", "fg_att_away", "3pt_away", "3pt_att_away",
                       "ft_away", "ft_att_away", "oreb_away", "dreb_away",
                       "ast_away", "stl_away", "blk_away", "to_away", "pf_away",
                       "wins_home_athome", "losses_home_athome","wins_home_away", "losses_home_away",
                       "wins_away_athome", "losses_away_athome","wins_away_away", "losses_away_away",
                       "away_bet", "home_bet", "OverUnder", "OddOver", "OddUnder", "HomeML", "AwayML", "id"]
        self.bad_dates = []
        self.session = requests.Session()
        self.headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
        'Origin': 'https://www.espn.com',
        'Referer': 'https://www.espn.com',
        'Connection': 'keep-alive'
        }
        self.max_workers = max_workers

    def fetch_game_data(self, current_date):
        url = f"https://www.espn.com/nba/scoreboard/_/date/{current_date.strftime('%Y%m%d')}"
        soup = BeautifulSoup(self.session.get(url, headers=self.headers).content, "html.parser")

        # Betting data (optional)
        if self.fetch_bet_data:
            url_bet = "https://site.web.api.espn.com/apis/v2/scoreboard/header"
            payload = {'sport': 'basketball', 'league': 'nba', 'region': 'us', 'dates': current_date.strftime("%Y%m%d")}
            bet_data = self.session.get(url_bet, headers=self.headers, params=payload).json()
            events = bet_data.get('sports', [{}])[0].get('leagues', [{}])[0].get('events', [])
            h_teams = [event.get('competitors', [{}])[0].get('displayName', "") for event in events]
            for ii in range(len(h_teams)):
                if h_teams[ii] == 'Philadelphia 76ers':
                    h_teams[ii] = 'Philadelphia 76Ers'
        else:
            events = []

        games_data = []

        for idx, scoreboard in enumerate(soup.select(".Scoreboard")):
            try:
                team_links = scoreboard.select(".ScoreboardScoreCell__Item a.AnchorLink")
                name_home = team_links[0]['href'].split("/")[-1].replace("-", " ").title()
                name_away = team_links[2]['href'].split("/")[-1].replace("-", " ").title()

                quarters_home = [q.get_text(strip=True) for q in scoreboard.select(".ScoreboardScoreCell__Item--home .ScoreboardScoreCell__Value")]
                quarters_away = [q.get_text(strip=True) for q in scoreboard.select(".ScoreboardScoreCell__Item--away .ScoreboardScoreCell__Value")]

                quarters_home += ["0"] * (5 - len(quarters_home))
                quarters_away += ["0"] * (5 - len(quarters_away))
                
                # Extract team records before the game
                record_away = scoreboard.select(".ScoreboardScoreCell__Item--away .ScoreboardScoreCell__Record")[0].get_text(strip=True)
                record_home = scoreboard.select(".ScoreboardScoreCell__Item--home .ScoreboardScoreCell__Record")[0].get_text(strip=True)
                
                
                
                record_away_away = scoreboard.select(".ScoreboardScoreCell__Item--away .ScoreboardScoreCell__Record")[1].get_text(strip=True)[:-4]
                record_home_athome = scoreboard.select(".ScoreboardScoreCell__Item--home .ScoreboardScoreCell__Record")[1].get_text(strip=True)[:-4]
                
                home_record_wins_athome, home_record_losses_athome = record_home_athome.split("-") # Home team wins and loses playing at home
                home_record_wins_away = str( int(record_home.split("-")[0]) - int(record_home_athome.split("-")[0]) ) # Home team wins playing away
                home_record_losses_away = str( int(record_home.split("-")[1]) - int(record_home_athome.split("-")[1]) ) # Home team losses playing away
                
                away_record_wins_away, away_record_losses_away = record_away_away.split("-") # Away team wins and loses playing away
                away_record_wins_athome = str( int(record_away.split("-")[0]) - int(record_away_away.split("-")[0]) ) # Home team wins playing away
                away_record_losses_athome = str( int(record_away.split("-")[1]) - int(record_away_away.split("-")[1]) ) # Home team losses playing away
                
                
                # Box score data
                game_id = scoreboard["id"]
                box_score_link = f"https://www.espn.com/nba/boxscore/_/gameId/{game_id}"
                box_score_soup = BeautifulSoup(self.session.get(box_score_link, headers=self.headers).content, "html.parser")
                boxscore_info = self.extract_boxscore_data(box_score_soup)

                # Betting data (if available)
                if self.fetch_bet_data:
                    idx_bet = h_teams.index(name_home)
                    event = events[idx_bet]
                    odds = event.get("odds", {})
                    over_under = odds.get("overUnder", "")
                    odd_over = odds.get("overOdds", "")
                    odd_under = odds.get("underOdds", "")
                    try:
                        odd_away_ML = odds.get('away')['moneyLine']
                        odd_home_ML = odds.get('home')['moneyLine']
                    except:
                        odd_away_ML = ""
                        odd_home_ML = ""
                    home_bet = event.get('competitors', [{}])[0].get('displayName', "")
                    away_bet = event.get('competitors', [{}])[1].get('displayName', "")
                else:
                    over_under = odd_over = odd_under = home_bet = away_bet = odd_away_ML = odd_home_ML = ""

                game_data = [
                    current_date.strftime("%Y-%m-%d"), current_date.year, current_date.month, current_date.day,
                    name_home, name_away, *quarters_home, *quarters_away,
                    *boxscore_info[len(boxscore_info)//2:],
                    *boxscore_info[:len(boxscore_info)//2],
                    home_record_wins_athome, home_record_losses_athome, home_record_wins_away, home_record_losses_away,
                    away_record_wins_athome, away_record_losses_athome, away_record_wins_away, away_record_losses_away,
                    home_bet, away_bet, over_under, odd_over, odd_under, odd_home_ML, odd_away_ML, game_id
                ]
                if len(game_data) != 58:
                    continue
                games_data.append(game_data)
            except Exception:
                self.bad_dates.append(current_date.strftime("%Y%m%d"))
                continue

        return games_data

    def extract_boxscore_data(self, soup):
        boxscore_info = []
    
        try:
            rows = soup.select("tr")
            for row in rows:
                columns = row.select(".Table__TD")
                
                # Ensure there are enough columns to avoid IndexError
                if len(columns) < 3:
                    continue
                
                # Check if first column is a player name and skip if so
                player_name = columns[0].text.strip()
                if len(player_name.replace(" ", "")) > 0:
                    continue
                
                # Check if the third column (three-point data) contains a percentage and skip if so
                three_pointers = columns[2].text.strip()
                if "%" in three_pointers:
                    continue
    
                # Loop over specified columns, handling values as needed
                for ii in range(1, 12):
                    if ii == 6:
                        continue  # Skip the 6th column as per original logic
                    elif ii <= 3:
                        # Add two parts if data is split by a dash (e.g., "2-5")
                        split_data = columns[ii].text.strip().split("-")
                        if len(split_data) == 2:
                            boxscore_info.extend(split_data)
                        else:
                            boxscore_info.append(split_data[0])  # Handle missing dash case
                    else:
                        boxscore_info.append(columns[ii].text.strip())
                    
        except IndexError as e:
            print(f"IndexError: {e} - Check column count in rows, especially for 'columns[{ii}]'")
        except AttributeError as e:
            print(f"AttributeError: {e} - Ensure BeautifulSoup object structure is as expected")
        except Exception as e:
            print(f"Unexpected error: {e} - Type: {type(e).__name__}")

        # Ensure the returned list has a length of at least 26 items
        return boxscore_info + ["N/A"] * (26 - len(boxscore_info))


    def remove_existing_rows(self):
        if not os.path.exists(self.csv_file):
            return
        updated_rows = []
        with open(self.csv_file, mode="r", newline="") as file:
            reader = csv.reader(file)
            headers = next(reader)
            for row in reader:
                row_date = row[0]
                if not (self.start_date.strftime("%Y-%m-%d") <= row_date <= self.end_date.strftime("%Y-%m-%d")):
                    updated_rows.append(row)
        with open(self.csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(updated_rows)

    def save_to_csv(self, games_data, append=False):
        mode = "a" if append and os.path.exists(self.csv_file) else "w"
        with open(self.csv_file, mode=mode, newline="") as file:
            writer = csv.writer(file)
            if mode == "w":
                writer.writerow(self.fields)
            writer.writerows(games_data)

    def scrape(self, start_date, end_date, append=False):
        self.start_date = start_date
        self.end_date = end_date
        if append:
            self.remove_existing_rows()

        all_games_data = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.fetch_game_data, self.start_date + timedelta(days=i)): i for i in range((self.end_date - self.start_date).days)}
            for future in tqdm(as_completed(futures), total=len(futures)):
                all_games_data.extend(future.result())

        self.save_to_csv(all_games_data, append=append)
        print(f"Data has been written to {self.csv_file}")
        if self.bad_dates:
            print("No data for dates:", ", ".join(self.bad_dates))
            
    def update(self, delta = 0):
        data = pd.read_csv(self.csv_file)
        data = data.sort_values(['Year', 'Month', 'Day', 'away']).reset_index(drop = True)

        last_existing_date = datetime(data.iloc[-1]["Year"], data.iloc[-1]["Month"], data.iloc[-1]["Day"])
        current_date = datetime.now() - timedelta(days = delta)
        self.scrape(last_existing_date, current_date, append = True)


if __name__ == '__main__':
	import sys
	# Example usage:
	start_date = datetime(int(sys.argv[1]),1,1)
	end_date = datetime(int(sys.argv[1]),12,31)
	csv_file = "/home/jsarrato/PersonalProjects/NBBA/Data/Complete_data/nba_games_"+str(sys.argv[1])+".csv"

	scraper = NBAGameScraper(csv_file, fetch_bet_data=True)
	scraper.scrape(start_date, end_date, append=False)





