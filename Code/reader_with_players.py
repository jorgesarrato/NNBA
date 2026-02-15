import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from time import time


import polars as pl

"""def sum_team_players_points_for_game(date, team, players_data):
    return players_data.filter(
        (players_data['date'] == date) & (players_data['team'] == team)
    )['pts'].sum()



def sum_team_players_points_for_game(date, team):
	return players_data[ (players_data['date'] == date) & (players_data['team'] == team)  ]
"""

"""real_home_pts = []
player_home_pts = []
real_away_pts = []
player_away_pts = []
for ii in range(35000, 35050):
	real_home_pts.append( np.array(data.iloc[ii][['q1_home', 'q2_home', 'q3_home', 'q4_home', 'OT_home']], dtype = int).sum())
	real_away_pts.append( np.array(data.iloc[ii][['q1_away', 'q2_away', 'q3_away', 'q4_away', 'OT_away']], dtype = int).sum())

	player_home_pts.append(sum_team_players_points_for_game(data.iloc[ii]['Date'], data.iloc[ii]['home']))
	player_away_pts.append(sum_team_players_points_for_game(data.iloc[ii]['Date'], data.iloc[ii]['away']))

	if (real_home_pts[ii-35000] != player_home_pts[ii-35000]) or (real_away_pts[ii-35000] != player_away_pts[ii-35000]):
		print(data.iloc[ii]['Date'], data.iloc[ii]['home'], data.iloc[ii]['away'])


plt.figure()
plt.plot(real_home_pts, player_home_pts, 'o', label = 'home')

plt.plot(real_away_pts, player_away_pts, 'o', label = 'away')

plt.legend()
plt.show()"""




nba_teams = ["Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets", "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets", "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers", "La Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat", "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks", "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76Ers", "Phoenix Suns", "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors", "Utah Jazz", "Washington Wizards"]

max_nba_len = max([len(x) for x in nba_teams])


# Mapping dictionary for correct team names
mapping = {
	'Knicks': 'New York Knicks',
	'Mavericks': 'Dallas Mavericks',
	'Lakers': 'Los Angeles Lakers',
	'Thunder': 'Oklahoma City Thunder',
	'Warriors': 'Golden State Warriors',
	'Wizards': 'Washington Wizards',
	'Hornets': 'Charlotte Hornets',
	'Magic': 'Orlando Magic',
	'Cavaliers': 'Cleveland Cavaliers',
	'Pacers': 'Indiana Pacers',
	'Timberwolves': 'Minnesota Timberwolves',
	'Spurs': 'San Antonio Spurs',
	'Suns': 'Phoenix Suns',
	'Kings': 'Sacramento Kings',
	'Trailblazers': 'Portland Trail Blazers',
	'NewJersey': 'Brooklyn Nets',
	'Heat': 'Miami Heat',
	'Bucks': 'Milwaukee Bucks',
	'Raptors': 'Toronto Raptors',
	'Hawks': 'Atlanta Hawks',
	'Pistons': 'Detroit Pistons',
	'Pelicans': 'New Orleans Pelicans',
	'Grizzlies': 'Memphis Grizzlies',
	'Nuggets': 'Denver Nuggets',
	'Rockets': 'Houston Rockets',
	'Celtics': 'Boston Celtics',
	'Jazz': 'Utah Jazz',
	'Clippers': 'La Clippers',
	'Bulls': 'Chicago Bulls',
	'Seventysixers': 'Philadelphia 76Ers',
	'Golden State': 'Golden State Warriors',
	'Oklahoma City': 'Oklahoma City Thunder',
	'LA Clippers': 'La Clippers'
}




class Reader_Polars:
	def __init__(self, path, player_path, latest_matches=5, latest_vs_matches=2, latest_other_matches=5, latest_player_dates = 5, predict_path=None):
		self.path = path
		self.player_path = player_path
		self.latest_matches = latest_matches
		self.latest_vs_matches = latest_vs_matches
		self.latest_other_matches = latest_other_matches
		self.latest_player_dates = latest_player_dates
		self.predict_path = predict_path
		self.teams = nba_teams
		self.current_game_fields = [
			"wins_home_athome", "losses_home_athome", "wins_home_away", "losses_home_away",
			"wins_away_athome", "losses_away_athome", "wins_away_away", "losses_away_away"
		]
		#self.current_game_fields = []
		self.other_game_fields_orig = [
			"q1_home", "q2_home", "q3_home", "q4_home", "OT_home", "q1_away", "q2_away", "q3_away",
			"q4_away", "OT_away", "fg_home","fg_att_home","3pt_home","3pt_att_home","ft_home","ft_att_home","oreb_home","dreb_home","ast_home","stl_home","blk_home","to_home",
			"pf_home","fg_away","fg_att_away","3pt_away","3pt_att_away","ft_away","ft_att_away","oreb_away","dreb_away","ast_away","stl_away","blk_away","to_away","pf_away",
			"wins_home_athome","losses_home_athome","wins_home_away","losses_home_away","wins_away_athome","losses_away_athome","wins_away_away","losses_away_away"
		]
		self.other_game_fields_inverted = [
			"q1_away", "q2_away", "q3_away", "q4_away", "OT_away", "q1_home", "q2_home", "q3_home",
			"q4_home", "OT_home", "fg_away","fg_att_away","3pt_away","3pt_att_away","ft_away","ft_att_away","oreb_away","dreb_away","ast_away","stl_away","blk_away","to_away",
			"pf_away","fg_home","fg_att_home","3pt_home","3pt_att_home","ft_home","ft_att_home","oreb_home","dreb_home","ast_home","stl_home","blk_home","to_home","pf_home",
			"wins_away_athome","losses_away_athome","wins_away_away","losses_away_away","wins_home_athome","losses_home_athome","wins_home_away","losses_home_away"
		]

		self.player_fields = ["age", "mp", "fg", "fga", "fg%", "3p", "3pa", "3p%", "ft", "fta", "ft%", "orb", "drb", "trb", "ast", "stl", "blk", "tov", "pf", "pts", "+/-"]

		self.other_game_fields = self.other_game_fields_orig
		self.N_current_game_fields = len(self.current_game_fields)
		self.N_other_game_fields = len(self.other_game_fields)
		self.N_player_fields = len(self.player_fields)
		self.N_players_per_team = 18

	def get_team_players(self, date, team):
		return self.players_data.filter(
			(self.players_data['date'] == date) & (self.players_data['team'] == team)
		)

	def get_players_past_results_for_all(self, date, players, Ndates, Nplayers):

		# Add placeholder players if the number of players is less than Nplayers
		players = players + [f"Placeholder_{i}" for i in range(Nplayers - len(players))]

		# Filter for the specific players and dates before the given date
		past_data = self.players_data.filter(
			(self.players_data['date'] < date) & (self.players_data['Player'].is_in(players))
		)


		# Drop unnecessary columns to prepare for transformation
		past_data = past_data.drop(['date', 'team', 'opp', 'Player', ''])

		# Initialize an empty list to store results for each player
		all_players_data = []

		for player in players:
			# Filter data for the specific player
			player_data = self.players_data.filter((self.players_data['Player'] == player) & (self.players_data['date'] < date))
			
			# Convert the filtered data to a NumPy array
			player_data_array = np.array(player_data.drop(['date', 'team', 'opp', 'Player', '']).to_numpy())

			# Check if rows are fewer than Ndates or no data is found
			if player_data_array.shape[0] == 0 or player.startswith("Placeholder"):
				# If no data is found or this is a placeholder player, fill with zeros
				player_past_data = np.zeros((Ndates, past_data.width))
			else:
				# Get the last Ndates rows for the player
				player_past_data = player_data_array[-Ndates:]
				
				# Add zero rows if fewer than Ndates rows are found
				if player_past_data.shape[0] < Ndates:
					missing_rows = Ndates - player_past_data.shape[0]
					zero_rows = np.zeros((missing_rows, player_past_data.shape[1]))
					player_past_data = np.vstack((zero_rows, player_past_data))
		
			# Append the player's data to the results
			all_players_data.append(player_past_data)

		# Convert the list of arrays to a single NumPy array with shape (Ndates, Nplayers, Ndimensions)
		all_players_data = np.stack(all_players_data[:Nplayers], axis=0)

		return all_players_data





	def make_dataset(self, only_new_games=False):
		# Load main data
		self.data = pl.read_csv(self.path)
		self.players_data = pl.read_csv(self.player_path)
		
		self.data = self.data.sort(["Year", "Month", "Day", "away"]).filter(
			(pl.col("home").is_in(self.teams)) & (pl.col("away").is_in(self.teams))
		)
		self.data = self.data.with_columns([
			pl.col(self.other_game_fields).cast(pl.Float64, strict=False)
		])
		self.data = self.data.with_columns([
			pl.col(['OverUnder', 'OddOver', 'OddUnder']).cast(pl.Float64, strict=False)
		])
			
		self.data = self.data.drop_nulls(subset=self.other_game_fields)

		# Load additional data
		self.data2 = pl.read_json("/home/jsarrato/PersonalProjects/NBBA/Data/sportsbook/nba_archive_10Y.json")
		self.data2 = self.data2.with_columns([
			pl.col("home_team").map_elements(lambda x: mapping.get(x, x)).alias("home"),
			pl.col("away_team").map_elements(lambda x: mapping.get(x, x)).alias("away"),
			pl.col("date").map_elements(lambda x: int(str(x)[:4]), return_dtype=pl.Int64).alias("Year"),
			pl.col("date").map_elements(lambda x: int(str(x)[4:6]), return_dtype=pl.Int64).alias("Month"),
			pl.col("date").map_elements(lambda x: int(str(x)[6:8]), return_dtype=pl.Int64).alias("Day")
		])

		self.data = self.data.join(
			self.data2.select(["Year", "Month", "Day", "home", "away", "home_close_ml", "away_close_ml"]),
			on=["Year", "Month", "Day", "home", "away"],
			how="left"
		)
		
		self.data = self.data.with_columns(pl.col("HomeML").cast(pl.Int64))
		self.data = self.data.with_columns(pl.col("AwayML").cast(pl.Int64))
		
		
		self.data = self.data.with_columns(pl.min_horizontal(["HomeML", "home_close_ml"]).alias("H_ML"))
		self.data = self.data.with_columns(pl.min_horizontal(["AwayML", "away_close_ml"]).alias("A_ML"))
		
		
		self.data = self.data.with_columns(
		    pl.when((pl.col("q1_home") + pl.col("q2_home") + pl.col("q3_home") + pl.col("q4_home") + pl.col("OT_home")) > ( pl.col("q1_away") + pl.col("q2_away") + pl.col("q3_away") + pl.col("q4_away") + pl.col("OT_away") ))
		    .then(pl.col("wins_home_athome") - 1)
		    .otherwise(pl.col("wins_home_athome"))
		    .alias("wins_home_athome")  # Overwrite the existing "A" column
		)


		self.data = self.data.with_columns(
		    pl.when((pl.col("q1_home") + pl.col("q2_home") + pl.col("q3_home") + pl.col("q4_home") + pl.col("OT_home")) > ( pl.col("q1_away") + pl.col("q2_away") + pl.col("q3_away") + pl.col("q4_away") + pl.col("OT_away") ))
		    .then(pl.col("losses_home_athome"))
		    .otherwise(pl.col("losses_home_athome") - 1)
		    .alias("losses_home_athome")  # Overwrite the existing "A" column
		)


		self.data = self.data.with_columns(
		    pl.when((pl.col("q1_home") + pl.col("q2_home") + pl.col("q3_home") + pl.col("q4_home") + pl.col("OT_home")) > ( pl.col("q1_away") + pl.col("q2_away") + pl.col("q3_away") + pl.col("q4_away") + pl.col("OT_away") ))
		    .then(pl.col("wins_away_away"))
		    .otherwise(pl.col("wins_away_away") - 1)
		    .alias("wins_away_away")  # Overwrite the existing "A" column
		)


		self.data = self.data.with_columns(
		    pl.when((pl.col("q1_home") + pl.col("q2_home") + pl.col("q3_home") + pl.col("q4_home") + pl.col("OT_home")) > ( pl.col("q1_away") + pl.col("q2_away") + pl.col("q3_away") + pl.col("q4_away") + pl.col("OT_away") ))
		    .then(pl.col("losses_away_away") - 1)
		    .otherwise(pl.col("losses_away_away"))
		    .alias("losses_away_away")  # Overwrite the existing "A" column
		)
		

		if isinstance(self.predict_path, str):
			self.predict_data = pl.read_csv(self.predict_path)
			
			self.predict_data = self.predict_data.sort(["Year", "Month", "Day", "away"]).filter(
				(pl.col("home").is_in(self.teams)) & (pl.col("away").is_in(self.teams))
			)
			
			self.predict_data = self.predict_data.with_columns([
				pl.col(self.other_game_fields).cast(pl.Float64, strict=False)
			])
			
			self.predict_data = self.predict_data.with_columns([
				pl.col(['OverUnder', 'OddOver', 'OddUnder']).cast(pl.Float64, strict=False)
			])
			
			self.predict_data = self.predict_data.with_columns(pl.col("HomeML").cast(pl.Int64))
			self.predict_data = self.predict_data.with_columns(pl.col("AwayML").cast(pl.Int64))
			
			self.predict_data = self.predict_data.with_columns(pl.col("HomeML").alias("home_close_ml"))
			self.predict_data = self.predict_data.with_columns(pl.col("AwayML").alias("away_close_ml"))
		
			self.predict_data = self.predict_data.with_columns(pl.min_horizontal(["HomeML", "home_close_ml"]).alias("H_ML"))
			self.predict_data = self.predict_data.with_columns(pl.min_horizontal(["AwayML", "away_close_ml"]).alias("A_ML"))
		
		
			self.data = pl.concat([self.data, self.predict_data], how="vertical")

		first_el = len(self.data) - len(self.predict_data) if only_new_games else 0

		dataset_current_game = np.zeros((
			len(self.data),
			2 + self.N_current_game_fields
		))

		dataset_old_games_home = np.zeros((
			len(self.data),
			self.latest_matches,
			3 + self.N_other_game_fields
		))

		dataset_old_games_away = np.zeros((
			len(self.data),
			self.latest_matches,
			3 + self.N_other_game_fields
		))

		dataset_old_games_vs = np.zeros((
			len(self.data),
			self.latest_vs_matches,
			2 + self.N_other_game_fields
		))

		dataset_old_games_other = np.zeros((
			len(self.data),
			self.latest_vs_matches,
			self.latest_other_matches,
			3 + self.N_other_game_fields
		))

		dataset_players_home = np.zeros((
			len(self.data),
			self.N_players_per_team,
			self.latest_player_dates,
			self.N_player_fields
		))

		dataset_players_away = np.zeros((
			len(self.data),
			self.N_players_per_team,
			self.latest_player_dates,
			self.N_player_fields
		))

		labels = np.zeros((len(self.data),))
		bet_pred_spread = np.zeros((len(self.data), 2))
		bet_pred_totals = np.zeros((len(self.data), 3))
		dates = np.zeros((len(self.data)))
		home_teams = np.zeros((len(self.data)), dtype = '<U' + str(max_nba_len))
		away_teams = np.zeros((len(self.data)), dtype = '<U' + str(max_nba_len))
		

		bad_rows = []

		field_indices = {field: self.data.columns.index(field) for field in self.data.columns}

		print("Loading games into dataset...")
		for ii in tqdm(range(first_el, len(self.data))):

			row = self.data.row(ii)
			labels[ii] = sum(row[field_indices[q]] for q in ["q1_home", "q2_home", "q3_home", "q4_home", "OT_home"]) - \
						 sum(row[field_indices[q]] for q in ["q1_away", "q2_away", "q3_away", "q4_away", "OT_away"])

			dates[ii] = 10000 * row[field_indices["Year"]] + 100 * row[field_indices["Month"]] + row[field_indices["Day"]]
			current_date = datetime(row[field_indices["Year"]], row[field_indices["Month"]], row[field_indices["Day"]])

			h_team = row[field_indices["home"]]
			a_team = row[field_indices["away"]]
			dataset_current_game[ii, 0] = self.teams.index(h_team)
			dataset_current_game[ii, 1] = self.teams.index(a_team)
			home_teams[ii] = h_team
			away_teams[ii] = a_team
						
			for kk in range(self.N_current_game_fields):
				dataset_current_game[ii, 2 + kk] = row[field_indices[self.current_game_fields[kk]]]


			prev_data = self.data.slice(0, ii)
			latest_home = prev_data.filter((prev_data["home"] == h_team) | (prev_data["away"] == h_team)).tail(self.latest_matches)
			latest_away = prev_data.filter((prev_data["home"] == a_team) | (prev_data["away"] == a_team)).tail(self.latest_matches)
			latest_vs = prev_data.filter(
				((prev_data["home"] == h_team) | (prev_data["away"] == h_team)) &
				((prev_data["home"] == a_team) | (prev_data["away"] == a_team))
			).tail(self.latest_vs_matches)
			latest_other = prev_data.filter(
				((prev_data["home"] != h_team) & (prev_data["away"] != h_team)) &
				((prev_data["home"] != a_team) & (prev_data["away"] != a_team))
			).tail(self.latest_other_matches)

			if len(latest_home) < self.latest_matches or len(latest_away) < self.latest_matches or \
			   len(latest_vs) < self.latest_vs_matches or len(latest_other) < self.latest_other_matches or \
				labels[ii] == 0:
				bad_rows.append(ii)
				continue


			# Similar logic for latest_home
			for jj in range(self.latest_matches):
				row2 = latest_home.row(jj)
				home_col_idx = field_indices["home"]
				year_col_idx = field_indices["Year"]
				month_col_idx = field_indices["Month"]
				day_col_idx = field_indices["Day"]

				if row2[home_col_idx] == h_team:
					home_flag = 1
					team_str = "home"
					other_str = "away"
					self.other_game_fields = self.other_game_fields_orig
				else:
					home_flag = 0
					team_str = "away"
					other_str = "home"
					self.other_game_fields = self.other_game_fields_inverted

				dataset_old_games_home[ii, jj, 0] = home_flag

				# Date processing
				date = datetime(row2[year_col_idx], row2[month_col_idx], row2[day_col_idx])
				date_difference = current_date - date
				dataset_old_games_home[ii, jj, 1] = date_difference.days

				dataset_old_games_home[ii, jj, 2] = self.teams.index(row2[field_indices[other_str]])

				for kk in range(self.N_other_game_fields):
					dataset_old_games_home[ii, jj, 3 + kk] = row2[field_indices[self.other_game_fields[kk]]]


			# Similar logic for latest_away
			for jj in range(self.latest_matches):
				row2 = latest_away.row(jj)
				home_col_idx = field_indices["home"]
				year_col_idx = field_indices["Year"]
				month_col_idx = field_indices["Month"]
				day_col_idx = field_indices["Day"]

				if row2[home_col_idx] == a_team:
					home_flag = 1
					team_str = "home"
					other_str = "away"
					self.other_game_fields = self.other_game_fields_orig
				else:
					home_flag = 0
					team_str = "away"
					other_str = "home"
					self.other_game_fields = self.other_game_fields_inverted

				dataset_old_games_away[ii, jj, 0] = home_flag

				# Date processing
				date = datetime(row2[year_col_idx], row2[month_col_idx], row2[day_col_idx])
				date_difference = current_date - date
				dataset_old_games_away[ii, jj, 1] = date_difference.days

				dataset_old_games_away[ii, jj, 2] = self.teams.index(row2[field_indices[other_str]])

				for kk in range(self.N_other_game_fields):
					dataset_old_games_away[ii, jj, 3 + kk] = row2[field_indices[self.other_game_fields[kk]]]


			# Similar logic for latest_vs
			for jj in range(self.latest_vs_matches):
				row2 = latest_vs.row(jj)
				home_col_idx = field_indices["home"]
				year_col_idx = field_indices["Year"]
				month_col_idx = field_indices["Month"]
				day_col_idx = field_indices["Day"]

				if row2[home_col_idx] == h_team:
					home_flag = 1
					team_str = "home"
					other_str = "away"
					self.other_game_fields = self.other_game_fields_orig
				else:
					home_flag = 0
					team_str = "away"
					other_str = "home"
					self.other_game_fields = self.other_game_fields_inverted

				dataset_old_games_vs[ii, jj, 0] = home_flag

				# Date processing
				date = datetime(row2[year_col_idx], row2[month_col_idx], row2[day_col_idx])
				date_difference = current_date - date
				dataset_old_games_vs[ii, jj, 1] = date_difference.days

				for kk in range(self.N_other_game_fields):
					dataset_old_games_vs[ii, jj, 2 + kk] = row2[field_indices[self.other_game_fields[kk]]]




			# Similar logic for latest_other
			for jj in range(self.latest_other_matches):
				row2 = latest_other.row(jj)
				year_col_idx = field_indices["Year"]
				month_col_idx = field_indices["Month"]
				day_col_idx = field_indices["Day"]

				# Date processing
				date = datetime(row2[year_col_idx], row2[month_col_idx], row2[day_col_idx])
				date_difference = current_date - date
				dataset_old_games_other[ii, jj, 0] = date_difference.days
				dataset_old_games_other[ii, jj, 1] = self.teams.index(row2[field_indices["home"]])
				dataset_old_games_other[ii, jj, 2] = self.teams.index(row2[field_indices["away"]])

				for kk in range(self.N_other_game_fields):
					dataset_old_games_other[ii, jj, 3 + kk] = row2[field_indices[self.other_game_fields[kk]]]
					
			if self.latest_player_dates > 0:


				players_home = self.get_team_players(row[field_indices['Date']], row[field_indices['home']])
				dataset_players_home[ii,:,:,:] = self.get_players_past_results_for_all(row[field_indices['Date']], list(players_home['Player']), self.latest_player_dates, self.N_players_per_team)
						
				players_away = self.get_team_players(row[field_indices['Date']], row[field_indices['away']])
				dataset_players_away[ii,:,:,:] = self.get_players_past_results_for_all(row[field_indices['Date']], list(players_away['Player']), self.latest_player_dates, self.N_players_per_team)

					

			# Perform the operation using the safe_min function
			bet_pred_spread[ii, :] = [row[field_indices['H_ML']], row[field_indices['A_ML']]]
			bet_pred_totals[ii, :] = [row[field_indices['OverUnder']], row[field_indices['OddOver']], row[field_indices['OddUnder']]]

		arrays = [dataset_current_game, dataset_old_games_home, dataset_old_games_away, dataset_old_games_vs, dataset_players_home, dataset_players_away, labels, bet_pred_spread, bet_pred_totals, dates, home_teams, away_teams]
		arrays = [np.delete(arr, bad_rows, axis=0) for arr in arrays]

		# Unpack the arrays back if needed
		dataset_current_game, dataset_old_games_home, dataset_old_games_away, dataset_old_games_vs, dataset_players_home, dataset_players_away, labels, bet_pred_spread, bet_pred_totals, dates, home_teams, away_teams = arrays

		return dataset_current_game, dataset_old_games_home, dataset_old_games_away, dataset_old_games_vs, dataset_players_home, dataset_players_away, labels, bet_pred_spread, bet_pred_totals, dates, home_teams, away_teams




