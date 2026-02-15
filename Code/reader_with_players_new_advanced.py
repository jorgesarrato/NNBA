import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from time import time

import random
import polars as pl
from polars import lit, when, col

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




nba_teams = ["Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets", "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets", "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers", "Los Angeles Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat", "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks", "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers", "Phoenix Suns", "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors", "Utah Jazz", "Washington Wizards"]

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
		self.current_game_fields = ["wins_at_home_HOME", "wins_away_HOME", "losses_at_home_HOME", "losses_away_HOME","wins_at_home_AWAY", "wins_away_AWAY", "losses_at_home_AWAY", "losses_away_AWAY"]
		#self.current_game_fields = []
		self.other_game_fields_orig = ["MIN","fg_home", "fg_att_home", "3pt_home", "3pt_att_home", "ft_home", "ft_att_home", "oreb_home", "dreb_home", "ast_home", "stl_home", "blk_home", "to_home", "pf_home", "pts_home", "plus_minus_home", "pts_qtr1_home", "pts_qtr2_home", "pts_qtr3_home", "pts_qtr4_home", "pts_ot1_home", "pts_ot2_home", "pts_ot3_home", "pts_ot4_home", "pts_paint_home", "pts_2nd_chance_home", "pts_fb_home", "largest_lead_home", "LEAD_CHANGES_HOME", "TIMES_TIED_HOME", "pts_off_to_home", "pct_pts_2pt_mr_home", "pct_pts_fb_home", "pct_pts_off_tov_home", "pct_pts_paint_home", "pct_ast_2pm_home", "pct_uast_2pm_home", "pct_ast_3pm_home", "pct_uast_3pm_home", "pct_ast_fgm_home", "pct_uast_fgm_home", "possessions_home", "wins_at_home_HOME", "wins_away_HOME", "losses_at_home_HOME", "losses_away_HOME", "fg_away", "fg_att_away", "3pt_away", "3pt_att_away", "ft_away", "ft_att_away", "oreb_away", "dreb_away", "ast_away", "stl_away", "blk_away", "to_away", "pf_away", "pts_away", "plus_minus_away", "pts_qtr1_away", "pts_qtr2_away", "pts_qtr3_away", "pts_qtr4_away", "pts_ot1_away", "pts_ot2_away", "pts_ot3_away", "pts_ot4_away", "pts_paint_away", "pts_2nd_chance_away", "pts_fb_away", "largest_lead_away", "LEAD_CHANGES_AWAY", "TIMES_TIED_AWAY", "pts_off_to_away", "pct_pts_2pt_mr_away", "pct_pts_fb_away", "pct_pts_off_tov_away", "pct_pts_paint_away", "pct_ast_2pm_away", "pct_uast_2pm_away", "pct_ast_3pm_away", "pct_uast_3pm_away", "pct_ast_fgm_away", "pct_uast_fgm_away", "possessions_away", "wins_at_home_AWAY", "wins_away_AWAY", "losses_at_home_AWAY", "losses_away_AWAY"]

		self.other_game_fields_inverted = ["MIN", "fg_away", "fg_att_away", "3pt_away", "3pt_att_away", "ft_away", "ft_att_away", "oreb_away", "dreb_away", "ast_away", "stl_away", "blk_away", "to_away", "pf_away", "pts_away", "plus_minus_away", "pts_qtr1_away", "pts_qtr2_away", "pts_qtr3_away", "pts_qtr4_away", "pts_ot1_away", "pts_ot2_away", "pts_ot3_away", "pts_ot4_away", "pts_paint_away", "pts_2nd_chance_away", "pts_fb_away", "largest_lead_away", "LEAD_CHANGES_AWAY", "TIMES_TIED_AWAY", "pts_off_to_away", "pct_pts_2pt_mr_away", "pct_pts_fb_away", "pct_pts_off_tov_away", "pct_pts_paint_away", "pct_ast_2pm_away", "pct_uast_2pm_away", "pct_ast_3pm_away", "pct_uast_3pm_away", "pct_ast_fgm_away", "pct_uast_fgm_away", "possessions_away", "wins_at_home_AWAY", "wins_away_AWAY", "losses_at_home_AWAY", "losses_away_AWAY", "fg_home", "fg_att_home", "3pt_home", "3pt_att_home", "ft_home", "ft_att_home", "oreb_home", "dreb_home", "ast_home", "stl_home", "blk_home", "to_home", "pf_home", "pts_home", "plus_minus_home", "pts_qtr1_home", "pts_qtr2_home", "pts_qtr3_home", "pts_qtr4_home", "pts_ot1_home", "pts_ot2_home", "pts_ot3_home", "pts_ot4_home", "pts_paint_home", "pts_2nd_chance_home", "pts_fb_home", "largest_lead_home", "LEAD_CHANGES_HOME", "TIMES_TIED_HOME", "pts_off_to_home", "pct_pts_2pt_mr_home", "pct_pts_fb_home", "pct_pts_off_tov_home", "pct_pts_paint_home", "pct_ast_2pm_home", "pct_uast_2pm_home", "pct_ast_3pm_home", "pct_uast_3pm_home", "pct_ast_fgm_home", "pct_uast_fgm_home", "possessions_home", "wins_at_home_HOME", "wins_away_HOME", "losses_at_home_HOME", "losses_away_HOME"]

		self.player_fields = ['mp', 'fg', 'fg_att', '3pt', '3pt_att', 'ft', 'ft_att', 'FT_PCT', 'oreb', 'dreb',
		       'ast', 'stl', 'blk', 'to', 'pf', 'pts', '+/-', 'date', 'pct_pts_2pt_mr',
		       'pct_pts_fb', 'pct_pts_off_tov', 'pct_pts_paint', 'pct_ast_2pm',
		       'pct_uast_2pm', 'pct_ast_3pm', 'pct_uast_3pm', 'pct_ast_fgm',
		       'pct_uast_fgm', 'pts_2nd_chance', 'possessions', "START_POSITION", 'days_since_date']

		self.other_game_fields = self.other_game_fields_orig
		self.N_current_game_fields = len(self.current_game_fields)
		self.N_other_game_fields = len(self.other_game_fields)
		self.N_player_fields = len(self.player_fields)
		self.N_players_per_team = 20

	def get_team_players(self, date, team):
		players = list(self.players_data.filter(
			(self.players_data['date'] == date) & (self.players_data['team'] == team)
		)['Player'])
		
		return players
		
	def get_nba_season(date_str):
		# Parse the date string
		date = datetime.strptime(date_str, "%Y-%m-%d")
		year = date.year
		month = date.month

		# Determine the season
		if month >= 10:  # October, November, December
			season = f"{year}-{(year + 1) % 100:02d}"
		else:  # January through September
			season = f"{year-1}-{year % 100:02d}"

		return season
		
	def get_all_team_players_in_season(self, date, team):
		return self.players_data.filter(
			(self.players_data['date'] == date) & (self.players_data['team'] == team)
		).sort("START_POSITION", descending = True  )[:self.N_players_per_team]

	def get_players_past_results_for_all(self, date, players, Ndates, Nplayers):

		# Add placeholder players if the number of players is less than Nplayers
		players = players + [f"Placeholder_{i}" for i in range(Nplayers - len(players))]

		# Filter for the specific players and dates before the given date
		past_data = self.players_data.filter(
			(self.players_data['date'] < date) & (self.players_data['Player'].is_in(players))
		)


		# Initialize an empty list to store results for each player
		all_players_data = []

		for player in players:
			# Filter data for the specific player
			if not player.startswith("Placeholder"):
				player_data = self.player_dict[player].filter(pl.col('date') < date)
				
				date_obj = datetime.strptime(date, '%Y-%m-%d')
				
				
				player_data = player_data.with_columns(
				(pl.col('date').str.strptime(pl.Date, format='%Y-%m-%d')  # Convert 'date' to a Polars Date type
				.cast(pl.Date)  # Ensure 'date' is a Polars Date column
				.alias('date')
				)
				)
				
				player_data = player_data.with_columns(
				    (pl.lit(date_obj).cast(pl.Datetime) - pl.col('date').cast(pl.Datetime)).dt.total_days().alias('days_since_date')
				)
								
				# Convert the filtered data to a NumPy array
				player_data_array = np.array(player_data.select(self.player_fields).to_numpy())

			# Check if rows are fewer than Ndates or no data is found
			if player_data_array.shape[0] == 0 or player.startswith("Placeholder"):
				# If no data is found or this is a placeholder player, fill with zeros
				player_past_data = np.zeros((Ndates, self.N_player_fields))
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
		print('Loading DataFrames...')
		self.data = pl.read_csv(self.path)
		
		self.data = self.data.filter(pl.col('Date') >= "1996-11-01")
		
		#with open('/home/jsarrato/PersonalProjects/NBBA/Data/DataNew_advanced/rosters.pickle', 'rb') as handle:
		#	self.rosters = pickle.load(handle)
		
		
		self.players_data = pl.read_csv(self.player_path)
		
		self.players_data = self.players_data.filter(pl.col('date') >= "1996-11-01")
		
		self.player_dict = {p: self.players_data.filter(pl.col('Player') == p) for p in self.players_data['Player'].unique()}
				
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
			self.data2.select(["Year", "Month", "Day", "home", "away", "home_close_ml", "away_close_ml", "open_over_under", "home_close_spread"]),
			on=["Year", "Month", "Day", "home", "away"],
			how="left"
		)
		
		self.data = self.data.with_columns(pl.col("HomeML").cast(pl.Float64).cast(pl.Int64))
		self.data = self.data.with_columns(pl.col("AwayML").cast(pl.Float64).cast(pl.Int64))
		
		
		# For H_ML
		self.data = self.data.with_columns(
		    when((col("HomeML") != 0) & (col("HomeML").is_not_null()))
			.then(col("HomeML"))
			.when((col("home_close_ml") != 0) & (col("home_close_ml").is_not_null()))
			.then(col("home_close_ml"))
			.otherwise(lit(float('nan')))
			.alias("H_ML")
		)

		# For A_ML
		self.data = self.data.with_columns(
		    when((col("AwayML") != 0) & (col("AwayML").is_not_null()))
			.then(col("AwayML"))
			.when((col("away_close_ml") != 0) & (col("away_close_ml").is_not_null()))
			.then(col("away_close_ml"))
			.otherwise(lit(float('nan')))
			.alias("A_ML")
		)
		
		# For OverUnder
		self.data = self.data.with_columns(
		    when((col("OverUnder") != 0) & (col("OverUnder").is_not_null()))
			.then(col("OverUnder"))
			.when((col("open_over_under") != 0) & (col("open_over_under").is_not_null()))
			.then(col("open_over_under"))
			.otherwise(lit(float('nan')))
			.alias("OverUnder")
		)
		
		"""
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
		"""
		

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
			self.N_current_game_fields
		))

		dataset_old_games_home = np.zeros((
			len(self.data),
			self.latest_matches,
			2 + self.N_other_game_fields
		))

		dataset_old_games_away = np.zeros((
			len(self.data),
			self.latest_matches,
			2 + self.N_other_game_fields
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
			1 + self.N_other_game_fields
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
		labels_total = np.zeros((len(self.data),))
		bet_pred = np.zeros((len(self.data), 2))
		bet_pred_totals = np.zeros((len(self.data),))
		dates = np.zeros((len(self.data)))
		home_teams = np.zeros((len(self.data)), dtype = '<U' + str(max_nba_len))
		away_teams = np.zeros((len(self.data)), dtype = '<U' + str(max_nba_len))
		

		bad_rows = []

		field_indices = {field: self.data.columns.index(field) for field in self.data.columns}

		print("Loading games into dataset...")
		for ii in tqdm(range(first_el, len(self.data))):

			row = self.data.row(ii)
			labels[ii] = sum(row[field_indices[q]] for q in ["pts_home"]) - \
						 sum(row[field_indices[q]] for q in ["pts_away"])
						 
			labels_total[ii] = sum(row[field_indices[q]] for q in ["pts_home"]) + \
						 sum(row[field_indices[q]] for q in ["pts_away"])

			dates[ii] = 10000 * row[field_indices["Year"]] + 100 * row[field_indices["Month"]] + row[field_indices["Day"]]
			current_date = datetime(row[field_indices["Year"]], row[field_indices["Month"]], row[field_indices["Day"]])

			h_team = row[field_indices["home"]]
			a_team = row[field_indices["away"]]

			home_teams[ii] = h_team
			away_teams[ii] = a_team
						
			for kk in range(self.N_current_game_fields):
				dataset_current_game[ii, 0 + kk] = row[field_indices[self.current_game_fields[kk]]]


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

				for kk in range(self.N_other_game_fields):
					dataset_old_games_home[ii, jj, 2 + kk] = row2[field_indices[self.other_game_fields[kk]]]


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

				for kk in range(self.N_other_game_fields):
					dataset_old_games_away[ii, jj, 2 + kk] = row2[field_indices[self.other_game_fields[kk]]]


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

				for kk in range(self.N_other_game_fields):
					dataset_old_games_other[ii, jj, 1 + kk] = row2[field_indices[self.other_game_fields[kk]]]
					
			if self.latest_player_dates > 0:


				players_home = self.get_team_players(row[field_indices['Date']], row[field_indices['home']])
				dataset_players_home[ii,:,:,:] = self.get_players_past_results_for_all(row[field_indices['Date']], players_home, self.latest_player_dates, self.N_players_per_team)
						
				players_away = self.get_team_players(row[field_indices['Date']], row[field_indices['away']])
				dataset_players_away[ii,:,:,:] = self.get_players_past_results_for_all(row[field_indices['Date']], players_away, self.latest_player_dates, self.N_players_per_team)

					

			# Perform the operation using the safe_min function
			bet_pred[ii, :] = [row[field_indices['H_ML']], row[field_indices['A_ML']]]
			bet_pred_totals[ii] = row[field_indices['OverUnder']]

		arrays = [dataset_current_game, dataset_old_games_home, dataset_old_games_away, dataset_old_games_vs, dataset_players_home, dataset_players_away, labels, labels_total, bet_pred, bet_pred_totals, dates, home_teams, away_teams]
		arrays = [np.delete(arr, bad_rows, axis=0) for arr in arrays]

		# Unpack the arrays back if needed
		dataset_current_game, dataset_old_games_home, dataset_old_games_away, dataset_old_games_vs, dataset_players_home, dataset_players_away, labels, labels_total, bet_pred, bet_pred_totals, dates, home_teams, away_teams = arrays

		return dataset_current_game, dataset_old_games_home, dataset_old_games_away, dataset_old_games_vs, dataset_players_home, dataset_players_away, labels, labels_total, bet_pred, bet_pred_totals, dates, home_teams, away_teams




