import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm


import polars as pl



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



class reader:
	def __init__(self, path, latest_matches = 5, latest_vs_matches = 2, latest_other_matches = 5, predict_path = None):
		self.path = path
		self.latest_matches = latest_matches
		self.latest_vs_matches = latest_vs_matches
		self.latest_other_matches = latest_other_matches
		self.predict_path = predict_path
		self.teams = ["Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets", "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets", "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers", "La Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat", "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks", "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76Ers", "Phoenix Suns", "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs", "Toronto Raptors", "Utah Jazz", "Washington Wizards"]
		self.current_game_fields = ["wins_home_athome","losses_home_athome","wins_home_away","losses_home_away","wins_away_athome","losses_away_athome","wins_away_away","losses_away_away"]
		self.other_game_fields_orig = ["q1_home","q2_home","q3_home","q4_home","OT_home","q1_away","q2_away","q3_away","q4_away","OT_away","fg_home","fg_att_home","3pt_home","3pt_att_home","ft_home","ft_att_home","oreb_home","dreb_home","ast_home","stl_home","blk_home","to_home","pf_home","fg_away","fg_att_away","3pt_away","3pt_att_away","ft_away","ft_att_away","oreb_away","dreb_away","ast_away","stl_away","blk_away","to_away","pf_away","wins_home_athome","losses_home_athome","wins_home_away","losses_home_away","wins_away_athome","losses_away_athome","wins_away_away","losses_away_away"]
		self.other_game_fields = self.other_game_fields_orig
		self.N_current_game_fields = len(self.current_game_fields)
		self.N_other_game_fields = len(self.other_game_fields)
	
	def make_dataset(self, only_new_games = False):
		self.data = pd.read_csv(self.path)
		#self.data = self.data[self.data["Year"]>2020].reset_index(drop = True)
		self.data = self.data.sort_values(['Year', 'Month', 'Day', 'away']).reset_index(drop = True)

		self.data = self.data[self.data["home"].isin(self.teams) & self.data["away"].isin(self.teams)].reset_index(drop = True)
		
		# Convert the specified columns to numeric, setting errors='coerce' will convert non-numeric values to NaN
		self.data[self.other_game_fields] = self.data[self.other_game_fields].apply(pd.to_numeric, errors='coerce')

		# Keep rows where these columns do not have NaN values (indicating they were all numeric)
		self.data = self.data.dropna(subset=self.other_game_fields).reset_index(drop = True)
		
		
		
		
		
		self.data2 = pd.read_json('/home/jsarrato/PersonalProjects/NBBA/Data/sportsbook/nba_archive_10Y.json')
		# Apply the mapping and drop rows where the value is 0
		self.data2['home_team'] = self.data2['home_team'].map(mapping).dropna()
		self.data2['away_team'] = self.data2['away_team'].map(mapping).dropna()

		self.data2['home'] = self.data2['home_team']
		self.data2['away'] = self.data2['away_team']
		self.data2["Year"] = [int(str(x)[:4]) for x in self.data2["date"]]
		self.data2["Month"] = [int(str(x)[4:6]) for x in self.data2["date"]]
		self.data2["Day"] = [int(str(x)[6:8]) for x in self.data2["date"]]
		
		self.data = self.data.merge(self.data2[['Year', 'Month', 'Day', 'home', 'away', 'home_close_ml', 'away_close_ml']], on = ['Year', 'Month', 'Day', 'home', 'away'], how = 'left')
		
		
		
		
		
		
		if isinstance(self.predict_path, str):
			self.predict_data = pd.read_csv(self.predict_path)
			self.predict_data = self.predict_data.sort_values(['away']).reset_index(drop = True)
			self.data = pd.concat((self.data, self.predict_data))
			
		if only_new_games:

			first_el = len(self.data)-len(self.predict_data)
		else:
			first_el = 0
			
		dataset = np.zeros((len(self.data),2 + self.N_current_game_fields + 2*(3 + self.N_other_game_fields)*self.latest_matches + (2 + self.N_other_game_fields)*self.latest_vs_matches + (3 + self.N_other_game_fields)*self.latest_other_matches))
		labels = np.zeros((len(self.data),))
		bet_pred = np.zeros((len(self.data), 2))
		dates = np.zeros((len(self.data)))
		home_teams = np.zeros((len(self.data)), dtype = '<U' + str(max_nba_len))
		away_teams = np.zeros((len(self.data)), dtype = '<U' + str(max_nba_len))
		
		bad_rows = []
		
		
		print("Loading games into dataset...")
		for ii in tqdm(range(first_el, len(self.data))):
			row = self.data.iloc[ii]
			labels[ii] = np.sum(row[["q1_home","q2_home","q3_home","q4_home","OT_home"]]) - np.sum(row[["q1_away","q2_away","q3_away","q4_away","OT_away"]])
			
			dates[ii] = 10000*row['Year']+100*row['Month']+row['Day']

				
			current_date = datetime(row["Year"], row["Month"], row["Day"])
			h_team = row['home']
			a_team = row['away']
			dataset[ii,0] = self.teams.index(h_team)
			dataset[ii,1] = self.teams.index(a_team)
			
			home_teams[ii] = h_team
			away_teams[ii] = a_team
			
			for kk in range(self.N_current_game_fields):
				dataset[ii,2 + kk] = row[self.current_game_fields[kk]]
				if kk == self.N_current_game_fields-1:
					start_index = 2 + kk


			prev_data = self.data.iloc[:ii]
			latest_home = prev_data[(prev_data['home']==h_team) | (prev_data['away']==h_team)].iloc[-self.latest_matches:].reset_index(drop = True)
			latest_away = prev_data[(prev_data['home']==a_team) | (prev_data['away']==a_team)].iloc[-self.latest_matches:].reset_index(drop = True)
			latest_vs = prev_data[((prev_data['home']==h_team) | (prev_data['away']==h_team)) & ((prev_data['home']==a_team) | (prev_data['away']==a_team))].iloc[-self.latest_vs_matches:].reset_index(drop = True)
			latest_other = prev_data[((prev_data['home']!=h_team) & (prev_data['away']!=h_team)) & ((prev_data['home']!=a_team) & (prev_data['away']!=a_team))].iloc[-self.latest_other_matches:].reset_index(drop = True)
			
			if (len(latest_home) < self.latest_matches) or (len(latest_away) < self.latest_matches) or (len(latest_vs) < self.latest_vs_matches) or (len(latest_other) < self.latest_other_matches):
				bad_rows.append(ii)
				continue
			
			for jj in range(self.latest_matches):
				row2 = latest_home.iloc[jj]
				if row2['home'] == h_team:
					home_flag = 1
					team_str = 'home'
					other_str = 'away'
					self.other_game_fields = self.other_game_fields_orig
				else:
					home_flag = 0
					team_str = 'away'
					other_str = 'home'
					self.other_game_fields = self.other_game_fields_inverted
					
				dataset[ii, start_index + 1 + jj*(3 + self.N_other_game_fields)] = home_flag
					
				date= datetime(row2["Year"], row2["Month"], row2["Day"])
				
				date_difference = current_date - date
				
				dataset[ii, start_index + 2 + jj*(3 + self.N_other_game_fields)] = date_difference.days
				dataset[ii, start_index + 3 + jj*(3 + self.N_other_game_fields)] = self.teams.index(row2[other_str])
				
				for kk in range(self.N_other_game_fields):
					dataset[ii, start_index + 4 + jj*(3 + self.N_other_game_fields) + kk] = row2[self.other_game_fields[kk]]
					if jj == self.latest_matches-1 and kk == self.N_other_game_fields-1:
						start_index = start_index + 4 + jj*(3 + self.N_other_game_fields) + kk
						
			
			for jj in range(self.latest_matches):
				row2 = latest_away.iloc[jj]
				if row2['home'] == a_team:
					home_flag = 1
					team_str = 'home'
					other_str = 'away'
					self.other_game_fields = self.other_game_fields_orig
				else:
					home_flag = 0
					team_str = 'away'
					other_str = 'home'
					self.other_game_fields = self.other_game_fields_inverted
					
				dataset[ii, start_index + 1 + jj*(3 + self.N_other_game_fields)] = home_flag
					
				date= datetime(row2["Year"], row2["Month"], row2["Day"])
				
				date_difference = current_date - date
				
				dataset[ii, start_index + 2 + jj*(3 + self.N_other_game_fields)] = date_difference.days
				dataset[ii, start_index + 3 + jj*(3 + self.N_other_game_fields)] = self.teams.index(row2[other_str])
				
				for kk in range(self.N_other_game_fields):
					dataset[ii, start_index + 4 + jj*(3 + self.N_other_game_fields) + kk] = row2[self.other_game_fields[kk]]
					if jj == self.latest_matches-1 and kk == self.N_other_game_fields-1:
						start_index = start_index + 4 + jj*(3 + self.N_other_game_fields) + kk

					
					
			for jj in range(self.latest_vs_matches):
				row2 = latest_vs.iloc[jj]
				if row2['home'] == h_team:
					home_flag = 1
					team_str = 'home'
					other_str = 'away'
					self.other_game_fields = self.other_game_fields_orig
				else:
					home_flag = 0
					team_str = 'away'
					other_str = 'home'
					self.other_game_fields = self.other_game_fields_inverted
					
				dataset[ii, start_index + 1 + jj*(2 + self.N_other_game_fields)] = home_flag
					
				date= datetime(row2["Year"], row2["Month"], row2["Day"])
				
				date_difference = current_date - date
				
				dataset[ii,  start_index + 2 + jj*(2 + self.N_other_game_fields)] = date_difference.days
				
				for kk in range(self.N_other_game_fields):
					dataset[ii, start_index + 3 + jj*(2 + self.N_other_game_fields) + kk] = row2[self.other_game_fields[kk]]
					if jj == self.latest_vs_matches-1 and kk == self.N_other_game_fields-1:
						start_index = start_index + 3 + jj*(2 + self.N_other_game_fields) + kk

	
					
			for jj in range(self.latest_other_matches):
				row2 = latest_other.iloc[jj]
					
				date= datetime(row2["Year"], row2["Month"], row2["Day"])
				
				date_difference = current_date - date
				
				dataset[ii, start_index + 1 + jj*(3 + self.N_other_game_fields)] = date_difference.days
				dataset[ii, start_index + 2 + jj*(3 + self.N_other_game_fields)] = self.teams.index(row2['home'])
				dataset[ii, start_index + 3 + jj*(3 + self.N_other_game_fields)] = self.teams.index(row2['away'])
				
				for kk in range(self.N_other_game_fields):
					dataset[ii, start_index + 4 + jj*(3 + self.N_other_game_fields) + kk] = row2[self.other_game_fields[kk]]
	

			bet_pred[ii, :] = [ np.nanmin([row['HomeML'], row['home_close_ml']]) , np.nanmin([row['AwayML'], row['away_close_ml']])]
			
		dataset = np.delete(dataset, bad_rows, axis=0)
		labels = np.delete(labels, bad_rows, axis=0)
		bet_pred = np.delete(bet_pred, bad_rows, axis=0)
		dates = np.delete(dates, bad_rows, axis=0)
		home_teams = np.delete(home_teams, bad_rows, axis=0)
		away_teams = np.delete(away_teams, bad_rows, axis=0)

		
		inputs = dataset
		return inputs, labels, bet_pred, dates, home_teams, away_teams











class Reader_Polars:
	def __init__(self, path, latest_matches=5, latest_vs_matches=2, latest_other_matches=5, predict_path=None):
		self.path = path
		self.latest_matches = latest_matches
		self.latest_vs_matches = latest_vs_matches
		self.latest_other_matches = latest_other_matches
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

		self.other_game_fields = self.other_game_fields_orig
		self.N_current_game_fields = len(self.current_game_fields)
		self.N_other_game_fields = len(self.other_game_fields)

	def make_dataset(self, only_new_games=False):
		def safe_min(value1, value2):
			# Replace None with np.nan
			value1 = np.nan if value1 is None else value1
			value2 = np.nan if value2 is None else value2
			return np.nanmin([value1, value2])
		
		# Load main data
		self.data = pl.read_csv(self.path)
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
		
		#self.data = self.data.with_columns(( pl.col("wins_home_athome") + pl.col("wins_home_away") ).alias("wins_home"))
		"""self.data = self.data.with_columns(
		    pl.when((pl.col("q1_home") + pl.col("q2_home") + pl.col("q3_home") + pl.col("q4_home") + pl.col("OT_home")) > ( pl.col("q1_away") + pl.col("q2_away") + pl.col("q3_away") + pl.col("q4_away") + pl.col("OT_away") ))
		    .then(pl.col("wins_home") - 1)
		    .otherwise(pl.col("wins_home"))
		    .alias("wins_home")  # Overwrite the existing "A" column
		)"""
		
		#self.data = self.data.with_columns(( pl.col("losses_home_athome") + pl.col("losses_home_away") ).alias("losses_home"))
		"""self.data = self.data.with_columns(
		    pl.when((pl.col("q1_home") + pl.col("q2_home") + pl.col("q3_home") + pl.col("q4_home") + pl.col("OT_home")) > ( pl.col("q1_away") + pl.col("q2_away") + pl.col("q3_away") + pl.col("q4_away") + pl.col("OT_away") ))
		    .then(pl.col("losses_home"))
		    .otherwise(pl.col("losses_home") - 1)
		    .alias("losses_home")  # Overwrite the existing "A" column
		)"""
		
		#self.data = self.data.with_columns(( pl.col("wins_away_athome") + pl.col("wins_away_away") ).alias("wins_away"))
		"""self.data = self.data.with_columns(
		    pl.when((pl.col("q1_home") + pl.col("q2_home") + pl.col("q3_home") + pl.col("q4_home") + pl.col("OT_home")) > ( pl.col("q1_away") + pl.col("q2_away") + pl.col("q3_away") + pl.col("q4_away") + pl.col("OT_away") ))
		    .then(pl.col("wins_away"))
		    .otherwise(pl.col("wins_away") - 1)
		    .alias("wins_away")  # Overwrite the existing "A" column
		)"""
		
		#self.data = self.data.with_columns(( pl.col("losses_away_athome") + pl.col("losses_away_away") ).alias("losses_away"))
		"""self.data = self.data.with_columns(
		    pl.when((pl.col("q1_home") + pl.col("q2_home") + pl.col("q3_home") + pl.col("q4_home") + pl.col("OT_home")) > ( pl.col("q1_away") + pl.col("q2_away") + pl.col("q3_away") + pl.col("q4_away") + pl.col("OT_away") ))
		    .then(pl.col("losses_away") - 1)
		    .otherwise(pl.col("losses_away"))
		    .alias("losses_away")  # Overwrite the existing "A" column
		)"""
		
		
		
		
		
		
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

		dataset = np.zeros((
			len(self.data),
			2 + self.N_current_game_fields + 2 * (3 + self.N_other_game_fields) * self.latest_matches +
			(2 + self.N_other_game_fields) * self.latest_vs_matches +
			(3 + self.N_other_game_fields) * self.latest_other_matches
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
			dataset[ii, 0] = self.teams.index(h_team)
			dataset[ii, 1] = self.teams.index(a_team)
			home_teams[ii] = h_team
			away_teams[ii] = a_team
			
			start_index = 1
			
			for kk in range(self.N_current_game_fields):
				dataset[ii, 2 + kk] = row[field_indices[self.current_game_fields[kk]]]
				if kk == self.N_current_game_fields - 1:
					start_index = 2 + kk

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
			   len(latest_vs) < self.latest_vs_matches or len(latest_other) < self.latest_other_matches:
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

				dataset[ii, start_index + 1 + jj * (3 + self.N_other_game_fields)] = home_flag

				# Date processing
				date = datetime(row2[year_col_idx], row2[month_col_idx], row2[day_col_idx])
				date_difference = current_date - date
				dataset[ii, start_index + 2 + jj * (3 + self.N_other_game_fields)] = date_difference.days

				dataset[ii, start_index + 3 + jj * (3 + self.N_other_game_fields)] = self.teams.index(row2[field_indices[other_str]])

				for kk in range(self.N_other_game_fields):
					dataset[ii, start_index + 4 + jj * (3 + self.N_other_game_fields) + kk] = row2[field_indices[self.other_game_fields[kk]]]

					if jj == self.latest_matches - 1 and kk == self.N_other_game_fields - 1:
						start_index = start_index + 4 + jj * (3 + self.N_other_game_fields) + kk

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

				dataset[ii, start_index + 1 + jj * (3 + self.N_other_game_fields)] = home_flag

				# Date processing
				date = datetime(row2[year_col_idx], row2[month_col_idx], row2[day_col_idx])
				date_difference = current_date - date
				dataset[ii, start_index + 2 + jj * (3 + self.N_other_game_fields)] = date_difference.days

				dataset[ii, start_index + 3 + jj * (3 + self.N_other_game_fields)] = self.teams.index(row2[field_indices[other_str]])

				for kk in range(self.N_other_game_fields):
					dataset[ii, start_index + 4 + jj * (3 + self.N_other_game_fields) + kk] = row2[field_indices[self.other_game_fields[kk]]]

					if jj == self.latest_matches - 1 and kk == self.N_other_game_fields - 1:
						start_index = start_index + 4 + jj * (3 + self.N_other_game_fields) + kk

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

				dataset[ii, start_index + 1 + jj * (2 + self.N_other_game_fields)] = home_flag

				# Date processing
				date = datetime(row2[year_col_idx], row2[month_col_idx], row2[day_col_idx])
				date_difference = current_date - date
				dataset[ii, start_index + 2 + jj * (2 + self.N_other_game_fields)] = date_difference.days

				for kk in range(self.N_other_game_fields):
					dataset[ii, start_index + 3 + jj * (2 + self.N_other_game_fields) + kk] = row2[field_indices[self.other_game_fields[kk]]]

					if jj == self.latest_vs_matches - 1 and kk == self.N_other_game_fields - 1:
						start_index = start_index + 3 + jj * (2 + self.N_other_game_fields) + kk

			# Similar logic for latest_other
			for jj in range(self.latest_other_matches):
				row2 = latest_other.row(jj)
				year_col_idx = field_indices["Year"]
				month_col_idx = field_indices["Month"]
				day_col_idx = field_indices["Day"]

				# Date processing
				date = datetime(row2[year_col_idx], row2[month_col_idx], row2[day_col_idx])
				date_difference = current_date - date
				dataset[ii, start_index + 1 + jj * (3 + self.N_other_game_fields)] = date_difference.days
				dataset[ii, start_index + 2 + jj * (3 + self.N_other_game_fields)] = self.teams.index(row2[field_indices["home"]])
				dataset[ii, start_index + 3 + jj * (3 + self.N_other_game_fields)] = self.teams.index(row2[field_indices["away"]])

				for kk in range(self.N_other_game_fields):
					dataset[ii, start_index + 4 + jj * (3 + self.N_other_game_fields) + kk] = row2[field_indices[self.other_game_fields[kk]]]
					

			# Perform the operation using the safe_min function
			bet_pred_spread[ii, :] = [row[field_indices['H_ML']], row[field_indices['A_ML']]]
			bet_pred_totals[ii, :] = [row[field_indices['OverUnder']], row[field_indices['OddOver']], row[field_indices['OddUnder']]]

		dataset = np.delete(dataset, bad_rows, axis=0)
		labels = np.delete(labels, bad_rows, axis=0)
		bet_pred_spread = np.delete(bet_pred_spread, bad_rows, axis=0)
		bet_pred_totals = np.delete(bet_pred_totals, bad_rows, axis=0)
		dates = np.delete(dates, bad_rows, axis=0)
		home_teams = np.delete(home_teams, bad_rows, axis=0)
		away_teams = np.delete(away_teams, bad_rows, axis=0)

		inputs = dataset
		return inputs, labels, bet_pred_spread, bet_pred_totals, dates, home_teams, away_teams




