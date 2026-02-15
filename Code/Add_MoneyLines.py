import bs4
from bs4 import BeautifulSoup
import requests
import pandas as pd
from tqdm import tqdm
import numpy as np

def extract_money_lines(url, dateStr):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'}
    payload = {
        'sport': 'basketball',
        'league': 'nba',
        'region': 'us',
        'lang': 'en',
        'contentorigin': 'espn',
        'buyWindow': '1m',
        'showAirings': 'buy,live,replay',
        'tz': 'America/New_York',
        'dates': dateStr}
    
    response = requests.get(url, headers=headers, params=payload).json()
    events = response['sports'][0]['leagues'][0]['events']
    
    df = pd.json_normalize(events,
                           record_path=['competitors'],
                           meta=['odds', ['odds', 'away', 'moneyLine'], ['odds', 'home', 'moneyLine'], ['odds', 'overUnder'], ['odds', 'underOdds'], ['odds', 'overOdds'], ['odds', 'spread']],
                           errors='ignore')
                           
    return df




games_df = pd.read_csv('/home/jsarrato/PersonalProjects/NBBA/Data/DataNew_advanced/processed_team_boxscores.csv')


games_df['HomeML'] = ''
games_df['AwayML'] = ''
games_df['OverUnder'] = ''
games_df['OddOver'] = ''
games_df['OddUnder'] = ''
games_df['Spread'] = ''


for date in tqdm(games_df['Date'].unique()):
	print(date)

	dateStr = date.split('-')[0] + date.split('-')[1] + date.split('-')[2]
	url = f"https://site.web.api.espn.com/apis/v2/scoreboard/header"
	
	df_date = games_df[games_df['Date']==date]

	# Extract money lines and team names
	try:
		df = extract_money_lines(url, dateStr)
		print('MLs found')
	except:
		print('MLs not found')
		continue
	
	df['full_name'] = df['location'] + ' ' + df['name']
	df['full_name'] = df['full_name'].replace('LA Clippers', 'Los Angeles Clippers')
	
	for ii in range(len(df)):
		if df.iloc[ii]['full_name'] in np.array(df_date['home']):
			games_df.loc[df_date.index[df_date['home'] == df.iloc[ii]['full_name']][0], 'HomeML'] = df['odds.' + df.iloc[ii]['homeAway'] + '.moneyLine'][ii]
			
			games_df.loc[df_date.index[df_date['home'] == df.iloc[ii]['full_name']][0], 'OverUnder'] = df['odds.overUnder'][ii]
			games_df.loc[df_date.index[df_date['home'] == df.iloc[ii]['full_name']][0], 'OddUnder'] = df['odds.underOdds'][ii]
			games_df.loc[df_date.index[df_date['home'] == df.iloc[ii]['full_name']][0], 'OddOver'] = df['odds.overOdds'][ii]
			
			games_df.loc[df_date.index[df_date['home'] == df.iloc[ii]['full_name']][0], 'Spread'] = df['odds.spread'][ii]
			
		elif df.iloc[ii]['full_name'] in np.array(df_date['away']):
			games_df.loc[df_date.index[df_date['away'] == df.iloc[ii]['full_name']][0], 'AwayML'] = df['odds.' + df.iloc[ii]['homeAway'] + '.moneyLine'][ii]
			

games_df.to_csv('/home/jsarrato/PersonalProjects/NBBA/Data/DataNew_advanced/processed_team_boxscores_ML.csv', index = False)
