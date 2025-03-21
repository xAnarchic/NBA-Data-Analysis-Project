from NBADataFunctions import data_collection, correlations, multiple_linear_regression_model, databaseConnection, team_names, points, stats, defense, team_names_insertions, points_insertions, stats_insertions, defense_insertions, curr_season_data_df


seasons = []

print('How many seasons are you interested in?')
num_of_seasons = int(input())
for num in range(num_of_seasons):
    num += 1
    print(f'Season {num}?')
    seasons.append(input())

df = data_collection(seasons)


teams_list = df.get('TEAM_NAME').values
fg3a_list = df.get('FG3A').values
efgpct_list = df.get('EFG_PCT').values
fgpct_list = df.get('FG_PCT').values
fg3pct_list = df.get('FG3_PCT').values
ftpct_list = df.get('FT_PCT').values
fta_list = df.get('FTA').values
fg2a_list = df.get('FG2A').values
fg2pct_list = df.get('FG2_PCT').values
wins_list = df.get('W').values

pts_list = df.get('PTS').values
ast_list = df.get('AST').values
reb_list = df.get('REB').values
stl_list = df.get('STL').values
blk_list = df.get('BLK').values
to_list = df.get('TOV').values

correlations(pts_list, ast_list, reb_list, stl_list, blk_list, to_list, fg3pct_list, fg3a_list, ftpct_list, fta_list, fgpct_list, fg2a_list, fg2pct_list, wins_list)

multiple_linear_regression_model(fg3a_list, efgpct_list, ftpct_list, fta_list, fg2a_list, fg2pct_list, fg3pct_list, fgpct_list, ast_list, reb_list, stl_list, blk_list, wins_list)

print('Username?')
user = input()

print('Password?')
passwd = input()

db_conn = databaseConnection(user, passwd)

team_names(db_conn)
points(db_conn)
stats(db_conn)
defense(db_conn)

merged_df = curr_season_data_df('2024-25')

teams_df = merged_df.get(['TEAM_NAME', 'Conference', 'Logos', 'W'])
points_df = merged_df.get(['TEAM_NAME', 'FG3A', 'FG3_PCT', 'FG2A', 'FG2_PCT', 'FTA', 'FT_PCT', 'FG_PCT', 'EFG_PCT'])
stats_df = merged_df.get(['TEAM_NAME', 'AST', 'REB', 'BLK', 'STL'])
defense_df = merged_df.get(
    ['TEAM_NAME', 'D_FGM', 'D_FGA', 'D_FG_PCT', 'NORMAL_FG_PCT', 'PLUS_MINUS', 'E_DEF_RATING'])


team_names_insertions(db_conn, teams_df)
points_insertions(db_conn, points_df)
stats_insertions(db_conn, stats_df)
defense_insertions(db_conn, defense_df)
