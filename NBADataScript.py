from NBADataFunctions import data_collection, correlations, multiple_linear_regression_model, database_connection, team_names, points, core_metrics, defense, team_names_insertions, points_insertions, core_metrics_insertion, defense_insertions, curr_season_data_df

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
fg3pct_list = df.get('FG3_PCT').values
fta_list = df.get('FTA').values
ftpct_list = df.get('FT_PCT').values
fg2a_list = df.get('FG2A').values
fg2pct_list = df.get('FG2_PCT').values
efgpct_list = df.get('EFG_PCT').values
wins_list = df.get('W').values

pts_list = df.get('PTS').values
ast_list = df.get('AST').values
reb_list = df.get('REB').values
stl_list = df.get('STL').values
blk_list = df.get('BLK').values

correlations(pts_list, ast_list, reb_list, stl_list, blk_list, fg3a_list, fg3pct_list, fg2a_list, fg2pct_list, fta_list, ftpct_list, efgpct_list, wins_list)

multiple_linear_regression_model(fg3a_list, fg2a_list, efgpct_list, fta_list, ftpct_list, ast_list, reb_list, stl_list, blk_list, wins_list)

print('Username?')
user = input()

print('Password?')
passwd = input()

db_conn = database_connection(user, passwd)

team_names(db_conn)
points(db_conn)
core_metrics(db_conn)
defense(db_conn)

merged_df = curr_season_data_df('2024-25')

teams_df = merged_df.get(['TEAM_NAME', 'Conference', 'Logos', 'W'])
points_df = merged_df.get(['TEAM_NAME', 'FG3A', 'FG3_PCT', 'FG2A', 'FG2_PCT', 'FTA', 'FT_PCT', 'EFG_PCT'])
core_metrics_df = merged_df.get(['TEAM_NAME', 'AST', 'REB', 'BLK', 'STL'])
defense_df = merged_df.get(['TEAM_NAME', 'D_FGM', 'D_FGA', 'D_FG_PCT', 'E_DEF_RATING'])


team_names_insertions(db_conn, teams_df)
points_insertions(db_conn, points_df)
core_metrics_insertion(db_conn, core_metrics_df)
defense_insertions(db_conn, defense_df)
