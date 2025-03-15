from pprint import pprint
from nba_api.stats.endpoints import playercareerstats, playerindex, homepageleaders, leaguedashteamstats, leaguedashptstats, leaguedashteamshotlocations, leaguedashteamclutch, leaguedashteamptshot, teamestimatedmetrics
from nba_api.stats.static import players
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import time

#Numbers all players that have participated in the NBA.
def all_players():

    all_players_historic = players.get_players()
    number_of_players = len(all_players_historic)

    for num in range(number_of_players):
        player_num = num + 1
        player_name = all_players_historic[num]['full_name']
        print(player_name, num)

    return print(all_players_historic[-1]['full_name'])

#Returns players (and teams) with a certain PTS/ AST/ REB statline.
def players_by_current_stats():

    players = playerindex.PlayerIndex(season = 2024)
    df = players.get_data_frames()[0]
    print('How many points should these players be averaging?')
    pts = int(input())
    print('How many assists should these players be averaging?')
    ast = int(input())
    print('How many rebounds should these players be averaging?')
    reb = int(input())

    df = df.loc[(df['PTS'] > pts) & (df['AST'] > ast) & (df['REB'] > reb) ]
    df = df.sort_values(by = 'PTS', ascending = False)
    playerlist = []
    for i, player in enumerate(df.to_numpy()):
        playerlist.append({
            player[3]:player[7] + ' ' + player[8]
        })
    return pprint(playerlist)

#Exploring core stats (PTS, AST, REB, STL, BLK) and turnovers (TO) of teams in the 2024-25 season.
def team_stats(season):

    teams = leaguedashteamstats.LeagueDashTeamStats(season = season)
    df = teams.get_data_frames()[0]
    ranksdf = df.get(['TEAM_NAME', 'PTS', 'AST', 'REB', 'STL','BLK','TOV', 'W'])

    # ranks7df = df['W'].tolist()
    # print(ranks7df)

    sorteddf = ranksdf.sort_values(by = 'W', ascending = False)
    alpha_sorteddf = ranksdf.sort_values(by = 'TEAM_NAME', ascending = True)
    alpha_to_sorteddf = alpha_sorteddf.get('TOV')
    n_array = sorteddf.to_numpy()
    alpha_to_n_array = alpha_to_sorteddf.to_numpy()

    wins= []
    points = []
    assists = []
    rebounds = []
    steals = []
    blocks = []
    turnovers = []


    for i, teams in enumerate(n_array):     # maybe can just get a column, then use the ".tolist()" method on it
        wins.append(teams[-1])
        points.append(teams[1])
        assists.append(teams[2])
        rebounds.append(teams[3])
        steals.append(teams[4])
        blocks.append(teams[5])
        turnovers.append(teams[6])

    p_res = stats.spearmanr(wins, points)
    #print(f'Points: \n Spearman correlation coefficient= {round(p_res.statistic,5)} \n p-value = {round(p_res.pvalue,5)}')

    a_res = stats.spearmanr(wins, assists)
    #print(f'Assists: \n Spearman correlation coefficient= {round(a_res.statistic, 5)} \n p-value = {round(a_res.pvalue,5)}')

    r_res = stats.spearmanr(wins, rebounds)
    #print(f'Rebounds: \n Spearman correlation coefficient= {round(r_res.statistic,5)} \n p-value = {round(r_res.pvalue,5)}')

    s_res = stats.spearmanr(wins, steals)
    #print(f'Steals: \n Spearman correlation coefficient= {round(s_res.statistic, 5)} \n p-value = {round(s_res.pvalue, 5)}')

    b_res = stats.spearmanr(wins, blocks)
    #print(f'Blocks: \n Spearman correlation coefficient= {round(b_res.statistic, 5)} \n p-value = {round(b_res.pvalue, 5)}')

    t_res = stats.spearmanr(wins, turnovers)
    #print(f'Turnovers: \n Spearman correlation coefficient= {round(t_res.statistic, 5)} \n p-value = {round(t_res.pvalue, 5)}')

    return {'Points': points, 'Assists': assists, 'Rebounds': rebounds, 'Steals': steals, 'Blocks': blocks, 'Turnovers': turnovers, 'Wins': wins, 'Turnovers by alpha' : alpha_to_n_array}

def teams_points(season):
    points = leaguedashteamptshot.LeagueDashTeamPtShot(season = season)
    points_df = points.get_data_frames()[0]
    points_df0 = points_df.sort_values(by = 'TEAM_NAME', ascending = True)
    points_df = points_df0.get(['TEAM_NAME', 'FG3A', 'EFG_PCT', 'FG_PCT', 'FG3_PCT']).to_numpy()
    points_df1 = points_df0.get(['TEAM_NAME', 'FG3A', 'EFG_PCT', 'FG_PCT', 'FG3_PCT'])

    teams = leaguedashteamstats.LeagueDashTeamStats(season=season)
    df = teams.get_data_frames()[0]
    ranksdf = df.get(['TEAM_NAME','W'])
    ranksdf = ranksdf.sort_values(by = 'TEAM_NAME', ascending = True)
    team_wins_list = ranksdf.get(['W']).squeeze().tolist()
    team_wins = {'Team wins' : team_wins_list}

    points_df1 = points_df1.assign(Wins = team_wins_list).to_numpy()

    return points_df1

point_stats = teams_points('2023-24')

teams_list = []
fg3a_list = []
efgpct_list = []
fgpct_list = []
fg3pct_list = []
wins_list = []

for i, team in enumerate(point_stats):
    teams_list.append(team[0])
    fg3a_list.append(team[1])
    efgpct_list.append(team[2])
    fgpct_list.append(team[3])
    fg3pct_list.append(team[4])
    wins_list.append(team[5])

data = pd.DataFrame([fg3pct_list, fgpct_list, efgpct_list, fg3a_list])
data= data.transpose()
data.columns = ['FG3PCT','FGPCT', 'EFGPCT', 'FG3A']
print(data)

data_y = pd.DataFrame([wins_list])
data_y = data_y.transpose()
data_y.columns = ['Wins']
print(data_y)

x = data
y = data_y

#y = np.array([wins_list])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state= 0)
lr = LinearRegression()
lr.fit(x_train, y_train)
intercept = lr.intercept_
gradients = lr.coef_
print(gradients)

y_pred_train = lr.predict(x_train)
plt.scatter(y_train, y_pred_train)
plt.xlabel('Actual wins- training')
plt.ylabel('Predicted wins- training')
plt.show()


r2_val_train = r2_score(y_train, y_pred_train)
print(r2_val_train)

y_pred_test = lr.predict(x_test)

plt.scatter(y_test, y_pred_test)
plt.xlabel('Actual wins- testing')
plt.ylabel('Predicted wins- testing')
plt.show()

r2_val_test = r2_score(y_test, y_pred_test)
print(r2_val_test)





# years_w = team_stats('2023-24')['Wins'] + team_stats('2022-23')['Wins'] + team_stats('2021-22')['Wins'] + team_stats('2020-21')['Wins'] + team_stats('2019-20')['Wins'] + team_stats('2018-19')['Wins'] + team_stats('2017-18')['Wins'] + team_stats('2016-17')['Wins'] + team_stats('2015-16')['Wins']
# time.sleep(5)
# years_pts = team_stats('2023-24')['Points'] + team_stats('2022-23')['Points'] + team_stats('2021-22')['Points'] + team_stats('2020-21')['Points'] + team_stats('2019-20')['Points'] + team_stats('2018-19')['Points'] + team_stats('2017-18')['Points'] + team_stats('2016-17')['Points'] + team_stats('2015-16')['Points']
# time.sleep(5)
# years_ast = team_stats('2023-24')['Assists'] + team_stats('2022-23')['Assists'] + team_stats('2021-22')['Assists'] + team_stats('2020-21')['Assists'] + team_stats('2019-20')['Assists'] + team_stats('2018-19')['Assists'] + team_stats('2017-18')['Assists'] + team_stats('2016-17')['Assists'] + team_stats('2015-16')['Assists']
# time.sleep(5)
# years_reb = team_stats('2023-24')['Rebounds'] + team_stats('2022-23')['Rebounds'] + team_stats('2021-22')['Rebounds'] + team_stats('2020-21')['Rebounds'] + team_stats('2019-20')['Rebounds'] + team_stats('2018-19')['Rebounds'] + team_stats('2017-18')['Rebounds'] + team_stats('2016-17')['Rebounds'] + team_stats('2015-16')['Rebounds']
# # time.sleep(5)
# years_stl = team_stats('2023-24')['Steals'] + team_stats('2022-23')['Steals'] + team_stats('2021-22')['Steals'] + team_stats('2020-21')['Steals'] + team_stats('2019-20')['Steals'] + team_stats('2018-19')['Steals'] + team_stats('2017-18')['Steals'] + team_stats('2016-17')['Steals'] + team_stats('2015-16')['Steals']
# time.sleep(5)
# years_blk = team_stats('2023-24')['Blocks'] + team_stats('2022-23')['Blocks'] + team_stats('2021-22')['Blocks'] + team_stats('2020-21')['Blocks'] + team_stats('2019-20')['Blocks'] + team_stats('2018-19')['Blocks'] + team_stats('2017-18')['Blocks'] + team_stats('2016-17')['Blocks'] + team_stats('2015-16')['Blocks']
# time.sleep(5)
# years_to = team_stats('2023-24')['Turnovers'] + team_stats('2022-23')['Turnovers'] + team_stats('2021-22')['Turnovers'] + team_stats('2020-21')['Turnovers'] + team_stats('2019-20')['Turnovers'] + team_stats('2018-19')['Turnovers'] + team_stats('2017-18')['Turnovers'] + team_stats('2016-17')['Turnovers'] + team_stats('2015-16')['Turnovers']
# time.sleep(5)
#
# p_res = stats.spearmanr(years_w, years_pts)
# print(f'Points: \n Spearman correlation coefficient= {round(p_res.statistic,5)} \n p-value = {p_res.pvalue}')
#
# a_res = stats.spearmanr(years_w, years_ast)
# print(f'Assists: \n Spearman correlation coefficient= {round(a_res.statistic, 5)} \n p-value = {a_res.pvalue}')
#
# r_res = stats.spearmanr(years_w, years_reb)
# print(f'Rebounds: \n Spearman correlation coefficient= {round(r_res.statistic, 5)} \n p-value = {r_res.pvalue}')
#
# s_res = stats.spearmanr(years_w, years_stl)
# print(f'Steals: \n Spearman correlation coefficient= {round(s_res.statistic,5)} \n p-value = {s_res.pvalue}')
#
# b_res = stats.spearmanr(years_w, years_blk)
# print(f'Blocks: \n Spearman correlation coefficient= {round(b_res.statistic, 5)} \n p-value = {b_res.pvalue}')
#
# t_res = stats.spearmanr(years_w, years_to)
# print(f'Turnovers: \n Spearman correlation coefficient= {round(t_res.statistic, 5)} \n p-value = {t_res.pvalue}')


#Uses all 5 core metrics: training r2 val = 0.240, testing r2 val = 0.270
# data = pd.DataFrame([years_pts, years_reb, years_stl, years_blk])
# data= data.transpose()
# data.columns = ['Points', 'Rebounds', 'Steals', 'Blocks']
# print(data)

#Uses all 5 core metrics: training r2 val = 0.244, testing r2 val = 0.260
# data = pd.DataFrame([years_pts, years_reb, years_stl, years_blk])
# data= data.transpose()
# data.columns = ['Points', 'Rebounds', 'Steals', 'Blocks']
# print(data)

#Uses all 5 core metrics: training r2 val = 0.217, testing r2 val = 0.224
# data = pd.DataFrame([years_pts])
# data= data.transpose()
# data.columns = ['Points']
# print(data)

#Uses 4 core metrics(not incl. points): training r2 val = 0.196, testing r2 val = 0.255
# data = pd.DataFrame([years_ast, years_reb, years_stl, years_blk])
# data= data.transpose()
# data.columns = ['Assists', 'Rebounds', 'Steals', 'Blocks']
# print(data)

#Uses 3 core metrics(not incl. points, blocks): training r2 val = 0.182, testing r2 val = 0.231
# data = pd.DataFrame([years_ast, years_reb, years_stl])
# data= data.transpose()
# data.columns = ['Assists', 'Rebounds', 'Steals']
# print(data)

#Uses 3 core metrics(not incl. points, assists): training r2 val = 0.177, testing r2 val = 0.228
# data = pd.DataFrame([years_reb, years_stl, years_blk])
# data= data.transpose()
# data.columns = ['Rebounds', 'Steals', 'Blocks']
# print(data)

#Uses 2 core metrics(not incl. points, assists, rebounds): training r2 val = 0.100, testing r2 val = 0.184
# data = pd.DataFrame([years_stl, years_blk])
# data= data.transpose()
# data.columns = ['Steals', 'Blocks']
# print(data)


#Uses 4 core metrics(not incl. steal): training r2 val = 0.244, testing r2 val = 0.255
# data = pd.DataFrame([years_pts, years_ast, years_reb, years_blk])
# data= data.transpose()
# data.columns = ['Points', 'Assists', 'Rebounds', 'Blocks']
# print(data)

#Uses 3 core metrics(not incl. steal, blocks): training r2 val = 0.225, testing r2 val = 0.227
# data = pd.DataFrame([years_pts, years_ast, years_reb])
# data= data.transpose()
# data.columns = ['Points', 'Assists', 'Rebounds']
# print(data)

#Uses 2 core metrics(not incl. steal, blocks, rebounds): training r2 val = 0.219, testing r2 val = 0.216
# data = pd.DataFrame([years_pts, years_ast])
# data= data.transpose()
# data.columns = ['Points', 'Assists']
# print(data)

# data_y = pd.DataFrame([years_w])
# data_y = data_y.transpose()
# data_y.columns = ['Wins']
# print(data_y)
#
# x = data
# y = data_y
#
# #y = np.array([years_w])
#
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state= 0)
# lr = LinearRegression()
# lr.fit(x_train, y_train)
# intercept = lr.intercept_
# gradients = lr.coef_
# print(gradients)
#
# y_pred_train = lr.predict(x_train)
# plt.scatter(y_train, y_pred_train)
# plt.xlabel('Actual wins- training')
# plt.ylabel('Predicted wins- training')
# plt.show()
#
#
# r2_val_train = r2_score(y_train, y_pred_train)
# print(r2_val_train)
#
# y_pred_test = lr.predict(x_test)
#
# plt.scatter(y_test, y_pred_test)
# plt.xlabel('Actual wins- testing')
# plt.ylabel('Predicted wins- testing')
# plt.show()
#
# r2_val_test = r2_score(y_test, y_pred_test)
# print(r2_val_test)

# slope, intercept, r, p, std_err = stats.linregress(years_w, years_pts)
#
# print(r**2)
# print(p)
# print(std_err)
#
# def myfunc(x):
#     return slope * x + intercept
#
# mymodel = list(map(myfunc, years_w))
#
# plt.scatter(years_w, years_pts)
# plt.plot(years_w, mymodel)
# plt.show()      # quality isn't great, there are some big outliers/ residuals from the fitted line





# plt.scatter(years_w, years_ast)
# plt.show()

# plt.scatter(years_w, years_to)
# plt.show()




# assists_list = team_stats()['Assists']
# turnovers_list = team_stats()['Turnovers']
# wins_list = team_stats()['Wins']
# alpha_turnovers_list = team_stats()['Turnovers by alpha']


def turnover_analysis(assists, turnovers, wins, alpha_turnovers):

    ind = -1
    ast_to_ratio_list = []
    for ast in assists:
        ind += 1
        ast_to_ratio = (ast / turnovers[ind])
        ast_to_ratio_list.append(ast_to_ratio)

    ast_to_ratio_res = stats.spearmanr(wins, ast_to_ratio_list)
    print(f'Assists-to-Turnover ratio: \n Spearman correlation coefficient= {round(ast_to_ratio_res.statistic, 5)} \n p-value = {round(ast_to_ratio_res.pvalue, 5)}')

    ast_to_res = stats.spearmanr(turnovers, assists)
    print(f'Assists-Turnover correlation: \n Spearman correlation coefficient= {round(ast_to_res.statistic, 5)} \n p-value = {round(ast_to_res.pvalue, 5)}')

    team_pace = leaguedashptstats.LeagueDashPtStats()
    pace_df = team_pace.get_data_frames()[0]
    pace_df = pace_df.sort_values(by = 'W', ascending = False)
    off_pace_df = pace_df.get('AVG_SPEED_OFF')
    off_pace = off_pace_df.to_numpy()

    off_pace_to_res = stats.spearmanr(turnovers, off_pace)
    print(f'Offensive pace-Turnover correlation: \n Spearman correlation coefficient= {round(off_pace_to_res.statistic, 5)} \n p-value = {round(off_pace_to_res.pvalue, 5)}')

    team_shot_loc = leaguedashteamshotlocations.LeagueDashTeamShotLocations()
    team_shot_loc_df = team_shot_loc.get_data_frames()[0]
    team_shot_loc_df = team_shot_loc_df.get(('Restricted Area','FGA'))
    ra_shots_made = team_shot_loc_df.to_numpy()

    ra_shots_to_res = stats.spearmanr(alpha_turnovers, ra_shots_made)
    print(f'Restricted area shots-Turnover correlation: \n Spearman correlation coefficient= {round(ra_shots_to_res.statistic, 5)} \n p-value = {round(ra_shots_to_res.pvalue, 5)}')



    return

# turnover_analysis(assists_list, turnovers_list, wins_list, alpha_turnovers_list)


#Trying to check out previous league stats