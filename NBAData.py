from pprint import pprint
from nba_api.stats.endpoints import playercareerstats, playerindex, homepageleaders, leaguedashteamstats, leaguedashptstats, leaguedashteamshotlocations
from nba_api.stats.static import players
import pandas as pd
from scipy import stats

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
def team_stats():

    teams = leaguedashteamstats.LeagueDashTeamStats()
    df = teams.get_data_frames()[0]
    ranksdf = df.get(['TEAM_NAME', 'PTS', 'AST', 'REB', 'STL','BLK','TOV', 'W'])
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


    for i, teams in enumerate(n_array):
        wins.append(teams[-1])
        points.append(teams[1])
        assists.append(teams[2])
        rebounds.append(teams[3])
        steals.append(teams[4])
        blocks.append(teams[5])
        turnovers.append(teams[6])

    p_res = stats.spearmanr(wins, points)
    print(f'Points: \n Spearman correlation coefficient= {round(p_res.statistic,5)} \n p-value = {round(p_res.pvalue,5)}')

    a_res = stats.spearmanr(wins, assists)
    print(f'Assists: \n Spearman correlation coefficient= {round(a_res.statistic, 5)} \n p-value = {round(a_res.pvalue,5)}')

    r_res = stats.spearmanr(wins, rebounds)
    print(f'Rebounds: \n Spearman correlation coefficient= {round(r_res.statistic,5)} \n p-value = {round(r_res.pvalue,5)}')

    s_res = stats.spearmanr(wins, steals)
    print(f'Steals: \n Spearman correlation coefficient= {round(s_res.statistic, 5)} \n p-value = {round(s_res.pvalue, 5)}')

    b_res = stats.spearmanr(wins, blocks)
    print(f'Blocks: \n Spearman correlation coefficient= {round(b_res.statistic, 5)} \n p-value = {round(b_res.pvalue, 5)}')

    t_res = stats.spearmanr(wins, turnovers)
    print(f'Turnovers: \n Spearman correlation coefficient= {round(t_res.statistic, 5)} \n p-value = {round(t_res.pvalue, 5)}')

    return {'Assists': assists, 'Turnovers': turnovers, 'Wins': wins, 'Turnovers by alpha' : alpha_to_n_array}

assists_list = team_stats()['Assists']
turnovers_list = team_stats()['Turnovers']
wins_list = team_stats()['Wins']
alpha_turnovers_list = team_stats()['Turnovers by alpha']


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

    #add the different kinds of 3's

    return

turnover_analysis(assists_list, turnovers_list, wins_list, alpha_turnovers_list)

#Use the homepageleaders endpoint to perform the points -> FT analysis