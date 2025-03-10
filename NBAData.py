from pprint import pprint
from nba_api.stats.endpoints import playercareerstats, playerindex, homepageleaders, leaguedashteamstats
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
    n_array = sorteddf.to_numpy()

    sorteddf1 = ranksdf.sort_values(by = 'REB', ascending = False)
    print(sorteddf1)

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

    return
