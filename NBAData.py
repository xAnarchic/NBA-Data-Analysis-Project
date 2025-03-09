import requests
from nba_api.stats.endpoints import playercareerstats, playerindex
from nba_api.stats.static import players

response = requests.post(url = "https://v2.nba.api-sports.io")
response.raise_for_status()
print(response.json())


all_players_historic = players.get_players()
number_of_players = len(all_players_historic)

for num in range(number_of_players):
    player_num = num + 1
    player_name = all_players_historic[num]['full_name']
    print(player_name, num)

print(all_players_historic[-1]['full_name'])

# Nikola Jokić
# career = playercareerstats.PlayerCareerStats(player_id='76003')
# print((career.get_data_frames()[0]).to_string())

# players = playerindex.PlayerIndex(season = "2024")
# #print((players.get_data_frames()[0]).to_string())
#
#
# print(players.get_dict()['resultSets'][0]['rowSet'][0])