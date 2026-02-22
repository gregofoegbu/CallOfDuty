team_one_bbox = {
    "y1": 0.03,
    "y2": 0.24,
    "x1": 0.02,
    "x2": 0.34
}

center_stats_bbox = {
    "y1": 0.02,
    "y2": 0.22,
    "x1": 0.37,
    "x2": 0.63
}

team_two_bbox = {
    "y1": 0.03,
    "y2": 0.24,
    "x1": 0.65,
    "x2": 0.98
}

kill_feed_bbox = {
    "y1": 0.40,
    "y2": 0.65,
    "x1": 0.02,
    "x2": 0.30
}

team_one_name_bbox = {
    "y1": 0.03,
    "y2": 0.08,
    "x1": 0.02,
    "x2": 0.34
}

team_two_name_bbox = {
    "y1": 0.03,
    "y2": 0.08,
    "x1": 0.65,
    "x2": 0.98
}

players_team_one = {
    "y1": 0.08,
    "y2": 0.24,
    "x1": 0.06,
    "x2": 0.14
}

players_team_two = {
    "y1": 0.08,
    "y2": 0.24,
    "x1": 0.69,
    "x2": 0.78
}

all_regions = [team_one_bbox, team_two_bbox, center_stats_bbox, kill_feed_bbox]

all_regions_dict = {"team_one": team_one_bbox, "team_two": team_two_bbox, "centre_stats": center_stats_bbox,
                    "kill_feed": kill_feed_bbox, "players_t1": players_team_one, "players_t2": players_team_two}