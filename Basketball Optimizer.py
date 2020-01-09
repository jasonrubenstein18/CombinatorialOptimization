from pulp import LpProblem, LpVariable, lpSum, LpMaximize, value
import pandas as pd
import cvxpy
import numpy as np
import pstats
import statistics

os.chdir("...")

# Read in payment data
bbal_data = pd.read_csv("Fantrax-Players-The Diverse Dozen-115.csv", delimiter=',', encoding = "ISO-8859-1")
# bbal_data = bbal_data[(bbal_data['GP'] > 0)]

# Z-Scores
bbal_data['pts_z'] = (bbal_data['PTS']-bbal_data['PTS'].mean()) / statistics.stdev(bbal_data['PTS'])
bbal_data['reb_z'] = (bbal_data['REB']-bbal_data['REB'].mean()) / statistics.stdev(bbal_data['REB'])
bbal_data['ast_z'] = (bbal_data['AST']-bbal_data['AST'].mean()) / statistics.stdev(bbal_data['AST'])
bbal_data['3pt_z'] = (bbal_data['3PTM']-bbal_data['3PTM'].mean()) / statistics.stdev(bbal_data['3PTM'])
bbal_data['st_z'] = (bbal_data['ST']-bbal_data['ST'].mean()) / statistics.stdev(bbal_data['ST'])
bbal_data['blk_z'] = (bbal_data['BLK']-bbal_data['BLK'].mean()) / statistics.stdev(bbal_data['BLK'])
bbal_data['fgp_z'] = (bbal_data['FG%']-bbal_data['FG%'].mean()) / statistics.stdev(bbal_data['FG%'])
bbal_data['ftp_z'] = (bbal_data['FT%']-bbal_data['FT%'].mean()) / statistics.stdev(bbal_data['FT%'])

# Avg Z
bbal_data['STD_Score'] = bbal_data[['pts_z', 'reb_z', 'ast_z', '3pt_z', 'st_z',
                                    'blk_z', 'fgp_z', 'ftp_z']].mean(axis=1)



prob = LpProblem("Basketball_Lineup", LpMaximize)

bbal_data['Fix'] = 1

from pulp import *
players = bbal_data['Player']
salary = bbal_data['Salary']
std_score = bbal_data['STD_Score']
points = bbal_data['PTS']
fix = bbal_data['Fix']
owner = bbal_data['Status']

P = range(len(players))
S = 200

prob = LpProblem("Optimal_Lineup", LpMaximize)

# Declare decision variable x, which is 1 if a
# player is part of the portfolio and 0 else
x = LpVariable.matrix("x", list(P), 0, 1, LpInteger)

# Objective function -> Maximize z-score
prob += sum(std_score[p] * x[p] for p in P)

# Constraint definition
prob += sum(x[p] for p in P) == 13
prob += sum(salary[p] * x[p] for p in P) <= 200

# Can add more constraints

# Start solving the problem instance
prob.solve()

# Extract solution
lineup = [players[p] for p in P if x[p].varValue]
salary = [salary[p] for p in P if x[p].varValue]

print(lineup)
full = pd.DataFrame(lineup)
full.columns = ['Player']

full = pd.merge(full, bbal_data, how='left', on='Player')
