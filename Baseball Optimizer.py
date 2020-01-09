from pulp import *
import pandas as pd
import cvxpy
import numpy as np
import pstats
import statistics

os.chdir("/Users/jasonrubenstein/Downloads")

# Read in payment data
baseball_data = pd.read_csv("BBal_WAR19.csv", delimiter=',', encoding = "ISO-8859-1")

baseball_data['POS'] = np.where(baseball_data['POS'] == "OF", "LF", baseball_data['POS'])
baseball_data = baseball_data.dropna(subset=['WAR'], inplace=False).reset_index()

def positional_assignment(df):
    df['SP'] = np.where(df['POS'] == "SP", 1, 0)
    df['RP'] = np.where(df['POS'] == "RP", 1, 0)
    df['CF'] = np.where(df['POS'] == "CF", 1, 0)
    df['RF'] = np.where(df['POS'] == "RF", 1, 0)
    df['LF'] = np.where(df['POS'] == "LF", 1, 0)
    df['3B'] = np.where(df['POS'] == "3B", 1, 0)
    df['SS'] = np.where(df['POS'] == "SS", 1, 0)
    df['2B'] = np.where(df['POS'] == "2B", 1, 0)
    df['1B'] = np.where(df['POS'] == "1B", 1, 0)
    df['C'] = np.where(df['POS'] == "C", 1, 0)
    df['DH'] = np.where(df['POS'] == "DH", 1, 0)
    return df
baseball_data = positional_assignment(baseball_data)


baseball_data['SALARY'] = pd.to_numeric(baseball_data['SALARY'])
# baseball_data['WAR'] = pd.to_numeric(baseball_data['WAR'])

players = baseball_data['Name']
salary = baseball_data['SALARY']
war = baseball_data['WAR']
pos = baseball_data['POS']
ip = baseball_data['Innings Pitched']
sp = baseball_data['SP']
rp = baseball_data['RP']
cf = baseball_data['CF']
rf = baseball_data['RF']
lf = baseball_data['LF']
_3B = baseball_data['3B']
ss = baseball_data['SS']
_2B = baseball_data['2B']
_1B = baseball_data['1B']
c = baseball_data['C']
dh = baseball_data['DH']


P = range(len(players))
S = 206000000
I = 1305
R = 40

prob = LpProblem("Optimal_25_Man_Roster", LpMaximize)

# Declare decision variable x, which is 1 if a
# player is part of the roster and 0 else
x = LpVariable.matrix("x", list(P), 0, 1, LpInteger)

# Objective function -> Maximize WAR
prob += sum(war[p] * x[p] for p in P)

# Constraint definition
prob += sum(x[p] for p in P) == R
prob += sum(salary[p] * x[p] for p in P) <= S

# Can positional minimum constraints
prob += sum(sp[p] * x[p] for p in P) >= 5
prob += sum(rp[p] * x[p] for p in P) >= 6
prob += sum(cf[p] * x[p] for p in P) >= 1
prob += sum(rf[p] * x[p] for p in P) >= 1
prob += sum(lf[p] * x[p] for p in P) >= 1
prob += sum(_3B[p] * x[p] for p in P) >= 1
prob += sum(ss[p] * x[p] for p in P) >= 1
prob += sum(_2B[p] * x[p] for p in P) >= 1
prob += sum(_1B[p] * x[p] for p in P) >= 1
prob += sum(c[p] * x[p] for p in P) >= 2

# Start solving the problem instance
prob.solve()

# Extract solution
lineup = [players[p] for p in P if x[p].varValue]
salary = [salary[p] for p in P if x[p].varValue]
war = [war[p] for p in P if x[p].varValue]

print(lineup, salary, war)
full = pd.DataFrame(lineup)
full.columns = ['Name']

full = pd.merge(full, baseball_data, how='left', on='Name')
full.to_csv('/Users/jasonrubenstein/Downloads/optimal_lineup.csv')


# bbal_data.to_csv('/Users/jasonrubenstein/Downloads/check.csv')
