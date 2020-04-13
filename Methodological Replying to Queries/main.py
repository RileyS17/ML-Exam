def compareAns(group1, group2):
    totalSame = 0
    for x in range(15):
        ans1 = groupData[group1]['Q{}'.format(x+1)]
        ans2 = groupData[group2]['Q{}'.format(x+1)]
        if (ans1 == ans2):
            totalSame += 1
    return totalSame/15

def compareTotals(a, b):
    yDif = abs(a[0]-b[0])/200
    nDif = abs(a[1]-b[1])/200
    uDif = abs(a[2]-b[2])/200
    similarity = 1 - (yDif + nDif + uDif)
    return similarity


import pandas as pd
import itertools
import numpy as np

data = pd.read_excel("Take home exam dataset.xlsx", header=1)

# Query 1
# Which group has provided more unified answers to the questions  (higher internal agreement)? How do you rank the groups based on this metric?
print("-- Query 1 --")
for x in range(4):
    curGroup = data.iloc[x, 2:].values
    totalAgr = 0
    for y in range(15):
        currAgr = max(curGroup[y*3], curGroup[(y*3)+1])
        totalAgr += currAgr
    print("Group {} average agreement value is {}%.".format(x+1, totalAgr/15))


# Query 2
# Determine that, in overall, the answers of every group is closer to which one of the other group’s answers?
print("\n-- Query 2 --")
groupData = {   'G1': {},
                'G2': {},
                'G3': {},
                'G4': {}}
for x in range(4):
    curGroup = data.iloc[x, 2:].values
    for y in range(15):
        Y = curGroup[y*3]
        N = curGroup[(y*3)+1]
        U = curGroup[(y*3)+2]

        if (Y > N and Y > U):
            answer = 'Y'    
        elif (N > Y and N > U):
            answer = 'N'
        elif (U > Y and U > N):
            answer = 'U'
        else:
            answer = 'U'

        groupData['G{}'.format(x+1)]['Q{}'.format(y+1)] = answer

compG1G2 = compareAns('G1', 'G2')
compG1G3 = compareAns('G1', 'G3')
compG1G4 = compareAns('G1', 'G4')
compG2G3 = compareAns('G2', 'G3')
compG2G4 = compareAns('G2', 'G4')
compG3G4 = compareAns('G3', 'G4')

print("Group 1 -")
print("     Group 2: {}%".format(compG1G2*100))
print("     Group 3: {}%".format(compG1G3*100))
print("     Group 4: {}%".format(compG1G4*100))
print("Group 2 -")
print("     Group 1: {}%".format(compG1G2*100))
print("     Group 3: {}%".format(compG2G3*100))
print("     Group 4: {}%".format(compG2G4*100))
print("Group 3 -")
print("     Group 1: {}%".format(compG1G3*100))
print("     Group 2: {}%".format(compG2G3*100))
print("     Group 4: {}%".format(compG3G4*100))
print("Group 4 -")
print("     Group 1: {}%".format(compG1G4*100))
print("     Group 2: {}%".format(compG2G4*100))
print("     Group 3: {}%".format(compG3G4*100))


# Query 3
# Rank the questions based on their discrimination power over the groups (i.e., sorting questions based on corresponding level of agreements of groups).
print("\n-- Query 3 --")
questionData = { 'Q1': {}, 'Q2': {}, 'Q3': {}, 'Q4': {}, 'Q5': {}, 'Q6': {}, 'Q7': {}, 'Q8': {}, 'Q9': {}, 'Q10': {}, 'Q11': {}, 'Q12': {}, 'Q13': {}, 'Q14': {}, 'Q15': {}}
avgAgreement = {}
for x in range(15):
    curQuestion = data.iloc[:, ((x*3)+2):(x*3+4)].values
    temp = 0
    for y in range(4):
        temp += max(curQuestion[y])
    avgAgreement['Q{}'.format(x+1)] = temp/4
sortedAgreement = {a: b for a, b in sorted(avgAgreement.items(), key=lambda item: item[1], reverse=True)}
outputString = 'Questions in order of discrimination power are: '
for key in sortedAgreement:
    outputString += '{} ({}%), '.format(key, sortedAgreement[key])
outputString = outputString[:-2]
print(outputString)

# Query 4
# Determine what is the closest question for a given question (Q1, Q2, …, Q15) based on results of the groups’ responses?
print("\n-- Query 4 --")
totalQuestionData = {}
for x in range(15):
    curQuestion = data.iloc[:, ((x*3)+2):(x*3+5)].values
    curQuestion = curQuestion.sum(axis=0)/4
    totalQuestionData[x+1] = curQuestion

similMatrix = np.zeros((15, 15))
for a, b in itertools.combinations(totalQuestionData, 2):
    similValue = compareTotals(totalQuestionData[a], totalQuestionData[b])
    similMatrix[a-1][b-1] = similValue
    similMatrix[b-1][a-1] = similValue

for x in range(15):
    bestMatch = np.argmax(similMatrix[:, x])
    matchValue = similMatrix[bestMatch, x]
    print("Q{}'s closest question is Q{}, with a {}% similarity".format(x+1, bestMatch+1, matchValue*100))

# Query 5
# What is the minimum number of questions to determine a person’s group with the accuracy of higher than 80%? What are those questions?
print("\n-- Query 5 --")
# Done on paper

# Query 6
# For each group, rank the questions based on the level of agreements among group members.
print("\n-- Query 6 --")
groupData = {   'G1': {},
                'G2': {},
                'G3': {},
                'G4': {}}
for x in range(15):
    curQuestion = data.iloc[:, ((x*3)+2):(x*3+4)].values
    for y in range(4):
        temp = max(curQuestion[y])
        groupData['G{}'.format(y+1)]['Q{}'.format(x+1)] = temp
for x in range(4):
    outputString = 'Group {} ranking for each question: '.format(x+1)
    sortedGroups = {a: b for a, b in sorted(groupData['G{}'.format(x+1)].items(), key=lambda item: item[1], reverse=True)}
    for key in sortedGroups:
        outputString += '{} ({}%), '.format(key, sortedGroups[key])
    outputString = outputString[:-2]
    print(outputString)