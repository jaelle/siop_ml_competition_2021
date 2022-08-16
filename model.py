import pandas as pd
import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn import linear_model

train = pd.read_csv("train.csv",na_values=" ",low_memory=True)
train.fillna(0, inplace=True)

allvars_train = train[0:7890]
allvars_train

retained_protected_train = train
retained_protected_train


perf_av = []
perf_rp = []

#performance variables
perf_av += [allvars_train['Overall_Rating']]
perf_av += [allvars_train['Technical_Skills']]
perf_av += [allvars_train['Teamwork']]
perf_av += [allvars_train['Customer_Service']]
perf_av += [allvars_train['Hire_Again']]
perf_av += [allvars_train['High_Performer']]
perf_rp += [retained_protected_train['Protected_Group']]
perf_rp += [retained_protected_train['Retained']]

#evaluation variables
evale = []

evale += [allvars_train['High_Performer']]
evale += [retained_protected_train['Protected_Group']]
evale += [retained_protected_train['Retained']]

X_allvars = allvars_train[allvars_train.columns[9:-1]]
X_retained_protected = retained_protected_train[retained_protected_train.columns[9:-1]]


predicted_av = []
predicted_rp = []

random_state = 0

for i in range(len(perf_av)):
    X_train_av, X_test_av, y_train_av, y_test_av = train_test_split(X_allvars, perf_av[i], test_size=0.2, random_state=random_state)

    clf = linear_model.LinearRegression()
    clf.fit(X_train_av, y_train_av)

    predicted_av += [clf.predict(X_test_av)]

predicted_av += [clf.predict(X_test_av)]
    
for i in range(len(perf_rp)):
    X_train_rp, X_test_rp, y_train_rp, y_test_rp = train_test_split(X_retained_protected, perf_rp[i], test_size=0.2, random_state=random_state)

    clf = linear_model.LinearRegression()
    clf.fit(X_train_rp, y_train_rp)
    
    predicted_rp += [clf.predict(X_test_rp)]
    
all_scores = []
for i in range(len(X_test_av)):
    
    score = 0
    score_vector = []
    
    for j in range(len(perf_av)):
        
        score += predicted_av[j][i]
        score_vector += [predicted_av[j][i]]
    
    for j in range(len(perf_rp)):
        
        score += predicted_rp[j][i]
        score_vector += [predicted_rp[j][i]]
    
    all_scores += [[score] + score_vector + [evale[0][X_test_av.index[i]],evale[1][X_test_av.index[i]],evale[2][X_test_av.index[i]]]]

df_scores = pd.DataFrame(all_scores,index=X_test_av.index,columns=['score','Overall_Rating','Technical_Skills','Teamwork','Customer_Service','Hire_Again','High_Performer','Protected_Group','Retained','Actual High Performer','Actual Protected Group','Actual Retained'])
sorted_scores = df_scores.sort_values(by='score',ascending=False,ignore_index=True)
half = int(sorted_scores.shape[0] / 2)

top50percent = sorted_scores[0:half]

high_performers_holdout = sorted_scores[sorted_scores['Actual High Performer'] == 1]
total_high_performers = high_performers_holdout.shape[0]

high_perfomers_selected = top50percent[top50percent['Actual High Performer'] == 1]
print("% of high performers selected")
selected_high_performers = high_perfomers_selected.shape[0]

percent_high_performers = selected_high_performers / total_high_performers

print(percent_high_performers)

retained_holdout = sorted_scores[sorted_scores['Actual Retained'] == 1]
total_retained = retained_holdout.shape[0]

retained_selected = top50percent[top50percent['Actual Retained'] == 1]
print("% of retained in selected")
selected_retained = retained_selected.shape[0]

percent_retained = selected_retained / total_retained

print(percent_retained)

high_perf_retained_holdout = sorted_scores[(sorted_scores['Actual Retained'].astype('int') == 1) & (sorted_scores['Actual High Performer'].astype('int') == 1)]
total_combined = high_perf_retained_holdout.shape[0]

high_perf_retained_selected = top50percent[(top50percent['Actual Retained'].astype('int') == 1) & (top50percent['Actual High Performer'].astype('int') == 1)]
print("% of high performing retained in test set")
selected_combined = high_perf_retained_selected.shape[0]

percent_combined = selected_combined / total_combined
print(percent_combined)

protected_holdout = sorted_scores[sorted_scores['Actual Protected Group'] == 1]
total_protected = protected_holdout.shape[0]

protected_selected = top50percent[top50percent['Actual Protected Group'] == 1]
selected_protected = protected_selected.shape[0]

percent_protected = selected_protected / total_protected

print(percent_protected)

nonprotected_holdout = sorted_scores[sorted_scores['Actual Protected Group'] == 0]
total_nonprotected = nonprotected_holdout.shape[0]

nonprotected_selected = top50percent[top50percent['Actual Protected Group'] == 0]
selected_nonprotected = nonprotected_selected.shape[0]

percent_nonprotected = selected_nonprotected / total_nonprotected

print(percent_nonprotected)

adverse_impact = percent_protected / percent_nonprotected
print(adverse_impact)

print("Final accuracy")

accuracy = (percent_high_performers * 0.25 + percent_retained * 0.25 + percent_combined * 0.5) - abs(1 - adverse_impact)
print(accuracy)

