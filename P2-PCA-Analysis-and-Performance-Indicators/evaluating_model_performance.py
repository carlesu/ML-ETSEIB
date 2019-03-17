# xlrd is needed in order to open excel files
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_excel('./files/Data_accuracyTrap.xlsx', sheet_name='Hoja1')


# We set threshold to 0.70 (our score goes from 0 to 1)
# So scores above 0.70 will be marked as infected.
trh = 0.70
# TruePositive, FalsePositive, TrueNegative, FalseNegative
tp=fp=tn=fn=0
# Now we check all the data to see how many tp, fp, tn, fn we have
# Since we have the real status of patient in Actual (1 infected, -1 Not infected).
real_state = df['Actual']
score = df['Model score']
for i, element in enumerate(score):
    # Si esta por encima del treshold y su estado real es infectado, es truepositive
    if element>=trh and real_state[i]==1:
        tp += 1
    # Si esta por encima del treshold y su estado es no infectado, es falepositive
    elif element>=trh and real_state[i]==-1:
        fp += 1
    elif element<=trh and real_state[i]==1:
        fn += 1
    elif element<=trh and real_state[i]==-1:
        tn += 1
# Performance indicators for theshold 0.70
accuracy = (tp + tn)/(tp + tn + fp + fn)
recall = tp/(tp + fn)
precision = tp/(tp + fp)
f_score = 2*recall*(precision/(recall + precision))
print('Performance indicators for threshold 0.70:')
print('Accuracy:', accuracy)
print('Recall:', recall)
print('Precision:', precision)
print('F-score:', f_score, '\n')


# For threshold 0.80.
trh = 0.80
# TruePositive, FalsePositive, TrueNegative, FalseNegative
tp=fp=tn=fn=0
# Now we check all the data to see how many tp, fp, tn, fn we have
# Since we have the real status of patient in Actual (1 infected, -1 Not infected).
real_state = df['Actual']
score = df['Model score']
for i, element in enumerate(score):
    # Si esta por encima del treshold y su estado real es infectado, es truepositive
    if element>=trh and real_state[i]==1:
        tp += 1
    # Si esta por encima del treshold y su estado es no infectado, es falepositive
    elif element>=trh and real_state[i]==-1:
        fp += 1
    elif element<=trh and real_state[i]==1:
        fn += 1
    elif element<=trh and real_state[i]==-1:
        tn += 1
# Performance indicators for teshold 0.70
accuracy = (tp + tn)/(tp + tn + fp + fn)
recall = tp/(tp + fn)
precision = tp/(tp + fp)
f_score = 2*recall*(precision/(recall + precision))
print('Performance indicators for threshold 0.80:')
print('Accuracy:', accuracy)
print('Recall:', recall)
print('Precision:', precision)
print('F-score:', f_score, '\n')


# False negatives are not tolerable here. A fn means that someone is infected
# but he treat it as he is healthy.
# Since a fp means someone who is healthy is treated as infected, is not that bad
# because he won't die, as in the case of fn; which untreated would die.
# So in this case we cant recall = 1.0
# In case of spam mail, we can tolerate fn (message which are spam, but are not marked
# as spam) but we cannot tolerate fp (message which is not spam, treated as spam).
# In this second case we would aim for precision = 1.0.
# Let's test for threshold = [0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95]
# Introducing FPR (false positive rate) = recall
# TPR (true positive rate
# The curve x-asxis(FPR) y-axis(TPR) is the ROC (Reciver operating Characteristics)
trh = [0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95]
real_state = df['Actual']
score = df['Model score']
TPR = []
FPR = []
for trhvalue in trh:
    tp = fp = tn = fn = 0.0
    for i, scorevalue in enumerate(score):
        if scorevalue >= trhvalue and real_state[i] == 1:
            tp += 1
        elif scorevalue >= trhvalue and real_state[i] == -1:
            fp += 1
        elif scorevalue <= trhvalue and real_state[i] == 1:
            fn += 1
        elif scorevalue <= trhvalue and real_state[i] == -1:
            tn += 1
    recall = tp/(tp+fn)
    TPR.append(recall)
    fpr = fp/(tn+fp)
    FPR.append(fpr)

plt.plot(FPR, TPR, 'g')
# So what matters here is that we want fn = 0, which implies reacall = TPR = 1
# on the graphic we can see that for trh = 0.70 we have the TPR = 1.0.
# So that would be okay to set the threshold to 0.70.
