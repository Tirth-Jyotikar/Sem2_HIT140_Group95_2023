#Objective 1: Distinguishing People with PD

#Importing Libraires
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

#Importing Dataset
data = pd.read_csv('po1_data.txt', header=None, delimiter=',')

# Separating PPD and non-PPD groups
ppd_group = data[data[28] == 1]
non_ppd_group = data[data[28] == 0]



# Extract relevant columns (acoustic features)
acoustic_columns = list(range(1, 28))
ppd_acoustic_features = ppd_group[acoustic_columns]
non_ppd_acoustic_features = non_ppd_group[acoustic_columns]

#Calculating means and standard deviations for each group
ppd_means = ppd_group.mean()
non_ppd_means = non_ppd_group.mean()
ppd_stdevs = ppd_group.std()
non_ppd_stdevs = non_ppd_group.std()


# Performing t-test statistical tests for each feature
p_values = []
for col in acoustic_columns:
    t_statistic, p_value = stats.ttest_ind(ppd_acoustic_features[col], non_ppd_acoustic_features[col])
    p_values.append(p_value)

t_statistic, p_value = stats.ttest_ind(ppd_group, non_ppd_group)

# Extracting jitter and shimmer columns
jitter_columns = list(range(2, 7))
shimmer_columns = list(range(7, 13))
jitter_features = ppd_group[jitter_columns]
shimmer_features = ppd_group[shimmer_columns]

# Extracting pitch and pulse columns
pitch_columns = list(range(16, 21))
pulse_columns = list(range(21, 25))
pitch_features = ppd_group[pitch_columns]
pulse_features = ppd_group[pulse_columns]

# Extracting harmonicity and voice columns
harmonicity_columns = list(range(13, 16))
voice_columns = list(range(25, 28))
harmonicity_features = ppd_group[harmonicity_columns]
voice_features = ppd_group[voice_columns]

# Comparing feature distributions using box plots for jitter and shimmer - PPD group
plt.figure(figsize=(12, 6))
sns.boxplot(data=ppd_group.iloc[:, jitter_columns], orient='h')
plt.title("Jitter Features - PPD Group")
plt.show()

# Jitter feature distributions for non-PPD group
plt.figure(figsize=(12, 6))
sns.boxplot(data=non_ppd_group.iloc[:, jitter_columns], orient='h')
plt.title("Jitter Features - Non-PPD Group")
plt.show()

# Shimmer feature distributions for PPD group
plt.figure(figsize=(12, 6))
sns.boxplot(data=ppd_group.iloc[:, shimmer_columns], orient='h')
plt.title("Shimmer Features - PPD Group")
plt.show()

# Shimmer feature distributions for non-PPD group
plt.figure(figsize=(12, 6))
sns.boxplot(data=non_ppd_group.iloc[:, shimmer_columns], orient='h')
plt.title("Shimmer Features - Non-PPD Group")
plt.show()


# Comparing feature distributions using box plots for pitch and pulse - PPD group
plt.figure(figsize=(12, 6))
sns.boxplot(data=ppd_group.iloc[:, pitch_columns], orient='h')
plt.title("pitch Features - PPD Group")
plt.show()

# pitch feature distributions for non-PPD group
plt.figure(figsize=(12, 6))
sns.boxplot(data=non_ppd_group.iloc[:, pitch_columns], orient='h')
plt.title("pitch Features - Non-PPD Group")
plt.show()

# pulse feature distributions for PPD group
plt.figure(figsize=(12, 6))
sns.boxplot(data=ppd_group.iloc[:, pulse_columns], orient='h')
plt.title("pulse Features - PPD Group")
plt.show()

# pulse feature distributions for non-PPD group
plt.figure(figsize=(12, 6))
sns.boxplot(data=non_ppd_group.iloc[:, pulse_columns], orient='h')
plt.title("pulse Features - Non-PPD Group")
plt.show()




# Comparing feature distributions using box plots for harmonicity and voice - PPD group
plt.figure(figsize=(12, 6))
sns.boxplot(data=ppd_group.iloc[:, harmonicity_columns], orient='h')
plt.title("harmonicity Features - PPD Group")
plt.show()

# harmonicity feature distributions for non-PPD group
plt.figure(figsize=(12, 6))
sns.boxplot(data=non_ppd_group.iloc[:, harmonicity_columns], orient='h')
plt.title("harmonicity Features - Non-PPD Group")
plt.show()

# voice feature distributions for PPD group
plt.figure(figsize=(12, 6))
sns.boxplot(data=ppd_group.iloc[:, voice_columns], orient='h')
plt.title("voice Features - PPD Group")
plt.show()

# voice feature distributions for non-PPD group
plt.figure(figsize=(12, 6))
sns.boxplot(data=non_ppd_group.iloc[:, voice_columns], orient='h')
plt.title("voice Features - Non-PPD Group")
plt.show()



# Performing t-test statistical tests for jitter and shimmer features
jitter_p_values = []
shimmer_p_values = []

for col in jitter_columns:
    t_statistic, p_value = stats.ttest_ind(ppd_group[col], non_ppd_group[col])
    jitter_p_values.append(p_value)
for col in shimmer_columns:
    t_statistic, p_value = stats.ttest_ind(ppd_group[col], non_ppd_group[col])
    shimmer_p_values.append(p_value)
alpha_jitter = 0.05 / len(jitter_columns)
alpha_shimmer = 0.05 / len(shimmer_columns)
significant_jitter = [col for col, p_value in zip(jitter_columns, jitter_p_values) if p_value < alpha_jitter]
significant_shimmer = [col for col, p_value in zip(shimmer_columns, shimmer_p_values) if p_value < alpha_shimmer]

# Performing t-test statistical tests for pitch and pulse features
pitch_p_values = []
pulse_p_values = []

for col in pitch_columns:
    t_statistic, p_value = stats.ttest_ind(ppd_group[col], non_ppd_group[col])
    pitch_p_values.append(p_value)
for col in pulse_columns:
    t_statistic, p_value = stats.ttest_ind(ppd_group[col], non_ppd_group[col])
    pulse_p_values.append(p_value)
alpha_pitch = 0.05 / len(pitch_columns)
alpha_pulse = 0.05 / len(pulse_columns)
significant_pitch = [col for col, p_value in zip(pitch_columns, pitch_p_values) if p_value < alpha_pitch]
significant_pulse = [col for col, p_value in zip(pulse_columns, pulse_p_values) if p_value < alpha_pulse]

#Performing t-test statistical tests for harmonicity and voice features
harmonicity_p_values = []
voice_p_values = []

for col in harmonicity_columns:
    t_statistic, p_value = stats.ttest_ind(ppd_group[col], non_ppd_group[col])
    harmonicity_p_values.append(p_value)
for col in voice_columns:
    t_statistic, p_value = stats.ttest_ind(ppd_group[col], non_ppd_group[col])
    voice_p_values.append(p_value)
alpha_harmonicity = 0.05 / len(harmonicity_columns)
alpha_voice = 0.05 / len(voice_columns)
significant_harmonicity = [col for col, p_value in zip(harmonicity_columns, harmonicity_p_values) if p_value < alpha_harmonicity]
significant_voice = [col for col, p_value in zip(voice_columns, voice_p_values) if p_value < alpha_voice]



# Adjusting for multiple comparisons using Bonferroni correction
alpha = 0.05 / len(acoustic_columns)
significant_features = [col for col, p_value in zip(acoustic_columns, p_values) if p_value < alpha]
# Printing significant features
print("Significant jitter features (Bonferroni corrected p-value < {}):".format(alpha_jitter))
for col in significant_jitter:
    print("Column {}: {}".format(col, data.columns[col]))

print("Significant shimmer features (Bonferroni corrected p-value < {}):".format(alpha_shimmer))
for col in significant_shimmer:
    print("Column {}: {}".format(col, data.columns[col]))

print("Significant pitch features (Bonferroni corrected p-value < {}):".format(alpha_pitch))
for col in significant_pitch:
    print("Column {}: {}".format(col, data.columns[col]))

print("Significant pulse features (Bonferroni corrected p-value < {}):".format(alpha_pulse))
for col in significant_pulse:
    print("Column {}: {}".format(col, data.columns[col]))

print("Significant harmonicity features (Bonferroni corrected p-value < {}):".format(alpha_harmonicity))
for col in significant_harmonicity:
    print("Column {}: {}".format(col, data.columns[col]))

print("Significant voice features (Bonferroni corrected p-value < {}):".format(alpha_voice))
for col in significant_voice:
    print("Column {}: {}".format(col, data.columns[col]))




#Comparing feature distributions using box plots for voice samples ppd group
ppd_group
plt.figure(figsize=(12, 6))
sns.boxplot(data=ppd_group.iloc[:, 1:29], orient='h')
plt.title("Voice Sample Features - PPD Group")
plt.show()

#Comparing feature distributions using box plots for voice samples non-ppd group
plt.figure(figsize=(12, 6))
sns.boxplot(data=non_ppd_group.iloc[:, 1:29], orient='h')
plt.title("Voice Sample Features - Non-PPD Group")
plt.show()



print("PPD Group Means:\n", ppd_means)
print("Non-PPD Group Means:\n", non_ppd_means)
print("PPD Group Standard Deviations:\n", ppd_stdevs)
print("Non-PPD Group Standard Deviations:\n", non_ppd_stdevs)
print("T-statistic:", t_statistic)
print("P-value:", p_value)



