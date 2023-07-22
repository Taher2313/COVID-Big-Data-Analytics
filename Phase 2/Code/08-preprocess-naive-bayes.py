import pandas as pd

# read the CSV file into a DataFrame
df_train = pd.read_csv('dataset/04-preprocessed/train.csv')
df_test = pd.read_csv('dataset/04-preprocessed/test.csv')

# define the bin edges
bin_edges = list(range(0, 81, 10)) + [float('inf')]
bin_labels = [f'{i}-{i+9}' for i in range(0, 80, 10)] + ['80+']

# create the age bins
df_train['AGE_BIN'] = pd.cut(df_train['AGE'], bins=bin_edges, labels=bin_labels)
df_test['AGE_BIN'] = pd.cut(df_test['AGE'], bins=bin_edges, labels=bin_labels)

# drop the AGE column
df_train = df_train.drop(columns=['AGE'])
df_test = df_test.drop(columns=['AGE'])

# rename the AGE_BIN column to AGE
df_train = df_train.rename(columns={'AGE_BIN': 'AGE'})
df_test = df_test.rename(columns={'AGE_BIN': 'AGE'})

# move the DIED column to the end
died_col = df_train.pop('DIED')
df_train['DIED'] = died_col
died_col = df_test.pop('DIED')
df_test['DIED'] = died_col


# save the DataFrame to a new CSV file
df_train.to_csv('dataset/05-naive-bayes/train.csv', index=False)
df_test.to_csv('dataset/05-naive-bayes/test.csv', index=False)