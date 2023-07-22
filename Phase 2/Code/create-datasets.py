import subprocess

scripts = [
    '01-prepare-dataset.py',
    '02-split-dataset.py',
    '04-preprocess.py',
    '08-preprocess-naive-bayes.py',
]

for script in scripts:
    print('Running script: {}'.format(script))
    subprocess.call(['python', script])
    print('Done!')

print('All datasets created!')