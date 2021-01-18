import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys



sns.set_style("darkgrid")

sns.set_theme()


file = 'results.csv'

if len(sys.argv) > 1:
    file = sys.argv[1]

results = np.recfromcsv(file).tolist()


results_avg = []

i = 0
sum = 0
for r in results:
    sum += r[0]
    i += 1
    if (i == 10):
        results_avg.append((sum/10))
        i = 0
        sum = 0

# df = pd.DataFrame(results)
 
# g = sns.relplot(kind="line", data=df)
# g.fig.autofmt_xdate()

# sns.lineplot(data=df, x="year", y="passengers")

plt.plot(results_avg, linewidth=0.9)
# plt.gca().set_xticklabels([0, 0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000])

plt.show()



