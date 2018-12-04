import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

column_label = 'clkD'
#column_label = 'traffic'

df = pd.read_csv('gpsdata')
#df = pd.read_csv('https://github.com/selva86/datasets/raw/master/AirPassengers.csv')
#data = df['clkD'].values
data = df[column_label].values

doublediff = np.diff(np.sign(np.diff(data)))
peak_locations = np.where(doublediff == -2)[0] + 1

doublediff2 = np.diff(np.sign(np.diff(-1*data)))
trough_locations = np.where(doublediff2 == -2)[0] + 1

# Draw Plot
plt.figure(figsize=(16,10), dpi= 80)
x = range(len(data))
plt.plot(x, column_label, data=df, color='tab:blue', label='Air clkD')
#plt.scatter(df.week[peak_locations], df.clkD[peak_locations], marker=mpl.markers.CARETUPBASE, color='tab:green', s=100, label='Peaks')
#plt.scatter(df.week[trough_locations], df.clkD[trough_locations], marker=mpl.markers.CARETDOWNBASE, color='tab:red', s=100, label='Troughs')

"""
# Annotate
for t, p in zip(trough_locations[1::5], peak_locations[::3]):
    plt.text(x[p], df.clkD[p]+15, x[p], horizontalalignment='center', color='darkgreen')
    plt.text(x[t], df.clkD[t]-35, x[t], horizontalalignment='center', color='darkred')
"""

# Decoration
plt.ylim(50,750)
xtick_location = df.index.tolist()[::6]
xtick_labels = df.week.tolist()[::6]
#plt.xticks(ticks=xtick_location, labels=xtick_labels, rotation=90, fontsize=12, alpha=.7)
plt.title("Peak and Troughs of Air Passengers clkD (1949 - 1969)", fontsize=22)
plt.yticks(fontsize=12, alpha=.7)

# Lighten borders
plt.gca().spines["top"].set_alpha(.0)
plt.gca().spines["bottom"].set_alpha(.3)
plt.gca().spines["right"].set_alpha(.0)
plt.gca().spines["left"].set_alpha(.3)

plt.legend(loc='upper left')
plt.grid(axis='y', alpha=.3)
plt.show()