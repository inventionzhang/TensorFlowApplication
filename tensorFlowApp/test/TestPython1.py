# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt


fig = plt.figure()
ax = fig.add_subplot(111)
plot_data=[1.7,1.7,1.7,1.54,1.52]
xdata = range(len(plot_data))
labels = ["2009-June","2009-Dec","2010-June","2010-Dec","2011-June"]
ax.plot(xdata,plot_data,"b-")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticks([1.4,1.6,1.8])

# grow the y axis down by 0.05
ax.set_ylim(1.35, 1.8)
# expand the x axis by 0.5 at two ends
ax.set_xlim(-0.5, len(labels)-0.5)
ax.set_title(u'行程',fontsize=18)

plt.show()
print ("done")