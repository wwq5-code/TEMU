import matplotlib.pyplot as plt
import numpy as np

# user num = 50
labels = ['SISA', 'VBU', 'RFU', 'HBU']
#unl_fr = [10*10*0.22 *5, 10*10*0.22*5, 10*10*0.22 *5, 10*10*0.22*5 , 10*10*0.22*5  , 10*10*0.22*5  ]

unl_ss_in = [0.97303  , 0.972769 , 0.971316 , 0.97288 ]
unl_ss_not_in = [0.973511   , 0.97564    , 0.97500     , 0.976009]


unl_ms_in = [0.96608     , 0.966605   , 0.9661485 , 0.966252]
unl_ms_not_in = [0.967100   , 0.97029 , 0.97007  , 0.970016]


sisa_unl = [0.97303, 0.973511, 0.96608, 0.967100]
vbu_unl = [0.972769, 0.97564, 0.966605, 0.97029]
rfu_unl = [0.971316, 0.97500,  0.9661485, 0.97007]
hbu_unl = [0.97288, 0.976009, 0.966252,  0.970016]

x = np.arange(len(labels))  # the label locations
width = 0.7  # the width of the bars
# no_noise = np.around(no_noise,0)
# samping = np.around(samping,0)
# ldp = np.around(ldp,0)

# plt.style.use('bmh')

plt.style.use('seaborn')

plt.figure()
#plt.subplots(figsize=(8, 5.3))
#plt.bar(x - width / 2 - width / 8 + width / 8, unl_fr, width=0.168, label='Retrain', color='dodgerblue', hatch='/')

plt.bar(x - width / 2 - width / 8 + width / 8 , unl_ss_in,   width=0.21, label='SS In', color='#9BC985', edgecolor='black', hatch='/')
plt.bar(x - width / 8 - width / 16,  unl_ss_not_in, width=0.21, label='SS Not In', color='#F7D58B', edgecolor='black', hatch='*')
plt.bar(x + width / 8, unl_ms_in, width=0.21, label='MS In', color='#B595BF',edgecolor='black', hatch='\\')
plt.bar( x + width / 2 - width / 8 + width / 16, unl_ms_not_in, width=0.21, label='MS Not In', color='#797BB7', edgecolor='black', hatch='x')





# plt.bar(x - width / 2.5 ,  unl_br, width=width/3, label='VBU', color='orange', hatch='\\')
# plt.bar(x,unl_self_r, width=width/3, label='RFU-SS', color='g', hatch='x')
# plt.bar(x + width / 2.5,  unl_hess_r, width=width/3, label='HBU', color='tomato', hatch='-')


# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel('Rec. Similarity', fontsize=20)
# ax.set_title('Performance of Different Users n')
plt.xticks(x, labels, fontsize=20)
# ax.set_xticklabels(labels,fontsize=15)

my_y_ticks = np.arange(0.9, 1.05, 0.02)
plt.yticks(my_y_ticks, fontsize=20)
plt.ylim(0.9,1.02)
# ax.set_yticklabels(my_y_ticks,fontsize=15)

# Set the background of the axes, which is the area of the plot, to grey
# plt.gca().set_facecolor('grey')

# Set the grid with white color and a specific linestyle and linewidth
# plt.grid(color='white', linestyle='-', linewidth=0.5)

# plt.grid(axis='y')

plt.legend(loc='upper left', fontsize=20)
# plt.xlabel('$\it{ESS}$' ,fontsize=20)
# ax.bar_label(rects1, padding=1)
# ax.bar_label(rects2, padding=3)
# ax.bar_label(rects3, padding=3)

plt.tight_layout()

plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('cifar10_rec_mse_unl_analysis_bar.pdf', format='pdf', dpi=200)
plt.show()
