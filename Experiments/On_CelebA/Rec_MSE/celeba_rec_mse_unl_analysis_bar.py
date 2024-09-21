import matplotlib.pyplot as plt
import numpy as np

# user num = 50
labels = ['SISA', 'VBU', 'RFU', 'HBU']
#unl_fr = [10*10*0.22 *5, 10*10*0.22*5, 10*10*0.22 *5, 10*10*0.22*5 , 10*10*0.22*5  , 10*10*0.22*5  ]

unl_ss_in = [0.977335  , 0.97762 , 0.964420 , 0.977701 ]
unl_ss_not_in = [0.979339   , 0.98005    , 0.966258     , 0.979951]


unl_ms_in = [0.96394     , 0.9641662   , 0.963810 , 0.965032]
unl_ms_not_in = [0.9669511   , 0.96662 , 0.9661702  , 0.9672170]


sisa_unl = [0.977335, 0.979339, 0.96394, 0.9669511]
vbu_unl = [0.97762, 0.98005, 0.9641662, 0.96662]
rfu_unl = [0.964420, 0.966258, 0.963810, 0.9661702]
hbu_unl = [0.977701, 0.979951, 0.965032,  0.9672170]

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

my_y_ticks = np.arange(0.9, 1.03, 0.02)
plt.yticks(my_y_ticks, fontsize=20)
plt.ylim(0.9,1.03)
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
plt.savefig('celeba_rec_mse_unl_analysis_bar.pdf', format='pdf', dpi=200)
plt.show()
