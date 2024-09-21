import matplotlib.pyplot as plt
import numpy as np

# user num = 50
labels = ['1', '20', '40', '60', '80', '100']
#unl_fr = [10*10*0.22 *5, 10*10*0.22*5, 10*10*0.22 *5, 10*10*0.22*5 , 10*10*0.22*5  , 10*10*0.22*5  ]

unl_hess_r = [516*20/6000*0.22*5 +2.2*5 , 338*20/6000*0.22*5 +2.2*5 , 338*20/6000*0.22*5 +2.2*5 , 439*20/6000*0.22*5 +2.2*5 , 389*20/6000*0.22*5 +2.2 *5, 1200*20/6000*0.22*2*5]
unl_br = [72*20/6000*0.22*5           , 72*20/6000*0.22   *5    , 43*20/6000*0.22 *5      , 45*20/6000*0.22*5, 59*20/6000*0.22*5, 1200*20/6000*0.22*2*5 ]


unl_muv = [37.612, 37.5137  ,32.11904, 37.4105, 37.3426, 37.1530 ]
unl_mib = [1232.582, 1131.4467 , 1225.5774, 1350.929, 1327.0406, 1505.699]


for i in range(len(labels)):
    unl_muv[i] = unl_muv[i]/1000
    unl_mib[i] = unl_mib[i] / 1000


x = np.arange(len(labels))  # the label locations
width = 0.9 # the width of the bars
# no_noise = np.around(no_noise,0)
# samping = np.around(samping,0)
# ldp = np.around(ldp,0)

plt.style.use('seaborn')
plt.figure()
#plt.subplots(figsize=(8, 5.3))
#plt.bar(x - width / 2 - width / 8 + width / 8, unl_fr, width=0.168, label='Retrain', color='dodgerblue', hatch='/')

plt.bar(x - width / 8 - width / 16, unl_muv, width=0.268,  label='MUV', color='g', hatch='x')
plt.bar(x + width / 8, unl_mib, width=0.268, label='MIB', color='orange', hatch='\\')


# plt.bar(x - width / 2 - width / 8 + width / 8 , unl_br,   width=0.168, label='VBU', color='r', hatch='/')
# plt.bar(x - width / 8 - width / 16, unl_vib, width=0.168, label='PriMU$_{w}$', color='cornflowerblue', hatch='*')
# plt.bar(x + width / 8, unl_self_r, width=0.168, label='PriMU$_{w/o}$', color='g', hatch='x')
# plt.bar(x + width / 2 - width / 8 + width / 16, unl_hess_r, width=0.168, label='HBFU', color='orange', hatch='\\')


# plt.bar(x - width / 2.5 ,  unl_br, width=width/3, label='VBU', color='orange', hatch='\\')
# plt.bar(x,unl_self_r, width=width/3, label='RFU-SS', color='g', hatch='x')
# plt.bar(x + width / 2.5,  unl_hess_r, width=width/3, label='HBU', color='tomato', hatch='-')


# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel('Running Time (s)', fontsize=20)
# ax.set_title('Performance of Different Users n')
plt.xticks(x, labels, fontsize=20)
# ax.set_xticklabels(labels,fontsize=15)

my_y_ticks = np.arange(0, 1.3, 0.2)
plt.yticks(my_y_ticks, fontsize=20)
# ax.set_yticklabels(my_y_ticks,fontsize=15)
# plt.grid(axis='y')
plt.legend(loc='upper left', fontsize=20)
plt.xlabel('$\it{ESS}$' ,fontsize=20)
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
plt.savefig('celeba_rt_sample_size_bar.pdf', format='pdf', dpi=200)
plt.show()
