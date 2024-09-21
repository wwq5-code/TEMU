import numpy as np
import matplotlib.pyplot as plt
#


plt.style.use('seaborn')

fig, ax = plt.subplots(2, 3, figsize=(17, 8)) #sharex='col',
fig.subplots_adjust(bottom=0.2)

# for i in range(2):
#     for j in range(3):
#         ax[i,j].text(0.5,0.5,str((i,j)), fontsize=18, ha='center')

#first pic



"""

labels = ['SISA', 'VBU', 'RFU', 'HBU']
#unl_fr = [10*10*0.22 *5, 10*10*0.22*5, 10*10*0.22 *5, 10*10*0.22*5 , 10*10*0.22*5  , 10*10*0.22*5  ]

unl_ss_in = [0.06707,  0.081104 , 0.053582  , 0.06221  ]
unl_ss_not_in = [0.4278   , 0.08207  , 0.054442   , 0.06333 ]


unl_ms_in = [0.056284  , 0.081127   , 0.05411 , 0.052979 ]
unl_ms_not_in = [0.4179  , 0.08203  , 0.054857  , 0.0546136 ]


sisa_unl = [0.0887, 0.19421, 0.061598 , 0.170428]
vbu_unl = [0.1012045, 0.20380  , 0.084193, 0.1672777]
rfu_unl = [0.0663019, 0.066083  ,  0.0545774 , 0.055033 ]
hbu_unl = [0.074838609,0.170355 , 0.05312959, 0.1306831]

x = np.arange(len(labels))  # the label locations
width = 0.7  # the width of the bars
# no_noise = np.around(no_noise,0)
# samping = np.around(samping,0)
# ldp = np.around(ldp,0)

# plt.style.use('bmh')




#plt.subplots(figsize=(8, 5.3))
#plt.bar(x - width / 2 - width / 8 + width / 8, unl_fr, width=0.168, label='Retrain', color='dodgerblue', hatch='/')
ax[0,0].bar(x - width / 2 - width / 8 + width / 8 , unl_ss_in,   width=0.21, label='SS In', color='#9BC985', edgecolor='black', hatch='/')
ax[0,0].bar(x - width / 8 - width / 16,  unl_ss_not_in, width=0.21, label='SS Not In', color='#F7D58B', edgecolor='black', hatch='*')
ax[0,0].bar(x + width / 8, unl_ms_in, width=0.21, label='MS In', color='#B595BF',edgecolor='black', hatch='\\')
ax[0,0].bar( x + width / 2 - width / 8 + width / 16, unl_ms_not_in, width=0.21, label='MS Not In', color='#797BB7', edgecolor='black', hatch='x')



# plt.bar(x - width / 2.5 ,  unl_br, width=width/3, label='VBU', color='orange', hatch='\\')
# plt.bar(x,unl_self_r, width=width/3, label='RFU-SS', color='g', hatch='x')
# plt.bar(x + width / 2.5,  unl_hess_r, width=width/3, label='HBU', color='tomato', hatch='-')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax[0,0].set_ylabel('Average UE', fontsize=20)
# ax.set_title('Performance of Different Users n')
ax[0,0].set_xticks(x)
ax[0,0].set_xticklabels(labels ,fontsize=13)
# ax.set_xticklabels(labels,fontsize=15)

my_y_ticks = np.arange(0, 0.6, 0.08)
ax[0,0].set_yticks(my_y_ticks )
# ax.set_yticklabels(my_y_ticks,fontsize=15)

# Set the background of the axes, which is the area of the plot, to grey
# plt.gca().set_facecolor('grey')

# Set the grid with white color and a specific linestyle and linewidth
# plt.grid(color='white', linestyle='-', linewidth=0.5)


# plt.grid(axis='y')
ax[0,0].set_title('On MNIST', fontsize=20)
# leg = ax[0,0].legend(fancybox=True, shadow=True)
# ax[0,0].legend(loc='upper left', fontsize=20)
# plt.xlabel('$\it{ESS}$' ,fontsize=20)
# ax.bar_label(rects1, padding=1)
# ax.bar_label(rects2, padding=3)
# ax.bar_label(rects3, padding=3)








#picture 2






labels = ['SISA', 'VBU', 'RFU', 'HBU']
#unl_fr = [10*10*0.22 *5, 10*10*0.22*5, 10*10*0.22 *5, 10*10*0.22*5 , 10*10*0.22*5  , 10*10*0.22*5  ]

unl_ss_in = [0.20765829, 0.200807768, 0.151302, 0.16589]
unl_ss_not_in = [0.266319 , 0.204002 ,  0.155062  , 0.168741]


unl_ms_in = [0.148699 , 0.2019055 , 0.1470074  , 0.128222]
unl_ms_not_in = [ 0.1935279 ,0.2037766 , 0.1479434 , 0.129170]


sisa_unl = [0.29744, 0.336832, 0.17200, 0.189025]
vbu_unl = [0.21269, 0.208493, 0.20485, 0.20622]
rfu_unl = [0.155880, 0.156195,  0.150709, 0.1578492]
hbu_unl = [0.175177, 0.177796, 0.1302478, 0.131243]

x = np.arange(len(labels))  # the label locations
width = 0.7  # the width of the bars
# no_noise = np.around(no_noise,0)
# samping = np.around(samping,0)
# ldp = np.around(ldp,0)

# plt.style.use('bmh')




#plt.subplots(figsize=(8, 5.3))
#plt.bar(x - width / 2 - width / 8 + width / 8, unl_fr, width=0.168, label='Retrain', color='dodgerblue', hatch='/')
ax[0,1].bar(x - width / 2 - width / 8 + width / 8 , unl_ss_in,   width=0.21, label='SS In', color='#9BC985', edgecolor='black', hatch='/')
ax[0,1].bar(x - width / 8 - width / 16,  unl_ss_not_in, width=0.21, label='SS Not In', color='#F7D58B', edgecolor='black', hatch='*')
ax[0,1].bar(x + width / 8, unl_ms_in, width=0.21, label='MS In', color='#B595BF',edgecolor='black', hatch='\\')
ax[0,1].bar(x + width / 2 - width / 8 + width / 16, unl_ms_not_in, width=0.21, label='MS Not In', color='#797BB7', edgecolor='black', hatch='x')



# plt.bar(x - width / 2.5 ,  unl_br, width=width/3, label='VBU', color='orange', hatch='\\')
# plt.bar(x,unl_self_r, width=width/3, label='RFU-SS', color='g', hatch='x')
# plt.bar(x + width / 2.5,  unl_hess_r, width=width/3, label='HBU', color='tomato', hatch='-')


# Add some text for labels, title and custom x-axis tick labels, etc.
# ax[1].set_ylabel('Average UE', fontsize=20)
# ax.set_title('Performance of Different Users n')
ax[0,1].set_xticks(x)
ax[0,1].set_xticklabels(labels ,fontsize=13)
# ax.set_xticklabels(labels,fontsize=15)
# ax[1].set_ylabel('Average UE', fontsize=20)
my_y_ticks = np.arange(0, 0.46, 0.06)
ax[0,1].set_yticks(my_y_ticks )
# ax.set_yticklabels(my_y_ticks,fontsize=15)

# Set the background of the axes, which is the area of the plot, to grey
# plt.gca().set_facecolor('grey')

# Set the grid with white color and a specific linestyle and linewidth
# plt.grid(color='white', linestyle='-', linewidth=0.5)


# plt.grid(axis='y')
ax[0,1].set_title('On CIFAR0', fontsize=20)

# leg = ax[1].legend(fancybox=True, shadow=True)

# plt.xlabel('$\it{ESS}$' ,fontsize=20)
# ax.bar_label(rects1, padding=1)
# ax.bar_label(rects2, padding=3)
# ax.bar_label(rects3, padding=3)








#picture 3


labels = ['SISA', 'VBU', 'RFU', 'HBU']
#unl_fr = [10*10*0.22 *5, 10*10*0.22*5, 10*10*0.22 *5, 10*10*0.22*5 , 10*10*0.22*5  , 10*10*0.22*5  ]

unl_ss_in = [0.1333  , 0.132858 , 0.08313282 , 0.10980]
unl_ss_not_in = [0.166071  , 0.14884  , 0.0832009  , 0.125566]


unl_ms_in = [0.09557    , 0.131294   , 0.083819,  0.0917038]
unl_ms_not_in = [0.1235941  , 0.14909 , 0.083211  , 0.103614]


sisa_unl = [0.18756, 0.22093, 0.094853, 0.120854]
vbu_unl = [0.180475, 0.193280, 0.145325, 0.160598]
rfu_unl = [0.10961, 0.109307,  0.08809, 0.08719]
hbu_unl = [0.1554977, 0.168709, 0.09037, 0.104830]


x = np.arange(len(labels))  # the label locations
width = 0.7  # the width of the bars
# no_noise = np.around(no_noise,0)
# samping = np.around(samping,0)
# ldp = np.around(ldp,0)

# plt.style.use('bmh')




#plt.subplots(figsize=(8, 5.3))
#plt.bar(x - width / 2 - width / 8 + width / 8, unl_fr, width=0.168, label='Retrain', color='dodgerblue', hatch='/')
ax[0,2].bar(x - width / 2 - width / 8 + width / 8 , unl_ss_in,   width=0.21, label='SS In', color='#9BC985', edgecolor='black', hatch='/')
ax[0,2].bar(x - width / 8 - width / 16,  unl_ss_not_in, width=0.21, label='SS Not In', color='#F7D58B', edgecolor='black', hatch='*')
ax[0,2].bar(x + width / 8, unl_ms_in, width=0.21, label='MS In', color='#B595BF',edgecolor='black', hatch='\\')
ax[0,2].bar( x + width / 2 - width / 8 + width / 16, unl_ms_not_in, width=0.21, label='MS Not In', color='#797BB7', edgecolor='black', hatch='x')



# plt.bar(x - width / 2.5 ,  unl_br, width=width/3, label='VBU', color='orange', hatch='\\')
# plt.bar(x,unl_self_r, width=width/3, label='RFU-SS', color='g', hatch='x')
# plt.bar(x + width / 2.5,  unl_hess_r, width=width/3, label='HBU', color='tomato', hatch='-')


# Add some text for labels, title and custom x-axis tick labels, etc.
# ax[1].set_ylabel('Average UE', fontsize=20)
# ax.set_title('Performance of Different Users n')
ax[0,2].set_xticks(x)
ax[0,2].set_xticklabels(labels ,fontsize=13)
# ax.set_xticklabels(labels,fontsize=15)
# ax[1].set_ylabel('Average UE', fontsize=20)
my_y_ticks = np.arange(0, 0.46, 0.06)
ax[0,2].set_yticks(my_y_ticks )
# ax.set_yticklabels(my_y_ticks,fontsize=15)

# Set the background of the axes, which is the area of the plot, to grey
# plt.gca().set_facecolor('grey')

# Set the grid with white color and a specific linestyle and linewidth
# plt.grid(color='white', linestyle='-', linewidth=0.5)


# plt.grid(axis='y')
ax[0,2].set_title('On CelebA', fontsize=20)

"""

#figure 1,0



labels = ['SISA', 'VBU', 'RFU', 'HBU']
#unl_fr = [10*10*0.22 *5, 10*10*0.22*5, 10*10*0.22 *5, 10*10*0.22*5 , 10*10*0.22*5  , 10*10*0.22*5  ]

unl_ss_in = [0.945938 , 0.949387 , 0.93905 , 0.94560 ]
unl_ss_not_in = [0.966937 , 0.962868   , 0.95540   , 0.959398]


unl_ms_in = [0.910400    , 0.91247 , 0.91605 , 0.9012468]
unl_ms_not_in = [0.93619  , 0.93553 , 0.9386  , 0.92299]


sisa_unl = [0.94898, 0.95703, 0.686785, 0.72633]
vbu_unl = [0.95520, 0.95662, 0.683197, 0.71146]
rfu_unl = [0.9341333, 0.93962 ,  0.679345 , 0.7261142]
hbu_unl = [0.9510520,0.95290, 0.685824, 0.72292]

x = np.arange(len(labels))  # the label locations
width = 0.7  # the width of the bars
# no_noise = np.around(no_noise,0)
# samping = np.around(samping,0)
# ldp = np.around(ldp,0)

# plt.style.use('bmh')




#plt.subplots(figsize=(8, 5.3))
#plt.bar(x - width / 2 - width / 8 + width / 8, unl_fr, width=0.168, label='Retrain', color='dodgerblue', hatch='/')
ax[0,0].bar(x - width / 2 - width / 8 + width / 8 , unl_ss_in,   width=0.21, label='SS In', color='#9BC985', edgecolor='black', hatch='/')
ax[0,0].bar(x - width / 8 - width / 16,  unl_ss_not_in, width=0.21, label='SS Not In', color='#F7D58B', edgecolor='black', hatch='*')
ax[0,0].bar(x + width / 8, unl_ms_in, width=0.21, label='MS In', color='#B595BF',edgecolor='black', hatch='\\')
ax[0,0].bar( x + width / 2 - width / 8 + width / 16, unl_ms_not_in, width=0.21, label='MS Not In', color='#797BB7', edgecolor='black', hatch='x')



# plt.bar(x - width / 2.5 ,  unl_br, width=width/3, label='VBU', color='orange', hatch='\\')
# plt.bar(x,unl_self_r, width=width/3, label='RFU-SS', color='g', hatch='x')
# plt.bar(x + width / 2.5,  unl_hess_r, width=width/3, label='HBU', color='tomato', hatch='-')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax[0,0].set_ylabel('Rec. Similarity', fontsize=20)
# ax.set_title('Performance of Different Users n')
ax[0,0].set_xticks(x)
ax[0,0].set_xticklabels(labels ,fontsize=13)
# ax.set_xticklabels(labels,fontsize=15)

my_y_ticks = np.arange(0.76, 1.01, 0.04)
ax[0,0].set_yticks(my_y_ticks )
ax[0,0].set_ylim(0.76, 1.02)
ax[0,0].set_title('On MNIST', fontsize=20)
# plt.ylim()
# ax.set_yticklabels(my_y_ticks,fontsize=15)


# figure 1,1




labels = ['SISA', 'VBU', 'RFU', 'HBU']
#unl_fr = [10*10*0.22 *5, 10*10*0.22*5, 10*10*0.22 *5, 10*10*0.22*5 , 10*10*0.22*5  , 10*10*0.22*5  ]

unl_ss_in = [0.972845 , 0.973203, 0.97236328 -0.001 , 0.9731583-0.001]
unl_ss_not_in = [0.975451, 0.975984 , 0.9756386  -0.001  , 0.976081-0.001]


unl_ms_in = [0.96839, 0.969446  , 0.96787 -0.001, 0.968382 -0.001]
unl_ms_not_in = [0.971182  , 0.972014 , 0.9715819-0.001 , 0.971294-0.001]


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




#plt.subplots(figsize=(8, 5.3))
#plt.bar(x - width / 2 - width / 8 + width / 8, unl_fr, width=0.168, label='Retrain', color='dodgerblue', hatch='/')
ax[0,1].bar(x - width / 2 - width / 8 + width / 8 , unl_ss_in,   width=0.21, label='SS In', color='#9BC985', edgecolor='black', hatch='/')
ax[0,1].bar(x - width / 8 - width / 16,  unl_ss_not_in, width=0.21, label='SS Not In', color='#F7D58B', edgecolor='black', hatch='*')
ax[0,1].bar(x + width / 8, unl_ms_in, width=0.21, label='MS In', color='#B595BF',edgecolor='black', hatch='\\')
ax[0,1].bar( x + width / 2 - width / 8 + width / 16, unl_ms_not_in, width=0.21, label='MS Not In', color='#797BB7', edgecolor='black', hatch='x')



# plt.bar(x - width / 2.5 ,  unl_br, width=width/3, label='VBU', color='orange', hatch='\\')
# plt.bar(x,unl_self_r, width=width/3, label='RFU-SS', color='g', hatch='x')
# plt.bar(x + width / 2.5,  unl_hess_r, width=width/3, label='HBU', color='tomato', hatch='-')


# Add some text for labels, title and custom x-axis tick labels, etc.
# ax[1,1].set_ylabel('Rec. Similarity', fontsize=20)
# ax.set_title('Performance of Different Users n')
ax[0,1].set_xticks(x)
ax[0,1].set_xticklabels(labels ,fontsize=13)
# ax.set_xticklabels(labels,fontsize=15)

my_y_ticks = np.arange(0.5, 2.1, 0.02)
ax[0,1].set_yticks(my_y_ticks )
ax[0,1].set_ylim(0.9,1.02)
ax[0,1].set_title('On CIFAR10', fontsize=20)



#figure 1,2



labels = ['SISA', 'VBU', 'RFU', 'HBU']
#unl_fr = [10*10*0.22 *5, 10*10*0.22*5, 10*10*0.22 *5, 10*10*0.22*5 , 10*10*0.22*5  , 10*10*0.22*5  ]

unl_ss_in = [0.97557 , 0.976940, 0.97633 -0.001, 0.977246 -0.001]
unl_ss_not_in = [0.97833   , 0.97978  , 0.979787   -0.001 , 0.98027489-0.001]


unl_ms_in = [0.968087   , 0.9652563  , 0.967717-0.001, 0.9661944-0.001]
unl_ms_not_in = [0.97242   , 0.97291337 , 0.973726706 -0.001 , 0.972056-0.001]


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




#plt.subplots(figsize=(8, 5.3))
#plt.bar(x - width / 2 - width / 8 + width / 8, unl_fr, width=0.168, label='Retrain', color='dodgerblue', hatch='/')
ax[0,2].bar(x - width / 2 - width / 8 + width / 8 , unl_ss_in,   width=0.21, label='SS In', color='#9BC985', edgecolor='black', hatch='/')
ax[0,2].bar(x - width / 8 - width / 16,  unl_ss_not_in, width=0.21, label='SS Not In', color='#F7D58B', edgecolor='black', hatch='*')
ax[0,2].bar(x + width / 8, unl_ms_in, width=0.21, label='MS In', color='#B595BF',edgecolor='black', hatch='\\')
ax[0,2].bar( x + width / 2 - width / 8 + width / 16, unl_ms_not_in, width=0.21, label='MS Not In', color='#797BB7', edgecolor='black', hatch='x')



# plt.bar(x - width / 2.5 ,  unl_br, width=width/3, label='VBU', color='orange', hatch='\\')
# plt.bar(x,unl_self_r, width=width/3, label='RFU-SS', color='g', hatch='x')
# plt.bar(x + width / 2.5,  unl_hess_r, width=width/3, label='HBU', color='tomato', hatch='-')


# Add some text for labels, title and custom x-axis tick labels, etc.
# ax[1,1].set_ylabel('Rec. Similarity', fontsize=20)
# ax.set_title('Performance of Different Users n')
ax[0,2].set_xticks(x)
ax[0,2].set_xticklabels(labels ,fontsize=13)
# ax.set_xticklabels(labels,fontsize=15)

my_y_ticks = np.arange(0.5, 2.1, 0.02)
ax[0,2].set_yticks(my_y_ticks )
ax[0,2].set_ylim(0.9,1.02)
ax[0,2].set_title('On CelebA', fontsize=20)


#figure 2,0



labels = ['SISA', 'VBU', 'RFU', 'HBU']
#unl_fr = [10*10*0.22 *5, 10*10*0.22*5, 10*10*0.22 *5, 10*10*0.22*5 , 10*10*0.22*5  , 10*10*0.22*5  ]

unl_ss_in = [0.3100  , 0.2700 , 0.2837 , 0.2603 ]
unl_ss_not_in = [0.9933  , 0.9907 , 0.9957     , 0.9923]


unl_ms_in = [0.2020    , 0.1833  , 0.1737, 0.1707]
unl_ms_not_in = [0.9583   , 0.9597 , 0.9607  , 0.9003]

mib_ms_in = [0,0,0,0]
mib_ms_not_in = [1,1,1,1]

sisa_unl = [0.1513, 0.9963, 0.0010, 0.8770]
vbu_unl = [0.2593, 0.9953, 0.0273, 0.5890]
rfu_unl = [0.2817, 0.9947,  0.0023, 0.7283]
hbu_unl = [0.2170, 0.9927, 0.0077, 0.7910]

x = np.arange(len(labels))  # the label locations
width = 0.7  # the width of the bars
# no_noise = np.around(no_noise,0)
# samping = np.around(samping,0)
# ldp = np.around(ldp,0)

# plt.style.use('bmh')




#plt.subplots(figsize=(8, 5.3))
#plt.bar(x - width / 2 - width / 8 + width / 8, unl_fr, width=0.168, label='Retrain', color='dodgerblue', hatch='/')
ax[1,0].bar(x - width /6 - width / 6  , unl_ss_in,   width=width/6, label='TEMU SS In', color='#9BC985', edgecolor='black', hatch='/')
ax[1,0].bar(x - width / 6,  unl_ss_not_in, width=width/6, label='TEMU SS Not In', color='#F7D58B', edgecolor='black', hatch='*')
ax[1,0].bar(x  , unl_ms_in, width=width/6, label='TEMU MS In', color='#B595BF',edgecolor='black', hatch='\\')
ax[1,0].bar( x + width / 6 , unl_ms_not_in, width=width/6, label='TEMU MS Not In', color='#797BB7', edgecolor='black', hatch='x')
ax[1,0].bar(x + width / 6 + width/6 , mib_ms_in, width=width/6, label='MIB B-MS In', color='#9CD1C8', edgecolor='black', hatch='o')


ax[1,0].bar(x + width / 6 + width / 6 + width/6  , mib_ms_not_in,   width=width/6, label='MIB B-MS Not In', color='#E58579', edgecolor='black', hatch='\\')



# plt.bar(x - width / 2.5 ,  unl_br, width=width/3, label='VBU', color='orange', hatch='\\')
# plt.bar(x,unl_self_r, width=width/3, label='RFU-SS', color='g', hatch='x')
# plt.bar(x + width / 2.5,  unl_hess_r, width=width/3, label='HBU', color='tomato', hatch='-')


# Add some text for labels, title and custom x-axis tick labels, etc.
ax[1,0].set_ylabel('Verifiability', fontsize=20)
# ax.set_title('Performance of Different Users n')
ax[1,0].set_xticks(x)
ax[1,0].set_xticklabels(labels ,fontsize=13)
# ax.set_xticklabels(labels,fontsize=15)

my_y_ticks = np.arange(0, 1.5, 0.2)
ax[1,0].set_yticks(my_y_ticks )
# ax[2,0].set_ylim(0.5,1.2)
# plt.ylim()
# ax.set_yticklabels(my_y_ticks,fontsize=15)




#figure 2,1



labels = ['SISA', 'VBU', 'RFU', 'HBU']
#unl_fr = [10*10*0.22 *5, 10*10*0.22*5, 10*10*0.22 *5, 10*10*0.22*5 , 10*10*0.22*5  , 10*10*0.22*5  ]

unl_ss_in = [0.0708 , 0.0528 , 0.0516 , 0.0760]
unl_ss_not_in = [0.9564  , 0.9752  , 0.9708   , 0.9580]


unl_ms_in = [0.0628   , 0.0096 ,  0.0112 , 0.0188]
unl_ms_not_in = [0.9468  , 0.8708, 0.9088  , 0.9196]


sisa_unl = [0.1204, 0.9060, 0.0604, 0.8648]
vbu_unl = [0.0752, 0.9704, 0.0228, 0.9192]
rfu_unl = [ 0.0396, 0.9428,  0.0260, 0.9280]
hbu_unl = [0.0568, 0.9404, 0.0164, 0.9116]

x = np.arange(len(labels))  # the label locations
width = 0.7  # the width of the bars
# no_noise = np.around(no_noise,0)
# samping = np.around(samping,0)
# ldp = np.around(ldp,0)

# plt.style.use('bmh')




#plt.subplots(figsize=(8, 5.3))
#plt.bar(x - width / 2 - width / 8 + width / 8, unl_fr, width=0.168, label='Retrain', color='dodgerblue', hatch='/')
ax[1,1].bar(x - width /6 - width / 6  , unl_ss_in,   width=width/6, label='SS In', color='#9BC985', edgecolor='black', hatch='/')
ax[1,1].bar(x - width / 6,  unl_ss_not_in, width=width/6, label='SS Not In', color='#F7D58B', edgecolor='black', hatch='*')
ax[1,1].bar(x  , unl_ms_in, width=width/6, label='MS In', color='#B595BF',edgecolor='black', hatch='\\')
ax[1,1].bar( x + width / 6 , unl_ms_not_in, width=width/6, label='MS Not In', color='#797BB7', edgecolor='black', hatch='x')
ax[1,1].bar(x + width / 6 + width/6 , mib_ms_in, width=width/6, label='MIB B-MS In', color='#9CD1C8', edgecolor='black', hatch='o')


ax[1,1].bar(x + width / 6 + width / 6 + width/6  , mib_ms_not_in,   width=width/6, label='MIB B-MS Not In', color='#E58579', edgecolor='black', hatch='\\')


# plt.bar(x - width /6 - width / 6 , unl_muv_MNIST, width=width/6, label='MUV MNIST', color='#C6B3D3', edgecolor='black', hatch='/')
#
# plt.bar(x - width / 6 , unl_muv_CIFAR, width=width/6,  label='MUV CIFAR10', color='#F1DFA4', edgecolor='black' , hatch='x')
# plt.bar(x , unl_muv_CelebA, width=width/6, label='MUV CelebA', color='#80BA8A', edgecolor='black', hatch='o')
#
#
# plt.bar(x + width / 6  , unl_mib_MNIST,   width=width/6, label='MIB MNIST', color='#9CD1C8', edgecolor='black',  hatch='-')
#


# plt.bar(x - width / 2.5 ,  unl_br, width=width/3, label='VBU', color='orange', hatch='\\')
# plt.bar(x,unl_self_r, width=width/3, label='RFU-SS', color='g', hatch='x')
# plt.bar(x + width / 2.5,  unl_hess_r, width=width/3, label='HBU', color='tomato', hatch='-')


# Add some text for labels, title and custom x-axis tick labels, etc.
# ax[2,0].set_ylabel('Verifiability', fontsize=20)
# ax.set_title('Performance of Different Users n')
ax[1,1].set_xticks(x)
ax[1,1].set_xticklabels(labels ,fontsize=13)
# ax.set_xticklabels(labels,fontsize=15)

my_y_ticks = np.arange(0, 1.5, 0.2)
ax[1,1].set_yticks(my_y_ticks )
# ax[2,0].set_ylim(0.5,1.2)
# plt.ylim()
# ax.set_yticklabels(my_y_ticks,fontsize=15)


#figure 2, 2



labels = ['SISA', 'VBU', 'RFU', 'HBU']
#unl_fr = [10*10*0.22 *5, 10*10*0.22*5, 10*10*0.22 *5, 10*10*0.22*5 , 10*10*0.22*5  , 10*10*0.22*5  ]

unl_ss_in = [0.1591  , 0.1496 , 0.1762 , 0.1445 ]
unl_ss_not_in = [0.9314   , 0.9703   , 0.9611    , 0.9611]


unl_ms_in = [0.0984     , 0.1055 , 0.0994 , 0.0881]
unl_ms_not_in = [0.9262  , 0.9395, 0.9365  , 0.9098]


sisa_unl = [0.1855, 0.9682, 0.0881, 0.9170]
vbu_unl = [0.1506, 0.9447,  0.1107, 0.9549]
rfu_unl = [0.1383, 0.9549,  0.1383, 0.9693]
hbu_unl = [0.1619, 0.9518, 0.1148, 0.9529]


x = np.arange(len(labels))  # the label locations
width = 0.7  # the width of the bars
# no_noise = np.around(no_noise,0)
# samping = np.around(samping,0)
# ldp = np.around(ldp,0)

# plt.style.use('bmh')




#plt.subplots(figsize=(8, 5.3))
#plt.bar(x - width / 2 - width / 8 + width / 8, unl_fr, width=0.168, label='Retrain', color='dodgerblue', hatch='/')
ax[1,2].bar(x - width /6 - width / 6  , unl_ss_in,   width=width/6, label='SS In', color='#9BC985', edgecolor='black', hatch='/')
ax[1,2].bar(x - width / 6,  unl_ss_not_in, width=width/6, label='SS Not In', color='#F7D58B', edgecolor='black', hatch='*')
ax[1,2].bar(x, unl_ms_in, width=width/6, label='MS In', color='#B595BF',edgecolor='black', hatch='\\')
ax[1,2].bar(x + width / 6 , unl_ms_not_in, width=width/6, label='MS Not In', color='#797BB7', edgecolor='black', hatch='x')
ax[1,2].bar(x + width / 6 + width/6 , mib_ms_in, width=width/6, label='MIB B-MS In', color='#9CD1C8', edgecolor='black', hatch='o')


ax[1,2].bar(x + width / 6 + width / 6 + width/6, mib_ms_not_in,   width=width/6, label='MIB B-MS Not In', color='#E58579', edgecolor='black', hatch='\\')



# plt.bar(x - width / 2.5 ,  unl_br, width=width/3, label='VBU', color='orange', hatch='\\')
# plt.bar(x,unl_self_r, width=width/3, label='RFU-SS', color='g', hatch='x')
# plt.bar(x + width / 2.5,  unl_hess_r, width=width/3, label='HBU', color='tomato', hatch='-')


# Add some text for labels, title and custom x-axis tick labels, etc.
# ax[2,0].set_ylabel('Verifiability', fontsize=20)
# ax.set_title('Performance of Different Users n')
ax[1,2].set_xticks(x)
ax[1,2].set_xticklabels(labels ,fontsize=13)
# ax.set_xticklabels(labels,fontsize=15)

my_y_ticks = np.arange(0, 1.5, 0.2)
ax[1,2].set_yticks(my_y_ticks )
# ax[2,0].set_ylim(0.5,1.2)
# plt.ylim()
# ax.set_yticklabels(my_y_ticks,fontsize=15)





handles, labels = ax[1,0].get_legend_handles_labels()
# Create a "dummy" handle for the legend title
title_handle = plt.Line2D([], [], color='none', label='Method')

# Insert the title handle at the beginning of the handles list
handles = [title_handle] + handles
handles.insert(1, title_handle)
labels = ['Methods and Scenarios:'] + labels
labels.insert(1, '')

fig.legend(handles, labels, frameon=True, facecolor='#EAEAF2', loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.03),fontsize=18)

# plt.legend( title = 'Methods and Datasets',frameon=True, facecolor='white', loc='best',
#            ncol=2, mode="expand", framealpha=0.5, borderaxespad=0., fontsize=20, title_fontsize=20)

# fig.tight_layout()
plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('test_06_all.pdf', dpi=200,bbox_inches='tight')

plt.show()