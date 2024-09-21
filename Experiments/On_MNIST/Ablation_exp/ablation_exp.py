import numpy as np
import  matplotlib.pyplot as plt
#


plt.style.use('seaborn')

fig, ax = plt.subplots(1, 3, figsize=(17, 4)) #sharex='col',
fig.subplots_adjust(bottom=0.16)

# for i in range(2):
#     for j in range(3):
#         ax[i,j].text(0.5,0.5,str((i,j)), fontsize=18, ha='center')




#picture 1





x=[1, 2, 3, 4, 5, 6]
# validation_for_plt =[97,95.8600, 94.9400, 93.5400, 93.2400]
# attack_for_plt=[0, 0.3524, 0, 0.1762, 0.1762]
# basic_for_plt=[99.8, 99.8, 99.8, 99.8, 99.8]

labels = ['1', '20', '40', '60', '80', '100' ]
# unl_org = [97.77, 97.55, 97.35, 97.29, 97.21, 97.21]

UEV = [0.96693, 0.93609, 0.913304, 0.88554, 0.8717471, 0.809093]
UEV_no_mask = [0.96581, 0.93200, 0.90528, 0.8762532, 0.86064, 0.80541]
# unl_hess_r = [96.6, 96.66, 96.04, 95.94, 95.85, 97.21]
UEV_no_division = [0.968303, 0.76445, 0.752777, 0.75964, 0.75321, 0.75159]

UEV_no_both = [0.96632, 0.739603, 0.736120, 0.73471, 0.73322, 0.73321]



l_w=3.5
m_s=8
marker_s = 2
markevery=1
#plt.figure(figsize=(8, 5.3))
#plt.plot(x, unl_fr, color='blue', marker='^', label='Retrain',linewidth=l_w, markersize=m_s)
ax[0].plot(x, UEV, linestyle='-', color='#797BB7', marker='o', fillstyle='full', markevery=markevery,
         label='Includes', linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

#plt.plot(x, unl_ss_w, color='g',  marker='*',  label='PriMU$_{w}$',linewidth=l_w, markersize=m_s)
ax[0].plot(x, UEV_no_mask, linestyle='--', color='#9BC985',  marker='s', fillstyle='full', markevery=markevery,
         label='Not Includes',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)


# plt.plot(x, unl_mib_bck, linestyle=':', color='r',  marker='^', fillstyle='none', markevery=markevery,
#          label='MIB (bac.)', linewidth=l_w,  markersize=m_s, markeredgewidth=marker_s)


ax[0].plot(x, UEV_no_division, linestyle='-.', color='#2A5522',  marker='D', fillstyle='full', markevery=markevery,
         label='MIB (Normal Not In.)',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)


ax[0].plot(x, UEV_no_both, linestyle='-.', color='#E1C855',  marker='^', fillstyle='full', markevery=markevery,
         label='MIB (Normal Not In.)',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)



#plt.plot(x, unl_vibu, color='silver',  marker='d',  label='VIBU',linewidth=4,  markersize=10)

# plt.plot(x, y_sa03, color='r',  marker='2',  label='AAAI21 A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_sa05, color='darkblue',  marker='4',  label='AAAI21 A_acc, pr=0.5',linewidth=3, markersize=8)
# plt.plot(x, y_ma03, color='darkviolet',  marker='3',  label='FedMC A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_ma05, color='cyan',  marker='p',  label='FedMC A_acc, pr=0.5',linewidth=3, markersize=8)


# plt.grid()
leg = plt.legend(fancybox=True, shadow=True)
# plt.xlabel('Malicious Client Ratio (%)' ,fontsize=16)
ax[0].set_ylabel('Rec. Similarity', fontsize=18)
my_y_ticks = np.arange(0.6, 1.01, 0.08)
ax[0].set_yticks(my_y_ticks )
ax[0].set_xlabel('$\it{ESS}$' ,fontsize=18)

ax[0].set_xticklabels(labels ,fontsize=13)
ax[0].set_xticks(x)
# plt.title('CIFAR10 IID')









#picture 2

x=[1, 2, 3, 4, 5, 6]
# validation_for_plt =[97,95.8600, 94.9400, 93.5400, 93.2400]
# attack_for_plt=[0, 0.3524, 0, 0.1762, 0.1762]
# basic_for_plt=[99.8, 99.8, 99.8, 99.8, 99.8]

labels = ['1', '20', '40', '60', '80', '100' ]
# unl_org = [97.77, 97.55, 97.35, 97.29, 97.21, 97.21]

UEV = [0.9933, 0.9583, 0.9083, 0.8550, 0.8700, 0.8410]
UEV_no_mask = [0.9857, 0.9340, 0.8803, 0.6980, 0.7347, 0.5997]
# unl_hess_r = [96.6, 96.66, 96.04, 95.94, 95.85, 97.21]
UEV_no_division = [0.9717, 0.8967, 0.8977, 0.7097, 0.7943, 0.5703]

UEV_no_both = [0.9790, 0.9293, 0.7603,0.5900, 0.6453, 0.5697]



#plt.figure(figsize=(8, 5.3))
#plt.plot(x, unl_fr, color='blue', marker='^', label='Retrain',linewidth=l_w, markersize=m_s)
ax[1].plot(x, UEV, linestyle='-', color='#797BB7', marker='o', fillstyle='full', markevery=markevery,
         label='Includes', linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

#plt.plot(x, unl_ss_w, color='g',  marker='*',  label='PriMU$_{w}$',linewidth=l_w, markersize=m_s)
ax[1].plot(x, UEV_no_mask, linestyle='--', color='#9BC985',  marker='s', fillstyle='full', markevery=markevery,
         label='Not Includes',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)


# plt.plot(x, unl_mib_bck, linestyle=':', color='r',  marker='^', fillstyle='none', markevery=markevery,
#          label='MIB (bac.)', linewidth=l_w,  markersize=m_s, markeredgewidth=marker_s)


ax[1].plot(x, UEV_no_division, linestyle='-.', color='#2A5522',  marker='D', fillstyle='full', markevery=markevery,
         label='MIB (Normal Not In.)',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)


ax[1].plot(x, UEV_no_both, linestyle='-.', color='#E1C855',  marker='^', fillstyle='full', markevery=markevery,
         label='MIB (Normal Not In.)',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)




#plt.plot(x, unl_vibu, color='silver',  marker='d',  label='VIBU',linewidth=4,  markersize=10)

# plt.plot(x, y_sa03, color='r',  marker='2',  label='AAAI21 A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_sa05, color='darkblue',  marker='4',  label='AAAI21 A_acc, pr=0.5',linewidth=3, markersize=8)
# plt.plot(x, y_ma03, color='darkviolet',  marker='3',  label='FedMC A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_ma05, color='cyan',  marker='p',  label='FedMC A_acc, pr=0.5',linewidth=3, markersize=8)


# plt.grid()
leg = plt.legend(fancybox=True, shadow=True)
# plt.xlabel('Malicious Client Ratio (%)' ,fontsize=16)
ax[1].set_ylabel('Verifiability', fontsize=18)
my_y_ticks = np.arange(0, 1.1, 0.2)
ax[1].set_yticks(my_y_ticks )
ax[1].set_xlabel('$\it{ESS}$' ,fontsize=18)

ax[1].set_xticklabels(labels ,fontsize=13)
ax[1].set_xticks(x)
# plt.title('CIFAR10 IID')


# figure 3


x=[1, 2, 3, 4, 5, 6]
# validation_for_plt =[97,95.8600, 94.9400, 93.5400, 93.2400]
# attack_for_plt=[0, 0.3524, 0, 0.1762, 0.1762]
# basic_for_plt=[99.8, 99.8, 99.8, 99.8, 99.8]

labels = ['1', '20', '40', '60', '80', '100']
# unl_org = [97.77, 97.55, 97.35, 97.29, 97.21, 97.21]

UEV = [141.4497, 116.9057  , 118.2206 , 119.850, 118.458, 116.92136]
UEV_no_mask = [143.49714, 115.250388, 115.3491835, 114.92731, 115.585058, 115.58505]
# unl_hess_r = [96.6, 96.66, 96.04, 95.94, 95.85, 97.21]
UEV_no_division = [146.170001, 118.3346, 116.60770, 117.58471, 117.281131, 117.198461]

UEV_no_both = [142.9, 118.9, 117.9, 116.00222, 116.92408, 116.9]
# unl_ss_wo = [94.32, 94.53, 94.78, 93.38, 94.04, 97.21]


#plt.figure(figsize=(8, 5.3))
#plt.plot(x, unl_fr, color='blue', marker='^', label='Retrain',linewidth=l_w, markersize=m_s)
ax[2].plot(x, UEV, linestyle='-', color='#797BB7', marker='o', fillstyle='full', markevery=markevery,
         label='MUA-PD', linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

#plt.plot(x, unl_ss_w, color='g',  marker='*',  label='PriMU$_{w}$',linewidth=l_w, markersize=m_s)
ax[2].plot(x, UEV_no_mask, linestyle='--', color='#9BC985',  marker='s', fillstyle='full', markevery=markevery,
         label='MUA-PD w/o UDP',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)


# plt.plot(x, unl_mib_bck, linestyle=':', color='r',  marker='^', fillstyle='none', markevery=markevery,
#          label='MIB (bac.)', linewidth=l_w,  markersize=m_s, markeredgewidth=marker_s)


ax[2].plot(x, UEV_no_division, linestyle='-.', color='#2A5522',  marker='D', fillstyle='full', markevery=markevery,
         label='MUA-PD w/o UID',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)


ax[2].plot(x, UEV_no_both, linestyle='-.', color='#E1C855',  marker='^', fillstyle='full', markevery=markevery,
         label='MUA-PD w/o both',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

#F7D58B

#plt.plot(x, unl_vibu, color='silver',  marker='d',  label='VIBU',linewidth=4,  markersize=10)

# plt.plot(x, y_sa03, color='r',  marker='2',  label='AAAI21 A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_sa05, color='darkblue',  marker='4',  label='AAAI21 A_acc, pr=0.5',linewidth=3, markersize=8)
# plt.plot(x, y_ma03, color='darkviolet',  marker='3',  label='FedMC A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_ma05, color='cyan',  marker='p',  label='FedMC A_acc, pr=0.5',linewidth=3, markersize=8)


# plt.grid()
# leg = plt.legend(fancybox=True, shadow=True)
# plt.xlabel('Malicious Client Ratio (%)' ,fontsize=16)
ax[2].set_ylabel('Running Time', fontsize=18)
my_y_ticks = np.arange(108, 152, 8)
ax[2].set_yticks(my_y_ticks )
ax[2].set_xlabel('$\it{ESS}$' ,fontsize=18)

ax[2].set_xticklabels(labels ,fontsize=13)
ax[2].set_xticks(x)
# plt.title('CIFAR10 IID')





# Set the background of the axes, which is the area of the plot, to grey
# plt.gca().set_facecolor('grey')

# Set the grid with white color and a specific linestyle and linewidth
# plt.grid(color='white', linestyle='-', linewidth=0.5)


# plt.grid(axis='y')
# ax[2].set_title('On CelebA', fontsize=20)



handles, labels = ax[2].get_legend_handles_labels()
# Create a "dummy" handle for the legend title
# title_handle = plt.Line2D([], [], color='none', label='Method')
#
# # Insert the title handle at the beginning of the handles list
# handles = [title_handle] + handles
# handles.insert(1, title_handle)
# labels = ['Methods and Scenarios:'] + labels
# labels.insert(1, '')

fig.legend(handles, labels,  loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.13),fontsize=18)

#facecolor='#EAEAF2', frameon=True,
# plt.legend( title = 'Methods and Datasets',frameon=True, facecolor='white', loc='best',
#            ncol=2, mode="expand", framealpha=0.5, borderaxespad=0., fontsize=20, title_fontsize=20)

# fig.tight_layout()
plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('ablation_exp.pdf', dpi=200,bbox_inches='tight')

plt.show()