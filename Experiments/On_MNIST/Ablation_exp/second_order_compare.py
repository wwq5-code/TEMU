import numpy as np
import  matplotlib.pyplot as plt
#


plt.style.use('seaborn')

fig, ax = plt.subplots(1, 3, figsize=(15, 3.8)) #sharex='col',
fig.subplots_adjust(bottom=0.2)

# for i in range(2):
#     for j in range(3):
#         ax[i,j].text(0.5,0.5,str((i,j)), fontsize=18, ha='center')




#picture 1


"""
x=[1, 2, 3, 4, 5, 6]
# validation_for_plt =[97,95.8600, 94.9400, 93.5400, 93.2400]
# attack_for_plt=[0, 0.3524, 0, 0.1762, 0.1762]
# basic_for_plt=[99.8, 99.8, 99.8, 99.8, 99.8]

labels = ['1', '20', '40', '60', '80', '100' ]
# unl_org = [97.77, 97.55, 97.35, 97.29, 97.21, 97.21]

UEV = [0.4278, 0.4113, 0.41067, 0.410123, 0.409559, 0.4098]
UEV_second = [0.434302, 0.416475, 0.4174561, 0.4169029, 0.4166158, 0.405258]
# unl_hess_r = [96.6, 96.66, 96.04, 95.94, 95.85, 97.21]
UEV_in = [0.0670, 0.05614, 0.05513, 0.054879, 0.05501, 0.054093]

UEV_second_in = [0.0661944, 0.05628, 0.05532, 0.0552045, 0.055168, 0.0547215]



l_w=3.5
m_s=8
marker_s = 2
markevery=1
#plt.figure(figsize=(8, 5.3))
#plt.plot(x, unl_fr, color='blue', marker='^', label='Retrain',linewidth=l_w, markersize=m_s)
ax[0].plot(x, UEV, linestyle='-', color='#797BB7', marker='o', fillstyle='full', markevery=markevery,
         label='Includes', linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

#plt.plot(x, unl_ss_w, color='g',  marker='*',  label='PriMU$_{w}$',linewidth=l_w, markersize=m_s)
ax[0].plot(x, UEV_second, linestyle='--', color='#9BC985',  marker='s', fillstyle='full', markevery=markevery,
         label='Not Includes',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)


# plt.plot(x, unl_mib_bck, linestyle=':', color='r',  marker='^', fillstyle='none', markevery=markevery,
#          label='MIB (bac.)', linewidth=l_w,  markersize=m_s, markeredgewidth=marker_s)


#ax[0].plot(x, UEV_in, linestyle='-.', color='#2A5522',  marker='D', fillstyle='full', markevery=markevery,label='MIB (Normal Not In.)',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)


#ax[0].plot(x, UEV_second_in, linestyle='-.', color='#E1C855',  marker='^', fillstyle='full', markevery=markevery, label='MIB (Normal Not In.)',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)



#plt.plot(x, unl_vibu, color='silver',  marker='d',  label='VIBU',linewidth=4,  markersize=10)

# plt.plot(x, y_sa03, color='r',  marker='2',  label='AAAI21 A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_sa05, color='darkblue',  marker='4',  label='AAAI21 A_acc, pr=0.5',linewidth=3, markersize=8)
# plt.plot(x, y_ma03, color='darkviolet',  marker='3',  label='FedMC A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_ma05, color='cyan',  marker='p',  label='FedMC A_acc, pr=0.5',linewidth=3, markersize=8)


# plt.grid()
leg = plt.legend(fancybox=True, shadow=True)
# plt.xlabel('Malicious Client Ratio (%)' ,fontsize=16)
ax[0].set_ylabel('Average UE', fontsize=20)
my_y_ticks = np.arange(0.0, 0.46, 0.09)
ax[0].set_yticks(my_y_ticks )
ax[0].set_xlabel('$\it{ESS}$' ,fontsize=20)

ax[0].set_xticklabels(labels ,fontsize=13)
ax[0].set_xticks(x)
# plt.title('CIFAR10 IID')

"""



# figure 2



x=[1, 2, 3, 4, 5, 6]
# validation_for_plt =[97,95.8600, 94.9400, 93.5400, 93.2400]
# attack_for_plt=[0, 0.3524, 0, 0.1762, 0.1762]
# basic_for_plt=[99.8, 99.8, 99.8, 99.8, 99.8]

labels = ['1', '20', '40', '60', '80', '100' ]
# unl_org = [97.77, 97.55, 97.35, 97.29, 97.21, 97.21]

UEV = [0.96693, 0.93609, 0.913304, 0.88554, 0.8717471, 0.809093]
UEV_second = [0.9667912, 0.93423, 0.91603, 0.880405, 0.85692, 0.810767]
# unl_hess_r = [96.6, 96.66, 96.04, 95.94, 95.85, 97.21]
UEV_in = [ 0.9459, 0.9097, 0.85268, 0.856668, 0.818231, 0.732786]

UEV_second_in = [0.947320, 0.91448736, 0.85384, 0.8462974, 0.8267230, 0.764254]



l_w=3.5
m_s=8
marker_s = 2
markevery=1
#plt.figure(figsize=(8, 5.3))
#plt.plot(x, unl_fr, color='blue', marker='^', label='Retrain',linewidth=l_w, markersize=m_s)
ax[0].plot(x, UEV, linestyle='-', color='#797BB7', marker='o', fillstyle='full', markevery=markevery,
         label='Includes', linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

#plt.plot(x, unl_ss_w, color='g',  marker='*',  label='PriMU$_{w}$',linewidth=l_w, markersize=m_s)
ax[0].plot(x, UEV_second, linestyle='--', color='#9BC985',  marker='s', fillstyle='full', markevery=markevery,
         label='Not Includes',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)


# plt.plot(x, unl_mib_bck, linestyle=':', color='r',  marker='^', fillstyle='none', markevery=markevery,
#          label='MIB (bac.)', linewidth=l_w,  markersize=m_s, markeredgewidth=marker_s)


#ax[1].plot(x, UEV_in, linestyle='-.', color='#2A5522',  marker='D', fillstyle='full', markevery=markevery, label='MIB (Normal Not In.)',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)


#ax[1].plot(x, UEV_second_in, linestyle='-.', color='#E1C855',  marker='^', fillstyle='full', markevery=markevery, label='MIB (Normal Not In.)',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)



#plt.plot(x, unl_vibu, color='silver',  marker='d',  label='VIBU',linewidth=4,  markersize=10)

# plt.plot(x, y_sa03, color='r',  marker='2',  label='AAAI21 A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_sa05, color='darkblue',  marker='4',  label='AAAI21 A_acc, pr=0.5',linewidth=3, markersize=8)
# plt.plot(x, y_ma03, color='darkviolet',  marker='3',  label='FedMC A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_ma05, color='cyan',  marker='p',  label='FedMC A_acc, pr=0.5',linewidth=3, markersize=8)


# plt.grid()
leg = plt.legend(fancybox=True, shadow=True)
# plt.xlabel('Malicious Client Ratio (%)' ,fontsize=16)
ax[0].set_ylabel('Rec. Similarity', fontsize=20)
my_y_ticks = np.arange(0.7, 1.01, 0.06)
ax[0].set_yticks(my_y_ticks )
ax[0].set_xlabel('$\it{ESS}$' ,fontsize=20)

ax[0].set_xticklabels(labels ,fontsize=13)
ax[0].set_xticks(x)
# plt.title('CIFAR10 IID')








#picture 3

x=[1, 2, 3, 4, 5, 6]
# validation_for_plt =[97,95.8600, 94.9400, 93.5400, 93.2400]
# attack_for_plt=[0, 0.3524, 0, 0.1762, 0.1762]
# basic_for_plt=[99.8, 99.8, 99.8, 99.8, 99.8]

labels = ['1', '20', '40', '60', '80', '100' ]
# unl_org = [97.77, 97.55, 97.35, 97.29, 97.21, 97.21]

UEV = [0.9933, 0.9583, 0.9083, 0.8550, 0.8600, 0.8410]
UEV_second = [0.9907, 0.9607, 0.9423, 0.8810, 0.8777, 0.8533]
# unl_hess_r = [96.6, 96.66, 96.04, 95.94, 95.85, 97.21]
UEV_in= [0.3100, 0.2073, 0.15, 0.0600, 0.0320, 0.0000]

UEV_second_in = [0.3197, 0.2100, 0.1567, 0.1513, 0.0217, 0.0033]



#plt.figure(figsize=(8, 5.3))
#plt.plot(x, unl_fr, color='blue', marker='^', label='Retrain',linewidth=l_w, markersize=m_s)
ax[1].plot(x, UEV, linestyle='-', color='#797BB7', marker='o', fillstyle='full', markevery=markevery,
         label='First Order Not In', linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

#plt.plot(x, unl_ss_w, color='g',  marker='*',  label='PriMU$_{w}$',linewidth=l_w, markersize=m_s)
ax[1].plot(x, UEV_second, linestyle='--', color='#9BC985',  marker='s', fillstyle='full', markevery=markevery,
         label='Second Order Not In',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)


# plt.plot(x, unl_mib_bck, linestyle=':', color='r',  marker='^', fillstyle='none', markevery=markevery,
#          label='MIB (bac.)', linewidth=l_w,  markersize=m_s, markeredgewidth=marker_s)


#ax[2].plot(x, UEV_in, linestyle='-.', color='#2A5522',  marker='D', fillstyle='full', markevery=markevery, label='First Order In',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)


#ax[2].plot(x, UEV_second_in, linestyle='-.', color='#E1C855',  marker='^', fillstyle='full', markevery=markevery, label='Second Order In',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)




#plt.plot(x, unl_vibu, color='silver',  marker='d',  label='VIBU',linewidth=4,  markersize=10)

# plt.plot(x, y_sa03, color='r',  marker='2',  label='AAAI21 A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_sa05, color='darkblue',  marker='4',  label='AAAI21 A_acc, pr=0.5',linewidth=3, markersize=8)
# plt.plot(x, y_ma03, color='darkviolet',  marker='3',  label='FedMC A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_ma05, color='cyan',  marker='p',  label='FedMC A_acc, pr=0.5',linewidth=3, markersize=8)


# plt.grid()
leg = plt.legend(fancybox=True, shadow=True)
# plt.xlabel('Malicious Client Ratio (%)' ,fontsize=16)
ax[1].set_ylabel('Verifiability', fontsize=20)
my_y_ticks = np.arange(0.0, 1.01, 0.2)
ax[1].set_yticks(my_y_ticks )
ax[1].set_xlabel('$\it{ESS}$' ,fontsize=20)

ax[1].set_xticklabels(labels ,fontsize=13)
ax[1].set_xticks(x)
# plt.title('CIFAR10 IID')


# figure 4


x=[1, 2, 3, 4, 5, 6]
# validation_for_plt =[97,95.8600, 94.9400, 93.5400, 93.2400]
# attack_for_plt=[0, 0.3524, 0, 0.1762, 0.1762]
# basic_for_plt=[99.8, 99.8, 99.8, 99.8, 99.8]

labels = ['1', '20', '40', '60', '80', '100']
# unl_org = [97.77, 97.55, 97.35, 97.29, 97.21, 97.21]

UEV = [141.4497, 116.9057  , 118.2206 , 119.850, 118.458, 116.92136]
UEV_second = [162.60409, 119.6657, 118.8308, 117.604, 118.85385, 116.829053]
# unl_hess_r = [96.6, 96.66, 96.04, 95.94, 95.85, 97.21]
# UEV_no_division = [146.170001, 118.3346, 116.60770, 117.58471, 117.281131, 117.198461]
#
# UEV_no_both = [142.9, 118.9, 117.9, 116.00222, 116.92408, 116.9]
# unl_ss_wo = [94.32, 94.53, 94.78, 93.38, 94.04, 97.21]


#plt.figure(figsize=(8, 5.3))
#plt.plot(x, unl_fr, color='blue', marker='^', label='Retrain',linewidth=l_w, markersize=m_s)
ax[2].plot(x, UEV, linestyle='-', color='#797BB7', marker='o', fillstyle='full', markevery=markevery,
         label='UEV', linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

#plt.plot(x, unl_ss_w, color='g',  marker='*',  label='PriMU$_{w}$',linewidth=l_w, markersize=m_s)
ax[2].plot(x, UEV_second, linestyle='--', color='#9BC985',  marker='s', fillstyle='full', markevery=markevery,
         label='UEV w/o masking',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)


# plt.plot(x, unl_mib_bck, linestyle=':', color='r',  marker='^', fillstyle='none', markevery=markevery,
#          label='MIB (bac.)', linewidth=l_w,  markersize=m_s, markeredgewidth=marker_s)


# ax[3].plot(x, UEV_no_division, linestyle='-.', color='#2A5522',  marker='D', fillstyle='full', markevery=markevery,
#          label='UEV w/o division',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)
#
#
# ax[3].plot(x, UEV_no_both, linestyle='-.', color='#E1C855',  marker='^', fillstyle='full', markevery=markevery,
#          label='UEV w/o both',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

#F7D58B

#plt.plot(x, unl_vibu, color='silver',  marker='d',  label='VIBU',linewidth=4,  markersize=10)

# plt.plot(x, y_sa03, color='r',  marker='2',  label='AAAI21 A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_sa05, color='darkblue',  marker='4',  label='AAAI21 A_acc, pr=0.5',linewidth=3, markersize=8)
# plt.plot(x, y_ma03, color='darkviolet',  marker='3',  label='FedMC A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_ma05, color='cyan',  marker='p',  label='FedMC A_acc, pr=0.5',linewidth=3, markersize=8)


# plt.grid()
# leg = plt.legend(fancybox=True, shadow=True)
# plt.xlabel('Malicious Client Ratio (%)' ,fontsize=16)
ax[2].set_ylabel('Running Time', fontsize=20)
my_y_ticks = np.arange(100, 180, 15)
ax[2].set_yticks(my_y_ticks )
ax[2].set_xlabel('$\it{ESS}$' ,fontsize=20)

ax[2].set_xticklabels(labels ,fontsize=13)
ax[2].set_xticks(x)
# plt.title('CIFAR10 IID')





# Set the background of the axes, which is the area of the plot, to grey
# plt.gca().set_facecolor('grey')

# Set the grid with white color and a specific linestyle and linewidth
# plt.grid(color='white', linestyle='-', linewidth=0.5)


# plt.grid(axis='y')
# ax[2].set_title('On CelebA', fontsize=20)



handles, labels = ax[1].get_legend_handles_labels()
# Create a "dummy" handle for the legend title
# title_handle = plt.Line2D([], [], color='none', label='Method')
#
# # Insert the title handle at the beginning of the handles list
# handles = [title_handle] + handles
# handles.insert(1, title_handle)
# labels = ['Methods and Scenarios:'] + labels
# labels.insert(1, '')

fig.legend(handles, labels,  loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.13),fontsize=20)

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
plt.savefig('second_order_compare.pdf', dpi=200,bbox_inches='tight')

plt.show()