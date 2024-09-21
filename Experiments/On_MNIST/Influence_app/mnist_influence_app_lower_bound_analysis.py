

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


epsilon = 3
beta = 1 / epsilon


x=[0, 0.1, 0.5, 0.9]

labels = ['0', '0.1', '0.5', '0.9']
# unl_org = [97.77, 97.55, 97.35, 97.29, 97.21, 97.21]

# unl_hess_r = [96.6, 96.66, 96.04, 95.94, 95.85, 97.21]
unl_mib = [0.0043, 0.0044, 0.018409, 0.0573]

unl_muv = [0.03600, 0.04310, 0.079888, 0.442221]
# unl_ss_wo = [94.32, 94.53, 94.78, 93.38, 94.04, 97.21]

unl_multi_lower_in=[0.006774, 0.00792, 0.01958, 0.05628]
unl_multi_lower_not_in =[0.0108651, 0.01275, 0.049736, 0.417920]

for i in range(len(x)):
    unl_mib[i] = unl_mib[i]
    unl_muv[i] = unl_muv[i]
    unl_multi_lower_in[i] = unl_multi_lower_in[i]
    unl_multi_lower_not_in[i] = unl_multi_lower_not_in[i]

plt.style.use('seaborn')
plt.figure(figsize=(5.5, 5.3))
l_w=5
m_s=15
marker_s = 3
markevery=1
#plt.figure(figsize=(8, 5.3))
#plt.plot(x, unl_fr, color='blue', marker='^', label='Retrain',linewidth=l_w, markersize=m_s)
plt.plot(x, unl_mib, linestyle='-', color='#797BB7', marker='o', fillstyle='full', markevery=markevery,
         label='TEMU In (SS)', linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

#plt.plot(x, unl_ss_w, color='g',  marker='*',  label='PriMU$_{w}$',linewidth=l_w, markersize=m_s)
plt.plot(x, unl_muv, linestyle='--', color='#9BC985',  marker='s', fillstyle='full', markevery=markevery,
         label='TEMU Not In (SS)',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

# plt.plot(x, unl_multi_lower_in, linestyle='-.', color='#2A5522',  marker='D', fillstyle='full', markevery=markevery,
#          label='MS In',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)
#
#
# plt.plot(x, unl_multi_lower_not_in, linestyle=':', color='#E07B54',  marker='^', fillstyle='full', markevery=markevery,
#          label='MS Not In', linewidth=l_w,  markersize=m_s, markeredgewidth=marker_s)
#


#plt.plot(x, unl_vibu, color='silver',  marker='d',  label='VIBU',linewidth=4,  markersize=10)

# plt.plot(x, y_sa03, color='r',  marker='2',  label='AAAI21 A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_sa05, color='darkblue',  marker='4',  label='AAAI21 A_acc, pr=0.5',linewidth=3, markersize=8)
# plt.plot(x, y_ma03, color='darkviolet',  marker='3',  label='FedMC A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_ma05, color='cyan',  marker='p',  label='FedMC A_acc, pr=0.5',linewidth=3, markersize=8)


# plt.grid()
leg = plt.legend(fancybox=True, shadow=True)
# plt.xlabel('Malicious Client Ratio (%)' ,fontsize=16)
plt.ylabel('Average UE' ,fontsize=24)
my_y_ticks = np.arange(0, 0.52, 0.1)
plt.yticks(my_y_ticks,fontsize=20)
# plt.yscale('log')

plt.xlabel('Task Weight $\\alpha$' ,fontsize=20)

plt.xticks(x, labels, fontsize=20)
# plt.title('CIFAR10 IID')

# Annotating the y-axis with the unit at its end using scientific notation

# Create a ScalarFormatter object
# Scale the y-values by a factor of 10


#plt.annotate(r"1e-1", xy=(0.1, 1.01), xycoords='axes fraction', xytext=(-10, 10),textcoords='offset points', ha='right', va='center', fontsize=15)

plt.title('On MNIST', fontsize=24 )

# plt.text(-0.1, 0.5, 'Title Beside Y', transform=ax[0].transAxes,
#            va='center', ha='left', rotation='vertical')
plt.legend(loc='best',fontsize=20)
plt.tight_layout()
#plt.title("MNIST")
plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('mnist_influence_app_lower_bound_analysis.pdf', format='pdf', dpi=200)
plt.show()