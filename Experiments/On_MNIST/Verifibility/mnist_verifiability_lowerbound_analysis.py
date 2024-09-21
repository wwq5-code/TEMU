

import numpy as np
import matplotlib.pyplot as plt

epsilon = 3
beta = 1 / epsilon


x=[0, 0.1, 0.5, 0.9]

labels = ['0', '0.1', '0.5', '0.9']
# unl_org = [97.77, 97.55, 97.35, 97.29, 97.21, 97.21]

unl_mib = [0, 0, 0, 0]
unl_mib_bck = [0, 1, 1, 1]

# unl_hess_r = [96.6, 96.66, 96.04, 95.94, 95.85, 97.21]
unl_muv_includes = [0.7247, 0.9747, 0.9973, 0.996]

unl_muv = [0.3823, 0.6520, 0.7057, 0.9217]
# unl_ss_wo = [94.32, 94.53, 94.78, 93.38, 94.04, 97.21]

unl_lower_in = [0.0000, 0.1830, 0.2813, 0.3100]
unl_lower_not_in = [0.8793, 0.9307, 0.9547, 0.9933 ]

unl_multi_lower_in = [0, 0.0067, 0.1097, 0.2020]
unl_multi_lower_not_in = [0.8350, 0.9447, 0.9623, 0.9583 ]

plt.style.use('seaborn')

plt.figure(figsize=(5.5, 5.3))
l_w=5
m_s=15
marker_s = 3
markevery=1
#plt.figure(figsize=(8, 5.3))
#plt.plot(x, unl_fr, color='blue', marker='^', label='Retrain',linewidth=l_w, markersize=m_s)
plt.plot(x, unl_lower_in, linestyle='-', color='#797BB7', marker='o', fillstyle='full', markevery=markevery,
         label='TEMU In (SS)', linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

#plt.plot(x, unl_ss_w, color='g',  marker='*',  label='PriMU$_{w}$',linewidth=l_w, markersize=m_s)
plt.plot(x, unl_lower_not_in, linestyle='--', color='#9BC985',  marker='s', fillstyle='full', markevery=markevery,
         label='TEMU Not In (SS)',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

plt.plot(x, unl_multi_lower_in, linestyle='-.', color='#2A5522',  marker='D', fillstyle='full', markevery=markevery,
         label='TEMU In (MS)',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)


plt.plot(x, unl_multi_lower_not_in, linestyle=':', color='#E07B54',  marker='^', fillstyle='full', markevery=markevery,
         label='TEMU Not In (MS)', linewidth=l_w,  markersize=m_s, markeredgewidth=marker_s)




#plt.plot(x, unl_vibu, color='silver',  marker='d',  label='VIBU',linewidth=4,  markersize=10)

# plt.plot(x, y_sa03, color='r',  marker='2',  label='AAAI21 A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_sa05, color='darkblue',  marker='4',  label='AAAI21 A_acc, pr=0.5',linewidth=3, markersize=8)
# plt.plot(x, y_ma03, color='darkviolet',  marker='3',  label='FedMC A_acc, pr=0.3',linewidth=3, markersize=8)
# plt.plot(x, y_ma05, color='cyan',  marker='p',  label='FedMC A_acc, pr=0.5',linewidth=3, markersize=8)


# plt.grid()
leg = plt.legend(fancybox=True, shadow=True)
# plt.xlabel('Malicious Client Ratio (%)' ,fontsize=16)
plt.ylabel('Verifiability' ,fontsize=24)
my_y_ticks = np.arange(0, 1.1, 0.2)
plt.yticks(my_y_ticks,fontsize=20)
plt.xlabel('Task Weight $\\alpha$' ,fontsize=20)

plt.xticks(x, labels, fontsize=20)
# plt.title('CIFAR10 IID')

#plt.annotate(r"1e0", xy=(0.1, 1.01), xycoords='axes fraction', xytext=(-10, 10),textcoords='offset points', ha='right', va='center', fontsize=15)


# plt.title('(c) Utility Preservation', fontsize=24)
plt.legend(loc=(0.15,0.4),fontsize=20)
plt.tight_layout()
#plt.title("MNIST")
plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('mnist_verifiability_lowerbound_analysis.pdf', format='pdf', dpi=200)
plt.show()