

import numpy as np
import matplotlib.pyplot as plt

epsilon = 3
beta = 1 / epsilon


x=[0, 0.1, 0.5, 0.9]

labels = ['0', '0.1', '0.5', '0.9']
# unl_org = [97.77, 97.55, 97.35, 97.29, 97.21, 97.21]

unl_mib = [0, 0, 0, 0 ]
unl_mib_bck = [0, 0.7, 0.7, 0.7 ]


# unl_hess_r = [96.6, 96.66, 96.04, 95.94, 95.85, 97.21]
unl_muv_includes = [0.0, 0.2428, 0.1281, 0.0891 ]

unl_muv = [0.0, 0.9221, 0.9467,  0.9314 ]
# unl_ss_wo = [94.32, 94.53, 94.78, 93.38, 94.04, 97.21]

unl_muv_lowerb_multi_in = [0, 0.2592, 0.1025, 0.0984]
unl_muv_lowerb_multi_not_in = [0,  0.9293, 0.9344, 0.9262]


plt.style.use('seaborn')
plt.figure(figsize=(5.5, 5.3))
l_w=5
m_s=15
marker_s = 3
markevery=1
#plt.figure(figsize=(8, 5.3))
#plt.plot(x, unl_fr, color='blue', marker='^', label='Retrain',linewidth=l_w, markersize=m_s)
plt.plot(x, unl_muv_includes, linestyle='-', color='#797BB7', marker='o', fillstyle='full', markevery=markevery,
         label='TEMU In (SS)', linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)

#plt.plot(x, unl_ss_w, color='g',  marker='*',  label='PriMU$_{w}$',linewidth=l_w, markersize=m_s)
plt.plot(x, unl_muv, linestyle='--', color='#9BC985',  marker='s', fillstyle='full', markevery=markevery,
         label='TEMU Not In (SS)',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)




plt.plot(x, unl_muv_lowerb_multi_in, linestyle='-.', color='#2A5522',  marker='D', fillstyle='full', markevery=markevery,
         label='TEMU In (MS)',linewidth=l_w, markersize=m_s, markeredgewidth=marker_s)


plt.plot(x, unl_muv_lowerb_multi_not_in, linestyle=':', color='#E07B54',  marker='^', fillstyle='full', markevery=markevery,
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
plt.legend(loc=(0.151, 0.35),fontsize=20)
plt.tight_layout()
#plt.title("MNIST")
plt.rcParams['figure.figsize'] = (2.0, 1)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['figure.subplot.left'] = 0.11
plt.rcParams['figure.subplot.bottom'] = 0.08
plt.rcParams['figure.subplot.right'] = 0.977
plt.rcParams['figure.subplot.top'] = 0.969
plt.savefig('celeba_verifiability_lower_bound_analysis.pdf', format='pdf', dpi=200)
plt.show()