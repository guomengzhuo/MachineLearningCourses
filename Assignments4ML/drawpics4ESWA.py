import matplotlib.pyplot as plt

x0 = [0, 1, 2, 3, 4, 5, 6, 7, 8]
y0 = [1, 2, 3, 2.5, 0.5, 1, 2.5, 3, 1]

x1 = [ [0,0], [4, 4], [8,8] ]
y1 = [ [0.2,3.5], [0.2, 3.5], [0.2,3.5] ]

plt.scatter(x0,y0,s=20, edgecolors='b', color='b')
plt.plot(x0,y0)
for i in range(len(x1)):
    plt.plot(x1[i], y1[i], 'k',linewidth=2)
plt.ylim(0,4)
plt.xlim(-0.5,9)
plt.annotate(r'$x_j^{*1}$', xy=(2,3),  fontsize = 20 )  #xy=() determine the coordinates of the annotation!
plt.annotate(r'$x_j^{*2}$', xy=(7,3),  fontsize = 20 )
plt.annotate(r'$x_{j,{*1}}}$', xy=(0,1),  fontsize = 20 )
plt.annotate(r'$x_{j,{*2}}}$', xy=(4,0.5),  fontsize = 20 )
plt.annotate(r'$x_{j,{*3}}}$', xy=(8,1),  fontsize = 20 )
plt.gca()
plt.savefig('F://Manuscript//Ejor_manuscript//Latex//example_multiple_peaks')
plt.show()
