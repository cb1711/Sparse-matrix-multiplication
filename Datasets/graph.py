import matplotlib.pyplot as plt
fname=raw_input("Enter the name of file")
x_axis=[]
y_axis=[]
flag=True
i=0
with open(fname,"r") as f:
    while flag: 
        x=f.readline().split()
        if len(x)>0:
            x_axis.append(x[0])
            
            y_axis.append(x[1])
        else:
            flag=False
            plt.plot(x_axis,y_axis,'ro', markersize=01)

#print i
#plt.axis([0,x_axis[0],0,y_axis[0]])
plt.show()
