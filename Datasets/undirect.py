fname=raw_input("Enter the name of file")
name=fname.split('.')
outName=str(name[0])+"u.mtx"
flag=True
i=0
out=[]
numRows=0
numCols=0
with open(fname,"r") as f:
    x=f.readline().split()

    numRows=x[0]
    numCols=x[1]
    while flag: 
        x=f.readline().split()
        if len(x)>0:
            if x[0]==x[1]:
                i=i+1
                out.append((int(x[0]),int(x[1]),float(x[2])))
            else:
                i=i+2
                out.append((int(x[0]),int(x[1]),float(x[2])))
                out.append((int(x[1]),int(x[0]),float(x[2])))
        else:
            flag=False
out.sort(key=lambda x:x[1])
f=open(outName,"w") 
f.write(str(numRows)+" "+str(numCols)+" "+str(i)+"\n")
for j in range(i):
    f.write(str(out[j][0])+" "+str(out[j][1])+" "+str(out[j][2])+"\n")
    
