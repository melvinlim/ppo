import os

def getNextTarget(path='recordings'):
    files=os.listdir(path)
    #print(files)
    files.remove('target')
    #print(files)
    files.sort()
    print(files)
    lastidx=files[-1][-1]
    lastidx=int(lastidx)
    #print(lastidx)
    nextidx=lastidx+1
    #print(nextidx)
    lasttarget=path+'/record'+str(lastidx)
    assert(os.path.exists(lasttarget)==True)
    nexttarget=path+'/record'+str(nextidx)
    print(nexttarget)
    assert(os.path.exists(nexttarget)==False)
    return nexttarget

#target=getNextTarget()
#print(target)
