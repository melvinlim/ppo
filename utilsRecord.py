import os
import argparse
import retro

def getArgs(defaultgame='Tetris-GameBoy'):
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default=defaultgame)
    parser.add_argument("--state", default=retro.State.DEFAULT)
    parser.add_argument("--scenario", default=None)
    args = parser.parse_args()
    return args

def getNextTarget(path='recordings'):
    files=os.listdir(path)
    if(len(files)==0):
        return path+'/record0'
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
