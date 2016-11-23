import numpy as np
import struct
import math
from numpy import linalg as la
NUM_RAND_MDP = 30
RMDP_STATES = 30
RMDP_ACTIONS = 3
RMDP_BRANCH_FACT = 4
NUM_ALIASED_STATES = 5
NUM_TERMINATIONS = 2
SEED = 3141
MIN_PROB_VAL = 0.0002


#-----------------------Feature Representation--------------------------------

# The taublar representation where each state is represented with a binary vector with a single one corresponding the current state index
def makeTaubularFeature(nStates):
    taubular = np.eye(nStates)
    return taubular


# The Aliased representation is that we alias five state to have the same feature vector

def makeAliasedFeatures(nStates,nAliased):
    PHI = np.eye(nStates)
    sameIndex = np.random.choice(nStates,nAliased+1,replace = False)
    # print sameIndex
    for i in range(1,nAliased+1):
        PHI[sameIndex[i]] = PHI[sameIndex[0]]
    # rePHI = np.zeros(nStates,nStates-nAliased)
    rePHI = PHI[:,~np.all(PHI == 0, axis=0)]

    return rePHI

# The binary representation: 30 states,  using five numbers to represent. 2^% = 32

def makeBinaryFeatures(nStates):
    nBits = math.floor(math.log(nStates+1,2)+1)

    PHI = np.zeros((nStates,int(nBits)), dtype=float)
    for i in range(1,nStates+1):

        # print i
        # bitSrting = binary(i)[1:9]
        bit = bin(i)[2:].zfill(64)
        bitSrting1 = str(bit)
        bitSrting = bitSrting1[64-(int(nBits)):64]
        for j in range(len(bitSrting)):
            # print len(bitSrting)
            if bitSrting[j] == '1':
                PHI[i-1,j] = 1

    s = PHI.sum(axis=1)
    for i in range(PHI.shape[0]):
        for j in range(PHI.shape[1]):
            PHI[i,j] = PHI[i,j] / s[i]
    # print PHI
    return PHI


# # generate a set of states without replacement



def makeRandomMDP(featureType,nStates,nActions,branchingFactor,pi_v,mu_v,gamma):
    if(featureType == "tabular"):
        PHI = makeTaubularFeature(RMDP_STATES)

    elif(featureType == "aliased"):
        PHI = makeAliasedFeatures(RMDP_STATES)

    elif(featureType == "binary"):
        PHI = makeBinaryFeatures(RMDP_STATES)

    else:
        print "FeatureType Error"

    d = PHI.shape[1]
    rewards = np.random.random([RMDP_STATES,RMDP_STATES])
    Pmat = makeTransMatrix(nStates,nActions,branchingFactor)
    piA = makeBiasedPolicy(nStates,nActions,pi_v,mu_v).get('piA')
    mu = makeBiasedPolicy(nStates,nActions,pi_v,mu_v).get('mu')
    Ppi = makePolicyTransMatrix(Pmat,piA)
    Pmu = makePolicyTransMatrix(Pmat, mu)
    dpi = getStationaryDist(Ppi)
    dmu = getStationaryDist(Pmu)
    #
    while (checkStationaryDist(dmu) == -1):
        Pmat = makeTransMatrix(nStates, nActions, branchingFactor)
        Ppi = np.zeros((nStates,nActions),dtype=float)
        Ppi = makePolicyTransMatrix(Pmat,Ppi)
        Pmu = np.zeros((nStates,nActions),dtype=float)
        Pmu = makePolicyTransMatrix(Pmu,mu)
        dpi = getStationaryDist(Ppi)
        dmu = getStationaryDist(Pmu)


    DD = np.eye(nStates) * dmu
    makeGammaMatrix(nStates,gamma,Pmu)



def makeGammaMatrix(nStates,gamma0,Pmu,numTerinations = NUM_TERMINATIONS):
    gamma = np.ones((nStates, nStates), dtype= float)*gamma0
    sameIndex = np.random.choice(nStates, numTerinations,replace= False)
    termStateSet = np.copy(sameIndex)
    print termStateSet
    for i in range(numTerinations):
        # posstemp = Pmu[sameIndex[i], :]
        possibletemp = np.reshape(Pmu[sameIndex[i], :], nStates, order='F')
        # possibletemp = np.reshape(posstemp, nStates, order='F')
        possible = np.nonzero(possibletemp)
        possible = np.array(possible)
        possible = np.reshape(possible,possible.shape[1] * possible.shape[0], order='F')
        termStatedInd = np.random.choice(possible,1)
        while ((Pmu[sameIndex[i], termStatedInd] <= 0.0) or ([] == np.where(termStateSet== termStatedInd))):
            termStatedInd = np.random.choice(possible, 1)
        gamma[sameIndex[i],termStatedInd] = 0
        termStateSet = np.append(termStateSet,termStatedInd)

        print "-------------------------------"
    print gamma
    return {'gamma':gamma,'sameIndex':sameIndex}


# random policies selecting one action in each state to have majority of probability  // piA is the matrix 0.9 && 0.05, mu is the same but it depends on mu

def makeBiasedPolicy(nStates,nActions,pi_v,mu_v):
    highProbActionIndex = np.random.randint(1,nActions,size = nStates)
    if nActions == 1 :
        pi_v = 1.0
        mu_v = 1.0
        piA = np.ones((nStates,nActions), dtype= float)
        mu = np.ones((nStates,nActions), dtype= float)
        return {'piA':piA,'mu':mu}


    piA = np.ones((nStates, nActions), dtype=float)
    piA = piA * (1-pi_v)/(nActions-1)
    mu = np.ones((nStates, nActions), dtype=float)
    mu = mu * (1-mu_v)/(nActions-1)

    for i in range(nStates):
        piA[i,highProbActionIndex[i]] = pi_v
        mu[i,highProbActionIndex[i]] = mu_v

    return {'piA':piA,'mu':mu}

##Give policy piA and transition matrix P, compute Pp(i,j) = sum_a pi(a|i) P(i,a,j)
def makePolicyTransMatrix(P,piA):
    ns = P.shape[0]
    na = P.shape[1]
    Ppi = np.zeros((ns,ns),dtype=float)
    for i in range(ns):
        for j in range(ns):
            Ppi[i,j] = 0
            for k in range(na):
                Ppi[i,j] = Ppi[i,j] +P[i,k,j] * piA[i,k]
    # print Ppi.shape
    return Ppi


# works with continuing or episodic tasks
# eigen vector corresponding to eigen value of 1.

def getStationaryDist(Ppi):
    (d,v)= la.eig(Ppi.transpose())
    realD= d.real
    index = realD.argmax()
    # print Ppi.shape
    # print "vvvv"
    # print realD.shape
    if abs(realD[index]-1) > 0.000001:
        print "Max eigenvalue must be 1.0, must be a problem with policy or transtion matrix"
        return 0
    dist = v[:,index].real
    distsum = sum(dist)
    dist = dist / distsum
    return dist


def checkStationaryDist(dpi):
    if dpi.min() < MIN_PROB_VAL:
        return -1
    return 0

def makeTransMatrix(nStates,nActions,branchingFactor):
    Pmat = np.zeros((nStates,nActions,nStates), dtype=float )

    for i in range(nActions):
        for j in range(nStates):
            # nextStates = np.random.randint(0,nStates,size=branchingFactor)
            nextStates = np.random.choice(nStates,branchingFactor,replace =False)
            TransProbs = np.random.random(nextStates.shape)
            for k in range(branchingFactor):
                if(TransProbs[k] >= MIN_PROB_VAL):
                    Pmat[j,i,nextStates[k]] = TransProbs[k]
                else:
                    Pmat[j, i, nextStates[k]] = MIN_PROB_VAL
            Total = 0
            for s in range(nStates):
                # PatTemp = Pmat.sum(1)
                # Total = PatTemp.sum(0)[nStates-1]
                # # print Total
                # Pmat[j,i,s] = Pmat[j,i,s] / Total
                Total = Total + Pmat[j,i,s];
            # print Total
            for s in range(nStates):
                if(Pmat[j, i, s] != 0):
                    Pmat[j, i, s] = Pmat[j, i, s] / Total
            # print "Next State "
            # print nextStates
            # print "Trans Pr "
            # print TransProbs
            # print "Pmat"
            # print Pmat

    return Pmat

if __name__ == '__main__':
    # makeRandomMDP("tabular", RMDP_STATES, RMDP_ACTIONS, RMDP_BRANCH_FACT, 0.9, 0.9,0.95)
    # makeRandomMDP("tabular", 6, 3, 4, 0.9, 0.9)
    # makeTransMatrix(10,1,4)
    # makeAliasedFeatures(6,3)
    # makeBinaryFeatures(5)
    # makeBiasedPolicy(6,3,0.9,0.9)
    # P = np.ones((3,4,5),dtype=float)
    # pia = np.array([[1,2,3.1,4],[1,2,3,4.4],[5,6,7.2,8.1]])
    # makePolicyTransMatrix(P,pia)
    # print aa
