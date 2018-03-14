import matplotlib
import numpy as np
import matplotlib.pyplot as plt

class Site:
    def __init__(self,spin):
        self.spin = spin
        self.clusterID = 0

    def flip(self):
        self.spin = -self.spin
        
    def getSpin(self):
        return self.spin 
    
    def addToCluster(self,name):
        self.clusterID = name
    
    def getClusterID(self):
        return self.clusterID
    
class simulation:
    def __init__(self,temperature,c_strength,dimension):
        self.dim = dimension
        self.lattice = np.array( [ [Site(np.random.choice([-1,1])) for i in range(self.dim)] for j in range(self.dim) ], dtype=object)
        self.J = c_strength
        self.T = temperature
        
    def lattice_as_array(self):
        LAT = np.zeros((self.dim,self.dim))
        for i in range(0,self.dim):
            for j in range(0,self.dim):
                LAT[i][j]=self.lattice[i][j].getSpin()
        return LAT
    
    def cluster_as_array(self):
        LAT = np.zeros((self.dim,self.dim))
        for i in range(0,self.dim):
            for j in range(0,self.dim):
                LAT[i][j]=self.lattice[i][j].getClusterID()
        return LAT
    
    def get_spin_energy(self,i,j):
        return self.J*(self.lattice[(i+1)%self.dim][j].getSpin()+self.lattice[(i-1)%self.dim][j].getSpin()+self.lattice[i][(j+1)%self.dim].getSpin()+self.lattice[i][(j-1)%self.dim].getSpin())*self.lattice[i][j].getSpin()

    def get_magnetization(self):
        return np.abs(np.sum(self.lattice_as_array()))/self.dim**2
    
    def get_total_energy(self):
        total = 0
        for i in range(0,self.dim):
            for j in range(0,self.dim):
                total += -self.get_spin_energy(i,j)
        return total / self.dim**2 / 4

    def plot_Lattice(self):
        plt.imshow(self.lattice_as_array(),cmap="plasma") #Greys
    
    def plot_Cluster(self):
        plt.imshow(self.cluster_as_array(),cmap="plasma")
        
    def list_by_Cluster_ID(self,cID):
        X = []
        Y = []
        for i in range(0,self.dim):
            for j in range(0,self.dim):
                if self.lattice[i][j].getClusterID() == cID:
                    X.append(i)
                    Y.append(j)
        return X,Y
    
    def create_Cluster(self):
        tmp = 1
        prob = 1-np.exp(-2/(self.T)*self.J)
        for i in range(0,self.dim):
            for j in range(0,self.dim):
                X = []
                Y = []
                ID_List = []
                if self.lattice[i][j].getSpin() == self.lattice[(i+1)%self.dim][j].getSpin():
                    if np.random.random() < prob:
                        X.append((i+1)%self.dim)
                        Y.append(j)
                        ID_List.append(self.lattice[(i+1)%self.dim][j].getClusterID())
                if self.lattice[i][j].getSpin() == self.lattice[(i-1)%self.dim][j].getSpin():
                    if np.random.random() < prob:
                        X.append((i-1)%self.dim)
                        Y.append(j)
                        ID_List.append(self.lattice[(i-1)%self.dim][j].getClusterID())
                if self.lattice[i][j].getSpin() == self.lattice[i][(j+1)%self.dim].getSpin():
                    if np.random.random() < prob:
                        X.append(i)
                        Y.append((j+1)%self.dim)
                        ID_List.append(self.lattice[i][(j+1)%self.dim].getClusterID())
                if self.lattice[i][j].getSpin() == self.lattice[i,(j-1)%self.dim].getSpin():
                    if np.random.random() < prob:
                        X.append(i)
                        Y.append((j-1)%self.dim)
                        ID_List.append(self.lattice[i][(j-1)%self.dim].getClusterID())
                X.append(i)
                Y.append(j)
                ID_List.append(self.lattice[i][j].getClusterID())
                if max(ID_List) == 0:
                    for point in range(0,len(ID_List)):
                        self.lattice[X[point]][Y[point]].addToCluster(tmp)
                        tmp += 1
                else:
                    for point in range(0,len(ID_List)):
                        ID_List = sorted(ID_List)
                        index = 0
                        while ID_List[index] == 0:
                            index+=1
                        self.lattice[X[point]][Y[point]].addToCluster(ID_List[index])
                        
    def swendsen_wang(self):
        self.create_Cluster()
        IDS = []
        prob=0.5
        for i in range(0,self.dim):
            for j in range(0,self.dim):
                if self.lattice[i][j].getClusterID() not in IDS:
                    IDS.append(self.lattice[i][j].getClusterID())
                    if np.random.random() < prob:
                        X,Y = self.list_by_Cluster_ID(self.lattice[i][j].getClusterID())
                        for it in range(len(X)):
                            self.lattice[X[it]][Y[it]].flip()
        for i in range(0,self.dim):
            for j in range(0,self.dim):
                self.lattice[i][j].addToCluster(0)
                
    def metropolis(self):
        i = np.random.randint(0,self.dim)
        j = np.random.randint(0,self.dim)
        delta_E = 2 * self.get_spin_energy(i,j)
        if delta_E < 0:
            self.lattice[i][j].flip()
        elif np.exp(-delta_E/(self.T)) >= np.random.random():
            self.lattice[i][j].flip()
            
    def wolff(self):
        prob = 1-np.exp(-2/(self.T)*self.J)
        i = np.random.randint(0,self.dim)
        j = np.random.randint(0,self.dim)
        X = []
        Y = []
        X.append(i)
        Y.append(j)
        self.lattice[i][j].addToCluster(1)
        
        while len(X) != 0:
            x=X[0]
            y=Y[0]
            X.pop(0)
            Y.pop(0)

            if self.lattice[x][y].getSpin() == self.lattice[(x+1)%self.dim][y].getSpin() and not self.lattice[(x+1)%self.dim][y].getClusterID() == 1:
                if np.random.random() < prob:
                    X.append((x+1)%self.dim)
                    Y.append(y)
                    self.lattice[(x+1)%self.dim][y].addToCluster(1)
            if self.lattice[x][y].getSpin() == self.lattice[(x-1)%self.dim][y].getSpin() and not self.lattice[(x-1)%self.dim][y].getClusterID() == 1:
                if np.random.random() < prob:
                    X.append((x-1)%self.dim)
                    Y.append(y)
                    self.lattice[(x-1)%self.dim][y].addToCluster(1)
            if self.lattice[x][y].getSpin() == self.lattice[x][(y+1)%self.dim].getSpin() and not self.lattice[x][(y+1)%self.dim].getClusterID() == 1:
                if np.random.random() < prob:
                    Y.append((y+1)%self.dim)
                    X.append(x)
                    self.lattice[x][(y+1)%self.dim].addToCluster(1)
            if self.lattice[x][y].getSpin() == self.lattice[x][(y-1)%self.dim].getSpin() and not self.lattice[x][(y-1)%self.dim].getClusterID() == 1:
                if np.random.random() < prob:
                    Y.append((y-1)%self.dim)
                    X.append(x)
                    self.lattice[x][(y-1)%self.dim].addToCluster(1)
        
        X_cl, Y_cl = self.list_by_Cluster_ID(1)
        for i in range(len(X_cl)):
            self.lattice[X_cl[i]][Y_cl[i]].flip()
            self.lattice[X_cl[i]][Y_cl[i]].addToCluster(0)