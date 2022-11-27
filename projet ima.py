import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

def affine_transfer(u,v):
    w = np.zeros(u.shape)
    for i in range(0,3):
        w[:,:,i] = (u[:,:,i] -np.mean(u[:,:,i]))/np.std(u[:,:,i])*np.std(v[:,:,i])+ np.mean(v[:,:,i])
    return w

def todo_specification_separate_channels(u,v):
    nrowu,ncolu,nchu = u.shape
    w = np.zeros(u.shape)
    for i in range(3):
        uch = u[:,:,i]
        vch = v[:,:,i]
        u_sort,index_u=np.sort(uch,axis=None),np.argsort(uch,axis=None)
        v_sort,index_v=np.sort(vch,axis=None),np.argsort(vch,axis=None)
        uspecifv= np.zeros(nrowu*ncolu)
        uspecifv[index_u] = v_sort
        uspecifv = uspecifv.reshape(nrowu,ncolu)   
        w[:,:,i] = uspecifv.reshape(nrowu,ncolu)
    return w

w = todo_specification_separate_channels(imrgb1,imrgb2)
w = (w>1)+(w<=1)*(w>0)*w      # w should be in [0,1]

def cov(a, b):

    if len(a) != len(b):
        return

    a_mean = np.mean(a)
    b_mean = np.mean(b)

    sum = 0

    for i in range(0, len(a)):
        sum += ((a[i] - a_mean) * (b[i] - b_mean))

    return sum/(len(a)-1)

def transport1D(X,Y):
    sx = np.argsort(X) #argsort retourne les indices des valeurs s'ils étaient ordonnés par ordre croissant   
    sy = np.argsort(Y)
    return((sx,sy)) 

def get_optimal_axes(v):
    cov_var=np.zeros((3,3))
    rouge=[]
    vert=[]
    bleu=[]
    for i in range(3):
        for liste in v[:,:,i]:
            for element in liste :
                if i == 0 :
                    rouge.append(element)
                elif i == 1:
                    vert.append(element)
                elif i == 2:
                    bleu.append(element)

    couleur =[rouge,vert,bleu]
    for i in range(3):
        for j in range(3):
            if i==j :
                cov_var[i,j]=np.var(couleur[i])
            else :
                cov_var[i,j]= cov(couleur[i],couleur[j])

    m=np.linalg.eig(cov_var)[1]
    q=np.linalg.qr(m)[0]
    
    return q

def todo_specification_separate_channels_upgrade(u,v,q):
    usubsample = np.copy(u)
    vsubsample = np.copy(v)
    X = usubsample.reshape((usubsample.shape[0]*usubsample.shape[1],3))
    Y = vsubsample.reshape((vsubsample.shape[0]*vsubsample.shape[1],3))
    W = np.copy(X) # output
    for i in range(3):
        Wt = np.dot(W,q[:,i])
        Yt = np.dot(Y,q[:,i])
        [sW,sY]=transport1D(Wt,Yt)
        W[sW,:] += (Yt[sY]-Wt[sW])[:,None] * q[:,i][None,:] 
    return W

q=get_optimal_axes(imrgb2)
W = todo_specification_separate_channels_upgrade(imrgb1,imrgb2,q)
w_bis = W.reshape(imrgb1.shape[0],imrgb1.shape[1],3)
w_bis = (w_bis>1)+(w_bis<=1)*(w_bis>0)*w_bis      # w should be in [0,1]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
axes[0].imshow(w)
axes[0].set_title('Standard')
axes[1].imshow(w_bis)
axes[1].set_title('Upgrade')
fig.tight_layout()