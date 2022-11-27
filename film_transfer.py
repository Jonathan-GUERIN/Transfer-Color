# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 12:32:32 2022

@author: jonathan
"""

path1 = 'im/Mahana.jpg'
path2 = 'C:/Users/jonathan/Documents/Telecom2A/IMA201/Projet/video/input2/frame000002.jpeg'
path3 = 'C:/Users/jonathan/Documents/Telecom2A/IMA201/Projet/video/output2/frame000002.jpeg'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#%matplotlib inline
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from PIL import Image
import matplotlib.image as ima
import os
import ffmpeg
from scipy.ndimage import interpolation


def read_frame(path1,path2):
    """
    image = Image.open(lien)
    arr = np.asarray(image)
    img = Image.fromarray(arr)
    img1 = img.resize(size=(200,66),resample=Image.BILINEAR)
    return np.array(img1)
    
    """
    imrgb1 = plt.imread(path2)/255
    imrgb2 = plt.imread(path1)/255
    imrgb1=imrgb1[:,:,0:3] # useful if the image is a png with a transparency channel
    imrgb2=imrgb2[:,:,0:3] 
    
    """
    #we display the images
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7))
    axes[0].imshow(imrgb1)
    axes[0].set_title('First image')
    axes[1].imshow(imrgb2)
    axes[1].set_title('Second image')
    fig.tight_layout()
    
    # Separable color transfer
    print(imrgb1.shape)
    print(imrgb2.shape)
    """
    
    
    return imrgb1,imrgb2



def todo_specification_separate_channels(imrgb1,imrgb2):
    w = np.zeros(imrgb1.shape)
    for i in range(0,3):
        #w[:,:,i] = (u[:,:,i] -np.mean(u[:,:,i]))/np.std(u[:,:,i])*np.std(v[:,:,i])+ np.mean(v[:,:,i])
        
        u = imrgb1[:,:,i]
        v = imrgb2[:,:,i]
        ushape=u.shape
        uligne=u.reshape((-1,)) #transforme en ligne
        vligne=v.reshape((-1,))
        #print(uligne.shape)
        #print(vligne.shape)
        ind=np.argsort(uligne)
        n = len(ind)
        m = vligne.shape[0]
        unew=np.zeros(uligne.shape,uligne.dtype)
        if n < m:
            pos = np.arange(0,int(m/n)*n,int(m/n))
            unew[ind]=np.sort(vligne)[pos]
        elif n == m:
            unew[ind]=np.sort(vligne)
        elif n > m:
            z = np.sort(vligne)
            #print("LEN Z")
            #print(len(z))
            for j in range(n//m-1):
                z = np.concatenate((z,z),axis=0)
            reste = n%m
            restant = np.sort(vligne)[:reste]
            #print("LEN 2*Z")
            #print(len(z))
            #print("RESTE")
            #print(reste)
            #print("RESTANT")
            #print(len(restant))
            z = np.concatenate((z,restant),axis=0)
            #print(unew.shape[0])
            #print(len(z))
            unew[ind]=np.sort(z)
            
            #unew[ind]=np.sort(vligne)[pos]
            
            
        # on remet a la bonne taille
        unew=unew.reshape(ushape)
        w[:,:,i] = unew
    return w

def todo_specification_separate_channels_3(imrgb1,imrgb2):
    w = np.zeros(imrgb1.shape)
    for i in range(0,3):
        #w[:,:,i] = (u[:,:,i] -np.mean(u[:,:,i]))/np.std(u[:,:,i])*np.std(v[:,:,i])+ np.mean(v[:,:,i])
        
        u = imrgb1[:,:,i]
        v = imrgb2[:,:,i]
        ushape=u.shape
        uligne=u.reshape((-1,)) #transforme en ligne
        vligne=v.reshape((-1,))
        
        n = uligne.shape[0]
        m = vligne.shape[0]
        print(n)
        print(m)
        z = n / m

        v_int = interpolation.zoom(vligne,z)
        
        ind=np.argsort(uligne)
        unew=np.zeros(uligne.shape,uligne.dtype)
        unew[ind]=np.sort(v_int)
        # on remet a la bonne taille
        unew=unew.reshape(ushape)
        w[:,:,i] = unew
    return w

def affine_transfer(u,v):
    w = np.zeros(u.shape)
    for i in range(0,3):
        w[:,:,i] = (u[:,:,i] -np.mean(u[:,:,i]))/np.std(u[:,:,i])*np.std(v[:,:,i])+ np.mean(v[:,:,i])
    return w

def apply_transfer_frame_affine(imrgb1,imrgb2):
    w = affine_transfer(imrgb1,imrgb2)
    w = (w>1)+(w<=1)*(w>0)*w      # w should be in [0,1]
    return w

def apply_transfer_frame(imrgb1,imrgb2):
    w = todo_specification_separate_channels_3(imrgb1,imrgb2)
    w = (w>1)+(w<=1)*(w>0)*w      # w should be in [0,1]
    return w

def apply_transfer_frame_axes(imrgb1,imrgb2):
    q=get_optimal_axes(imrgb2)
    W = todo_specification_separate_channels_upgrade(imrgb1,imrgb2,q)
    w_bis = W.reshape(imrgb1.shape[0],imrgb1.shape[1],3)
    w_bis = (w_bis>1)+(w_bis<=1)*(w_bis>0)*w_bis      # w should be in [0,1]
    return w_bis

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
        
        n = Wt.shape[0]
        m = Yt.shape[0]
        print(n)
        print(m)
        z = n / m
        Y_int = interpolation.zoom(Yt,z)
        
        [sW,sY]=transport1D(Wt,Y_int)
        W[sW,:] += (Y_int[sY]-Wt[sW])[:,None] * q[:,i][None,:] 
    return W


def register_frame(w,path3,name):
    #plt.figure(figsize=(7, 7))
    #plt.title('result of color separable transfer')
    #plt.imshow(w)
    #plt.savefig
    
    #rgb_im = w.convert("RGB")
    #rgb_im = Image.fromarray(w, 'RGB')
    #rgb_im.show()
    #rgb_im.save("geeksforgeeks_jpg.jpg")
    
    path3 = 'C:/Users/jonathan/Documents/Telecom2A/IMA201/Projet/video/output2/'
    path4 = path3 + name
    print("REGISTERING")
    print(path4)
    #w = todo_specification_separate_channels_3(imrgb1,imrgb2)
    w = (w>1)+(w<=1)*(w>0)*w      # w should be in [0,1]
    
    ima.imsave(path4, w)
    
    
    
    
    
    
    
def transfer_one_frame(path1,path2,path3,name):
    imrgb1, imrgb2 = read_frame(path1, path2)
    w = apply_transfer_frame(imrgb1, imrgb2)
    #smpath = 'C:/Users/jonathan/Documents/Telecom2A/IMA201/Projet/video/temp.jpeg'
    #ima.imsave(smpath, w)
    wsliced = w
    #wsliced = plt.imread('C:/Users/jonathan/Documents/Telecom2A/IMA201/Projet/video/temp.jpeg')
    #wsliced = wsliced[:,:,0:3] 
    w = apply_guided_filter(imrgb1,wsliced)
    register_frame(w,path3,name)
    
  
    
path1 = 'im/Mahana.jpg'
paths2 = 'C:/Users/jonathan/Documents/Telecom2A/IMA201/Projet/video/input2/'
paths3 = 'C:/Users/jonathan/Documents/Telecom2A/IMA201/Projet/video/output2/'

def run_transfer_over_frames(path1,paths2,paths3):
    base = 'frame'
    for i in range(1,len(os.listdir(paths2))-1):
        if i < 10:
            name =  base + '00000' + str(i) + '.jpeg'
            path2 = paths2 + name
            path3 = paths3 + name
        elif i >= 10:
            name =  base + '0000' + str(i) + '.jpeg'
            path2 = paths2 + name
            path3 = paths3 + name
        transfer_one_frame(path1, path2, path3,name)
        #print(name)
 
#run_transfer_over_frames(path1,paths2,paths3)




paths2 = 'C:/Users/jonathan/Documents/Telecom2A/IMA201/Projet/video/input2/'
video_name_input = 'fish_soup.mp4'
fps = 10

def split_input_film(paths2, video_name_input,fps):
    video_input = paths2 + video_name_input
    frame_input = paths2 + 'frame%06d.jpeg'
    os.system('cmd /k "ffmpeg -i {} -vf fps={} {}"'.format(video_input,str(fps),frame_input))

#split_input_film(paths2, video_name_input,fps)



entry_path = 'C:/Users/jonathan/Documents/Telecom 2A/IMA201/Projet/video/input2/fish.mp4'
output_path = 'C:/Users/jonathan/Documents/Telecom 2A/IMA201/Projet/video/input2/fish_soup.mp4'
begin_trim = 0
ending_trim = 5
    
def trim(in_file, out_file, start, end):
    if os.path.exists(out_file):
        os.remove(out_file)
        
    probe_result = ffmpeg.probe(in_file)
    in_file_duration = probe_result.get("format", {}).get("duration",None)
    print(in_file_duration)
    
    input_stream = ffmpeg.input(in_file)
    
    pts = "PTS-STARTPTS"
    video = input_stream.trim(start=start,end=end).setpts(pts)
    audio = (input_stream
             .filter_("atrim", start = start, end = end)
             .filter_("asetpts", pts))
    
    video_and_audio = ffmpeg.concat(video,audio,v=1,a=1)
    output = ffmpeg.output(video, out_file,format="mp4")
    output.run()
    
#trim(entry_path,output_path,begin_trim,ending_trim)

def trim2():
    os.system('cmd /k "ffmpeg -i C:/Users/jonathan/Documents/Telecom2A/IMA201/Projet/video/input2/fish.mp4 -ss 00:00:01 -t 00:00:05 -c:v copy -an C:/Users/jonathan/Documents/Telecom2A/IMA201/Projet/video/input2/fish_soup.mp4"')

def build():
    os.system('cmd /k "ffmpeg -r 10 -i C:/Users/jonathan/Documents/Telecom2A/IMA201/Projet/video/output2/frame%06d.jpeg -pix_fmt yuv420p C:/Users/jonathan/Documents/Telecom2A/IMA201/Projet/video/output2/output_video.mp4"')


#
def average_filter(u,r):
    # uniform filter with a square (2*r+1)x(2*r+1) window 
    # u is a 2d image
    # r is the radius for the filter
   
    (nrow, ncol)                                      = u.shape
    big_uint                                          = np.zeros((nrow+2*r+1,ncol+2*r+1))
    big_uint[r+1:nrow+r+1,r+1:ncol+r+1]               = u
    big_uint                                          = np.cumsum(np.cumsum(big_uint,0),1)       # integral image
        
    out = big_uint[2*r+1:nrow+2*r+1,2*r+1:ncol+2*r+1] + big_uint[0:nrow,0:ncol] - big_uint[0:nrow,2*r+1:ncol+2*r+1] - big_uint[2*r+1:nrow+2*r+1,0:ncol]
    out = out/(2*r+1)**2
    
    return out


def todo_guided_filter(u,guide,r,epsilon):
    mean_u = average_filter(u,r)
    mean_guide = average_filter(guide,r)
    guide_u = average_filter(guide*u,r)
    sigma_sqr = average_filter((guide-mean_guide)*(guide-mean_guide),r)
    a = (guide_u-mean_u*mean_guide)/(sigma_sqr+epsilon)
    b = mean_u - a*mean_guide
    
    q = average_filter(a,r)*guide + average_filter(b,r)
    
    return q
    

#usubsample = plt.imread(path+'renoir.jpg')/255
#wsliced    = plt.imread(path+'renoir_by_gauguin.png')
#wsliced    = wsliced[:,:,0:3]

def apply_guided_filter(usubsample,wsliced):
    diff = wsliced-usubsample
    out = np.zeros_like(usubsample)
    
    for i in range(3):
        out[:,:,i] = todo_guided_filter(diff[:,:,i], usubsample[:,:,i], 20,1e-4 )
    w = out + usubsample
    return w
#


def overall_run():
    #trim(entry_path,output_path,begin_trim,ending_trim)
    trim2()
    split_input_film(paths2, video_name_input,fps)
    run_transfer_over_frames(path1,paths2,paths3)
    build()
    