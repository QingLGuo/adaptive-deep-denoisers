# Utility functions required for denoisers

import numpy as np

def integral_img_sq_diff(v,dx,dy):
    #sd,diff,t = integral_img_sq_diff(padded_guide,dx,dy)
    # 对图像进行平移，得到平移后的图像
    t = img2Dshift(v,dx,dy)
    # 计算平移后的图像与原图像的差值的平方
    diff = (v-t)**2
    # 在垂直方向上对差值的平方进行累积和
    sd = np.cumsum(diff,axis=0)
    # 在水平方向上对垂直累积和的结果进行累积和
    sd = np.cumsum(sd,axis=1)
    #sd是一个二维的张量，张量中每一个数值表示diff中（0，0）到（i，j）区间内的累积和
    # 返回计算得到的积分图像差异、差值平方、以及平移后的图像
    return(sd,diff,t)

#根据距离计算权重   实现了一个函数 triangle(dx, dy, Ns)，用于计算一个三角形权重的值，
# 其目的是在非局部均值（NLM）图像去噪方法中用于平滑权重调整
def triangle(dx,dy,Ns):
    r1 = np.abs(1 - np.abs(dx)/(Ns+1))
    r2 = np.abs(1 - np.abs(dy)/(Ns+1))
    return r1*r2

#temp1 = img2Dshift(sd,patch_rad,patch_rad)
def img2Dshift(v,dx,dy):
    row,col = v.shape
    t = np.zeros((row,col))
    typ = (1 if dx>0 else 0)*2 + (1 if dy>0 else 0)
    if(typ==0):
        t[-dx:,-dy:] = v[0:row+dx,0:col+dy]
    elif(typ==1):
        t[-dx:,0:col-dy] = v[0:row+dx,dy:]
    elif(typ==2):
        t[0:row-dx,-dy:] = v[dx:,0:col+dy]
    elif(typ==3):
        t[0:row-dx,0:col-dy] = v[dx:,dy:]
    return t

