"""Generate reproducible 512x512 synthetic test drawings, no external dataset."""
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'examples'
OUT.mkdir(exist_ok=True)

def canvas(): return np.full((512,512,3),255,np.uint8)

def save(name,img,gt):
    cv2.imwrite(str(OUT/(name+'.png')),img)
    cv2.imwrite(str(OUT/(name+'_gt.png')),gt)

def main():
    im=canvas()
    cv2.line(im,(70,240),(440,240),(0,0,0),28,cv2.LINE_AA)
    cv2.line(im,(255,240),(255,440),(0,0,0),26,cv2.LINE_AA)
    gt=canvas()
    cv2.line(gt,(70,240),(440,240),(0,0,0),1,cv2.LINE_8)
    cv2.line(gt,(255,240),(255,440),(0,0,0),1,cv2.LINE_8)
    save('t_junction',im,gt)
    im=canvas()
    cv2.line(im,(64,70),(442,445),(0,0,0),31,cv2.LINE_AA)
    cv2.line(im,(443,72),(70,443),(0,0,0),24,cv2.LINE_AA)
    gt=canvas()
    cv2.line(gt,(64,70),(442,445),(0,0,0),1)
    cv2.line(gt,(443,72),(70,443),(0,0,0),1)
    save('x_junction',im,gt)
    im=canvas()
    pts=np.array([[60,390],[145,300],[210,210],[280,148],[400,100]],np.int32)
    cv2.polylines(im,[pts],False,(0,0,0),25,cv2.LINE_AA)
    cv2.line(im,(211,209),(370,365),(0,0,0),20,cv2.LINE_AA)
    gt=canvas()
    cv2.polylines(gt,[pts],False,(0,0,0),1)
    cv2.line(gt,(211,209),(370,365),(0,0,0),1)
    save('y_junction',im,gt)
    im=canvas()
    cv2.circle(im,(255,254),145,(0,0,0),26,cv2.LINE_AA)
    gt=canvas()
    cv2.circle(gt,(255,254),145,(0,0,0),1)
    save('closed_loop',im,gt)
    im=canvas()
    t=np.linspace(0,1,170)
    x=40+430*t
    y=260+90*np.sin(2*np.pi*t)
    pts=np.round(np.c_[x,y]).astype(np.int32)
    for i in range(len(pts)-1):
        cv2.line(im,tuple(pts[i]),tuple(pts[i+1]),(0,0,0),int(8+28*i/(len(pts)-1)),cv2.LINE_AA)
    gt=canvas()
    cv2.polylines(gt,[pts],False,(0,0,0),1)
    save('varying_width',im,gt)
    print('Generated 5 examples in',OUT)

if __name__ == '__main__':main()
