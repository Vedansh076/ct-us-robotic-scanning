#!/usr/bin/env python3
"""
SONOBOT – FINAL PUBLICATION‑READY SIMULATION (COMPLETE)
=======================================================a
- PyBullet hospital scene (bed, cart, human mesh)
- 3D abdominal anatomy phantom (liver, kidney, etc.)
- Fan‑beam ultrasound rendering (impedance boundaries, TGC, speckle)
- Image‑content quality metric (target visibility, contrast, sharpness)
- Riemannian Matérn‑3/2 vs Euclidean RBF Bayesian Optimisation
- Safe orientation clamp (≤15° from vertical) + max step 5°
- Lateral scan over liver (world Y axis)
- Live US display (OpenCV) in interactive mode
- Headless experiment mode with statistical analysis
- Keys: 1=Euclidean  2=Riemannian  A=Auto both  Q=Quit & save plots

PLOTS ARE SAVED AFTER EVERY SCAN AND ON QUIT.
"""

import os, sys, time, tempfile, argparse
import numpy as np
import pybullet as p
import pybullet_data
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.stats import wilcoxon

# ---------- Optional trimesh ----------
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    print("⚠ trimesh not installed – human mesh will not be shown.")

try:
    import rtree
    HAS_RTREE = True
except ImportError:
    HAS_RTREE = False

# ═══════════════════ CONSTANTS ═══════════════════
HUMAN_MESH_PATH = (
    '/Users/lakshaybatta/Downloads/'
    'ar525-master 2/a3/assest/human_body/'
    'mosh_cmu_0511_f_lbs_10_207_0_v1.0.2.obj'
)

ROBOT_BASE = [0.0, -0.5, 0.0]
CART_DIMS  = [0.35, 0.35, 0.25]
BED_X, BED_Y = 0.5, 0.0
BED_LENGTH, BED_WIDTH = 2.0, 1.1
LEG_HEIGHT, MATTRESS_THICK = 0.15, 0.08
MATTRESS_TOP_Z = LEG_HEIGHT + 0.05 + 0.04 + MATTRESS_THICK
HUMAN_Z_OFFSET = MATTRESS_TOP_Z
HUMAN_CENTER_X = BED_X - 0.1

SCAN_X = 0.43                         # fixed head‑foot position (over liver)
SCAN_Y0, SCAN_Y1 = -0.07, 0.01        # lateral sweep (world Y)
N_WP = 30
HOVER_Z = MATTRESS_TOP_Z + 0.55
TOOL_LEN = 0.10
DT = 1/240.

JL    = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
JU    = [ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973]
JR    = [u - l for u, l in zip(JU, JL)]
JREST = [0.0, -0.4, 0.0, -2.4, 0.0, 1.8, 0.8]

# ═══════════════════ SONOGYM‑STYLE RENDERING ═══════════════════
class Tissue:
    AIR,SKIN,FAT,MUSCLE,FASCIA,LIVER,LIVER_VES,GALLBLADDER,KIDNEY_COR,KIDNEY_MED, \
    BOWEL_WALL,BOWEL_GAS,AORTA,IVC,SPINE,BLADDER_W,BLADDER_F = range(17)
    Z = {0:0.0004,1:1.68,2:1.38,3:1.70,4:1.72,5:1.65,6:1.62,7:1.51,
         8:1.62,9:1.58,10:1.68,11:0.0004,12:1.62,13:1.62,14:7.80,15:1.65,16:1.52}
    ECHO = {0:0.00,1:0.15,2:0.12,3:0.22,4:0.08,5:0.25,6:0.00,7:0.00,
            8:0.22,9:0.08,10:0.20,11:0.85,12:0.00,13:0.00,14:0.55,15:0.08,16:0.00}
    ATTEN = {0:10.0,1:0.35,2:0.48,3:0.57,4:0.80,5:0.50,6:0.18,7:0.05,
             8:0.55,9:0.55,10:0.60,11:8.00,12:0.18,13:0.18,14:2.50,15:0.40,16:0.08}
    POST = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:+1,8:0,9:0,10:0,11:-1,12:+1,13:+1,14:-1,15:0,16:+1}

class SyntheticTorsoVolume:
    DX,NX=15.0,150; DY,NY=18.0,180; DZ,NZ=12.0,120
    def __init__(self,verbose=True):
        if verbose: print("Building anatomy volume …",end='',flush=True)
        t0=time.time()
        x=np.linspace(-self.DX/2,self.DX/2,self.NX)
        y=np.linspace(-self.DY/2,self.DY/2,self.NY)
        z=np.linspace(0,self.DZ,self.NZ)
        X,Y,Z=np.meshgrid(x,y,z,indexing='ij')
        self.vol=self._build(X,Y,Z)
        self.x_ax,self.y_ax,self.z_ax=x,y,z
        self.Zvol=self._lut(Tissue.Z)
        self.echvol=self._lut(Tissue.ECHO)
        self.attvol=self._lut(Tissue.ATTEN)
        self.postvol=np.array([Tissue.POST.get(i,0) for i in range(17)],dtype=np.float32)[self.vol]
        if verbose: print(f" done ({time.time()-t0:.1f}s)")
    def _lut(self,d): return np.array([d.get(i,0.) for i in range(17)],dtype=np.float32)[self.vol]
    @staticmethod
    def _ell(X,Y,Z,cx,cy,cz,rx,ry,rz): return ((X-cx)**2/rx**2+(Y-cy)**2/ry**2+(Z-cz)**2/rz**2)<1.
    @staticmethod
    def _cyl(X,Y,Z,cx,cz,r,y0,y1): return ((X-cx)**2+(Z-cz)**2<r**2)&(Y>y0)&(Y<y1)
    def _build(self,X,Y,Z):
        T=Tissue; v=np.zeros((self.NX,self.NY,self.NZ),dtype=np.uint8)
        v[Z<0.4]=T.SKIN; v[(Z>=0.4)&(Z<2.5)&(np.abs(X)<7.)]=T.FAT
        for xc in [-1.8,1.8]: v[(Z>=2.5)&(Z<5.0)&(np.abs(X-xc)<1.5)&(Y>-8.)&(Y<7.)]=T.MUSCLE
        for s in [-1,1]: v[(Z>=1.5)&(Z<4.5)&(s*X>2.5)&(s*X<7.)]=T.MUSCLE
        v[(Z>=2.5)&(Z<5.5)&(np.abs(X)<0.3)&(Y>-8.)&(Y<7.)]=T.FASCIA
        v[(np.abs(Z-2.5)<0.25)&(np.abs(X)<5.)&(Y>-8.)&(Y<7.)]=T.FASCIA
        liv=self._ell(X,Y,Z,-3.,-3.5,7.5,5.5,5.0,4.0); liv|=self._ell(X,Y,Z,1.,-4.0,6.5,2.5,3.5,3.0)
        v[liv&(Z>4.5)]=T.LIVER
        pv=self._cyl(X,Y,Z,-3.,7.0,0.55,-6.,-1.); v[pv&(v==T.LIVER)]=T.LIVER_VES
        hv=self._cyl(X,Y,Z,-1.5,6.5,0.40,-8.,-2.); v[hv&(v==T.LIVER)]=T.LIVER_VES
        v[self._ell(X,Y,Z,-4.5,-1.5,7.0,1.8,1.5,1.8)]=T.GALLBLADDER
        v[self._ell(X,Y,Z,-5.,-1.0,9.5,2.5,2.8,2.0)]=T.KIDNEY_COR; v[self._ell(X,Y,Z,-5.,-1.0,9.5,1.4,1.7,1.1)]=T.KIDNEY_MED
        v[self._ell(X,Y,Z,5.,0.5,9.0,2.3,2.6,1.9)]=T.KIDNEY_COR; v[self._ell(X,Y,Z,5.,0.5,9.0,1.3,1.5,1.1)]=T.KIDNEY_MED
        v[self._cyl(X,Y,Z,0.8,9.5,0.80,-9.,9.)]=T.AORTA; v[self._cyl(X,Y,Z,-0.8,9.2,0.65,-9.,9.)]=T.IVC
        rng=np.random.RandomState(42)
        for _ in range(8):
            bx=rng.uniform(-3,3); by=rng.uniform(-1,4); bz=rng.uniform(5,7.5); br=rng.uniform(0.6,1.0)
            loop=(np.sqrt((X-bx)**2+(Z-bz)**2)<br)&(np.abs(Y-by)<1.2)
            inner=(np.sqrt((X-bx)**2+(Z-bz)**2)<br*.5)&(np.abs(Y-by)<1.0)
            v[loop&(v==0)]=T.BOWEL_WALL; v[inner&(v==T.BOWEL_WALL)]=T.BOWEL_GAS
        v[self._ell(X,Y,Z,0.,6.5,7.0,3.0,2.0,3.0)]=T.BLADDER_W; v[self._ell(X,Y,Z,0.,6.5,7.0,2.6,1.6,2.6)]=T.BLADDER_F
        v[(np.abs(X)<2.)&(Z>10.)]=T.SPINE; v[(Z<0.15)&(v>T.FAT)]=T.SKIN
        return v

class FanBeamUSRenderer:
    N_LINES=192; N_DEPTH=512; DEPTH_CM=15.0; FAN_DEG=60.0; FREQ=3.5; IM=512; DYN_DB=60.0
    def __init__(self,volume,seed=0):
        self.vol=volume; self.rng=np.random.RandomState(seed)
        self.FH=np.radians(self.FAN_DEG/2.); self.angles=np.linspace(-self.FH,self.FH,self.N_LINES)
        self.depths=np.linspace(0.0,self.DEPTH_CM,self.N_DEPTH); self.dz=self.DEPTH_CM/(self.N_DEPTH-1.)
        self.spk_sigma=max(1.5,0.44/(self.DEPTH_CM*10./self.N_DEPTH))
        self._build_sc_lut()
    def _build_sc_lut(self):
        IM,FH,DC=self.IM,self.FH,self.DEPTH_CM; half_w=DC*np.sin(FH)
        px,py=np.arange(IM,dtype=np.float32),np.arange(IM,dtype=np.float32); PX,PY=np.meshgrid(px,py,indexing='ij')
        lat_cm=(PX/(IM-1.)-0.5)*2.*half_w; depth_cm=(PY/(IM-1.))*DC
        r=np.sqrt(lat_cm**2+depth_cm**2); theta=np.arctan2(lat_cm,depth_cm+1e-6)
        self._mask=(np.abs(theta)<=FH)&(r<=DC)&(depth_cm>=0.4)
        line_f=(theta+FH)/(2.*FH)*(self.N_LINES-1.); depth_f=r/DC*(self.N_DEPTH-1.)
        self._sc_line=np.clip(line_f,0.,self.N_LINES-1.).astype(np.float32)
        self._sc_depth=np.clip(depth_f,0.,self.N_DEPTH-1.).astype(np.float32)
    def _interp(self,prop,wx,wy,wz):
        V=self.vol
        ix=np.clip((wx-V.x_ax[0])/(V.x_ax[-1]-V.x_ax[0])*(V.NX-1),0.,V.NX-1.)
        iy=np.clip((wy-V.y_ax[0])/(V.y_ax[-1]-V.y_ax[0])*(V.NY-1),0.,V.NY-1.)
        iz=np.clip((wz-V.z_ax[0])/(V.z_ax[-1]-V.z_ax[0])*(V.NZ-1),0.,V.NZ-1.)
        return map_coordinates(prop,[ix.ravel(),iy.ravel(),iz.ravel()],order=1,mode='constant',cval=0.).reshape(wx.shape)
    def _speckle(self,shape):
        xi=self.rng.randn(*shape).astype(np.float32); xq=self.rng.randn(*shape).astype(np.float32)
        xi=gaussian_filter(xi,sigma=(0.,self.spk_sigma)); xq=gaussian_filter(xq,sigma=(0.,self.spk_sigma))
        env=np.sqrt(xi**2+xq**2); return env/(env.mean()+1e-8)
    def render(self,probe_pos_cm,probe_R):
        NL,ND=self.N_LINES,self.N_DEPTH; sin_a,cos_a=np.sin(self.angles),np.cos(self.angles)
        dirs_probe=np.stack([sin_a,np.zeros_like(sin_a),cos_a],axis=1); dirs_body=dirs_probe@probe_R.T
        pos=(probe_pos_cm[:,None,None]+dirs_body.T[:,:,None]*self.depths[None,None,:]); wx,wy,wz=pos[0],pos[1],pos[2]
        Z_s=self._interp(self.vol.Zvol,wx,wy,wz); echo_s=self._interp(self.vol.echvol,wx,wy,wz)
        att_s=self._interp(self.vol.attvol,wx,wy,wz); post_s=self._interp(self.vol.postvol,wx,wy,wz)
        Z_next,Z_prev=np.roll(Z_s,-1,axis=1),np.roll(Z_s,1,axis=1)
        R=np.abs(Z_next-Z_prev)/(Z_next+Z_prev+1e-6); boundary=np.clip(R/0.65,0.,1.)
        backscatt=echo_s*self._speckle((NL,ND)); raw=0.70*boundary+0.30*backscatt
        cum_att=np.cumsum(att_s,axis=1)*self.dz*self.FREQ; lin_att=np.power(10.,-cum_att/20.); raw*=lin_att
        tgc=np.power(10.,(0.50*self.FREQ*self.depths)/20.); raw*=tgc[None,:]
        cum_post=np.cumsum(post_s,axis=1); enh,shad=np.clip(cum_post,0.,None),np.clip(-cum_post,0.,None)
        raw*=(1.+0.50*np.tanh(enh*0.4))*np.exp(-0.70*shad)
        nf_px=max(2,int(0.6/self.dz)); rev=np.exp(-np.arange(nf_px)/(nf_px*0.25))*0.15; raw[:,:nf_px]+=rev[None,:]
        raw=np.clip(raw,0.,None); top=np.percentile(raw[raw>0.001],99) if (raw>0.001).any() else 1.; raw/=(top+1e-8)
        dyn_lin=10.**(self.DYN_DB/20.); log_sig=np.log1p(raw*(dyn_lin-1.))/np.log(dyn_lin)
        scan_data=np.clip(log_sig,0.,1.).astype(np.float32); mask=self._mask
        coord=np.array([self._sc_line[mask].ravel(),self._sc_depth[mask].ravel()])
        vals=map_coordinates(scan_data,coord,order=1,mode='constant',cval=0.)
        img_f=np.zeros((self.IM,self.IM),dtype=np.float32); img_f[mask]=vals
        img_f=1./(1.+np.exp(-7.*(img_f-0.42))); img_f[~mask]=0.; img_f=gaussian_filter(img_f,sigma=(0.8,0.)); img_f[~mask]=0.
        return np.clip(img_f*255.,0.,255.).astype(np.uint8)

class ImageQualityAnalyzer:
    _TARGETS={'liver':(45,130),'gallbladder':(0,35),'kidney':(40,135),'bladder':(0,30),'vessels':(0,35)}
    def __init__(self,target='liver'): self.target=target
    def score(self,image):
        if image.mean()<5.: return dict(quality=0.,visibility=0.,contrast=0.,confidence=0.,sharpness=0.)
        f=image.astype(np.float32)/255.
        d=np.exp(-3.*np.linspace(0,1,image.shape[1])); l=np.exp(-4.*np.linspace(-1,1,image.shape[0])**2)
        conf=l[:,None]*d[None,:]; conf*=0.4+0.6*(f>0.04); q_conf=float(conf.mean())
        gx,gy=np.gradient(f); q_sharp=float(np.clip(np.sum(np.hypot(gx,gy)*conf)/(conf.sum()+1e-8)*12.,0.,1.))
        lo,hi=self._TARGETS.get(self.target,(40,180)); tpx=(image>=lo)&(image<=hi)
        q_vis=float(np.clip(tpx.mean()*8.,0.,1.)); q_cont=0.
        if tpx.any() and (~tpx).any(): q_cont=float(np.clip(abs(f[tpx].mean()-f[~tpx].mean())/(f[tpx].mean()+f[~tpx].mean()+1e-8)*2.,0.,1.))
        quality=0.30*q_conf+0.25*q_sharp+0.25*q_vis+0.20*q_cont
        return dict(quality=float(quality),visibility=q_vis,contrast=q_cont,confidence=q_conf,sharpness=q_sharp)

class SonoSimEnv:
    BODY_ORIGIN_M=np.array([0.45,0.0,0.32])
    def __init__(self,target='liver',seed=0,verbose=True):
        self.target=target; self.volume=SyntheticTorsoVolume(verbose=verbose)
        self.renderer=FanBeamUSRenderer(self.volume,seed=seed)
        self.quality=ImageQualityAnalyzer(target=target); self.history=[]
    def _to_body(self,pos_m,R_w):
        d=pos_m-self.BODY_ORIGIN_M; pos_cm=np.array([d[1]*100.,d[0]*100.,max(0.,-d[2]*100.+0.5)])
        R_wb=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]]); return pos_cm,R_wb@R_w
    def step(self,probe_pos_m,probe_quat_wxyz):
        q=np.asarray(probe_quat_wxyz,float); q/=np.linalg.norm(q)+1e-9
        w,x,y,z=q; R=np.array([[1-2*y*y-2*z*z,2*x*y-2*w*z,2*x*z+2*w*y],[2*x*y+2*w*z,1-2*x*x-2*z*z,2*y*z-2*w*x],[2*x*z-2*w*y,2*y*z+2*w*x,1-2*x*x-2*y*y]])
        pos_cm,R_body=self._to_body(np.asarray(probe_pos_m,float),R); image=self.renderer.render(pos_cm,R_body)
        metrics=self.quality.score(image); obs=dict(image=image,quality=metrics['quality'],metrics=metrics,pos_cm=pos_cm,R=R_body)
        self.history.append(obs); return obs

class LiveUSDisplay:
    def __init__(self,wn="SonoSim"):
        self._ok=False
        try: cv2.namedWindow(wn,cv2.WINDOW_NORMAL); cv2.resizeWindow(wn,520,540); self._ok=True; self._wn=wn
        except: pass
    def update(self,image,quality,metrics=None):
        if not self._ok: return
        d=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR); bw=int(quality*image.shape[1])
        cv2.rectangle(d,(0,image.shape[0]-20),(bw,image.shape[0]),(0,int(255*quality),255-int(255*quality)),-1)
        cv2.putText(d,f"Q={quality:.3f}",(8,24),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,0),2)
        if metrics:
            for i,(k,v) in enumerate(metrics.items()):
                if k!='quality': cv2.putText(d,f"{k[:4]}={v:.2f}",(8,46+i*18),cv2.FONT_HERSHEY_SIMPLEX,0.38,(200,200,200),1)
        cv2.imshow(self._wn,d); cv2.waitKey(1)

# ═══════════════════ SO(3) MATH ═══════════════════
def qnorm(q): return q/(np.linalg.norm(q)+1e-9)
def qmul(a,b): w1,x1,y1,z1=a; w2,x2,y2,z2=b; return np.array([w1*w2-x1*x2-y1*y2-z1*z2,w1*x2+x1*w2+y1*z2-z1*y2,w1*y2-x1*z2+y1*w2+z1*x2,w1*z2+x1*y2-y1*x2+z1*w2])
def qconj(q): return np.array([q[0],-q[1],-q[2],-q[3]])
def log_so3(q): q=qnorm(q); w=np.clip(q[0],-1,1); th=2.*np.arccos(w); s=np.sqrt(1-w*w)+1e-9; return q[1:4]*(th/s)
def exp_so3(omega): th=np.linalg.norm(omega); return np.array([1.,omega[0]*.5,omega[1]*.5,omega[2]*.5]) if th<1e-9 else (np.array([np.cos(th/2),(omega/th)[0]*np.sin(th/2),(omega/th)[1]*np.sin(th/2),(omega/th)[2]*np.sin(th/2)]))
def rot_between(v1,v2): v1,v2=v1/(np.linalg.norm(v1)+1e-9),v2/(np.linalg.norm(v2)+1e-9); axis=np.cross(v1,v2); angle=np.arccos(np.clip(np.dot(v1,v2),-1,1)); return np.array([1,0,0,0]) if np.linalg.norm(axis)<1e-9 else qnorm(np.array([np.cos(angle/2),*(axis/np.linalg.norm(axis))*np.sin(angle/2)]))
def quat_dist(q1,q2): return 2.*np.arccos(float(np.clip(abs(np.dot(qnorm(q1),qnorm(q2))),0.,1.)))
def probe_down(): return np.array([0.,1.,0.,0.])

MAX_TILT_DEG=15
def k_riemannian(Q1,Q2,l=0.60):
    n,m=len(Q1),len(Q2); K=np.empty((n,m))
    for i in range(n):
        for j in range(m):
            d=quat_dist(Q1[i],Q2[j]); r=np.sqrt(3.)*d/l; K[i,j]=(1.+r)*np.exp(-r)
    return K
def k_euclidean(Q1,Q2,l=0.60):
    X1=np.vstack([log_so3(q) for q in Q1]); X2=np.vstack([log_so3(q) for q in Q2])
    diff=X1[:,None,:]-X2[None,:,:]; return np.exp(-0.5*np.sum(diff**2,axis=-1)/(l**2))

class ProbeBO:
    def __init__(self,riemannian=True,noise=0.01,beta=2.5,l=0.60):
        self.kern=k_riemannian if riemannian else k_euclidean; self.riemannian=riemannian; self.l,self.noise,self.beta=l,noise,beta
        self.Xo,self.Yo,self.best_hist=[],[],[]
        self.candidates=self._build_grid()
    def _build_grid(self):
        q0=probe_down(); grid=[q0.copy()]
        for tilt in np.linspace(2,MAX_TILT_DEG,8):
            for az in np.linspace(0,360,20,endpoint=False):
                ax=np.array([np.cos(np.radians(az)),np.sin(np.radians(az)),0.])
                grid.append(qnorm(qmul(exp_so3(ax*np.radians(tilt)),q0)))
        return np.array(grid)
    def suggest(self):
        n=len(self.Xo)
        if n<5: return self.candidates[n*13%len(self.candidates)]
        X,Y=np.array(self.Xo),np.array(self.Yo); Y=(Y-Y.mean())/(Y.std()+1e-8)
        Kxx=self.kern(X,X,self.l)+self.noise*np.eye(n); Ksx=self.kern(self.candidates,X,self.l)
        Kss=np.array([self.kern(self.candidates[i:i+1],self.candidates[i:i+1],self.l)[0,0] for i in range(len(self.candidates))])
        try:
            L=np.linalg.cholesky(Kxx); alpha=np.linalg.solve(L.T,np.linalg.solve(L,Y))
            v=np.linalg.solve(L,Ksx.T); mu=Ksx@alpha; var=np.clip(Kss-np.einsum('ij,ij->j',v,v),1e-8,None)
        except: mu,var=np.zeros(len(self.candidates)),np.ones(len(self.candidates))
        return self.candidates[np.argmax(mu+np.sqrt(self.beta)*np.sqrt(var))]
    def observe(self,q,quality): self.Xo.append(qnorm(q)); self.Yo.append(float(quality)); self.best_hist.append(max(self.Yo))

# ═══════════════════ PyBullet Scene ═══════════════════
class Scene:
    def __init__(self): self.robot=self.human_id=None; self.client=-1
    def build(self):
        self.client=p.connect(p.GUI); p.configureDebugVisualizer(p.COV_ENABLE_GUI,0); p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.81); p.setPhysicsEngineParameter(fixedTimeStep=DT,numSolverIterations=100,numSubSteps=4,erp=0.8)
        p.loadURDF("plane.urdf",[0,0,0])
        # Cart
        cart_col=p.createCollisionShape(p.GEOM_BOX,halfExtents=CART_DIMS); cart_vis=p.createVisualShape(p.GEOM_BOX,halfExtents=CART_DIMS,rgbaColor=[0.3,0.3,0.35,1])
        p.createMultiBody(0,cart_col,cart_vis,[ROBOT_BASE[0],ROBOT_BASE[1],CART_DIMS[2]])
        wheel_radius=0.03; wheel_width=0.015; wheel_col=p.createCollisionShape(p.GEOM_CYLINDER,radius=wheel_radius,height=wheel_width)
        wheel_vis=p.createVisualShape(p.GEOM_CYLINDER,radius=wheel_radius,length=wheel_width,rgbaColor=[0.1,0.1,0.1,1])
        wheel_orn=p.getQuaternionFromEuler([np.pi/2,0,0])
        for dx,dy in [[-0.25,-0.25],[-0.25,0.25],[0.25,-0.25],[0.25,0.25]]: p.createMultiBody(0,wheel_col,wheel_vis,[ROBOT_BASE[0]+dx,ROBOT_BASE[1]+dy,0.01],wheel_orn)
        # Bed
        bed_x,bed_y=BED_X,BED_Y; leg_radius=0.03; leg_height=LEG_HEIGHT
        leg_col=p.createCollisionShape(p.GEOM_CYLINDER,radius=leg_radius,height=leg_height); leg_vis=p.createVisualShape(p.GEOM_CYLINDER,radius=leg_radius,length=leg_height,rgbaColor=[0.7,0.7,0.7,1])
        wheel_radius_bed=0.04; wheel_width_bed=0.02; wheel_col_bed=p.createCollisionShape(p.GEOM_CYLINDER,radius=wheel_radius_bed,height=wheel_width_bed)
        wheel_vis_bed=p.createVisualShape(p.GEOM_CYLINDER,radius=wheel_radius_bed,length=wheel_width_bed,rgbaColor=[0.2,0.2,0.2,1]); wheel_orn_bed=p.getQuaternionFromEuler([np.pi/2,0,0])
        for dx,dy in [[1.0,-0.55],[1.0,0.55],[-1.0,-0.55],[-1.0,0.55]]:
            p.createMultiBody(0,leg_col,leg_vis,[bed_x+dx,bed_y+dy,leg_height/2]); p.createMultiBody(0,wheel_col_bed,wheel_vis_bed,[bed_x+dx,bed_y+dy,0.01],wheel_orn_bed)
        frame_thick=0.03; beam_col=p.createCollisionShape(p.GEOM_BOX,halfExtents=[1.0,0.02,frame_thick]); beam_vis=p.createVisualShape(p.GEOM_BOX,halfExtents=[1.0,0.02,frame_thick],rgbaColor=[0.4,0.4,0.4,1])
        p.createMultiBody(0,beam_col,beam_vis,[bed_x,bed_y+0.55,leg_height+frame_thick]); p.createMultiBody(0,beam_col,beam_vis,[bed_x,bed_y-0.55,leg_height+frame_thick])
        cross_col=p.createCollisionShape(p.GEOM_BOX,halfExtents=[0.02,0.57,frame_thick]); cross_vis=p.createVisualShape(p.GEOM_BOX,halfExtents=[0.02,0.57,frame_thick],rgbaColor=[0.4,0.4,0.4,1])
        p.createMultiBody(0,cross_col,cross_vis,[bed_x-1.0,bed_y,leg_height+frame_thick]); p.createMultiBody(0,cross_col,cross_vis,[bed_x+1.0,bed_y,leg_height+frame_thick])
        board_col=p.createCollisionShape(p.GEOM_BOX,halfExtents=[0.02,0.55,0.25]); board_vis=p.createVisualShape(p.GEOM_BOX,halfExtents=[0.02,0.55,0.25],rgbaColor=[0.2,0.3,0.5,1])
        p.createMultiBody(0,board_col,board_vis,[bed_x+1.0,bed_y,leg_height+frame_thick+0.25])
        foot_col=p.createCollisionShape(p.GEOM_BOX,halfExtents=[0.02,0.55,0.15]); foot_vis=p.createVisualShape(p.GEOM_BOX,halfExtents=[0.02,0.55,0.15],rgbaColor=[0.2,0.3,0.5,1])
        p.createMultiBody(0,foot_col,foot_vis,[bed_x-1.0,bed_y,leg_height+frame_thick+0.15])
        mattress_half=[BED_LENGTH/2,BED_WIDTH/2,MATTRESS_THICK]; mattress_col=p.createCollisionShape(p.GEOM_BOX,halfExtents=mattress_half)
        mattress_vis=p.createVisualShape(p.GEOM_BOX,halfExtents=mattress_half,rgbaColor=[1,1,1,1]); p.createMultiBody(0,mattress_col,mattress_vis,[bed_x,bed_y,MATTRESS_TOP_Z])
        print("✓ Hospital bed")
        # Robot
        robot_base_pos=[ROBOT_BASE[0],ROBOT_BASE[1],CART_DIMS[2]*2]; self.robot=p.loadURDF("franka_panda/panda.urdf",robot_base_pos,useFixedBase=True)
        p.enableJointForceTorqueSensor(self.robot,6,True)
        self._load_human()
        col=p.createCollisionShape(p.GEOM_BOX,halfExtents=[0.2,0.1,0.005]); vis=p.createVisualShape(p.GEOM_BOX,halfExtents=[0.2,0.1,0.005],rgbaColor=[0.6,0.8,1.0,0.4])
        p.createMultiBody(0,col,vis,basePosition=[0.45,SCAN_Y0,0.47])
        p.resetDebugVisualizerCamera(1.8,45,-25,[0.45,0.0,0.5])
        return self
    def _load_human(self):
        if not HAS_TRIMESH: return
        try:
            m=trimesh.load(HUMAN_MESH_PATH)
            if isinstance(m,trimesh.Scene): m=m.geometry[list(m.geometry.keys())[0]]
            m.apply_scale(1.70/max(m.extents)); m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-90.),[0,0,1]))
            m.apply_translation([0,0,-m.vertices[:,2].min()]); c=m.centroid
            m.apply_translation([HUMAN_CENTER_X-c[0],-c[1],HUMAN_Z_OFFSET])
            with tempfile.NamedTemporaryFile(suffix='.obj',delete=False) as f: m.export(f.name); vis=p.createVisualShape(p.GEOM_MESH,fileName=f.name,rgbaColor=[1,0.8,0.7,1])
            os.unlink(f.name); self.human_id=p.createMultiBody(0,baseVisualShapeIndex=vis,basePosition=[0,0,0])
            print("✓ Human mesh on bed")
        except Exception as e: print(f"Human mesh failed: {e}")
    def reset_robot_to_home(self):
        print("  Resetting robot to home …"); q=probe_down(); qx=[q[1],q[2],q[3],q[0]]; pos=[0.45,SCAN_Y0,HOVER_Z]
        j=p.calculateInverseKinematics(self.robot,11,pos,qx,lowerLimits=JL,upperLimits=JU,jointRanges=JR,restPoses=JREST,maxNumIterations=500,residualThreshold=1e-5)
        for i in range(7): p.resetJointState(self.robot,i,j[i])
        for _ in range(200): p.stepSimulation()
        print("  Home position restored.")
    def close(self):
        if self.client>=0: p.disconnect(self.client)

# ═══════════════════ Robot Helpers ═══════════════════
def compute_ik(robot,pos,q_wxyz): qx=[q_wxyz[1],q_wxyz[2],q_wxyz[3],q_wxyz[0]]; return list(p.calculateInverseKinematics(robot,11,pos,qx,lowerLimits=JL,upperLimits=JU,jointRanges=JR,restPoses=JREST,maxNumIterations=200,residualThreshold=1e-4)[:7])
def drive(robot,joints,force=50.,velocity=0.07): [p.setJointMotorControl2(robot,i,p.POSITION_CONTROL,targetPosition=joints[i],force=force,maxVelocity=velocity) for i in range(7)]; p.stepSimulation()
def ee_pose(robot): s=p.getLinkState(robot,11,computeForwardKinematics=True); pos=np.array(s[4]); xyzw=s[5]; return pos,np.array([xyzw[3],xyzw[0],xyzw[1],xyzw[2]])
def quat_to_rotmat(q): w,x,y,z=q; return np.array([[1-2*y*y-2*z*z,2*x*y-2*w*z,2*x*z+2*w*y],[2*x*y+2*w*z,1-2*x*x-2*z*z,2*y*z-2*w*x],[2*x*z-2*w*y,2*y*z+2*w*x,1-2*x*x-2*y*y]])
def probe_tip_pos(flange_pos,flange_quat_wxyz): R=quat_to_rotmat(flange_quat_wxyz); return flange_pos+R@np.array([0,0,TOOL_LEN])

def clamp_orientation(q,max_tilt_deg,downward=np.array([0.,0.,-1.])):
    w,x,y,z=q; R=np.array([[1-2*y*y-2*z*z,2*x*y-2*w*z,2*x*z+2*w*y],[2*x*y+2*w*z,1-2*x*x-2*z*z,2*y*z-2*w*x],[2*x*z-2*w*y,2*y*z+2*w*x,1-2*x*x-2*y*y]])
    probe_z=R[:,2]; angle=np.degrees(np.arccos(np.clip(np.dot(probe_z,downward),-1,1)))
    if angle>max_tilt_deg: t=max_tilt_deg/angle; return qnorm(q+t*(rot_between(probe_z,downward)-q))
    return q

class ScanSession:
    def __init__(self,scene,env,riemannian=True,show_live=True):
        self.scene=scene; self.env=env; self.bo=ProbeBO(riemannian=riemannian)
        self.name="Riemannian Matérn‑3/2 BO" if riemannian else "Euclidean RBF BO"
        self.color='#1D9E75' if riemannian else '#E24B4A'; self.riemannian=riemannian
        self.hist=dict(iter=[],quality=[],best_q=[]); self.show_live=show_live
        self.display=LiveUSDisplay() if show_live else None
    def run(self):
        robot=self.scene.robot; q_vert=probe_down(); print(f"\n{'='*60}\n  {self.name}\n{'='*60}")
        start_joints=compute_ik(robot,[SCAN_X,SCAN_Y0,HOVER_Z],q_vert)
        for _ in range(200): drive(robot,start_joints,force=60.,velocity=0.05)
        ys=np.linspace(SCAN_Y0,SCAN_Y1,N_WP)
        for k,wy in enumerate(ys):
            _,q_curr=ee_pose(robot); q_bo=self.bo.suggest(); q_bo=clamp_orientation(q_bo,MAX_TILT_DEG)
            omega_des=log_so3(qmul(qconj(q_curr),q_bo)); nrm=np.linalg.norm(omega_des)
            max_step=np.radians(5)
            if nrm>max_step: omega_des*=max_step/nrm
            q_new=qmul(q_curr,exp_so3(omega_des)); wp_pos=[SCAN_X,wy,HOVER_Z]
            joints=compute_ik(robot,wp_pos,q_new)
            for _ in range(50): drive(robot,joints,force=30.,velocity=0.03)
            pos_ee,q_ee=ee_pose(robot); tip_pos=probe_tip_pos(pos_ee,q_ee)
            obs=self.env.step(tip_pos,q_ee); qual=obs['quality']; self.bo.observe(q_bo,qual)
            self.hist['iter'].append(k+1); self.hist['quality'].append(qual); self.hist['best_q'].append(self.bo.best_hist[-1])
            if self.show_live and self.display: self.display.update(obs['image'],qual,obs['metrics'])
            print(f"    wp {k+1:2d}  Q(sono)={qual:.3f}  bestQ={self.bo.best_hist[-1]:.3f}")
        final_best=self.bo.best_hist[-1] if self.bo.best_hist else 0.; print(f"\n  ✓ Scan complete.  Final best quality = {final_best:.3f}"); return self

def make_plots(sessions):
    plt.rcParams.update({'figure.dpi':200,'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':0.3,'lines.linewidth':2.2})
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for sess in sessions: h=sess.hist; t=np.array(h['iter']); axes[0,0].plot(t,h['quality'],color=sess.color,label=sess.name); axes[1,0].plot(t,h['best_q'],color=sess.color,label=sess.name)
    axes[0,0].set(ylabel='Image Quality',ylim=(0,1.05),title='Ultrasound Image Quality per Waypoint')
    axes[1,0].set(ylabel='Best Quality Found',ylim=(0,1.05),xlabel='BO Iteration',title='BO Convergence (Image‑Content Metric)')
    axes[0,0].legend(fontsize=8); axes[1,0].legend(fontsize=8); axes[0,1].set_visible(False); axes[1,1].set_visible(False)
    plt.suptitle('SonoBot: 3D Anatomy‑Aware Bayesian Optimisation',fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig('sonobot_anatomy_bo.png',dpi=300); plt.close()
    print("  Saved → sonobot_anatomy_bo.png")

def run_experiment():
    N_TRIALS=10; euc,rie=[],[]
    for t in range(N_TRIALS):
        print(f"Trial {t+1}")
        sc=Scene(); sc.client=p.connect(p.DIRECT); sc.build(); se=ScanSession(sc,SonoSimEnv(target='liver',seed=t,verbose=False),riemannian=False,show_live=False); se.run(); euc.append(se.hist['best_q'][-1]); p.disconnect(sc.client)
        sc=Scene(); sc.client=p.connect(p.DIRECT); sc.build(); se=ScanSession(sc,SonoSimEnv(target='liver',seed=t+100,verbose=False),riemannian=True,show_live=False); se.run(); rie.append(se.hist['best_q'][-1]); p.disconnect(sc.client)
    euc,rie=np.array(euc),np.array(rie); _,p=wilcoxon(euc,rie); imp=(rie.mean()-euc.mean()); rel=imp/(euc.mean()+1e-6)*100
    print(f"Euclidean: {euc.mean():.4f}±{euc.std():.4f}  Riemannian: {rie.mean():.4f}±{rie.std():.4f}  p={p:.4f}  +{rel:.1f}%")
    plt.boxplot([euc,rie],labels=['Euclidean','Riemannian']); plt.ylabel('Final Best Quality'); plt.title('10 Trials'); plt.savefig('sonobot_experiment_boxplot.png',dpi=300); plt.close()

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--experiment',action='store_true'); args=parser.parse_args()
    if args.experiment: run_experiment(); return
    print("="*60+"\n  SONOBOT – ANATOMY‑AWARE SCAN + BO COMPARISON\n  Keys: 1=Euclidean  2=Riemannian  A=Auto both  Q=Quit & save\n"+"="*60)
    env=SonoSimEnv(target='liver'); scene=Scene().build(); results=[]
    while True:
        keys=p.getKeyboardEvents()
        for k,st in keys.items():
            if not (st&p.KEY_WAS_TRIGGERED): continue
            ch=chr(k) if 32<=k<=126 else ''
            if ch in 'qQ':
                if results: print("\n  Saving final plots …"); make_plots(results)
                scene.close(); cv2.destroyAllWindows(); return
            if ch=='1':
                sess=ScanSession(scene,env,riemannian=False).run(); results.append(sess)
                print("  Saving plot …"); make_plots(results); scene.reset_robot_to_home()
            if ch=='2':
                sess=ScanSession(scene,env,riemannian=True).run(); results.append(sess)
                print("  Saving plot …"); make_plots(results); scene.reset_robot_to_home()
            if ch in 'aA':
                for r in [False,True]: results.append(ScanSession(scene,env,riemannian=r).run()); scene.reset_robot_to_home() if not r else None
                print("  Saving final plots …"); make_plots(results); scene.close(); cv2.destroyAllWindows(); return
        p.stepSimulation(); time.sleep(DT)

if __name__=='__main__': main()