import time

import cv2
import numpy as np

cv2.setNumThreads(0)  # measure single-core, worst case per worker

rng = np.random.default_rng(0)

def make_scene(w=2000, h=1400):
    # richly textured synthetic scene so ORB finds plenty of corners
    img = rng.integers(0, 255, (h, w), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (0,0), 1.2)
    for _ in range(400):
        x, y = rng.integers(50, w-50), rng.integers(50, h-50)
        cv2.rectangle(img, (x,y), (x+int(rng.integers(10,60)), y+int(rng.integers(10,60))),
                      int(rng.integers(0,255)), -1)
    return img

def warp(scene, i, W, H):
    # simulate forward+lateral camera motion via homography crop
    cx, cy = scene.shape[1]/2 + i*3.0, scene.shape[0]/2 + i*0.8
    s = 1.0 + i*0.004
    M = np.array([[s,0,cx-(W/2)*s],[0,s,cy-(H/2)*s]], np.float32)
    return cv2.warpAffine(scene, M, (W,H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)

def bench(W, H, nfeat, N=60):
    scene = make_scene()
    orb = cv2.ORB_create(nfeatures=nfeat, scaleFactor=1.2, nlevels=8, fastThreshold=12)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING)
    K = np.array([[max(W,H)*0.9,0,W/2],[0,max(W,H)*0.9,H/2],[0,0,1]], np.float64)

    frames = [warp(scene, i, W, H) for i in range(N)]
    t_det=t_mat=t_pose=t_tri=0.0; kp_tot=0; inl_tot=0; pairs=0
    prev=None
    for f in frames:
        t0=time.perf_counter(); kp,des = orb.detectAndCompute(f,None); t1=time.perf_counter()
        t_det += t1-t0; kp_tot += len(kp)
        if prev is not None and des is not None and prev[1] is not None:
            t0=time.perf_counter()
            m = bf.knnMatch(prev[1], des, k=2)
            good=[a for a,b in (p for p in m if len(p)==2) if a.distance < 0.75*b.distance]
            t1=time.perf_counter(); t_mat += t1-t0
            if len(good) >= 12:
                p0=np.float32([prev[0][g.queryIdx].pt for g in good])
                p1=np.float32([kp[g.trainIdx].pt for g in good])
                t0=time.perf_counter()
                E,mask=cv2.findEssentialMat(p0,p1,K,cv2.RANSAC,0.999,1.0)
                if E is not None and E.shape==(3,3):
                    _,R,t,mask2=cv2.recoverPose(E,p0,p1,K,mask=mask)
                    t1=time.perf_counter(); t_pose += t1-t0
                    inl=int(mask2.sum()) if mask2 is not None else 0
                    inl_tot+=inl; pairs+=1
                    # triangulate the inliers
                    t0=time.perf_counter()
                    P0=K@np.hstack([np.eye(3),np.zeros((3,1))])
                    P1=K@np.hstack([R,t])
                    sel=(mask2.ravel()>0) if mask2 is not None else np.ones(len(p0),bool)
                    if sel.sum()>0:
                        cv2.triangulatePoints(P0,P1,p0[sel].T,p1[sel].T)
                    t_tri += time.perf_counter()-t0
                else:
                    t_pose += time.perf_counter()-t0
        prev=(kp,des)

    per = lambda x: x/N*1000
    tot = per(t_det)+per(t_mat)+per(t_pose)+per(t_tri)
    print(f"  {W}x{H} nfeat={nfeat:<5} kp/frame={kp_tot//N:<5} "
          f"detect={per(t_det):5.1f}ms match={per(t_mat):5.1f}ms "
          f"pose={per(t_pose):5.1f}ms tri={per(t_tri):4.1f}ms "
          f"=> {tot:5.1f}ms/frame ({1000/tot:5.1f} fps 1-core) inliers~{inl_tot//max(pairs,1)}")

print("Single-core per-frame cost of the ORB front-end (cv2 threads disabled):")
for W,H,nf in [(640,360,1000),(640,360,1500),(960,540,1500),(960,540,2000),(1280,720,2000)]:
    bench(W,H,nf)
