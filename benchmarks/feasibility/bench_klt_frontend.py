import cv2, numpy as np, time
rng = np.random.default_rng(0)

def make_scene(w=2000,h=1400):
    img = rng.integers(0,255,(h,w),dtype=np.uint8)
    img = cv2.GaussianBlur(img,(0,0),1.2)
    for _ in range(400):
        x,y = rng.integers(50,w-50), rng.integers(50,h-50)
        cv2.rectangle(img,(x,y),(x+int(rng.integers(10,60)),y+int(rng.integers(10,60))),int(rng.integers(0,255)),-1)
    return img

def warp(scene,i,W,H):
    cx,cy = scene.shape[1]/2+i*3.0, scene.shape[0]/2+i*0.8
    s=1.0+i*0.004
    M=np.array([[s,0,cx-(W/2)*s],[0,s,cy-(H/2)*s]],np.float32)
    return cv2.warpAffine(scene,M,(W,H),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)

def bench_klt(W,H,npts,N=100,threads=0):
    cv2.setNumThreads(threads)
    scene=make_scene(); frames=[warp(scene,i,W,H) for i in range(N)]
    lk=dict(winSize=(21,21),maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,0.01))
    t_track=t_redetect=0.0; prev=None; pts=None; redetects=0; tracked=[]
    for f in frames:
        if prev is None or pts is None or len(pts)<npts*0.5:
            t0=time.perf_counter()
            pts=cv2.goodFeaturesToTrack(f,maxCorners=npts,qualityLevel=0.01,
                                        minDistance=7,blockSize=7)
            t_redetect+=time.perf_counter()-t0; redetects+=1
        else:
            t0=time.perf_counter()
            nxt,st,_=cv2.calcOpticalFlowPyrLK(prev,f,pts,None,**lk)
            t_track+=time.perf_counter()-t0
            st=st.ravel().astype(bool)
            pts=nxt[st].reshape(-1,1,2); tracked.append(int(st.sum()))
        prev=f
    print(f"  {W}x{H} npts={npts:<5} threads={'auto' if threads==0 else threads}  "
          f"KLT track={t_track/max(N-redetects,1)*1000:5.2f}ms/frame  "
          f"redetect={t_redetect/max(redetects,1)*1000:5.2f}ms (x{redetects})  "
          f"avg tracked={int(np.mean(tracked)) if tracked else 0}")

print("KLT optical-flow front-end (alternative to per-frame descriptor matching):")
for W,H,n in [(640,360,800),(640,360,1200),(960,540,1200),(1280,720,1500)]:
    bench_klt(W,H,n,threads=1)
print()
print("Same, with OpenCV's own threading enabled (all cores):")
for W,H,n in [(640,360,800),(960,540,1200),(1280,720,1500)]:
    bench_klt(W,H,n,threads=0)
print()
# ORB descriptor cost at keyframes only
cv2.setNumThreads(1)
scene=make_scene()
for W,H,nf in [(640,360,1000),(960,540,1500)]:
    orb=cv2.ORB_create(nfeatures=nf); fr=[warp(scene,i,W,H) for i in range(30)]
    t0=time.perf_counter()
    for f in fr: orb.detectAndCompute(f,None)
    print(f"  ORB detect+describe {W}x{H} nfeat={nf}: {(time.perf_counter()-t0)/30*1000:.1f}ms "
          f"(paid only at keyframes)")
