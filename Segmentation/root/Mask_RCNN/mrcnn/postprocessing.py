import colorspacious
from root.Mask_RCNN.glasbey import Glasbey
from skimage.color import label2rgb
from skimage.measure import label, regionprops
import math
import pytz
import datetime
from pytz import timezone
import numpy as np
import cv2

utc = pytz.utc
utc_dt = datetime.datetime.now()
eastern = timezone('US/Eastern')
loc_dt = utc_dt.astimezone(eastern)

color = np.array(([1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],[1,0.5,0],[0.5,1,0],[0,1,0.5],[0,0.5,1],[1,0,0.5],[0.5,0,1],[1,0.5,0.25],[0.25,0.5,1],[1,0.25,0.5],[0.5,0.25,1],[0.5,1,0.25],[0.25,1,0.5]),np.float32)
gb = Glasbey(base_palette=color, chroma_range = (60,100), no_black=True)
c4 = gb.generate_palette(size=18)
color4 = c4[1:]

def normalized(rgb):
    norm=np.zeros((512,512,3),np.float32)
    norm_rgb=np.zeros((512,512,3),np.uint8)

    b=rgb[:,:,0]
    g=rgb[:,:,1]
    r=rgb[:,:,2]

    sum=b+g+r

    norm[:,:,0]=b/sum*255.0
    norm[:,:,1]=g/sum*255.0
    norm[:,:,2]=r/sum*255.0

    norm_rgb=cv2.convertScaleAbs(norm)
    return norm_rgb

def overlay(mask, orig, clr=color4):
    maskPR = label(mask)
    labels = label2rgb(label=maskPR, bg_label=0, bg_color=(0, 0, 0), colors=clr)
    L2 = normalized(labels)
    if len(orig.shape) < 3: 
        O2 = cv2.cvtColor(orig.astype('uint8'), cv2.COLOR_GRAY2BGR)
    else:
        O2 = orig
    comb = cv2.addWeighted(L2.astype('float64'),0.5,O2.astype('float64'),0.5,0)
    return comb

PROC = True #@param {type:"boolean"}
def remove_overlay(r):
    masks = r['masks'].astype(np.uint8)
    mask = np.zeros([masks.shape[0], masks.shape[1]], dtype='uint8')
    maskD = np.zeros([masks.shape[0], masks.shape[1]], dtype='uint8')
    diff = np.zeros([masks.shape[0], masks.shape[1]], dtype='uint8')
    props = np.zeros((masks.shape[2]))
    for n in range(0,masks.shape[2]):
        M2 = label(masks[:,:,n])
        props2 = regionprops(M2)
        for m in range(0,M2.max()):
            if props2[m].area < 100:
                M2[M2==props2[m].label] = 0
        M2[M2 > 0] = 1
        masks[:,:,n] = M2*masks[:,:,n]
        props2 = regionprops(masks[:,:,n])

        maskD = maskD + masks[:,:,n]

        if maskD.max() <= 1:
            mask = mask + (n+1)*masks[:,:,n]
        else:
            try:
                diff[maskD > 1] = 1
                diff2 = diff.copy()
                pd = regionprops(diff)

                area2 = props2[0].area 
                aread = pd[0].area
                Vals = diff*mask # Find value of existing region label, under new overlap
                vals = Vals[Vals>0] # Not zero
                vals = vals[vals != n+1] # Not the current label
                vals = list(set(vals)) # Really should only be one left
                props1 = regionprops(masks[:,:,vals[0]])
                area1 = props1[0].area
                div1 = aread/area1
                div2 = aread/area2
                dd = vals[0] + n+1

                mask = mask + (n+1)*masks[:,:,n]
                if div1 < 0.15 and div2 < 0.15:
                    mask[diff > 0] = vals[0]
                elif div1 < 0.15 and div2 > 0.15:
                    mask[diff > 0] = n+1
                    mask[mask==vals[0]] = n+1
                elif div1 > 0.15 and div2 < 0.15:
                    mask[diff > 0] = vals[0]
                    mask[mask==n+1] = vals[0]
                elif div1 > 0.15 and div2 > 0.15 and div1 < 0.6 and div2 < 0.6:

                    y0, x0 = pd[0].centroid
                    orientation = pd[0].orientation

                    x1 = x0 - math.sin(orientation) * 0.55 * pd[0].major_axis_length
                    y1 = y0 - math.cos(orientation) * 0.55 * pd[0].major_axis_length
                    x2 = x0 + math.sin(orientation) * 0.55 * pd[0].major_axis_length
                    y2 = y0 + math.cos(orientation) * 0.55 * pd[0].major_axis_length 

                    cv2.line(diff, (int(x2),int(y2)), (int(x0),int(y0)), (0, 0, 0), thickness=2)
                    cv2.line(diff, (int(x1),int(y1)), (int(x0),int(y0)), (0, 0, 0), thickness=2)

                    lbl1 = label(diff)
                    lbl1 = lbl1.astype('uint8')
                    cv2.line(lbl1, (int(x2),int(y2)), (int(x0),int(y0)), (1, 1, 1), thickness=2)
                    cv2.line(lbl1, (int(x1),int(y1)), (int(x0),int(y0)), (1, 1, 1), thickness=2)
                    lbl2 = lbl1*diff2
                    mask[lbl2 == 2] = n+1
                    mask[lbl2 == 1] = vals[0]

                elif div1 > 0.6 or div2 > 0.6:
                    if area1 > area2:
                        mask[diff > 0] = vals[0]
                        mask[mask==n+1] = vals[0]
                    elif area2 > area1:
                        mask[diff > 0] = n+1
                        mask[mask==vals[0]] = n+1
            except Exception:
                continue
        maskD[maskD > 1] = 1
        diff = np.zeros([masks.shape[0], masks.shape[1]], dtype='uint8')
    return mask