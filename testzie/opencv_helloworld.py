import numpy as np
from matplotlib import pyplot as plt
import cv2


# show
img = cv2.imread('f:/index.jpg', 1)
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow('img', img)
k = cv2.waitKey(0)
if k == ord('s'):
    print(chr(k))
cv2.destroyAllWindows()


# draw
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param[0] = True
        param[1] = (x, y)
        return
    if event == cv2.EVENT_LBUTTONUP:
        param[0] = False
        param[1] = None
    if param[0] and param[1] and event == cv2.EVENT_MOUSEMOVE:
        cv2.line(img, param[1], (x, y), (255, 0, 0), 10)
        param[1] = (x, y)

img = np.zeros((512, 512, 3), np.uint8) + 255
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle, [False, None])

while True:
    cv2.imshow('image', img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()


# basic
img1 = cv2.imread('f:/index.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('f:/123.jpg', cv2.IMREAD_COLOR)


# threshold
def nothing(x):
    pass
gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
cv2.namedWindow('img')
cv2.createTrackbar('num', 'img', 0, 255, nothing)
while True:
    cv2.imshow('img',img)
    if cv2.waitKey(1) == ord('q'):
        break
    num = cv2.getTrackbarPos('num', 'img')
    ret, img = cv2.threshold(gray, num, 255, cv2.THRESH_BINARY)
cv2.destroyAllWindows()


# bit calculation
rows, cols, _ = img2.shape
roi = img1[0:rows, 0:cols]
ret, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
img2_bg = cv2.bitwise_and(img2, img2, mask=mask)
dst = cv2.add(img1_bg, img2_bg)
cv2.imshow('dst', dst)

img1[0:rows, 0:cols] = dst
cv2.imshow('img', img1)


# inRange
cv2.imshow('img', img1)
hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, np.array([0, 50, 0]), np.array([255, 255, 255]))
cv2.imshow('mask', mask)
res = cv2.bitwise_and(hsv, hsv, mask=mask)
res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
cv2.imshow('res', res)


# resize
res = cv2.resize(img1, None, fx=2, fy=2)
cv2.imshow('img', res)


# Perspective Transform
def mouse_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        cv2.circle(img1, (x, y), 10, (0, 255, 0), 2)
        param.append([y, x])
img1 = cv2.imread('f:/index.jpg', cv2.IMREAD_COLOR)
l = []
cv2.namedWindow('img')
cv2.setMouseCallback('img', mouse_event, l)
while True:
    cv2.imshow('img', img1)
    if cv2.waitKey(1) == ord('q'):
        break
    if len(l) == 4:
        w1 = l[2][1] - l[0][1]
        w2 = l[3][1] - l[1][1]
        h1 = l[1][0] - l[0][0]
        h2 = l[3][0] - l[2][0]
        w = int((w1 + w2) / 2)
        h = int((h1 + h2) / 2)
        p1 = np.float32(l)
        p2 = np.float32([[0, 0], [h, 0], [0, w], [h, w]])
        M = cv2.getPerspectiveTransform(p1, p2)
        dst = cv2.warpPerspective(img1, M, (w, h))
        cv2.imshow('dst', dst)
cv2.destroyAllWindows()


# pyramid
img = cv2.imread('f:/index.jpg', cv2.IMREAD_COLOR)
img = cv2.resize(img, (688, 400))
gp = [img]
for i in range(4):
    img = cv2.pyrDown(img)
    gp.append(img)
for i, v in enumerate(gp):
    cv2.imshow(str(i), v)
lp = [gp[4]]
for i in range(4, 0, -1):
    pu = cv2.pyrUp(gp[i])
    sub = cv2.subtract(gp[i-1], pu)
    lp.append(sub)
for i, v in enumerate(lp):
    cv2.imshow(str(i), v)
ls = lp[0]
for i, v in enumerate(lp):
    if i == 0:
        continue
    ls = cv2.pyrUp(ls)
    ls = cv2.add(ls, v)


# contour
img = cv2.imread('f:/index.jpg', 0)
shape = img.shape
img = cv2.bitwise_not(img)
ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
th = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
ret, th = cv2.threshold(th, 0, 255, cv2.THRESH_OTSU)
cv2.imshow('th', th)
image, contours, hierarchy = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
img = cv2.imread('f:/index.jpg')
img = cv2.drawContours(img, contours, 7, (0, 0, 255), -1)
cv2.imshow('img', img)


# moment
cnt = contours[7]
m = cv2.moments(cnt)
cx = int(m['m10']/m['m00'])
cy = int(m['m01']/m['m00'])
cv2.circle(img, (cx, cy), 2, (0, 0, 255), 2)
cv2.imshow('img', img)
cv2.contourArea(cnt)
cv2.arcLength(cnt, True)

epsilon = 0.01 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)
img = cv2.polylines(img, approx.swapaxes(0, 1), True, (0, 255, 0), 2)
cv2.imshow('img', img)

hull = cv2.convexHull(cnt)
img = cv2.polylines(img, hull.swapaxes(0, 1), True, (0, 255, 0), 2)
cv2.imshow('img', img)

mask = np.zeros(shape, np.uint8)
cv2.drawContours(mask, [cnt], 0, 255, -1)
cv2.minMaxLoc(img, mask)

cv2.mean(img, mask)

cv2.circle(img, tuple(cnt[cnt[:, :, 0].argmin()][0]), 2, (0, 0, 255), 2)
cv2.circle(img, tuple(cnt[cnt[:, :, 0].argmax()][0]), 2, (0, 0, 255), 2)
cv2.circle(img, tuple(cnt[cnt[:, :, 1].argmin()][0]), 2, (0, 0, 255), 2)
cv2.circle(img, tuple(cnt[cnt[:, :, 1].argmax()][0]), 2, (0, 0, 255), 2)
cv2.imshow('img', img)


# region
hull = cv2.convexHull(cnt, returnPoints=False)
defects = cv2.convexityDefects(cnt, hull)
for i in range(defects.shape[0]):
    s, e, f, d = defects[i, 0]
    # cv2.circle(img, tuple(cnt[s][0]), 2, (0, 0, 255), 2)
    # cv2.circle(img, tuple(cnt[e][0]), 2, (0, 255, 0), 2)
    cv2.line(img, tuple(cnt[s][0]), tuple(cnt[e][0]), (0, 0, 255), 1)
    cv2.circle(img, tuple(cnt[f][0]), 2, (0, 255, 0), 2)
cv2.imshow('img', img)

dist = cv2.pointPolygonTest(cnt, (50, 50), True)
inner = cv2.pointPolygonTest(cnt, (343, 218), False)

img = cv2.imread('f:/index.jpg')
img = cv2.drawContours(img, contours, 7, (0, 0, 255), 2)
img = cv2.drawContours(img, contours, 9, (0, 0, 255), 2)
cv2.imshow('img', img)
cnt1 = contours[7]
cnt2 = contours[9]
cv2.matchShapes(cnt1, cnt2, cv2.CONTOURS_MATCH_I1, 0.0)


# retrieval
image, contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# histogram
img = cv2.imread('f:/a.jpg', 0)
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
cdf = hist.cumsum()
cdf_norm = cdf * hist.max() / cdf.max()
plt.hist(img.ravel(), 256, [0, 256])
plt.plot(cdf_norm, 'r')

cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
cdf = np.ma.filled(cdf_m, 0).astype(np.uint8)
img2 = cdf[img]
cv2.imshow('img', img)
cv2.imshow('img2', img2)

equ = cv2.equalizeHist(img)
cv2.imshow('equ', equ)

clahe = cv2.createCLAHE(2, (8, 8))
cl = clahe.apply(img)
cv2.imshow('cl', cl)


# histogram 2D
img = cv2.imread('f:/a.jpg')
cv2.imshow('img', img)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
plt.imshow(hist)


# back project
img = cv2.imread('f:/se.jpg')
p1 = 100
p2 = 200
roi = img[p1:p2, p1:p2].copy()
# cv2.imshow('roi', roi)
imgc = img.copy()
imgc = cv2.rectangle(imgc, (p1, p1), (p2, p2), (255, 255, 255), 2)
cv2.imshow('imgc', imgc)

hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
hsvt = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
M = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
I = cv2.calcHist([hsvt], [0, 1], None, [180, 256], [0, 180, 0, 256])

Im = np.ma.masked_equal(I, 0.0)
Rm = M/Im
R = np.ma.filled(Rm, 1.0)
h, s, v = cv2.split(hsvt)
B = R[h.ravel(), s.ravel()]
B = np.minimum(B, 1)
B = B.reshape(hsvt.shape[:2])
cv2.imshow('B', B)
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dst = cv2.filter2D(B, -1, disc)
dst = np.uint8(dst)
cv2.normalize(dst, dst, 0, 255, cv2.NORM_MINMAX)
cv2.imshow('dst', dst)
ret, th = cv2.threshold(dst, 5, 255, cv2.THRESH_BINARY)
cv2.imshow('th', th)
imgt = cv2.bitwise_and(img, img, mask=th)
imgt = cv2.rectangle(imgt, (p1, p1), (p2, p2), (255, 255, 255), 2)
cv2.imshow('imgt', imgt)

B = cv2.calcBackProject([hsvt], [0, 1], M, [0, 180, 0, 256], 1)


# FFT
img = cv2.imread('f:/a.jpg', 0)
f = np.fft.fft2(img)
fs = np.fft.fftshift(f)
mag = np.log(np.abs(fs))
mag = cv2.normalize(mag, mag, 0, 255, cv2.NORM_MINMAX)
mag = mag.astype(int)
plt.imshow(mag, 'gray')

laplacian = np.array([[0, 1, 0],[1, -4, 1], [0, 1, 0]])
f = np.fft.fft2(laplacian)
fs = np.fft.fftshift(f)
mag = np.log(np.abs(fs)+1)
plt.imshow(mag, 'gray')


# match
img = cv2.imread('f:/1.jpg', 0)
ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
th = cv2.bitwise_not(th)
image, contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
x, y, w, h = cv2.boundingRect(contours[11])
img = cv2.imread('f:/1.jpg')
cv2.rectangle(img, (x-3, y-3), (x+w+3, y+h+3), (0, 255, 0), 2)
cv2.imshow('img', img)

img = cv2.imread('f:/1.jpg', 0)
roi = img[y-3:y+h+4,x-3:x+w+4].copy()
hs, ws = roi.shape

res = cv2.matchTemplate(img, roi, cv2.TM_CCOEFF_NORMED)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
img = cv2.imread('f:/1.jpg')
cv2.rectangle(img, max_loc, (max_loc[0]+ws, max_loc[1]+hs), (255, 0, 255), 2)
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow('img', img)

loc = np.where(res >= 0.95)
img = cv2.imread('f:/1.jpg')
pts = []
for i, pt in enumerate(zip(*loc[::-1])):
    cv2.rectangle(img, pt, (pt[0]+ws, pt[1]+hs), (255, 0, 255), 2)
    cv2.putText(img, str(i+1), pt, cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 255), 2)
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow('img', img)


# hough
img = cv2.imread('f:/pat1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edge = cv2.Canny(gray, 50, 150, apertureSize=3)
cv2.imshow('edge',edge)
lines = cv2.HoughLines(edge, 1, np.pi/360, 100)
lines = lines.reshape(-1, 2)
for r, theta in lines:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = r*a
    y0 = r*b
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
cv2.imshow('img', img)

img = cv2.imread('f:/pat1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edge = cv2.Canny(gray, 50, 150, apertureSize=3)
lines = cv2.HoughLinesP(edge, 1, np.pi/360, 50, minLineLength=50, maxLineGap=10)
lines = lines.reshape(-1, 4)
for x1, y1, x2, y2 in lines:
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
cv2.imshow('img', img)


# watershed
img = cv2.imread('f:/bi.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
cv2.imshow('th', th)
mp = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
cv2.imshow('mp', mp)
bg = cv2.dilate(mp, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=3)
cv2.imshow('bg', bg)
dist = cv2.distanceTransform(mp, cv2.DIST_L1, 5)
cv2.imshow('dist', dist)
ret, fg = cv2.threshold(dist, 0.7*dist.max(), 255, cv2.THRESH_BINARY)
fg = np.uint8(fg)
cv2.imshow('fg', fg)
unknown = cv2.subtract(bg, fg)
cv2.imshow('unknown', unknown)
ret, markers = cv2.connectedComponents(fg)
# markers = (markers - markers.min())*255/(markers.max() - markers.min())
# markers = np.uint8(markers)
# cv2.imshow('markers', markers)
markers = markers + 1
markers[unknown == 255] = 0
markers3 = cv2.watershed(img, markers)
img[markers3 == -1] = [255, 0, 0]
cv2.imshow('img', img)


# grabcut
def grabcut(event, x, y, flags, param):
    global imga, point, flag, mask
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(imga, (x, y), 5, (255, 0, 0, 64), -1)
        cv2.circle(mask, (x, y), 5, cv2.GC_PR_FGD, -1)
        point = (x, y)
    if event == cv2.EVENT_LBUTTONUP:
        point = None
        flag = True
    if point is not None and event == cv2.EVENT_MOUSEMOVE:
        cv2.line(imga, point, (x, y), (255, 0, 0, 64), 10)
        cv2.line(mask, point, (x, y), cv2.GC_PR_FGD, 10)
        point = (x, y)

img = cv2.imread('f:/photo.jpg')
imga = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('img', img)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
mask = np.zeros(img.shape[:2], np.uint8) + cv2.GC_PR_BGD

point = None
flag = True

cv2.namedWindow('image')
cv2.setMouseCallback('image', grabcut, [False, None])

while True:
    cv2.imshow('image', imga)
    cv2.imshow('mask', cv2.multiply(mask2, 255))
    if flag:
        cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)
        mask2_not = cv2.subtract(1, mask2)
        img_m = cv2.bitwise_and(img, img, mask=mask2)
        img_m = cv2.cvtColor(img_m, cv2.COLOR_BGR2BGRA)
        img_n = cv2.bitwise_and(gray, gray, mask=mask2_not)
        img_n = cv2.merge([img_n, img_n, img_n, cv2.add(np.zeros(gray.shape, np.uint8), 255)])
        imga = cv2.add(img_m, img_n)
        flag = False
    if cv2.waitKey(20) == ord('q'):
        break
cv2.destroyAllWindows()


# harris
img = cv2.imread('f:/1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.05)
dst = cv2.dilate(dst, None)
img[dst > 0.01*dst.max()] = [0, 0, 255]
cv2.imshow('img', img)

dst = np.uint8(dst)
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)
res = np.hstack((centroids, corners))
res = np.int0(res)
img[res[:, 1], res[:, 0]] = [0, 0, 255]
img[res[:, 3], res[:, 2]] = [0, 255, 0]
for x1, y1, x2, y2 in res:
    cv2.circle(img, (x1, y1), 2, (0, 0, 255), -1)
    cv2.circle(img, (x2, y2), 2, (0, 255, 0), -1)
cv2.imshow('img', img)


# shi-tomasi
img = cv2.imread('f:/1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
corners = np.int0(corners)
for i in corners:
    x, y = i.ravel()
    cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
cv2.imshow('img', img)


# SIFT
img = cv2.imread('f:/index.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# SURF


# FAST
img = cv2.imread('f:/1.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
fast = cv2.FastFeatureDetector_create()
kp = fast.detect(img, None)
cv2.drawKeypoints(img, kp, img, color=(0, 0, 255))
cv2.imshow('img', img)

fast.setNonmaxSuppression(False)
kp = fast.detect(img, None)
cv2.drawKeypoints(img, kp, img, color=(0, 0, 255))
cv2.imshow('img', img)


# BRIEF & CenSurE(Star)
img = cv2.imread('f:/a.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# ORB
img = cv2.imread('f:/a.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create()
kp = orb.detect(gray, None)
cv2.drawKeypoints(img, kp, img, color=(0, 0, 255))
cv2.imshow('img', img)


# match
img1 = cv2.imread('f:/m0.jpg', 0)
img2 = cv2.imread('f:/m2.jpg', 0)
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=0)
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.imshow('img', img3)


# meanshift


# camshift


# optical flow
