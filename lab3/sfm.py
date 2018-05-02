#!/usr/bin/env python2.7
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2 as cv
import matplotlib.pyplot as plt
from numpy import linalg as la 
import numpy as np 
from mpl_toolkits.mplot3d import Axes3D
import argparse

parser = argparse.ArgumentParser(description="CV_HW3")
parser.add_argument('-mode', type=int, default=2)
parser.add_argument('-ratio', type=float, default=.75)
parser.add_argument('-threshold', type=float, default=.5)
args = parser.parse_args()

def normalize(in_pts):
    centroid = np.array([in_pts[:, 0].mean(), in_pts[:, 1].mean(), 0])
    pts = in_pts - centroid

    meandist = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2).mean()
    scale = np.sqrt(2) / (meandist)

    matrix = np.array([[scale, 0, -scale * centroid[0]],
                  [0, scale, -scale * centroid[1]],
                  [0, 0, 1]])
    newpts = matrix.dot(in_pts.T).T
    return newpts, matrix

def fundmat(in_points_in_img1, in_points_in_img2):
    points_in_img1, T1 = normalize(in_points_in_img1)
    points_in_img2, T2 = normalize(in_points_in_img2)

    s, _ = points_in_img1.shape

    A = np.zeros((s, 9))
    for i in range(s):
        x1, y1 = points_in_img1[i][0], points_in_img1[i][1]
        x2, y2 = points_in_img2[i][0], points_in_img2[i][1]
        A[i] = [x1 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1]

    u, d, v = la.svd(A)
    F = v[-1].reshape(3, 3)

    u, d, v = la.svd(F)
    d[2] = 0
    F = u.dot(np.diag(d)).dot(v)
    F = T2.transpose().dot(F).dot(T1)

    return F / F[2, 2]

def calculateSampsonDistance(matches_kp1, matches_kp2, F):
    F1 = F.dot(matches_kp1.T)
    F2 = F.T.dot(matches_kp2.T)
    tmp = (F1[0]**2 + F1[1]**2 + F2[0]**2 + F2[1]**2).reshape(-1, 1)
    err = np.diag(matches_kp2.dot(F.dot(matches_kp1.T)))**2
    return err.reshape(-1, 1) / tmp

def findFundMatRansac(matches_kp1, matches_kp2, threshold):
    inlierThreshold=80
    data_points=10
    Iter=1000
    cnt_matches, _ = matches_kp1.shape
    best_fit = []
    best_error = np.Infinity
    best_kp1, best_kp2 = [], []
    best_total = 0

    for iter in xrange(Iter):
        tmp = np.arange(cnt_matches)
        np.random.shuffle(tmp)
        maybe_idxs, test_idxs = tmp[data_points:], tmp[:data_points]

        data_p1, data_p2 = matches_kp1.take(maybe_idxs, axis=0), matches_kp2.take(maybe_idxs, axis=0)
        F = fundmat(data_p1, data_p2)

        test_p1, test_p2 = matches_kp1.take(test_idxs, axis=0), matches_kp2.take(test_idxs, axis=0)
        errs = calculateSampsonDistance(test_p1, test_p2, F)

        inlier_indices = [errs[:, 0] < threshold]

        current_p1, current_p2 = np.append(data_p1, test_p1[inlier_indices], axis=0), np.append(data_p2, test_p2[inlier_indices], axis=0)
        current_total, _ = current_p1.shape

        if current_total > best_total and current_total >= inlierThreshold:
            better_fit = fundmat(current_p1, current_p2)
            better_err = calculateSampsonDistance(current_p1, current_p2, F)

            if (best_error > better_err.mean()):
                best_fit, best_kp1, best_kp2 = better_fit, current_p1, current_p2
                best_total, _ = current_p1.shape

    return best_fit, best_kp1, best_kp2

def triangulatePoint(point_1, point_2, p1, p2):
    # define A
    u1, v1 = point_1[0], point_1[1]
    u2, v2 = point_2[0], point_2[1]

    A = np.array([u1 * p1[2] - p1[0], v1 * p1[2] - p1[1], u2 * p2[2] - p2[0], v2 * p2[2] - p2[1]])

    _, _, v = la.svd(A) #u, d, v
    x = v[-1]
    x = x / x[-1]
    return x


def triangulate(pts1, pts2, p1, p2):
    R1, t1 = p1[:, :3], p1[:, 3]
    R2, t2 = p2[:, :3], p2[:, 3]

    C1 = -R1.T.dot(t1)
    C2 = -R2.T.dot(t2)

    V1, V2 = R1.T.dot(np.array([0, 0, 1])), R2.T.dot(np.array([0, 0, 1]))

    points = []
    for pt1, pt2 in zip(pts1, pts2):
        point_in_3d = triangulatePoint(pt1, pt2, p1, p2)[:3]
        test1 = (point_in_3d - C1).dot(V1)
        test2 = (point_in_3d - C2).dot(V2)
        if (test1 and test2):
            points.append(point_in_3d)

    return np.array(points)

def show_image(img, title=None):
    plt.figure()
    plt.imshow(img)
    if title != None:
        plt.title(title)
    plt.xticks([]), plt.yticks([])

    plt.show()

img1 = cv.imread("data/Mesona1.JPG",0)
img2 = cv.imread("data/Mesona2.JPG",0)
K1 = np.array([[1.4219, 0.0005, 0.5092],[0, 1.4219, 0.3802],[0,0,0.0010]])
K2 = np.array([[1.4219, 0.0005, 0.5092],[0, 1.4219, 0.3802],[0,0,0.0010]])

if args.mode == 2:
    img1 = cv.imread("data/Statue1.bmp", 0)
    img2 = cv.imread("data/Statue2.bmp", 0)
    K1 = np.array([[5426.566895, 0.678017, 330.096680], [0.000000, 5423.133301, 648.950012], [0, 0, 1]])
    K2 = np.array([[5426.566895, 0.678017, 387.430023], [0.000000, 5423.133301, 620.616699], [0, 0, 1]])


"""
    
    Find The Correspondences
    
"""
sift = cv.xfeatures2d.SIFT_create()
sift_kp1, des1 = sift.detectAndCompute(img1,None)
sift_kp2, des2 = sift.detectAndCompute(img2,None)
bf = cv.BFMatcher(cv.NORM_L2)

matches = bf.knnMatch(des1, des2, k=2)

kp1_tmp, kp2_tmp, matches_tmp = [], [], []
a = 0
for  (m, n) in matches:
    if m.distance < args.ratio * n.distance:
        kp1_tmp.append(sift_kp1[m.queryIdx])
        kp2_tmp.append(sift_kp2[m.trainIdx])
        matches_tmp.append([cv.DMatch(a, a, m.distance)])
        a += 1
matches, kp1, kp2 = matches_tmp, kp1_tmp, kp2_tmp

kp1_tmp, kp2_tmp = [], []
for dt in matches:
    kp1_tmp.append(kp1[dt[0].queryIdx].pt)
    kp2_tmp.append(kp2[dt[0].trainIdx].pt)
kp1 = np.array([[j for j in i] + [1] for i in kp1_tmp])
kp2 = np.array([[j for j in i] + [1] for i in kp2_tmp])

F, best_kp1, best_kp2 = findFundMatRansac(kp1, kp2, args.threshold)

E = K2.T.dot(F.dot(K1))

u, d, v = la.svd(E)
m = (d[0] + d[1]) / 2
E = u.dot(np.diag([m, m, 0])).dot(v)

u, d, v = la.svd(E)
w = np.array([[0, -1, -0], [1, 0, 0], [0, 0, 1]])

if la.det(v) < 0:
    v *= -1
if la.det(u) < 0:
    u *= -1

u3 = u[:, -1]
R1 = u.dot(w.dot(v))
R2 = u.dot(w.T.dot(v))
result = [np.vstack((R1.T, u3)).T, np.vstack((R1.T, -u3)).T, np.vstack((R2.T, u3)).T, np.vstack((R2.T, -u3)).T]

p1 = K1.dot(np.vstack((np.eye(3), np.zeros(3))).T)
best_p2_inliers = -np.Infinity
for i in result:
    p2 = K2.dot(i)
    temp = triangulate(best_kp1, best_kp2, p1, p2)
    if (best_p2_inliers < temp.shape[0]):
        best_p2_inliers, _ = temp.shape
        points = temp
plt.rcParams['figure.figsize'] = [8,8]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:,0], points[:,1], points[:,2], c='r', marker='o')
plt.show()
