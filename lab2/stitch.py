import sys
import cv2
import numpy as np
import imutils

def stitched_image(img1, img2, M):
   
    w1,h1 = img1.shape[:2]
    w2,h2 = img2.shape[:2]

    img1_dims = np.float32([ [0,0], [0,w1], [h1, w1], [h1,0] ]).reshape(-1,1,2)
    img2_dims_temp = np.float32([ [0,0], [0,w2], [h2, w2], [h2,0] ]).reshape(-1,1,2)

    img2_dims = cv2.perspectiveTransform(img2_dims_temp, M)

    result_dims = np.concatenate( (img1_dims, img2_dims), axis = 0)

    [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

    transform_dist = [-x_min,-y_min]
    transform_array = np.array([[1, 0, transform_dist[0]], 
                                [0, 1, transform_dist[1]], 
                                [0,0,1]]) 

    result = cv2.warpPerspective(img2, transform_array.dot(M), 
                                    (x_max-x_min, y_max-y_min))
    result[transform_dist[1]:w1+transform_dist[1], 
                transform_dist[0]:h1+transform_dist[0]] = img1

    return result

def get_sift_homography(img1, img2):

    sift = cv2.xfeatures2d.SIFT_create()

    k1, d1 = sift.detectAndCompute(img1, None)
    k2, d2 = sift.detectAndCompute(img2, None)

    matches = cv2.BFMatcher().knnMatch(d1,d2, k=2)

    verified_matches = []
    for m1,m2 in matches:
        if m1.distance < 1. * m2.distance:
            verified_matches.append(m1)

    min_matches = 10
    if len(verified_matches) > min_matches:

        img1_pts, img2_pts = [], []

        for match in verified_matches:
            img1_pts.append(k1[match.queryIdx].pt)
            img2_pts.append(k2[match.trainIdx].pt)
        img1_pts = np.float32(img1_pts).reshape(-1,1,2)
        img2_pts = np.float32(img2_pts).reshape(-1,1,2)

        M, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        draw_params = dict(matchColor = (255,255,33), singlePointColor = None, matchesMask = matchesMask, flags = 2)

        img3 = cv2.drawMatches(img1,k1,img2,k2,verified_matches,None,**draw_params)
        cv2.imwrite('match.png', img3)

        return M
    else:
        print ('Not Enough Matches Between Pictures')
        exit()

def equalize_histogram_color(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img

img1 = cv2.imread(sys.argv[1])
img2 = cv2.imread(sys.argv[2])

img1 = imutils.resize(img1, width=400)
img2 = imutils.resize(img2, width=400)

img1 = equalize_histogram_color(img1)
img2 = equalize_histogram_color(img2)

M =  get_sift_homography(img1, img2)

result = stitched_image(img2, img1, M)
    
argc = len(sys.argv)
for i in range(3,argc):

    img1 = result
    img2 = cv2.imread(sys.argv[i])

    img1 = imutils.resize(img1, width=400)
    img2 = imutils.resize(img2, width=400)

    img1 = equalize_histogram_color(img1)
    img2 = equalize_histogram_color(img2)

    M =  get_sift_homography(img1, img2)

    result = stitched_image(img2, img1, M)

cv2.imwrite('result.png', result)
