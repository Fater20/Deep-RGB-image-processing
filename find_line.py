# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

import math
# Usual color HSV value
color_dist = {
    'red1': {'lower': np.array([0, 43, 46]), 'upper': np.array([10, 255, 255])},
    'red2': {'lower': np.array([156, 43, 46]), 'upper': np.array([180, 255, 255])},
    'blue': {'lower': np.array([100, 43, 46]), 'upper': np.array([124, 255, 255])},
    'green': {'lower': np.array([35, 43, 46]), 'upper': np.array([77, 255, 255])},
    'yellow': {'lower': np.array([26, 43, 46]), 'upper': np.array([34, 255, 255])},
    'black': {'lower': np.array([0, 0, 0]), 'upper': np.array([180, 255, 46])},
    }

# Empty function
def empty(a):
    pass

# Image stack function
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

#if device_product_line == 'L500':
#    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
#else:
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Streaming loop
try:
    while True: 
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image
        
        # Get frames
        color_frame = frames.get_color_frame()

        # Validate that both frames are valid
        if  not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        imgResult = color_image.copy()

        # Transfer rgb to hsv
        grey_image=cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        #threshold(midImage, midImage, 100, 255, 0);
        gus_image = cv2.GaussianBlur(grey_image, (5, 5), 0)
        edges = cv2.Canny(gus_image, 50, 150, apertureSize=3)
        #lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80, 100, 10)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=80, maxLineGap=100)

        if lines is not None:
            # Sort according to the starting point of the line 
            # (from the lower to the upper, from the left to the right)

            print("######")
            theta = np.arctan2(lines[:,0,3] - lines[:,0,1],lines[:,0,2] - lines[:,0,0]) / np.pi *180
            theta = np.where(theta<=0, theta+90, 90-theta)

            lines = np.reshape(lines, (len(lines),4))
            theta = np.reshape(theta, (len(theta),1))
            lines_info = np.hstack((lines,theta.astype(int)))

            lines_infoSort_index = np.lexsort((479-lines_info[:,1],lines_info[:,4]))
            lines_infoSort=lines_info[lines_infoSort_index,:]

            lines_infoSort_slop = lines_infoSort[:,4]

            lines_infoSort_adjIndex = np.where(((lines_infoSort_slop>=2) & (lines_infoSort_slop<88)) | ((lines_infoSort_slop>-88) & (lines_infoSort_slop<=-2)))
            lines_infoSort_horIndex = np.where(((lines_infoSort_slop<=90) & (lines_infoSort_slop>=88)) | ((lines_infoSort_slop<=-88) & (lines_infoSort_slop>=-90)))
            lines_infoSort_verIndex = np.where((lines_infoSort_slop<2) & (lines_infoSort_slop>-2))
            

            lines_infoSort_adj = lines_infoSort[lines_infoSort_adjIndex,:]

            lines_infoSort_hor = lines_infoSort[lines_infoSort_horIndex,:]
            lines_infoSort_ver = lines_infoSort[lines_infoSort_verIndex,:]
            print("ADJ")
            print(lines_infoSort_adj)
            print("HOR")
            print(lines_infoSort_hor)
            print("VER")
            print(lines_infoSort_ver)

            if lines_infoSort_hor.any():
                print("Stop")
                for line in lines_infoSort_hor:
                    cv2.line(imgResult, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 255, 0), 3)
            elif lines_infoSort_ver.any():
                print("Go Forward")
                for line in lines_infoSort_ver:
                    cv2.line(imgResult, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 255, 0), 3)
            else :
                theta_ave = np.sum(lines_infoSort_adj[0,:,4])/lines_infoSort_adj.shape[1]
                print("Turn "+str(int(theta_ave)))
                for line in lines_infoSort_adj:
                    cv2.line(imgResult, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 255, 0), 3)




    #     imgResult = color_image.copy()
    #     if np.size(contours)>0:
    #         for i in range(0,np.size(contours)):
    #             area = cv2.contourArea(contours[i])
    #             if (area > maxArea) and (area<(imgResult.shape[1]-5)*(imgResult.shape[0]-5)):
    #                 maxArea = area
    #                 maxContour = contours[i]


    #         if maxArea > 0:
    #             x,y,w,h = cv2.boundingRect(maxContour) #计算点集的最外面（up-right）矩形边界
    #             if w - h<10 and w - h>-10:
    #             # 包围的矩形框
    #                 cv2.rectangle(imgResult, (x,y), (x+w,y+h), (0, 0, 255), 2)#draw rectangle

    #                 centerx = int(x + w / 2)
    #                 centery = int(y + h / 2)

    # #                print(centerx, centery)

    #                 cv2.circle(imgResult, (centerx,centery), 2, (255, 0, 0), 2)

        # rect = cv2.minAreaRect(contours)
        # imgResult = color_image
        # cv2.drawContours(imgResult, rect, -1, (0, 255, 255), 2)


        #imgResult = cv2.bitwise_and(color_image,color_image,mask=opening_mask)

        #images = np.hstack((color_image, imgResult,image_gus, erode_hsv))
        imgStack = stackImages(0.6,([color_image, imgResult],[gus_image, edges]))

        cv2.namedWindow('Color Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Color Example', imgStack)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()