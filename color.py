# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

color_dist = {
    'red1': {'lower': np.array([0, 43, 46]), 'upper': np.array([10, 255, 255])},
    'red2': {'lower': np.array([156, 43, 46]), 'upper': np.array([180, 255, 255])},
    'blue': {'lower': np.array([100, 80, 46]), 'upper': np.array([124, 255, 255])},
    'green': {'lower': np.array([35, 43, 35]), 'upper': np.array([90, 255, 255])},
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

# Creat new window 'Trackbar'
cv2.namedWindow('Trackbars')
cv2.resizeWindow("Trackbars",640,240)

# Trackbar seetings
cv2.createTrackbar("Hmin","Trackbars",  0,180,empty)
cv2.createTrackbar("Hmax","Trackbars",180,180,empty)
cv2.createTrackbar("Smin","Trackbars",  0,255,empty)
cv2.createTrackbar("Smax","Trackbars",255,255,empty)
cv2.createTrackbar("Vmin","Trackbars",  0,255,empty)
cv2.createTrackbar("Vmax","Trackbars",255,255,empty)

# Streaming loop
try:
    while True:
        # Get HSV range
        h_min = cv2.getTrackbarPos("Hmin","Trackbars")
        h_max = cv2.getTrackbarPos("Hmax","Trackbars")
        s_min = cv2.getTrackbarPos("Smin","Trackbars")
        s_max = cv2.getTrackbarPos("Smax","Trackbars")
        v_min = cv2.getTrackbarPos("Vmin","Trackbars")
        v_max = cv2.getTrackbarPos("Vmax","Trackbars")
        print(h_min,h_max,s_min,s_max,v_min,v_max)
        # Get lower and upper array
        lower = np.array([h_min,s_min,v_min])
        upper = np.array([h_max,s_max,v_max])
        
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image
        
        # Get frames
        color_frame = frames.get_color_frame()

        # Validate that both frames are valid
        if  not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        # Gaussian Blur
        image_gus = cv2.GaussianBlur(color_image, (5, 5), 0)

        # Transfer rgb to hsv
        image_hsv=cv2.cvtColor(image_gus, cv2.COLOR_BGR2HSV)

        # # Erode image
        # erode_hsv = cv2.erode(image_hsv, None, iterations=3)
        # # Dilate image
        # kernel = np.ones((3,3),np.uint8) 
        # dilate_hsv = cv2.dilate(erode_hsv,kernel,iterations = 3)

        # kernel = np.ones((3,3),np.uint8)
        # # Open operation( erode and then dilate)
        # opening_hsv = cv2.morphologyEx(image_hsv, cv2.MORPH_OPEN, kernel)

        # Generate mask
        #mask = cv2.inRange(image_hsv,lower,upper)
        mask1 = cv2.inRange(image_hsv,color_dist['red1']['lower'],color_dist['red1']['upper'])
        mask2 = cv2.inRange(image_hsv,color_dist['red2']['lower'],color_dist['red2']['upper'])
        mask = mask1+mask2

        # Set kernel as 3*3
        kernel = np.ones((3,3),np.uint8)
        # Erode image
        erode_mask = cv2.erode(mask, kernel, iterations=4)
        # Dilate image
        opening_mask = cv2.dilate(erode_mask, kernel, iterations=3)

        # # Open operation( erode and then dilate)
        # kernel = np.ones((5,5),np.uint8)
        # opening_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

        contours, hierarchy = cv2.findContours(erode_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        imgResult = color_image.copy()
        if np.size(contours)>0:
            c = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(c)
            #rect = cv2.boundingRect(c)
            box = cv2.boxPoints(rect)
            cv2.drawContours(imgResult, [np.int0(box)], -1, (0, 255, 255), 2)
            cv2.circle(imgResult, tuple(map(int,list(rect[0]))), 2, (255, 0, 0), 2)

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
        imgStack = stackImages(0.6,([color_image, imgResult],[image_gus, erode_mask]))

        cv2.namedWindow('Color Example', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Color Example', imgStack)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()