#####################################################
##         Color Locate and Depth Measure          ##
#####################################################

#

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

from cv2 import getTickCount, getTickFrequency

# Color information class
#   type : shape type
#   center: coordinates of the center
#   depth: depth of the center
#   vertices: 
class color_info:
    def __init__(self, center, depth, vertex):
      self.center = center
      self.depth = depth
      self.vertex = vertex

# Shape information class
#   type : shape type
#   center: coordinates of the center
#   depth: depth of the center
#   hull: the hull of the shape (For triangle and rectangle(square), hulls are the vertices; For circle, hull is the first point of contour)
class shape_info:
    def __init__(self, type, center, depth, hull):
      self.type = type
      self.center = center
      self.depth = depth
      self.hull = hull
    
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

# Measure the distance from the target center to the camera 
def depth_measure(depth_image, center, depth_scale):
    # Whether the center in the edge of the depth image
    if(center[0]>0 & center[0]<639 & center[1]>0 & center[1]<479):
        # Surrounding matrix
        s_array = np.array([[-1, -1],[-1,0],[-1,1],[0, -1],[0,0],[0,1],[1, -1],[1,0],[1,1]])
        
        # Center surrounding matrix
        cs_array = np.array(center)+s_array

        # Measure the distance
        distance = np.sum(depth_image[cs_array[:,1],cs_array[:,0]])/9 * depth_scale
        return distance
    
    # Can not measure
    return 0

# depth filter
def depth_filter(color_image_src, depth_image_src, depth_min, depth_max):
    # Remove background - Set pixels further than clipping_distance to grey
    white_color = 255
    depth_image_3d = np.dstack((depth_image_src,depth_image_src,depth_image_src)) #depth image is 1 channel, color is 3 channels
    bg_removed = np.where((depth_image_3d > depth_max) | (depth_image_3d <= 0) | (depth_image_3d < depth_min), white_color, color_image_src)
    return bg_removed

# Color detect and locate (locate all the color targets)
# Also filter the region not in the region of interest
def color_detect(color_image_src, depth_image_src, hsv_lower, hsv_upper, depth_min, depth_max, depth_scale):
    # Copy the color image
    imgResult = color_image_src.copy()

    # Create the empty color list
    color_list = list()

    # Gaussian Blur
    image_gus = cv2.GaussianBlur(color_image_src, (5, 5), 0)

    # Remove background - Set pixels further than clipping_distance to grey
    bg_removed = depth_filter(image_gus, depth_image_src, depth_min, depth_max)

    # white_color = 255
    # depth_image_3d = np.dstack((depth_image_src,depth_image_src,depth_image_src)) #depth image is 1 channel, color is 3 channels
    # bg_removed = np.where((depth_image_3d > clipping_distance_max) | (depth_image_3d <= 0) | (depth_image_3d < clipping_distance_min), white_color, image_gus)

    # Transfer rgb to hsv
    image_hsv=cv2.cvtColor(bg_removed, cv2.COLOR_BGR2HSV)

    # Generate mask ( The HSV value of red has two ranges)
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

    # Find all the contours in the erode_mask
    contours, hierarchy = cv2.findContours(opening_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    # Whether the contour exist
    if np.size(contours)>0:
        #area = cv2.contourArea(contours)
        contours_area = np.zeros((len(contours),))
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            
            if area < 200:
                cv2.drawContours(opening_mask,[contours[i]],0,0,-1)

        # Try to reduce the effects of highlights and chromatic aberration(色差)
        # After filling small areas, make valid areas stable and connected
        kernel5 = np.ones((5,5),np.uint8)
        opening_mask = cv2.dilate(opening_mask, kernel5, iterations=5)

        contours_filter, hierarchy_filter = cv2.findContours(opening_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours_info = np.zeros((len(contours_filter),3), dtype = int)
        for i in range(len(contours_filter)):
            # Find the minimum enclosing rectangle
            rect = cv2.minAreaRect(contours_filter[i])

            # Get the rectangle's four corner points
            box = cv2.boxPoints(rect)

            # Draw the contour in red
            cv2.drawContours(imgResult, [np.int0(box)], -1, (0, 255, 255), 2)
            
            # Draw the center of the rectangle in blue
            cv2.circle(imgResult, tuple(map(int,list(rect[0]))), 2, (255, 0, 0), 2)
        
            center = list(map(int,list(rect[0])))
            distance = depth_measure(depth_image_src, center, depth_scale)
            cv2.putText(imgResult,str(int(distance*100))+"cm",tuple(center),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
            # Write color information(center coordinates & depth) into color_list
            c = color_info([center[0],center[1]],int(distance*100),np.fix(box))
            color_list.append(c)

    return color_list, imgResult, opening_mask

# Color detect and locate (locate the biggest color target)
# Also filter the region not in the region of interest
def maxColor_locate(color_image_src, depth_image_src, hsv_lower, hsv_upper, depth_min, depth_max, depth_scale):
    # Copy the color image
    imgResult = color_image_src.copy()

    # Gaussian Blur
    image_gus = cv2.GaussianBlur(color_image_src, (5, 5), 0)

    # Remove background - Set pixels further than clipping_distance to grey
    bg_removed = depth_filter(image_gus, depth_image_src, depth_min, depth_max)

    # white_color = 255
    # depth_image_3d = np.dstack((depth_image_src,depth_image_src,depth_image_src)) #depth image is 1 channel, color is 3 channels
    # bg_removed = np.where((depth_image_3d > clipping_distance_max) | (depth_image_3d <= 0) | (depth_image_3d < clipping_distance_min), white_color, image_gus)

    # Transfer rgb to hsv
    image_hsv=cv2.cvtColor(bg_removed, cv2.COLOR_BGR2HSV)

    # Generate mask ( The HSV value of red has two ranges)
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

    # Find all the contours in the erode_mask
    contours, hierarchy = cv2.findContours(opening_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
    # Whether the contour exist
    if np.size(contours)>0:
        # Find the maximum contour
        c = max(contours, key=cv2.contourArea)
        # Find the minimum enclosing rectangle
        rect = cv2.minAreaRect(c)

        # Get the rectangle's four corner points
        box = cv2.boxPoints(rect)

        # Draw the contour in red
        cv2.drawContours(imgResult, [np.int0(box)], -1, (0, 255, 255), 2)
            
        # Draw the center of the rectangle in blue
        cv2.circle(imgResult, tuple(map(int,list(rect[0]))), 2, (255, 0, 0), 2)
        
        center = list(map(int,list(rect[0])))
        distance = depth_measure(depth_image_src, center, depth_scale)
        cv2.putText(imgResult,str(int(distance*100))+"cm",(20,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),2)
        # print(distance)
        return center, int(distance*100), imgResult

    return [0,0], 0, imgResult

# Shape detect and locate
def shape_detect(color_image_src, depth_image_src, depth_min, depth_max, depth_scale):
    # Copy the color image
    imgResult = color_image_src.copy()

    # Create the empty shape list
    shape_list = list()
    imgGray = cv2.cvtColor(color_image_src,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)
    imgCanny = cv2.Canny(imgBlur,50,50)

    # Find all the external contours in color_image
    contours,hierarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for contour0 in contours:
        area = cv2.contourArea(contour0)
        if area>500:
            cv2.drawContours(imgResult, contour0, -1, (255, 0, 0), 3)

            # Get closed contours length
            peri = cv2.arcLength(contour0,True)

            # Polyline the curve
            approx = cv2.approxPolyDP(contour0,0.02*peri,True)

            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            if objCor ==3: 
                objectType ="Tri"
                hull=cv2.convexHull(contour0)
            elif objCor == 4:
                aspRatio = w/float(h)
                if aspRatio >0.98 and aspRatio <1.03: objectType= "Square"
                else:objectType="Rectangle"
                hull=cv2.convexHull(contour0)
            elif objCor>4: 
                objectType= "Circle"
                hull=contour0[0]
            else:
                objectType="None"
                hull=contour0[0]
            

            distance = depth_measure(depth_image_src, [int(x+w/2),int(y+h/2)], depth_scale)
            distance_int = int(distance*100)
            cv2.rectangle(imgResult,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(imgResult,objectType+str(distance_int),
                        (x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.7,
                        (0,0,0),2)
            cv2.circle(imgResult, tuple([int(x+w/2),int(y+h/2)]), 2, (255, 0, 0), 2)

            # Write shape information(object type & center coordinates & depth) into shape_list
            s = shape_info(objectType, [int(x+w/2),int(y+h/2)], distance_int, hull)
            shape_list.append(s)

    return shape_list, imgResult

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
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
# clipping_distance_in_meters meters away
clipping_max_distance_in_meters = 6     #1 meter
clipping_min_distance_in_meters = 0.2   #0.2 meter

clipping_distance_max = clipping_max_distance_in_meters / depth_scale
clipping_distance_min = clipping_min_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Creat new window 'Trackbar'
cv2.namedWindow('Trackbars')
cv2.resizeWindow("Trackbars",640,480)

# Trackbar seetings
cv2.createTrackbar("Hmin","Trackbars",  0,180,empty)
cv2.createTrackbar("Hmax","Trackbars",180,180,empty)
cv2.createTrackbar("Smin","Trackbars",  0,255,empty)
cv2.createTrackbar("Smax","Trackbars",255,255,empty)
cv2.createTrackbar("Vmin","Trackbars",  0,255,empty)
cv2.createTrackbar("Vmax","Trackbars",255,255,empty)
cv2.createTrackbar("Dmin","Trackbars", 20,800,empty)
cv2.createTrackbar("Dmax","Trackbars",800,800,empty)

# Image showing window
cv2.namedWindow('Locate color and Measure distance', cv2.WINDOW_AUTOSIZE)

# Streaming loop
try:
    while True:
        # Frame starting time
        loop_start = getTickCount()

        # Get HSV range & depth range
        h_min = cv2.getTrackbarPos("Hmin","Trackbars")
        h_max = cv2.getTrackbarPos("Hmax","Trackbars")
        s_min = cv2.getTrackbarPos("Smin","Trackbars")
        s_max = cv2.getTrackbarPos("Smax","Trackbars")
        v_min = cv2.getTrackbarPos("Vmin","Trackbars")
        v_max = cv2.getTrackbarPos("Vmax","Trackbars")
        d_min = cv2.getTrackbarPos("Dmin","Trackbars")
        d_max = cv2.getTrackbarPos("Dmax","Trackbars")

        # Get HSV lower and upper array
        hsv_lower = np.array([h_min,s_min,v_min])
        hsv_upper = np.array([h_max,s_max,v_max])

        # Depth distance transform to depth color based on depth_scale
        clipping_distance_min = d_min / 100 / depth_scale
        clipping_distance_max = d_max / 100 / depth_scale

        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue
        
        # Transform depth_frame and color_frame to numpy array
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        #center, distance, imgResult, bg_removed, image_hsv, opening_mask = maxColor_locate(color_image, depth_image, hsv_lower, hsv_upper, clipping_distance_min, clipping_distance_max, depth_scale)

        color_list, imgResult, opening_mask = color_detect(color_image, depth_image, hsv_lower, hsv_upper, clipping_distance_min, clipping_distance_max, depth_scale)
        for i in range(len(color_list)):
            print(color_list[i].center)
            print(color_list[i].depth)
            print(color_list[i].vertex)
        print("######")

        # shape_list, imgResult=shape_detect(color_image, depth_image, clipping_distance_min, clipping_distance_max, depth_scale)
        # for i in range(len(shape_list)):
        #     print(shape_list[i].type)
        #     print(shape_list[i].center)
        #     print(shape_list[i].depth)
        #     print(shape_list[i].hull)
        # print("######")

        # Render images:
        #   depth align to color on left
        #   depth on right
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Frame ending time
        loop_time = getTickCount() - loop_start
        #print(loop_time)
        total_time=loop_time/(getTickFrequency())

        # Calculate the fps
        fps=int(1/total_time)
        cv2.putText(imgResult,"FPS: "+str(fps),(int(640*0.8),int(480*0.1)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2,cv2.LINE_AA)

        #imgStack = stackImages(0.5,([color_image, imgResult],[bg_removed, image_hsv],[opening_mask,depth_colormap]))

        imgStack = stackImages(0.8,([color_image, imgResult],[opening_mask, depth_colormap]))
        
        cv2.imshow('Locate color and Measure distance', imgStack)

        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()