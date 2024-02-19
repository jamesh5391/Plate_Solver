import cv2
import numpy as np

# Load the image
image = cv2.imread('stars.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#create filter for pixel intensity values < 210 
mask = cv2.inRange(gray_image, 210, 255)

#filter out less brighter pixels in image

result = cv2.bitwise_and(gray_image, gray_image, mask=mask)

res, threshold = cv2.threshold(result, 220, 255, 0); 

#get contours for stars 
contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:

    #find centroid coordinate of each eccentric star  
    M = cv2.moments(contour)

    x,y = contour[0,0]

    cv2.circle(threshold, (x, y), 1, (255, 255, 255), -1)

# Display the result

cv2.imshow('Detected Stars', threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()