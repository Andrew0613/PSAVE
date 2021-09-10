import cv2

im_gray = cv2.imread("C:\\Users\\hp\\Pictures\\123.jpg", cv2.IMREAD_GRAYSCALE)
im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)

cv2.imwrite('123.png',im_color)