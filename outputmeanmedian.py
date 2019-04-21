import mmv
import numpy as np
import cv2

f, _ = mmv.format()
m = mmv.mmv(f)
print('MNIST MMV Done.')


for num in range(10):
    m_img = np.reshape(m[num]['median'], (28, 28)) * 255

    width = int(m_img.shape[1] * 10)
    height = int(m_img.shape[0] * 10)
    dim = (width, height)
    # resize image
    resized = cv2.resize(m_img, dim, interpolation = cv2.INTER_AREA) 

    cv2.imwrite('images/' + str(num) + '-MEDIAN.jpg', resized)

    # cv2.imshow(str(num) + '-MEDIAN', resized)

    m_img = np.reshape(m[num]['mean'], (28, 28)) * 255

    width = int(m_img.shape[1] * 10)
    height = int(m_img.shape[0] * 10)
    dim = (width, height)
    # resize image
    resized = cv2.resize(m_img, dim, interpolation = cv2.INTER_AREA) 

    # cv2.imshow(str(num) + '-MEAN', resized)
    cv2.imwrite('images/' + str(num) + '-MEAN.jpg', resized)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

def img_resize(factor, img):
    width = int(img.shape[1] * factor)
    height = int(img.shape[0] * factor)
    return cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)

def vis_task1(m):
    row_05 = np.zeros((280, 1))
    row_16 = np.zeros((280, 1))
    row_27 = np.zeros((280, 1))
    row_38 = np.zeros((280, 1))
    row_49 = np.zeros((280, 1))
    
    for i in range(10):
        img1 = np.reshape(m[i]['mean'], (28,28)) * 255
        img1 = img_resize(10, img1)
        img2 = np.reshape(m[i]['median'], (28,28)) * 255
        img2 = img_resize(10, img2)
        img3 = np.reshape(m[i]['var'], (28,28)) * 255
        img3 = img_resize(10, img3)
        img4 = np.reshape(m[i]['std'], (28,28)) * 255
        img4 = img_resize(10, img4)
        
        if i % 5 == 0:
            row_05 = np.hstack([row_05, img1])
            row_05 = np.hstack([row_05, img2])
            row_05 = np.hstack([row_05, img3])
            row_05 = np.hstack([row_05, img4])
        elif i % 5 == 1:
            row_16 = np.hstack([row_16, img1])
            row_16 = np.hstack([row_16, img2])
            row_16 = np.hstack([row_16, img3])
            row_16 = np.hstack([row_16, img4])
        elif i % 5 == 2:
            row_27 = np.hstack([row_27, img1])
            row_27 = np.hstack([row_27, img2])
            row_27 = np.hstack([row_27, img3])
            row_27 = np.hstack([row_27, img4])
        elif i % 5 == 3:
            row_38 = np.hstack([row_38, img1])
            row_38 = np.hstack([row_38, img2])
            row_38 = np.hstack([row_38, img3])
            row_38 = np.hstack([row_38, img4])
        elif i % 5 == 4:
            row_49 = np.hstack([row_49, img1])
            row_49 = np.hstack([row_49, img2])
            row_49 = np.hstack([row_49, img3])
            row_49 = np.hstack([row_49, img4])
            
    all_rows = np.vstack([row_05, row_16])
    all_rows = np.vstack([all_rows, row_27])
    all_rows = np.vstack([all_rows, row_38])
    all_rows = np.vstack([all_rows, row_49])
    
    cv2.imwrite('images/All.jpg', all_rows)
