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
