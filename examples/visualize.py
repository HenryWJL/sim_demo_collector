import cv2
import zarr

with zarr.open("demos/lift_image/episodes.zarr", 'r') as f:
    images = f['agentview_image'][()]
    print(images.shape)
    for i in range(images.shape[0]):
        image = images[i]
        cv2.imshow("Img", cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), (256, 256)))
        cv2.waitKey(30)
