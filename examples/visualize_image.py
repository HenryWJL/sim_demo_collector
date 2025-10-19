import cv2
import zarr
import numpy as np


def visualize_image():
    with zarr.open("demos/lift_image/episodes.zarr", 'r') as f:
        images = f['agentview_image'][()]
        masks = f['agentview_image_mask'][()]
        
    for _, image in enumerate(images):
        image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), (512, 512))
        cv2.imshow("Image Sequence", image)
        cv2.waitKey(20)

    for _, mask in enumerate(masks):
        mask = cv2.resize(mask.astype(np.uint8) * 255, (512, 512))
        cv2.imshow("Image Mask Sequence", mask)
        cv2.waitKey(20)


if __name__ == "__main__":
    visualize_image()
