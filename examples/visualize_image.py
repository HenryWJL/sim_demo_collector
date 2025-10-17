import cv2
import zarr


def visualize_image():
    with zarr.open("demos/lift_image/episodes.zarr", 'r') as f:
        images = f['agentview_image'][()]
        
    for _, image in enumerate(images):
        image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), (512, 512))
        cv2.imshow("Image Sequence", image)
        cv2.waitKey(20)


if __name__ == "__main__":
    visualize_image()
