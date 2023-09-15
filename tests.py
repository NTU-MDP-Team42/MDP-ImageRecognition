import models_loading, img_utils
from PIL import Image

TEST_IMG_FOLDER = "test-imgs"

def get_test_img_path(img_name: str) -> str:
    return f"{TEST_IMG_FOLDER}/{img_name}"

def test_save_augmented_img():
    img_utils.show_image(img_utils.run_model_and_save_augmented(models_loading.week8, get_test_img_path("image.jpg"),\
                                                                get_test_img_path("image_output.jpg")))

def test_biggest_bbox():
    img_utils.show_image(img_utils.run_model_and_save_augmented_only_biggest_bbox(models_loading.week8,\
                                                            get_test_img_path("10.jpg"), get_test_img_path("10_output.jpg")))
    img_utils.show_image(img_utils.run_model_and_save_augmented_only_biggest_bbox(models_loading.week8,\
                                                            get_test_img_path("11.jpg"), get_test_img_path("11_output.jpg")))
    
def test_left_right():
    left_right_imgs = [("2.jpg", "left"), ("6.jpg", "right"), ("7.jpg", "left"), ("8.jpg", "left")]
    for img_name, direction in left_right_imgs:
        assert img_utils.is_left(get_test_img_path(img_name)) == (direction == "left")
        assert img_utils.is_right(get_test_img_path(img_name)) == (direction == "right")