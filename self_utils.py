
import os
import cv2
import main

def save_image(filename, img, log):
    output_path = os.path.join("output", filename)
    status = cv2.imwrite(output_path, img)

    if log:
        status_msg = "Saved" if status else "Failed to save"
        print(f"{status_msg} to {output_path}")

def improve_img(img, log):
    resized_img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    grayscaled_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    denoised_img = cv2.GaussianBlur(grayscaled_img, (5, 5), 0)

    save_image("i-original.png", img, log)
    save_image("i-resized.png", resized_img, log)
    save_image("i-grayscaled.png", grayscaled_img, log)
    save_image("i-denoised.png", denoised_img, log)

    return denoised_img
        
def test_image():
    img_paths = [
        "/Users/riniwtz/Documents/Coding/Python/PaddlePaddle/assets/nay2.png"
    ]

    for img in img_paths:
        print("\n")
        print("Testing original image")
        orig_score = main.run_ocr(img)
        print("\n")
        print("Testing resized image")
        resized_img = cv2.imread(img)
        resized_img = cv2.resize(resized_img, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
        resized_score = main.run_ocr(resized_img)
        print("\n")
        print("Testing enhanced image")
        enhanced_score = main.run_ocr(improve_img(img, False))
        print("\n")


        difference = resized_score-orig_score
        difference_txt = f'{difference:.20f}'
        print(f"{difference_txt}")