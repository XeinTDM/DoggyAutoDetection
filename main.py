import cv2
import numpy as np
import dxcam
import glob
import os
import time
import keyboard
import pyautogui

cam = dxcam.create()

def preprocess_template(template):
    if len(template.shape) == 3:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template[template > 240] = 0
    kernel = np.ones((3, 3), np.uint8)
    template = cv2.erode(template, kernel, iterations=1)
    return template

def preprocess_screenshot(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.equalizeHist(blurred)

def convert_image(img):
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return img

def get_combined_match_score(detection_img, template_img):
    res_gray = cv2.matchTemplate(detection_img, template_img, cv2.TM_CCOEFF_NORMED)
    score_gray = cv2.minMaxLoc(res_gray)[1]
    detection_edges = cv2.Canny(detection_img, 50, 150)
    template_edges = cv2.Canny(template_img, 50, 150)
    res_edges = cv2.matchTemplate(detection_edges, template_edges, cv2.TM_CCOEFF_NORMED)
    score_edges = cv2.minMaxLoc(res_edges)[1]
    return (0.7 * score_gray + 0.3 * score_edges)

def get_best_match_for_template(detection_img, template_img):
    best_score = -1
    for scale in np.linspace(0.1, 1.5, 15):
        scaled_template = cv2.resize(template_img, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scaled_template.shape[0] > detection_img.shape[0] or scaled_template.shape[1] > detection_img.shape[1]:
            print(f"Skipping scale {scale} because template shape {scaled_template.shape} is larger than detection shape {detection_img.shape}")
            continue
        score = get_combined_match_score(detection_img, scaled_template)
        print(f"Scale: {scale}, Score: {score}")
        if score > best_score:
            best_score = score
    return best_score

def detect_weapon():
    sw, sh = pyautogui.size()
    left, top = int(sw * 0.75), int(sh * 0.75)
    w, h = int(sw * 0.25), int(sh * 0.25)
    img = cam.grab(region=(left, top, left + w, top + h))
    img = convert_image(img)
    
    target, tol = (234, 255, 5), 15
    found = False
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if all(abs(int(img[y, x][i]) - target[i]) <= tol for i in range(3)):
                fx, fy = x, y
                found = True
                break
        if found:
            break
    if not found:
        print("Target color not found")
        return

    cont = 0
    for y in range(fy, img.shape[0]):
        if all(abs(int(img[y, fx][i]) - target[i]) <= tol for i in range(3)):
            cont += 1
        else:
            break

    new_left, new_top = left + fx, top + fy
    roi = cam.grab(region=(new_left, new_top, new_left + 200, new_top + cont))
    roi = convert_image(roi)
    cv2.imwrite("weaponhud.png", roi)
    
    start_x = 10
    detection_width = int(0.45 * roi.shape[1])
    detection_area = roi[:, start_x:start_x + detection_width]
    cv2.imwrite("detection.png", detection_area)
    
    processed_screen = preprocess_screenshot(detection_area)
    print("Processed screen shape:", processed_screen.shape)
    best_match = None
    best_score = -1
    
    for path in glob.glob("templates/*.png"):
        template = cv2.imread(path)
        if template is None:
            continue
        template = preprocess_template(template)
        score = get_best_match_for_template(processed_screen, template)
        if score > best_score:
            best_score = score
            best_match = path

    threshold = 0.3
    if best_match is not None and best_score > threshold:
        print("Detected weapon:", os.path.basename(best_match), "with score:", best_score)
    elif best_match is not None:
        print("No weapon detected, best score:", os.path.basename(best_match), "at", best_score)
    else:
        print("No weapon detected, best score:", best_score)

def main():
    print("Press 1 or 2 to scan (ESC to exit).")
    while True:
        key = keyboard.read_key()
        if key in ("1", "2"):
            time.sleep(2)
            detect_weapon()
        if key == "esc":
            break

if __name__ == "__main__":
    main()
