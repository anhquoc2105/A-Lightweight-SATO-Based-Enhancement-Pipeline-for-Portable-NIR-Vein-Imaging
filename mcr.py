import cv2
import numpy as np

# Ảnh đầu vào
path = r"F:\UIT\NCKH\new\anhlocnhieu\nguyen_p\figure7_step7_edge_preserving.png"
img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img_color = cv2.imread(path)

if img_gray is None:
    print("Không đọc được ảnh!")
    exit()

# ====== Tạo 5 ô xanh + 5 ô đỏ ======
rect_size = 20       # ô xanh lớn
rect_size_1 = 13     # ô đỏ nhỏ

# 5 ô xanh
rects_green = [
    [50, 50, 50+rect_size, 50+rect_size],
    [120, 80, 120+rect_size, 80+rect_size],
    [200, 120, 200+rect_size, 120+rect_size],
    [250, 200, 250+rect_size, 200+rect_size],
    [300, 300, 300+rect_size, 300+rect_size]
]

# 5 ô đỏ
rects_red = [
    [80, 40, 80+rect_size_1, 40+rect_size_1],
    [150, 90, 150+rect_size_1, 90+rect_size_1],
    [230, 160, 230+rect_size_1, 160+rect_size_1],
    [270, 260, 270+rect_size_1, 260+rect_size_1],
    [330, 350, 330+rect_size_1, 350+rect_size_1]
]

# ====== Trạng thái kéo thả ======
dragging = False
selected_group = None
selected_index = -1
offset_x = 0
offset_y = 0

def mouse_event(event, x, y, flags, param):
    global dragging, selected_group, selected_index, offset_x, offset_y

    if event == cv2.EVENT_LBUTTONDOWN:
        # kiểm tra nhóm xanh
        for i, (x1, y1, x2, y2) in enumerate(rects_green):
            if x1 <= x <= x2 and y1 <= y <= y2:
                dragging = True
                selected_group = "green"
                selected_index = i
                offset_x = x - x1
                offset_y = y - y1
                return

        # kiểm tra nhóm đỏ
        for i, (x1, y1, x2, y2) in enumerate(rects_red):
            if x1 <= x <= x2 and y1 <= y <= y2:
                dragging = True
                selected_group = "red"
                selected_index = i
                offset_x = x - x1
                offset_y = y - y1
                return

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        x1 = x - offset_x
        y1 = y - offset_y

        if selected_group == "green":
            rects_green[selected_index] = [x1, y1, x1 + rect_size, y1 + rect_size]
        else:
            rects_red[selected_index] = [x1, y1, x1 + rect_size_1, y1 + rect_size_1]

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        selected_group = None
        selected_index = -1


cv2.namedWindow("Keo tha 10 o")
cv2.setMouseCallback("Keo tha 10 o", mouse_event)

while True:
    temp = img_color.copy()

    # Vẽ ô xanh
    for (x1, y1, x2, y2) in rects_green:
        cv2.rectangle(temp, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # Vẽ ô đỏ
    for (x1, y1, x2, y2) in rects_red:
        cv2.rectangle(temp, (x1, y1), (x2, y2), (0, 0, 255), 1)

    cv2.imshow("Keo tha 10 o", temp)
    key = cv2.waitKey(20)

    # ===== ENTER: In mean =====
        # ===== ENTER: In mean =====
    if key == 13:
        print("\n======== GIÁ TRỊ TRUNG BÌNH Ô XANH ========")
        mean_green = []

        for i, (x1, y1, x2, y2) in enumerate(rects_green):
            roi = img_gray[y1:y2, x1:x2]
            mean_val = np.mean(roi)
            mean_green.append(mean_val)
            print(f"Ô xanh {i+1}: Mean = {mean_val:.3f}")

        print(f"=> Mean nhóm xanh = {np.mean(mean_green):.3f}")
        print("\n======== GIÁ TRỊ TRUNG BÌNH Ô ĐỎ ========")
        mean_red = []

        for i, (x1, y1, x2, y2) in enumerate(rects_red):
            roi = img_gray[y1:y2, x1:x2]
            mean_val = np.mean(roi)
            mean_red.append(mean_val)
            print(f"Ô đỏ {i+1}: Mean = {mean_val:.3f}")

        print(f"=> Mean nhóm đỏ = {np.mean(mean_red):.3f}")
        print("\n==========================================\n")
        xanh = np.mean(mean_green)
        do = np.mean(mean_red)
        mcr = (abs(xanh - do)) / (xanh+ do)
        print(f"=> MCR = {mcr:.6f}")
        print("\n==========================================\n")
    # ESC thoát
    if key == 27:
        break

cv2.destroyAllWindows()
