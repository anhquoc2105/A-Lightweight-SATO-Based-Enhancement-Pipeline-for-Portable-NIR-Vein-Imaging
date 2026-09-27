import cv2
import numpy as np

# Ảnh đầu vào
path = r"F:\UIT\NCKH\new\anhlocnhieu\nguyen_t\figure7_step7_edge_preserving.png"
img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img_color = cv2.imread(path)

if img_gray is None:
    print("Không đọc được ảnh!")
    exit()

# ====== Tạo 5 ô vuông ban đầu ======
rect_size = 22   # kích thước ô vuông (rộng x cao)
rects = [
    [50, 50, 50+rect_size, 50+rect_size],
    [120, 80, 120+rect_size, 80+rect_size],
    [200, 120, 200+rect_size, 120+rect_size],
    [250, 200, 250+rect_size, 200+rect_size],
    [300, 300, 300+rect_size, 300+rect_size]
]

dragging = False
selected = -1
offset_x = 0
offset_y = 0

def mouse_event(event, x, y, flags, param):
    global dragging, selected, offset_x, offset_y

    if event == cv2.EVENT_LBUTTONDOWN:
        # kiểm tra click trúng hình vuông nào
        for i, (x1, y1, x2, y2) in enumerate(rects):
            if x1 <= x <= x2 and y1 <= y <= y2:
                dragging = True
                selected = i
                offset_x = x - x1
                offset_y = y - y1
                break

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        # cập nhật vị trí khi kéo
        x1 = x - offset_x
        y1 = y - offset_y
        rects[selected] = [x1, y1, x1 + rect_size, y1 + rect_size]

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        selected = -1


cv2.namedWindow("Keo tha 5 o vuong")
cv2.setMouseCallback("Keo tha 5 o vuong", mouse_event)

while True:
    temp = img_color.copy()

    # vẽ 5 ô vuông màu xanh
    for (x1, y1, x2, y2) in rects:
        cv2.rectangle(temp, (x1, y1), (x2, y2), (0,255,0), 1)

    cv2.imshow("Keo tha 5 o vuong", temp)
    key = cv2.waitKey(20)

    # Nhấn ENTER -> tính mean/std/COV
    if key == 13:  # ENTER
        print("======== KẾT QUẢ 5 VÙNG ========")
        mean_list = []
        std_list = []

        for i, (x1, y1, x2, y2) in enumerate(rects):
            roi = img_gray[y1:y2, x1:x2]
            mean_val = np.mean(roi)
            std_val = np.std(roi)

            mean_list.append(mean_val)
            std_list.append(std_val)

            print(f"Vùng {i+1}: Mean = {mean_val:.3f} | Std = {std_val:.3f}")

        mean_avg = np.mean(mean_list)
        std_avg = np.mean(std_list)
        cov = std_avg / mean_avg

        print("\n======== TỔNG HỢP ========")
        print(f"Std trung bình  = {std_avg:.3f}")
        print(f"Mean trung bình = {mean_avg:.3f}")
        print(f"COV tổng (std/mean) = {cov:.3f}\n")

    # Nhấn ESC -> thoát
    if key == 27:
        break

cv2.destroyAllWindows()
