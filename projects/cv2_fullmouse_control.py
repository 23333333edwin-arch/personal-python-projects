import cv2
import numpy as np
import pyautogui
import time

# 加载人脸识别分类器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 启动摄像头
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 使用DirectShow以获得更稳定的视频流

# 设置摄像头参数以提高帧率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # 降低分辨率以提高帧率
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 60)            # 尝试设置更高的帧率
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # 减少缓冲区大小以减少延迟

# 获取屏幕分辨率
screen_w, screen_h = pyautogui.size()
print(f"屏幕分辨率: {screen_w}x{screen_h}")

# 获取摄像头分辨率
if cap.isOpened():
    cam_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cam_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"摄像头分辨率: {cam_w}x{cam_h}, 帧率: {cam_fps:.1f}FPS")
else:
    print("摄像头初始化失败！")
    exit()

# 灵敏度参数 - 无死区版本
SENSITIVITY_GAIN = 4.5  # 默认灵敏度(推荐2.5-5.0)
EDGE_BOOST_FACTOR = 1.5  # 边缘区域额外增益

# 平滑移动参数
smooth_factor = 0.5
prev_x, prev_y = screen_w/2, screen_h/2

# 帧率控制
TARGET_FPS = 60  # 目标帧率
frame_time = 1.0 / TARGET_FPS
last_time = time.time()

# 帧率统计
frame_count = 0
start_time = time.time()

# 主循环
while True:
    # 控制帧率
    current_time = time.time()
    elapsed = current_time - last_time
    if elapsed < frame_time:
        time.sleep(frame_time - elapsed)
    last_time = current_time
    
    # 清空缓冲区，获取最新帧
    for _ in range(2):  # 清空2帧缓冲区，确保获取最新帧
        cap.grab()
    
    ret, frame = cap.retrieve()
    if not ret:
        continue
    
    # 帧率计数
    frame_count += 1
    
    # 关键修正：水平镜像翻转（解决左右反向问题）
    frame = cv2.flip(frame, 1)
    
    # 创建更小的处理图像以提高速度
    process_frame = cv2.resize(frame, (320, 240))
    gray = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)
    
    # 在人脸检测中使用更小图像
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) > 0:
        # 取最大人脸
        main_face = max(faces, key=lambda face: face[2]*face[3])
        x, y, w, h = main_face
        
        # 将坐标映射回原始分辨率
        scale_x = cam_w / 320
        scale_y = cam_h / 240
        x, y, w, h = int(x*scale_x), int(y*scale_y), int(w*scale_x), int(h*scale_y)
        
        # 计算人脸中心点
        face_center_x = x + w//2
        face_center_y = y + h//2
        
        # 绘制视觉反馈（在原始帧上绘制）
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.circle(frame, (face_center_x, face_center_y), 5, (0, 255, 0), -1)
        
        # ============ 优化映射算法 ============
        # 1. 计算中心点偏移（归一化到[-1,1]）
        offset_x = (face_center_x - cam_w/2) / (cam_w/2)
        offset_y = (face_center_y - cam_h/2) / (cam_h/2)
        
        # 2. 应用非线性响应曲线（边缘区域更灵敏）
        # 使用立方曲线：f(x) = x^3 * a + x * (1-a)
        curve_factor = 0.7
        offset_x = (offset_x**3) * curve_factor + offset_x * (1 - curve_factor)
        offset_y = (offset_y**3) * curve_factor + offset_y * (1 - curve_factor)
        
        # 3. 应用灵敏度增益（边缘区域额外增强）
        edge_boost = 1.0 + EDGE_BOOST_FACTOR * (abs(offset_x) + abs(offset_y))/2
        offset_x *= SENSITIVITY_GAIN * edge_boost
        offset_y *= SENSITIVITY_GAIN * edge_boost
        
        # 4. 映射到屏幕坐标
        target_x = screen_w/2 + offset_x * (screen_w/2)
        target_y = screen_h/2 + offset_y * (screen_h/2)
        target_x = max(0, min(screen_w, target_x))
        target_y = max(0, min(screen_h, target_y))
        # ============ 修改结束 ============
        
        # 应用平滑移动
        smoothed_x = prev_x * (1 - smooth_factor) + target_x * smooth_factor
        smoothed_y = prev_y * (1 - smooth_factor) + target_y * smooth_factor
        
        # 移动鼠标
        pyautogui.moveTo(smoothed_x, smoothed_y)
        
        # 更新上一帧位置
        prev_x, prev_y = smoothed_x, smoothed_y
        
        # 显示坐标信息
        cv2.putText(frame, f"Face: ({face_center_x}, {face_center_y})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Screen: ({int(smoothed_x)}, {int(smoothed_y)})", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Sensitivity: {SENSITIVITY_GAIN:.1f}x", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 100, 255), 2)
    
    # 显示帧率
    elapsed_time = time.time() - start_time
    if elapsed_time > 0:
        fps = frame_count / elapsed_time
        cv2.putText(frame, f"FPS: {fps:.1f}", 
                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # 显示提示信息
    cv2.putText(frame, "Press +/-: Adjust Sensitivity", 
               (10, frame.shape[0]-50), 
               cv2.FONT_HERSHEY_SIMPLEX, 
               0.6, (200, 200, 0), 2)
    cv2.putText(frame, "Press ESC: Quit", 
               (10, frame.shape[0]-20), 
               cv2.FONT_HERSHEY_SIMPLEX, 
               0.6, (200, 200, 0), 2)
    
    cv2.imshow('Precision Face Mouse', frame)
    
    # 按键调整灵敏度
    key = cv2.waitKey(1)
    if key == 27:  # ESC退出
        break
    elif key == ord('+') or key == ord('='):  # 增加灵敏度
        SENSITIVITY_GAIN = min(6.0, SENSITIVITY_GAIN + 0.2)
    elif key == ord('-') or key == ord('_'):  # 减少灵敏度
        SENSITIVITY_GAIN = max(1.5, SENSITIVITY_GAIN - 0.2)
    elif key == ord('f'):  # 帧率控制
        TARGET_FPS = min(120, TARGET_FPS + 10)
        frame_time = 1.0 / TARGET_FPS
    elif key == ord('s'):  # 帧率控制
        TARGET_FPS = max(15, TARGET_FPS - 10)
        frame_time = 1.0 / TARGET_FPS

# 释放资源
cap.release()
cv2.destroyAllWindows()