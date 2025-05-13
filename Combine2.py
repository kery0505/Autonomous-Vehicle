import threading
import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
from picamera2 import Picamera2
import RPi.GPIO as GPIO
import time

# Thread-safe camera access
camera_lock = threading.Lock()

# Shared variables for GUI
symbol_frame = None
symbol_label = None
line_final = None
line_color1_final = None
line_color2_final = None
line_text = []

# Paths for TFLite model and labels
model_path = "/home/user/Desktop/Symbols/converted_tflite (1)/model_unquant.tflite"
label_path = "//home/user/Desktop/Symbols/converted_tflite (1)/labels.txt"

# Load labels
with open(label_path, 'r') as f:
    symbol_names = [line.strip() for line in f.readlines()]

# Load TFLite model
interpreter = Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set PWM duty cycle (0-100)
base_duty = 52
back_duty = 40

# Set GPIO Pins
enableR = 13  # Right PWM pin enB
enableL = 19  # Left PWM pin enA
Left1 = 12
Left2 = 16
Right1 = 20
Right2 = 21

# GPIO Setup
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)  # Set to BCM numbering
GPIO.setup(enableR, GPIO.OUT)
GPIO.setup(enableL, GPIO.OUT)
GPIO.setup(Left1, GPIO.OUT)
GPIO.setup(Left2, GPIO.OUT)
GPIO.setup(Right1, GPIO.OUT)
GPIO.setup(Right2, GPIO.OUT)

pwm1 = GPIO.PWM(enableR, 500)
pwm2 = GPIO.PWM(enableL, 500)
pwm1.start(0)
pwm2.start(0)

# PID Controller Kp, Ki, Kd
Kp = 0.8  # Proportional gain 0.95
Ki = 0.05   # Integral gain 0.050
Kd = 2.6 # Derivative gain 2.57

# PID variables
last_error = 0
integral = 0
setpoint = 0  # Desired position (center of the line)

# Color Detection parameters
lower_color1 = np.array([80, 100, 100])  # Lower boundary HSV (H, S, V)
upper_color1 = np.array([105, 255, 255])  # Upper boundary HSV (H, S, V)
lower_color2 = np.array([0, 100, 80])
upper_color2 = np.array([30, 255, 255])
blk_lower = np.array([0, 0, 0])
blk_upper = np.array([179, 255, 62])

# Motor Functions
def forward(left_speed, right_speed):
    left_speed = max(0, min(100, left_speed))
    right_speed = max(0, min(100, right_speed))
    pwm1.ChangeDutyCycle(right_speed)
    GPIO.output(Left1, GPIO.LOW)
    GPIO.output(Left2, GPIO.HIGH)
    pwm2.ChangeDutyCycle(left_speed)
    GPIO.output(Right1, GPIO.HIGH)
    GPIO.output(Right2, GPIO.LOW)

def backward():
    pwm1.ChangeDutyCycle(back_duty)
    GPIO.output(Left1, GPIO.HIGH)
    GPIO.output(Left2, GPIO.LOW)
    pwm2.ChangeDutyCycle(back_duty)
    GPIO.output(Right1, GPIO.LOW)
    GPIO.output(Right2, GPIO.HIGH)

def stop():
    pwm1.ChangeDutyCycle(0)
    GPIO.output(Left1, GPIO.LOW)
    GPIO.output(Left2, GPIO.LOW)
    pwm2.ChangeDutyCycle(0)
    GPIO.output(Right1, GPIO.LOW)
    GPIO.output(Right2, GPIO.LOW)

def pid_controller(error):
    global last_error, integral
    proportional = Kp * error
    integral += Ki * error
    derivative = Kd * (error - last_error)
    integral = max(-30, min(30, integral))
    output = proportional + integral + derivative
    last_error = error
    return output

# Initialize Camera
camera = Picamera2()
camera_config = camera.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
camera.configure(camera_config)
camera.start()
time.sleep(1)  # Allow camera to warm up

# Symbol Detection Thread
def symbol_detection():
    global symbol_frame, symbol_label
    while not stop_event.is_set():
        with camera_lock:
            frame = camera.capture_array()
        if frame is None:
            continue
        # Preprocess
        input_frame = cv2.resize(frame, (224, 224))
        input_frame = input_frame.astype(np.float32) / 255.0
        input_frame = np.expand_dims(input_frame, axis=0)
        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_frame)
        interpreter.invoke()
        # Get output
        output = interpreter.get_tensor(output_details[0]['index'])[0]
        predicted_class = np.argmax(output)
        confidence = output[predicted_class]
        # Update shared variables
        symbol_frame = frame.copy()
        if confidence > 0.5:
            label = symbol_names[predicted_class]
            if label != "noise":
                print(label)  # Print the detected label (excluding "noise")
            symbol_label = f"{label}: {confidence:.2f}"
        else:
            symbol_label = None
        time.sleep(0.01)  # Reduce CPU usage

# Line Following Thread
def line_following():
    global line_final, line_color1_final, line_color2_final, line_text
    while not stop_event.is_set():
        with camera_lock:
            frame = camera.capture_array()
        if frame is None:
            continue
        # Set up lower half of the frame as ROI
        height, width = frame.shape[:2]
        roi = frame[height//2:, :]
        # Convert RGB to BGR
        frame_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        # Process for black line
        blur = cv2.GaussianBlur(hsv, (5, 5), 0)
        blk_mask = cv2.inRange(blur, blk_lower, blk_upper)
        erosion = cv2.erode(blk_mask, None, iterations=2)
        final = cv2.dilate(erosion, None, iterations=2)
        # Process for color lines
        color_blur = cv2.GaussianBlur(hsv, (5, 5), 0)
        color1_mask = cv2.inRange(color_blur, lower_color1, upper_color1)
        color2_mask = cv2.inRange(color_blur, lower_color2, upper_color2)
        color1_erosion = cv2.erode(color1_mask, None, iterations=2)
        color1_final = cv2.dilate(color1_erosion, None, iterations=2)
        color2_erosion = cv2.erode(color2_mask, None, iterations=2)
        color2_final = cv2.dilate(color2_erosion, None, iterations=2)
        # Find contours
        contours, _ = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color1_contours, _ = cv2.findContours(color1_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color2_contours, _ = cv2.findContours(color2_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Find image center
        image_center = final.shape[1] // 2
        color_detect = 0
        cx = None
        line_text = []
        if color1_contours:
            largest_color1_contour = max(color1_contours, key=cv2.contourArea)
            M = cv2.moments(largest_color1_contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                color_detect = 1
                #print("1st color detected")
        elif color2_contours:
            largest_color2_contour = max(color2_contours, key=cv2.contourArea)
            M = cv2.moments(largest_color2_contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                color_detect = 1
                #print("2nd color detected")
        if not color_detect and contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                #print("Black line detected")
        if cx is not None:
            error = image_center - cx
            pid_output = pid_controller(error)
            scale = min(1.0, max(0.2, 1 - abs(error)/100))
            duty = base_duty * scale
            left_speed = (base_duty - pid_output)
            right_speed = (base_duty + pid_output)
            line_text.append(f"Error: {error}")
            line_text.append(f"PID: {pid_output:.2f}")
            line_text.append(f"L: {left_speed:.2f} R: {right_speed:.2f}")
            forward(left_speed, right_speed)
        else:
            backward()
            time.sleep(0.10)
            line_text.append("No Line")
        # Update shared variables
        line_final = final.copy()
        line_color1_final = color1_final.copy()
        line_color2_final = color2_final.copy()
        time.sleep(0.01)  # Reduce CPU usage

# Event to signal threads to stop
stop_event = threading.Event()

# Main thread for GUI
def main_gui():
    global symbol_frame, symbol_label, line_final, line_color1_final, line_color2_final, line_text
    while not stop_event.is_set():
        if symbol_frame is not None:
            display_frame = symbol_frame.copy()
            if symbol_label:
                cv2.putText(display_frame, symbol_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.imshow('Symbol Detection', display_frame)
        if line_final is not None:
            display_final = line_final.copy()
            for i, text in enumerate(line_text):
                color = (255, 0, 0) if text != "No Line" else (0, 0, 255)
                cv2.putText(display_final, text, (10, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imshow("binary mode", display_final)
        if line_color1_final is not None:
            cv2.imshow("Color1 line", line_color1_final)
        if line_color2_final is not None:
            cv2.imshow("Color2 line", line_color2_final)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('x'):
            stop_event.set()
            break

try:
    # Start threads
    symbol_thread = threading.Thread(target=symbol_detection)
    line_thread = threading.Thread(target=line_following)
    symbol_thread.start()
    line_thread.start()
    # Run GUI in main thread
    main_gui()
    # Wait for threads to complete
    symbol_thread.join()
    line_thread.join()

except KeyboardInterrupt:
    print("\n[INFO] Script stopped by user.")
    stop_event.set()

finally:
    # Clean up
    camera.stop()
    stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()
