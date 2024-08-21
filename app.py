from flask import Flask, request, render_template, jsonify, redirect, url_for
import cv2
import mediapipe as mp
import numpy as np
import os
from werkzeug.utils import secure_filename
import sqlite3
from datetime import datetime

app = Flask(__name__)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 角度を計算する関数
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle
        
    return angle

# リリースの瞬間を検出する関数
def is_release_moment(landmarks, prev_wrist_y):
    wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
    return prev_wrist_y is not None and wrist_y < prev_wrist_y

# 軸足が一直線になるかを確認する関数
def is_straight_line(hip, knee, ankle):
    return np.abs(np.cross(np.subtract(knee, hip), np.subtract(ankle, hip)))

# ビデオ処理関数
def video_processing_function(filepath, result_filepath):
    cap = cv2.VideoCapture(filepath)
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    result = cv2.VideoWriter(result_filepath, fourcc, fps, (width, height))
    elbow_angles = []
    knee_angles = []
    three_quarter_angles = []
    prev_wrist_y = None
    plate_x = None
    footplant_x = None
    is_phase_exceeded = False
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                # プレートとフットプラントの位置
                if plate_x is None:
                    plate_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x
                    footplant_x = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL.value].x

                # 3/4フェーズの位置を計算
                three_quarter_x = plate_x + (footplant_x - plate_x) * 3 / 4

                if results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL.value].x < three_quarter_x:
                    is_phase_exceeded = True

                # リリースの瞬間
                if is_release_moment(results.pose_landmarks.landmark, prev_wrist_y):
                    # 肘、手首、肩
                    shoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    wrist = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    
                    # 股関節、膝、足首
                    hip = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                    # 角度の計算
                    elbow_angle = calculate_angle(shoulder, elbow, wrist)
                    knee_angle = calculate_angle(hip, knee, ankle)
                    
                    # リリースの瞬間を保存
                    elbow_angles.append(elbow_angle)
                    knee_angles.append(knee_angle)

                # 3/4フェーズ
                if is_phase_exceeded:
                    hip = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x, 
                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    knee = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, 
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    ankle = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, 
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                    # 3/4フェーズの瞬間の軸足の角度を計算して追加
                    three_quarter_angle = calculate_angle(hip, knee, ankle)
                    three_quarter_angles.append(three_quarter_angle)
                    
                    is_phase_exceeded = False 
                prev_wrist_y = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
            result.write(image)

    cap.release()
    result.release()

    return elbow_angles, knee_angles, three_quarter_angles

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    
    if 'file1' not in request.files:
        return redirect(request.url)
    if 'file2' not in request.files:
        return redirect(request.url)
    
    file1 = request.files['file1']
    file2 = request.files['file2']
    if file1.filename == '':
        return redirect(request.url)
    if file2.filename == '':
        return redirect(request.url)

    # ファイル名の処理
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename1 = secure_filename(file1.filename)
    filename1_time = f"{timestamp}_{filename1}"
    filepath1 = os.path.join('static', filename1_time)
    file1.save(filepath1)

    filename2 = secure_filename(file2.filename)
    filename2_time = f"{timestamp}_{filename2}"
    filepath2 = os.path.join('static', filename2_time)
    file2.save(filepath2)

    result_filename1 = f"result1_{timestamp}_{filename1}"
    result_filepath1 = os.path.join('static', result_filename1)

    result_filename2 = f"result2_{timestamp}_{filename2}"
    result_filepath2 = os.path.join('static', result_filename2)

    elbow_angles1, knee_angles1, three_quarter_angles1 = video_processing_function(filepath1, result_filepath1)
    elbow_angles2, knee_angles2, three_quarter_angles2 = video_processing_function(filepath2, result_filepath2)

    # 動画の数が増え過ぎないように処理
    if os.path.exists(filepath1):
        os.remove(filepath1)

    def remove_old_files(directory, keep_latest=4):
        files = sorted(os.listdir(directory), key=lambda x: os.path.getctime(os.path.join(directory, x)))
        if len(files) > keep_latest:
            for file in files[:-keep_latest]:
                os.remove(os.path.join(directory, file))

    remove_old_files('static')

    # 3/4フェーズの平均角度を計算
    if three_quarter_angles1:
        average_three_quarter_angle1 = np.mean(three_quarter_angles1)
    else:
        average_three_quarter_angle1 = None
    if three_quarter_angles2:
        average_three_quarter_angle2 = np.mean(three_quarter_angles2)
    else:
        average_three_quarter_angle2 = None

    # 最大角度を計算
    max_elbow_angle1 = max([angle for angle in elbow_angles1])
    max_knee_angle1 = max([angle for angle in knee_angles1])
    max_elbow_angle2 = max([angle for angle in elbow_angles2])
    max_knee_angle2 = max([angle for angle in knee_angles2])

    # 結果
    return render_template('result.html', 
                           video_url1=url_for('static', filename=result_filename1), 
                           video_url2=url_for('static', filename=result_filename2),
                           max_elbow_angle1=max_elbow_angle1,
                           max_knee_angle1=max_knee_angle1,
                           max_elbow_angle2=max_elbow_angle2,
                           max_knee_angle2=max_knee_angle2,
                           average_three_quarter_angle1=average_three_quarter_angle1,
                           average_three_quarter_angle2=average_three_quarter_angle2)

if __name__ == "__main__":
    app.run(debug=True)