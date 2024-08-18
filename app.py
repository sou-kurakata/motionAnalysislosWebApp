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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    # ファイル名の処理
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = secure_filename(file.filename)
    filename_time = f"{timestamp}_{filename}"
    filepath = os.path.join('static', filename_time)
    file.save(filepath)

    result_filename = f"result_{timestamp}_{filename}"
    result_filepath = os.path.join('static', result_filename)

    # ビデオ処理
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

    # 動画の数が増え過ぎないように処理
    if os.path.exists(filepath):
        os.remove(filepath)

    def remove_old_files(directory, keep_latest=4):
        files = sorted(os.listdir(directory), key=lambda x: os.path.getctime(os.path.join(directory, x)))
        if len(files) > keep_latest:
            for file in files[:-keep_latest]:
                os.remove(os.path.join(directory, file))

    remove_old_files('static')

    # 3/4フェーズの平均角度を計算
    if three_quarter_angles:
        average_three_quarter_angle = np.mean(three_quarter_angles)
    else:
        average_three_quarter_angle = None

    # 最大角度を計算
    max_elbow_angle = max([angle for angle in elbow_angles])
    max_knee_angle = max([angle for angle in knee_angles])

    # 最大値を比較
    conn = sqlite3.connect('angles.db')
    c = conn.cursor()
    c.execute('SELECT MAX(elbow_angle), MAX(knee_angle) FROM angles')
    row = c.fetchone()
    previous_max_elbow = row[0] if row[0] else 0
    previous_max_knee = row[1] if row[1] else 0
    
    if max_elbow_angle > previous_max_elbow or max_knee_angle > previous_max_knee:
        c.execute('INSERT INTO angles (elbow_angle, knee_angle) VALUES (?, ?)', 
                  (max_elbow_angle, max_knee_angle))
        conn.commit()
    
    conn.close()

    # 結果
    return render_template('result.html', 
                           video_url=url_for('static', filename=result_filename), 
                           max_elbow_angle=max_elbow_angle,
                           max_knee_angle=max_knee_angle,
                           previous_max_elbow=previous_max_elbow,
                           previous_max_knee=previous_max_knee,
                           average_three_quarter_angle=average_three_quarter_angle)

if __name__ == "__main__":
    app.run(debug=True)
