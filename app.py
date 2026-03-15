import os
import cv2
import glob
import time
from datetime import datetime
from flask import Flask, render_template, request, send_from_directory
import provided_algorithm as provided_algo # 파일명 확인!

app = Flask(__name__)

BASE_DATA_DIR = 'data'
RESULT_FOLDER = 'results'
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 통합 로그 저장소
inspection_logs = []

@app.route('/')
def index():
    # [핵심] data 폴더 내의 이미지 목록을 다시 불러오는 로직입니다.
    image_paths = glob.glob(os.path.join(BASE_DATA_DIR, '**', '*.[jp][pn][g]'), recursive=True)
    images = [os.path.relpath(path, BASE_DATA_DIR) for path in image_paths]
    # images 변수를 반드시 templates로 넘겨줘야 목록이 뜹니다.
    return render_template('index.html', images=images)

@app.route('/get_logs')
def get_logs():
    return {"logs": inspection_logs}

@app.route('/inspect/<path:filename>')
def inspect(filename):
    start_time = time.time()
    try:
        safe_filename = filename.replace('/', os.sep)
        img_path = os.path.join(BASE_DATA_DIR, safe_filename)
        input_img = cv2.imread(img_path)

        if input_img is None:
            return {"status": "Error", "message": "Image not found"}

        # 알고리즘 실행 (중앙 영역 100x100 조각)
        h, w = input_img.shape[:2]
        template_all = input_img.copy()
        templates = [input_img[h//2:h//2+100, w//2:w//2+100]]
        top_left_points = [(w//2, h//2)]

        cropped_rois, scores, result_img = provided_algo.template_matching(
            input_img, template_all, templates, RESULT_FOLDER, top_left_points
        )

        avg_score = sum(scores) / len(scores) if scores else 0
        status = "PASS" if avg_score > 0.8 else "NG"
        
        res_name = f"res_{filename.replace(os.sep, '_')}"
        cv2.imwrite(os.path.join(RESULT_FOLDER, res_name), result_img)

        res_data = {
            "client": request.remote_addr,
            "timestamp": datetime.now().strftime('%H:%M:%S'),
            "status": status,
            "score": round(avg_score, 4),
            "processing_time": f"{round(time.time() - start_time, 3)}s",
            "image_path": filename,
            "result_url": f"/results/{res_name}"
        }
        inspection_logs.insert(0, res_data)
        return res_data
    except Exception as e:
        return {"status": "Error", "message": str(e)}

@app.route('/results/<filename>')
def get_result_image(filename):
    return send_from_directory(RESULT_FOLDER, filename)

@app.route('/data_img/<path:filename>')
def get_data_image(filename):
    return send_from_directory(BASE_DATA_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)