import cv2
import numpy as np
import time
import kociemba
from flask import Flask, render_template, Response, jsonify, request

app = Flask(__name__)

# --- CONFIGURAZIONE ---
current_camera_id = 0 # ID iniziale (0 = PC, 1 = Telefono di solito)
camera = cv2.VideoCapture(current_camera_id)

# Stato globale del cubo
faces_order = ['Up', 'Right', 'Front', 'Down', 'Left', 'Back']
current_face_index = 0
cube_state = {} 

# Mappatura Colori (HSV)
color_ranges = [
    ([0, 0, 130], [180, 80, 255], 'W'),   # Bianco
    ([20, 60, 100], [40, 255, 255], 'Y'), # Giallo
    ([40, 70, 70], [90, 255, 255], 'G'),  # Verde
    ([90, 70, 70], [130, 255, 255], 'B'), # Blu
    ([10, 70, 100], [20, 255, 255], 'O'), # Arancio
    ([0, 70, 70], [10, 255, 255], 'R'),   # Rosso 1
    ([160, 70, 70], [180, 255, 255], 'R') # Rosso 2
]

def get_color_name(hsv_pixel):
    h, s, v = hsv_pixel
    if s < 60:
        if v > 100: return 'W'
        return 'W' 
    if 10 <= h < 22: return 'O'
    if 22 <= h < 38: return 'Y'
    if 38 <= h < 95: return 'G'
    if 95 <= h < 130: return 'B'
    if (0 <= h < 10) or (160 <= h <= 180): return 'R'
    return 'U' 

def draw_grid_and_extract(frame, extract=False):
    height, width, _ = frame.shape
    cx, cy = width // 2, height // 2
    step = 60 
    start_x = cx - (step * 1.5)
    start_y = cy - (step * 1.5)
    
    detected_colors = []
    
    for i in range(3): 
        for j in range(3): 
            x1 = int(start_x + j * step)
            y1 = int(start_y + i * step)
            x2 = int(x1 + step)
            y2 = int(y1 + step)
            
            roi = frame[y1+10:y2-10, x1+10:x2-10]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            avg_hsv = np.mean(hsv_roi, axis=(0,1))
            color_code = get_color_name(avg_hsv)
            
            rect_color = (0, 0, 255) if color_code == 'U' else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), rect_color, 2)
            
            # Scritta piccola di debug dentro il quadratino (lasciamo questa per utilità)
            cv2.putText(frame, color_code, (x1 + 20, y1 + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if extract:
                detected_colors.append(color_code)
    
    return frame, detected_colors

def generate_frames():
    global camera
    while True:
        success, frame = camera.read()
        if not success:
            # Se la camera non legge, riprova o manda frame vuoto
            time.sleep(0.1)
            continue
        
        frame, _ = draw_grid_and_extract(frame, extract=False)
        
        # NOTA: ABBIAMO RIMOSSO LA SCRITTA GIALLA GRANDE DA QUI
        # La gestiremo in HTML per avere un font migliore.

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# --- ROTTE FLASK ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/change_camera', methods=['POST'])
def change_camera():
    global camera, current_camera_id
    new_id = int(request.form.get('camera_id'))
    
    # Rilascia la vecchia camera e avvia la nuova
    camera.release()
    current_camera_id = new_id
    camera = cv2.VideoCapture(new_id)
    
    # Aspetta un attimo che si inizializzi
    time.sleep(1.0)
    
    if camera.isOpened():
        return jsonify({'status': 'ok', 'msg': f'Camera {new_id} attivata'})
    else:
        # Se fallisce, prova a tornare alla 0
        camera = cv2.VideoCapture(0)
        return jsonify({'status': 'error', 'msg': 'Impossibile aprire la camera'})

@app.route('/solution_view')
def solution_view():
    return render_template('solution.html')

@app.route('/scan_face', methods=['POST'])
def scan_face():
    global current_face_index
    if current_face_index >= 6:
        return jsonify({'status': 'complete', 'msg': 'Tutte le facce scansionate!'})

    success, frame = camera.read()
    if success:
        _, colors = draw_grid_and_extract(frame, extract=True)
        if 'U' in colors:
            return jsonify({'status': 'error', 'msg': 'Colore non riconosciuto (U)!'})

        face_name = faces_order[current_face_index]
        cube_state[face_name] = colors
        current_face_index += 1
        
        return jsonify({'status': 'ok', 'face': face_name, 'colors': colors, 'next_index': current_face_index})
    return jsonify({'status': 'error', 'msg': 'Errore webcam'})

@app.route('/undo_last_face', methods=['POST'])
def undo_last_face():
    global current_face_index, cube_state
    if current_face_index > 0:
        current_face_index -= 1
        face_to_remove = faces_order[current_face_index]
        if face_to_remove in cube_state: del cube_state[face_to_remove]
        return jsonify({'status': 'ok', 'new_index': current_face_index, 'face_removed': face_to_remove})
    return jsonify({'status': 'error', 'msg': 'Nessuna faccia da annullare'})

@app.route('/solve', methods=['POST'])
def solve():
    if len(cube_state) < 6: return jsonify({'status': 'error', 'msg': 'Scansiona prima tutte le facce!'})
    try:
        centers = {
            'U': cube_state['Up'][4], 'R': cube_state['Right'][4], 'F': cube_state['Front'][4],
            'D': cube_state['Down'][4], 'L': cube_state['Left'][4], 'B': cube_state['Back'][4]
        }
        if len(set(centers.values())) < 6: return jsonify({'status': 'error', 'msg': 'Errore Centri duplicati!'})

        center_map_inv = {v: k for k, v in centers.items()}
        raw_string = ""
        for face in faces_order: raw_string += "".join(cube_state[face])
            
        kociemba_string = ""
        for char in raw_string: kociemba_string += center_map_inv[char]
            
        solution = kociemba.solve(kociemba_string)
        return jsonify({'status': 'solved', 'solution': solution, 'face_colors': centers})
    except Exception as e:
        print(e)
        return jsonify({'status': 'error', 'msg': 'Errore algoritmo.'})

@app.route('/reset', methods=['POST'])
def reset():
    global current_face_index, cube_state
    current_face_index = 0
    cube_state = {}
    return jsonify({'status': 'reset'})

if __name__ == '__main__':
    app.run(debug=True)