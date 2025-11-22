import cv2
import numpy as np
import kociemba
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

# --- CONFIGURAZIONE ---
camera = cv2.VideoCapture(0) # 0 è solitamente la webcam integrata

# Stato globale del cubo
# Ordine standard Kociemba: U, R, F, D, L, B
faces_order = ['Up', 'Right', 'Front', 'Down', 'Left', 'Back']
current_face_index = 0
cube_state = {} # Dizionario per salvare i colori: 'Up': ['W','W','W'...]

# Mappatura Colori (HSV ranges) - DA TARARE SE NECESSARIO
# Formato: [Hue_min, Sat_min, Val_min], [Hue_max, Sat_max, Val_max], 'NomeColore'
# --- INIZIO MODIFICA in app.py ---

# Nuova mappatura colori più tollerante per il Rosso
# [H_min, S_min, V_min], [H_max, S_max, V_max], 'Label'
color_ranges = [
    # BIANCO: Saturation molto bassa. Value (Luminosità) alta.
    ([0, 0, 130], [180, 80, 255], 'W'),

    # GIALLO: Hue tra 20 e 40.
    ([20, 60, 100], [40, 255, 255], 'Y'),

    # VERDE: Hue tra 40 e 90. 
    # Alziamo un po' la saturazione minima per non confonderlo con il bianco sporco
    ([40, 70, 70], [90, 255, 255], 'G'),

    # BLU: Hue tra 90 e 130.
    ([90, 70, 70], [130, 255, 255], 'B'),

    # ARANCIONE: Hue tra 10 e 20. 
    ([10, 70, 100], [20, 255, 255], 'O'),

    # ROSSO (Range 1): Hue vicino allo 0
    ([0, 70, 70], [10, 255, 255], 'R'),
    
    # ROSSO (Range 2): Hue vicino al 180 (la fine del cerchio cromatico)
    ([160, 70, 70], [180, 255, 255], 'R')
]

def get_color_name(hsv_pixel):
    h, s, v = hsv_pixel
    
    # 1. Controllo Prioritario per il BIANCO e il NERO
    # Se la saturazione è molto bassa, è probabilmente bianco (o grigio/nero se c'è poca luce)
    if s < 60:
        if v > 100: return 'W' # Bianco
        # Se V è basso e S è basso, è buio, ma assumiamo bianco sporco per ora
        return 'W' 

    # 2. Controllo Colori Standard
    # Arancione (stretto per non confondersi col rosso o giallo)
    if 10 <= h < 22: return 'O'
    
    # Giallo
    if 22 <= h < 38: return 'Y'
    
    # Verde
    if 38 <= h < 95: return 'G'
    
    # Blu (Attenzione: il blu scuro a volte sembra rosso su webcam scarse)
    if 95 <= h < 130: return 'B'
    
    # Rosso (Gestisce i due estremi dello spettro: 0-10 e 160-180)
    if (0 <= h < 10) or (160 <= h <= 180): return 'R'
    
    # Se non rientra in nessuno, cerchiamo il "più vicino"
    return 'U' 

# --- FINE MODIFICA ---
def draw_grid_and_extract(frame, extract=False):
    height, width, _ = frame.shape
    # Calcola centro
    cx, cy = width // 2, height // 2
    step = 60 # Dimensione quadratino
    start_x = cx - (step * 1.5)
    start_y = cy - (step * 1.5)
    
    detected_colors = []
    
    # Loop 3x3
    for i in range(3): # Righe
        for j in range(3): # Colonne
            x1 = int(start_x + j * step)
            y1 = int(start_y + i * step)
            x2 = int(x1 + step)
            y2 = int(y1 + step)
            
            # 1. Analisi SEMPRE attiva per visualizzazione
            # Prendi l'area centrale del quadratino
            roi = frame[y1+10:y2-10, x1+10:x2-10]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            avg_hsv = np.mean(hsv_roi, axis=(0,1))
            color_code = get_color_name(avg_hsv)
            
            # 2. Colore del contorno: ROSSO se Unknown, VERDE se OK
            rect_color = (0, 0, 255) if color_code == 'U' else (0, 255, 0)
            
            # Disegna rettangolo
            cv2.rectangle(frame, (x1, y1), (x2, y2), rect_color, 2)
            
            # 3. SCRIVI IL COLORE A VIDEO (Debug Visivo)
            # Scrive W, R, G, U... al centro del quadratino
            cv2.putText(frame, color_code, (x1 + 20, y1 + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            if extract:
                detected_colors.append(color_code)
    
    return frame, detected_colors

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success: break
        
        # Disegna solo la griglia per visualizzazione
        frame, _ = draw_grid_and_extract(frame, extract=False)
        
        # Overlay testo istruzioni
        if current_face_index < 6:
            txt = f"Mostra faccia: {faces_order[current_face_index]}"
        else:
            txt = "Scansione Completa! Premi Risolvi."
            
        cv2.putText(frame, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/scan_face', methods=['POST'])
def scan_face():
    global current_face_index
    if current_face_index >= 6:
        return jsonify({'status': 'complete', 'msg': 'Tutte le facce scansionate!'})

    success, frame = camera.read()
    if success:
        _, colors = draw_grid_and_extract(frame, extract=True)
        
        # --- CONTROLLO DI SICUREZZA ---
        # Se c'è anche solo un colore 'U' (Unknown), rifiuta la scansione
        if 'U' in colors:
            return jsonify({
                'status': 'error', 
                'msg': 'Colore non riconosciuto (U)! Migliora la luce e riprova.'
            })
        # -------------------------------

        face_name = faces_order[current_face_index]
        cube_state[face_name] = colors
        
        current_face_index += 1
        
        return jsonify({
            'status': 'ok', 
            'face': face_name, 
            'colors': colors,
            'next_index': current_face_index
        })
    return jsonify({'status': 'error', 'msg': 'Errore webcam'})

@app.route('/solve', methods=['POST'])
def solve():
    if len(cube_state) < 6:
        return jsonify({'status': 'error', 'msg': 'Scansiona prima tutte le facce!'})

    try:
        # 1. COSTRUISCI LA MAPPA DEI COLORI BASATA SUI CENTRI
        # L'algoritmo deve sapere: "Il colore del centro UP corrisponde a 'U'"
        # Esempio: Se il centro della faccia UP è Giallo (Y), allora Y = U.
        
        center_map = {}
        
        # L'ordine delle facce in cube_state deve essere quello standard: U, R, F, D, L, B
        # Le chiavi sono: 'Up', 'Right', 'Front', 'Down', 'Left', 'Back'
        
        # Mappiamo il colore del centro (indice 4) di ogni faccia alla notazione Kociemba
        center_map[cube_state['Up'][4]]    = 'U'
        center_map[cube_state['Right'][4]] = 'R'
        center_map[cube_state['Front'][4]] = 'F'
        center_map[cube_state['Down'][4]]  = 'D'
        center_map[cube_state['Left'][4]]  = 'L'
        center_map[cube_state['Back'][4]]  = 'B'

        # Verifica: abbiamo trovato 6 colori unici per i 6 centri?
        if len(center_map) < 6:
             return jsonify({'status': 'error', 'msg': 'Errore Centri: Hai scansionato due facce con lo stesso colore centrale! Riprova.'})

        # 2. TRADUCI TUTTA LA STRINGA
        raw_string = ""
        for face in faces_order: # Up, Right, Front, Down, Left, Back
            raw_string += "".join(cube_state[face])
            
        # Sostituisci ogni colore (es. 'Y') con la sua posizione (es. 'U')
        kociemba_string = ""
        for char in raw_string:
            kociemba_string += center_map[char]
            
        print(f"Stringa tradotta per Kociemba: {kociemba_string}") # Debug nel terminale

        # 3. RISOLVI
        solution = kociemba.solve(kociemba_string)
        return jsonify({'status': 'solved', 'solution': solution})
        
    except Exception as e:
        print(e)
        return jsonify({'status': 'error', 'msg': 'Errore: Cubo impossibile. Controlla di aver scansionato nell\'ordine corretto e che i colori siano giusti.'})

@app.route('/reset', methods=['POST'])
def reset():
    global current_face_index, cube_state
    current_face_index = 0
    cube_state = {}
    return jsonify({'status': 'reset'})

if __name__ == '__main__':
    app.run(debug=True)