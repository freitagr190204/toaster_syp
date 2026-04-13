import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import os
import datetime
import urllib.request
import time

# ================= KONFIGURATION =================
FILES = { "cfg": "yolov3-tiny.cfg", "weights": "yolov3-tiny.weights", "names": "coco.names" }
URLS = {
    "cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg",
    "weights": "https://pjreddie.com/media/files/yolov3-tiny.weights",
    "names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
}

# Dateiprüfung (nur falls noch nicht da)
if not os.path.exists("yolov3-tiny.weights"):
    print("Lade AI-Dateien...")
    for k, v in FILES.items():
        if not os.path.exists(v): urllib.request.urlretrieve(URLS[k], v)

# ================= GLOBALE VARIABLEN =================
cap = None
current_cam_index = 0
running = True
detect_objects = False
detect_faces = False
show_hud = True
recording = False
video_writer = None

# Standardwerte (Neutral)
mirror_mode = False 
brightness = 0
filter_index = 0
filter_modes = ["Normal", "Gray", "Night", "Edge"]

# Face & YOLO Init
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
try:
    net = cv2.dnn.readNet(FILES["weights"], FILES["cfg"])
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    with open(FILES["names"], "r") as f: classes = [line.strip() for line in f.readlines()]
except:
    print("Warnung: AI konnte nicht geladen werden.")
    net = None

# ================= KAMERA FUNKTIONEN (NEU & ROBUST) =================

def find_working_camera():
    """ Sucht automatisch nach einer funktionierenden Kamera """
    print("Suche funktionierende Kamera...")
    # Wir testen Index 0, 1 und 2
    priorities = [1, 0, 2] # Erst extern (1), dann intern (0)
    
    for idx in priorities:
        temp_cap = cv2.VideoCapture(idx) # KEIN DSHOW HIER (verursacht oft Fehler)
        if temp_cap.isOpened():
            # Test: Können wir wirklich ein Bild lesen?
            ret, frame = temp_cap.read()
            if ret and frame is not None:
                print(f"✅ Kamera {idx} funktioniert! Wird gestartet.")
                temp_cap.release()
                return idx
            else:
                print(f"❌ Kamera {idx} gibt nur schwarzes Bild.")
        temp_cap.release()
    
    print("⚠️ Keine Kamera gefunden. Versuche Standard 0.")
    return 0

def start_camera(index):
    global cap, current_cam_index
    if cap: cap.release()
    
    # WICHTIG: Hier kein CAP_DSHOW nutzen, wenn du schwarzes Bild hast!
    cap = cv2.VideoCapture(index)
    
    # Auflösung setzen (Vorsichtig)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    current_cam_index = index

def switch_cam():
    # Wechselt zwischen 0 und 1
    new_idx = 0 if current_cam_index == 1 else 1
    start_camera(new_idx)

# ================= BILD VERARBEITUNG =================

def process_frame(frame):
    # 1. Helligkeit (OpenCV optimiert)
    if brightness != 0:
        frame = cv2.convertScaleAbs(frame, alpha=1, beta=brightness)
    
    # 2. Spiegeln
    if mirror_mode:
        frame = cv2.flip(frame, 1)

    # 3. Filter
    mode = filter_modes[filter_index]
    if mode == "Gray":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif mode == "Night":
        frame[:, :, 0] = 0 # Blau weg
        frame[:, :, 2] = 0 # Rot weg
        frame = cv2.convertScaleAbs(frame, alpha=1, beta=60) # Grün heller
    elif mode == "Edge":
        frame = cv2.Canny(frame, 80, 150)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    return frame

def run_ai(frame):
    # Nur ausführen wenn eingeschaltet
    if detect_faces:
        small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x*2, y*2), ((x+w)*2, (y+h)*2), (0, 255, 0), 2)
            
    if detect_objects and net:
        h, w = frame.shape[:2]
        # 320x320 ist super schnell
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        
        boxes, confs, c_ids = [], [], []
        for out in outs:
            for det in out:
                scores = det[5:]
                class_id = np.argmax(scores)
                conf = scores[class_id]
                if conf > 0.5:
                    cx, cy = int(det[0]*w), int(det[1]*h)
                    bw, bh = int(det[2]*w), int(det[3]*h)
                    boxes.append([int(cx-bw/2), int(cy-bh/2), bw, bh])
                    confs.append(float(conf))
                    c_ids.append(class_id)
        
        indices = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, bw, bh = boxes[i]
                label = f"{classes[c_ids[i]]} {int(confs[i]*100)}%"
                cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 255), 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return frame

# ================= GUI LOOP =================
root = tk.Tk()
root.title("Kamera Fix Version")
video_label = tk.Label(root, bg="#111")
video_label.pack(fill="both", expand=True)

# Starten mit Auto-Suche
start_index = find_working_camera()
start_camera(start_index)

def update():
    global video_writer
    if not running: return

    ret, frame = cap.read()
    
    if not ret or frame is None:
        print("⚠️ Warnung: Leeres Bild empfangen.")
        # Kurze Pause und neustart versuch, um Crash zu verhindern
        root.after(100, update) 
        return

    frame = process_frame(frame)
    frame = run_ai(frame)

    # HUD
    if show_hud:
        info = f"CAM: {current_cam_index} | [C] Wechseln | [O] AI: {detect_objects} | [D] Face: {detect_faces}"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Recording
    if recording:
        if video_writer is None:
            name = datetime.datetime.now().strftime("recordings/rec_%H%M%S.avi")
            h, w = frame.shape[:2]
            video_writer = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'XVID'), 20, (w, h))
        video_writer.write(frame)
        cv2.circle(frame, (w-30, 30), 10, (0, 0, 255), -1) # Roter Punkt

    # Anzeigen
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update)

# ================= TASTEN =================
def key_handler(e):
    global running, detect_objects, detect_faces, recording, video_writer, brightness, filter_index, mirror_mode
    k = e.keysym.lower()
    
    if k == 'escape': running = False; root.destroy()
    elif k == 'c': switch_cam() # WICHTIG: Kamera wechseln
    elif k == 'o': detect_objects = not detect_objects
    elif k == 'd': detect_faces = not detect_faces
    elif k == 'r': 
        recording = not recording
        if not recording and video_writer: 
            video_writer.release()
            video_writer = None
    elif k == 'plus': brightness += 10
    elif k == 'minus': brightness -= 10
    elif k == 'f': filter_index = (filter_index + 1) % len(filter_modes)
    elif k == 'm': mirror_mode = not mirror_mode

root.bind("<Key>", key_handler)
os.makedirs("recordings", exist_ok=True)

update()
root.mainloop()
cap.release()