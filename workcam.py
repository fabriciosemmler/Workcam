import cv2
import time
import os
import urllib.request
import numpy as np
import keyboard
import winsound
import tkinter as tk  # Biblioteca de GUI nativa

# --- CONFIGURAÇÃO ---
META_HORAS = 0.004         # Sua meta
BUFFER_SEGURANCA = 5.0   # Tolerância
INTERVALO_LOOP = 0.5     # Loop (s)
CONFIANCA_MINIMA = 0.5   

# --- PREPARAÇÃO DO AMBIENTE ---
DIRETORIO_BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(DIRETORIO_BASE)

ARQUIVO_PROTO = "deploy.prototxt"
ARQUIVO_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
META_SEGUNDOS = META_HORAS * 3600 

# --- DOWNLOAD DE MODELOS ---
def verificar_modelos():
    arquivos = {
        ARQUIVO_PROTO: "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        ARQUIVO_MODEL: "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    }
    for nome, url in arquivos.items():
        if not os.path.exists(nome):
            try:
                urllib.request.urlretrieve(url, nome)
            except:
                exit()
verificar_modelos()

# --- SETUP DA GUI (TKINTER) ---
# Cria a janela, mas deixa ela escondida inicialmente
root = tk.Tk()
root.title("Workcam Monitor")
# Remove bordas e barra de título (Estilo Overlay)
root.overrideredirect(True)
# Mantém sempre no topo
root.attributes("-topmost", True)
# Cor de fundo padrão
root.configure(bg='black')

# Dimensões e Centralização da Janela
LARGURA, ALTURA = 400, 200
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_c = int((screen_width/2) - (LARGURA/2))
y_c = int((screen_height/2) - (ALTURA/2))
root.geometry(f"{LARGURA}x{ALTURA}+{x_c}+{y_c}")

# Elementos de Texto (Labels)
lbl_titulo = tk.Label(root, text="MONITORAMENTO", font=("Arial", 14), bg="black", fg="gray")
lbl_titulo.pack(pady=10)

lbl_tempo = tk.Label(root, text="00:00:00", font=("Arial", 35, "bold"), bg="black", fg="white")
lbl_tempo.pack()

lbl_meta = tk.Label(root, text=f"Meta: {META_HORAS}h", font=("Arial", 12), bg="black", fg="gray")
lbl_meta.pack(pady=5)

lbl_status = tk.Label(root, text="Ausente", font=("Arial", 10), bg="black", fg="red")
lbl_status.pack(pady=5)

# Esconde a janela ao iniciar
root.withdraw() 

# --- INICIALIZAÇÃO DA IA ---
print("--- MONITOR INVISÍVEL RODANDO ---")
print("Segure [F10] para ver o tempo.")
print("Pressione [Ctrl+C] no terminal para encerrar.")

net = cv2.dnn.readNetFromCaffe(ARQUIVO_PROTO, ARQUIVO_MODEL)
cap = cv2.VideoCapture(0)

# Variáveis
ultimo_visto = time.time()
tempo_sessao = 0
ja_comemorou = False
em_comemoracao = False

try:
    while True:
        # 1. ATUALIZA A GUI (Importante: sem mainloop, usamos update)
        root.update_idletasks()
        root.update()

        # 2. CAPTURA E DETECÇÃO (Rodando no fundo)
        ret, frame = cap.read()
        if not ret: break

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        rosto_encontrado = False
        for i in range(0, detections.shape[2]):
            confianca = detections[0, 0, i, 2]
            if confianca > CONFIANCA_MINIMA:
                rosto_encontrado = True
                break

        # 3. CONTABILIZAÇÃO
        agora = time.time()
        if rosto_encontrado: ultimo_visto = agora
        tempo_ausente = agora - ultimo_visto
        
        meta_atingida = tempo_sessao >= META_SEGUNDOS

        if tempo_ausente < BUFFER_SEGURANCA:
            tempo_sessao += INTERVALO_LOOP
            status_txt = "TRABALHANDO"
            status_fg = "#00FF00" # Verde Lime
        else:
            status_txt = "AUSENTE"
            status_fg = "#FF0000" # Vermelho

        # 4. LÓGICA DE EXIBIÇÃO
        tecla_f10 = keyboard.is_pressed('F10')
        
        # Formata o tempo
        hh = int(tempo_sessao // 3600)
        mm = int((tempo_sessao % 3600) // 60)
        ss = int(tempo_sessao % 60)
        str_tempo = f"{hh:02d}:{mm:02d}:{ss:02d}"

        # Cenário de Vitória
        if meta_atingida:
            if not ja_comemorou:
                winsound.PlaySound("SystemHand", winsound.SND_ALIAS | winsound.SND_ASYNC)
                ja_comemorou = True
                em_comemoracao = True
            
            # Muda estilo para vitória
            root.configure(bg='#1c1c1c') # Cinza escuro elegante
            lbl_titulo.config(text="PARABÉNS!", fg="#FFD700", bg='#1c1c1c')
            lbl_tempo.config(text=str_tempo, fg="#FFD700", bg='#1c1c1c')
            lbl_meta.config(text="META BATIDA!", fg="white", bg='#1c1c1c')
            lbl_status.config(text="Você é uma máquina.", fg="white", bg='#1c1c1c')
            
            # Mostra a janela forçadamente
            root.deiconify()
        
        # Cenário Normal (F10)
        elif tecla_f10:
            # Atualiza textos
            lbl_titulo.config(text="MONITORAMENTO", fg="gray", bg="black")
            lbl_tempo.config(text=str_tempo, fg="white", bg="black")
            lbl_status.config(text=status_txt, fg=status_fg, bg="black")
            root.configure(bg='black')
            
            # Mostra janela
            root.deiconify()
        
        # Esconde janela se não houver motivo para mostrar
        else:
            root.withdraw()

        # Pausa para não fritar CPU
        time.sleep(INTERVALO_LOOP)

except KeyboardInterrupt:
    print("\nEncerrando...")
finally:
    cap.release()
    root.destroy()