import cv2
import time
import os
import urllib.request
import numpy as np
import winsound
import tkinter as tk 
from pynput import keyboard 

# --- CONFIGURAÇÃO ---
META_HORAS = 36.0         
BUFFER_SEGURANCA = 5.0   
INTERVALO_LOOP = 0.5     
CONFIANCA_MINIMA = 0.5   

# --- PREPARAÇÃO DE ARQUIVOS ---
DIRETORIO_BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(DIRETORIO_BASE)
ARQUIVO_PROTO = "deploy.prototxt"
ARQUIVO_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
ARQUIVO_DADOS = "tempo_acumulado.txt" # Arquivo que guarda o banco de horas
META_SEGUNDOS = META_HORAS * 3600 

# --- VARIÁVEIS GLOBAIS ---
aviso_meta_fechado = False  
janela_visivel = False 
ctrl_pressionado = False # NOVA VARIÁVEL: Rastreador do Control

# --- LÓGICA DO INTERRUPTOR (TOGGLE) E RESET ---
def alternar_janela():
    global janela_visivel
    janela_visivel = not janela_visivel
    winsound.Beep(1000, 50) 

def zerar_contagem():
    """Função cirúrgica para reiniciar o ciclo da semana"""
    global tempo_sessao, ultimo_tempo_salvo, ja_comemorou, aviso_meta_fechado
    tempo_sessao = 0.0
    ultimo_tempo_salvo = 0.0
    ja_comemorou = False
    aviso_meta_fechado = False
    
    # Sobrescreve o arquivo com zero
    with open(ARQUIVO_DADOS, "w") as f:
        f.write("0.0")
        
    # Feedback sonoro duplo para confirmar o reset
    winsound.Beep(500, 150)
    time.sleep(0.1)
    winsound.Beep(500, 150)

def on_press(key):
    global ctrl_pressionado
    # Verifica se a tecla pressionada é o Ctrl (esquerdo ou direito)
    if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
        ctrl_pressionado = True
    elif key == keyboard.Key.f10:
        if ctrl_pressionado:
            zerar_contagem() # Ctrl + F10
        else:
            alternar_janela() # Apenas F10

def on_release(key):
    global ctrl_pressionado
    # Quando soltar o Ctrl, desativa a flag
    if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
        ctrl_pressionado = False

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# --- GUI (TKINTER) ---
root = tk.Tk()
root.overrideredirect(True) 
root.attributes("-topmost", True) 
root.configure(bg='black')

LARGURA, ALTURA = 400, 220 # Altura ajustada para caber a meta
x_c = int((root.winfo_screenwidth()/2) - (LARGURA/2))
y_c = int((root.winfo_screenheight()/2) - (ALTURA/2))
root.geometry(f"{LARGURA}x{ALTURA}+{x_c}+{y_c}")

def fechar_janela_meta():
    global aviso_meta_fechado, janela_visivel
    aviso_meta_fechado = True
    janela_visivel = False 
    root.withdraw() 

btn_fechar = tk.Button(root, text="X", command=fechar_janela_meta, bg="red", fg="white", bd=0)
btn_fechar.place(x=LARGURA-30, y=0, width=30, height=30)

lbl_titulo = tk.Label(root, text="MONITORAMENTO", font=("Arial", 14), bg="black", fg="gray")
lbl_titulo.pack(pady=15)

lbl_tempo = tk.Label(root, text="00:00:00", font=("Arial", 35, "bold"), bg="black", fg="white")
lbl_tempo.pack()

lbl_status = tk.Label(root, text="Ausente", font=("Arial", 10), bg="black", fg="red")
lbl_status.pack(pady=5)

# Rótulo da Meta (Adicionado abaixo do Status)
lbl_meta = tk.Label(root, text=f"Meta: {META_HORAS}h (0.0%)", font=("Arial", 12), bg="black", fg="gray")
lbl_meta.pack(pady=5)

root.withdraw() 

# --- IA ---
net = cv2.dnn.readNetFromCaffe(ARQUIVO_PROTO, ARQUIVO_MODEL)
cap = cv2.VideoCapture(0)

ultimo_visto = time.time()
tempo_sessao = 0.0

# Carrega o banco de horas se o arquivo já existir
if os.path.exists(ARQUIVO_DADOS):
    with open(ARQUIVO_DADOS, "r") as f:
        tempo_sessao = float(f.read().strip() or 0.0)

ultimo_tempo_salvo = tempo_sessao # Controle para economizar gravações no disco
ja_comemorou = False

ultimo_processamento_ia = 0
status_txt = "INICIANDO..."
status_fg = "gray"
str_tempo = "00:00:00"

print("--- MONITOR ATIVO (F10 para alternar) ---")

try:
    while True:
        root.update()
        agora = time.time()

        if agora - ultimo_processamento_ia > INTERVALO_LOOP:
            ultimo_processamento_ia = agora 
            ret, frame = cap.read()
            if ret:
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                net.setInput(blob)
                detections = net.forward()

                rosto_encontrado = any(detections[0, 0, i, 2] > CONFIANCA_MINIMA for i in range(detections.shape[2]))

                if rosto_encontrado: ultimo_visto = agora
                tempo_ausente = agora - ultimo_visto
                
                if tempo_ausente < BUFFER_SEGURANCA:
                    tempo_sessao += INTERVALO_LOOP 
                    status_txt, status_fg = "TRABALHANDO", "#00FF00"
                    
                    # Salva no disco a cada 10s de trabalho acumulado (Econômico)
                    if tempo_sessao - ultimo_tempo_salvo >= 10:
                        with open(ARQUIVO_DADOS, "w") as f:
                            f.write(str(tempo_sessao))
                        ultimo_tempo_salvo = tempo_sessao
                else:
                    status_txt, status_fg = "AUSENTE", "#FF0000"

                hh, mm, ss = int(tempo_sessao//3600), int((tempo_sessao%3600)//60), int(tempo_sessao%60)
                str_tempo = f"{hh:02d}:{mm:02d}:{ss:02d}"
                meta_atingida = tempo_sessao >= META_SEGUNDOS
                pct_concluida = (tempo_sessao / META_SEGUNDOS) * 100

        deve_mostrar = janela_visivel or (meta_atingida and not aviso_meta_fechado)
        
        if deve_mostrar:
            if meta_atingida:
                # Modo Vitória (Gold)
                if not ja_comemorou:
                    winsound.PlaySound("SystemHand", winsound.SND_ALIAS | winsound.SND_ASYNC)
                    ja_comemorou = True
                
                bg_victory = '#1c1c1c'
                fg_gold = "#FFD700"
                root.configure(bg=bg_victory)
                lbl_titulo.config(text="PARABÉNS!", fg=fg_gold, bg=bg_victory)
                lbl_tempo.config(text=str_tempo, fg=fg_gold, bg=bg_victory)
                lbl_status.config(text="Missão cumprida.", fg="white", bg=bg_victory)
                lbl_meta.config(text="META BATIDA!", fg="white", bg=bg_victory)
            else:
                # Modo Normal
                root.configure(bg='black')
                lbl_titulo.config(text="MONITORAMENTO", fg="gray", bg="black")
                lbl_tempo.config(text=str_tempo, fg="white", bg="black")
                lbl_status.config(text=status_txt, fg=status_fg, bg="black")
                lbl_meta.config(text=f"Meta: {META_HORAS}h ({pct_concluida:.1f}%)", fg="gray", bg="black")

            root.deiconify()
        else:
            root.withdraw()

        time.sleep(0.05)
except KeyboardInterrupt:
    pass
finally:
    # Salva o tempo exato antes de fechar a aplicação
    with open(ARQUIVO_DADOS, "w") as f:
        f.write(str(tempo_sessao))
    cap.release()
    root.destroy()