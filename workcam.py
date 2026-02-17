import cv2
import time
import os
import urllib.request
import numpy as np

# --- SOLUÇÃO DE CAMINHOS (Blindagem) ---
DIRETORIO_BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(DIRETORIO_BASE)

ARQUIVO_PROTO = "deploy.prototxt"
ARQUIVO_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"

# --- CONFIGURAÇÃO DO USUÁRIO ---
META_HORAS = 0.008         # <--- SUA META AQUI (pode usar quebrados, ex: 6.5)
BUFFER_SEGURANCA = 5.0   # Tolerância (s) para falha na detecção/piscar
INTERVALO_LOOP = 0.5     # Intervalo para poupar CPU
CONFIANCA_MINIMA = 0.5   # Sensibilidade da IA

# Conversão interna para segundos
META_SEGUNDOS = META_HORAS * 3600 

# --- MÓDULO 1: SETUP AUTOMÁTICO ---
def verificar_modelos():
    arquivos = {
        ARQUIVO_PROTO: "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        ARQUIVO_MODEL: "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    }
    print(f"--- Verificando diretório: {os.getcwd()} ---")
    for nome, url in arquivos.items():
        if not os.path.exists(nome):
            print(f"Baixando {nome}...")
            try:
                urllib.request.urlretrieve(url, nome)
            except Exception as e:
                print(f"Erro ao baixar {nome}: {e}")
                exit()
    print("Sistemas prontos.")

verificar_modelos()

# --- MÓDULO 2: INICIALIZAÇÃO ---
print("Carregando Rede Neural...")
net = cv2.dnn.readNetFromCaffe(ARQUIVO_PROTO, ARQUIVO_MODEL)
cap = cv2.VideoCapture(0)

# Variáveis de Estado
ultimo_visto = time.time()
tempo_sessao = 0
status = "AUSENTE"
hh, mm, ss = 0, 0, 0 

# Cores (BGR)
COR_VERDE = (0, 255, 0)
COR_VERMELHA = (0, 0, 255)
COR_AMARELA = (0, 255, 255)
COR_DOURADA = (0, 215, 255) # Ouro para a vitória

print(f"\n--- MONITOR DE PRODUTIVIDADE ---")
print(f"Meta de hoje: {META_HORAS} horas.")
print("Pressione 'q' na janela para encerrar.\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret: break

        # Prepara IA
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()

        rosto_encontrado = False
        
        # Analisa detecções
        for i in range(0, detections.shape[2]):
            confianca = detections[0, 0, i, 2]
            if confianca > CONFIANCA_MINIMA:
                rosto_encontrado = True
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Se meta batida, a caixa fica dourada, senão verde
                cor_box = COR_DOURADA if tempo_sessao >= META_SEGUNDOS else COR_VERDE
                
                cv2.rectangle(frame, (startX, startY), (endX, endY), cor_box, 2)
                
                # Mostra % confiança (discreto)
                text = f"{confianca*100:.0f}%"
                cv2.putText(frame, text, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, cor_box, 2)

        # --- LÓGICA TEMPORAL ---
        agora = time.time()
        
        if rosto_encontrado:
            ultimo_visto = agora
            
        tempo_ausente = agora - ultimo_visto

        # Verifica Meta
        meta_atingida = tempo_sessao >= META_SEGUNDOS

        if tempo_ausente < BUFFER_SEGURANCA:
            tempo_sessao += INTERVALO_LOOP
            
            if meta_atingida:
                status = "META BATIDA!"
                cor = COR_DOURADA
            else:
                status = "TRABALHANDO"
                cor = COR_VERDE
            
            # Barra de Buffer (Amarela) se perder o rosto momentaneamente
            if not rosto_encontrado:
                pct = 1 - (tempo_ausente / BUFFER_SEGURANCA)
                cv2.rectangle(frame, (20, 100), (20 + int(200 * pct), 110), COR_AMARELA, -1)
                cv2.putText(frame, "Procurando...", (20, 95), cv2.FONT_HERSHEY_PLAIN, 1, COR_AMARELA, 1)
        else:
            status = "AUSENTE"
            cor = COR_VERMELHA

        # --- EXIBIÇÃO (HUD) ---
        hh = int(tempo_sessao // 3600)
        mm = int((tempo_sessao % 3600) // 60)
        ss = int(tempo_sessao % 60)
        tempo_fmt = f"{hh:02d}h {mm:02d}m {ss:02d}s"

        # Texto Principal
        cv2.putText(frame, f"{status}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)
        cv2.putText(frame, f"{tempo_fmt} / {META_HORAS}h", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)

        # Mensagem Gigante de Vitória
        if meta_atingida and rosto_encontrado:
            texto_vit = "PARABENS!"
            # Centraliza o texto
            t_size = cv2.getTextSize(texto_vit, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
            t_x = (w - t_size[0]) // 2
            t_y = (h + t_size[1]) // 2
            cv2.putText(frame, texto_vit, (t_x, t_y), cv2.FONT_HERSHEY_SIMPLEX, 2, COR_DOURADA, 3)

        cv2.imshow("Monitor Workcam", frame)
        print(f"\r[{status}] {tempo_fmt}", end="")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        time.sleep(INTERVALO_LOOP)

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n\n--- RELATORIO FINAL ---\nTempo Total: {hh:02d}h {mm:02d}m {ss:02d}s")