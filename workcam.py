import cv2
import time
import os
import urllib.request
import numpy as np

# --- CONFIGURAÇÃO ---
BUFFER_SEGURANCA = 5.0    # Tempo (s) que tolera falha na detecção
INTERVALO_LOOP = 0.5      # Dorme para economizar CPU
CONFIANCA_MINIMA = 0.5    # 50% de certeza para considerar que é um rosto

# --- MÓDULO 1: SETUP AUTOMÁTICO (Baixa o cérebro da IA se não existir) ---
def verificar_modelos():
    arquivos = {
        "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "res10_300x300_ssd_iter_140000.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    }
    print("--- Verificando modelos de IA ---")
    for nome, url in arquivos.items():
        if not os.path.exists(nome):
            print(f"Baixando {nome} (apenas 1x)...")
            try:
                urllib.request.urlretrieve(url, nome)
                print("Download concluído.")
            except Exception as e:
                print(f"Falha ao baixar {nome}. Verifique internet.\nErro: {e}")
                exit()
    print("Modelos carregados.")

verificar_modelos()

# --- MÓDULO 2: INICIALIZAÇÃO DA REDE NEURAL ---
# Carrega a rede neural profunda (DNN) do disco
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
cap = cv2.VideoCapture(0)

# Variáveis de Estado
ultimo_visto = time.time()
tempo_sessao = 0
status = "AUSENTE"

print(f"\n--- MONITOR DNN INICIADO ---")
print("Este modelo funciona bem com óculos e pouca luz.")
print("Pressione 'q' na janela ou Ctrl+C para encerrar.\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret: break

        # Prepara a imagem para a Rede Neural (Blob)
        # Redimensiona para 300x300 e normaliza cores
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))

        # Detecção
        net.setInput(blob)
        detections = net.forward()

        rosto_encontrado = False
        
        # Loop sobre as detecções (a rede pode achar várias coisas, filtramos por confiança)
        for i in range(0, detections.shape[2]):
            confianca = detections[0, 0, i, 2]

            if confianca > CONFIANCA_MINIMA:
                rosto_encontrado = True
                
                # Desenha quadrado (apenas visual)
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Borda verde mais fina e elegante
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                texto_score = f"{confianca * 100:.0f}%"
                cv2.putText(frame, texto_score, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        # --- LÓGICA DE TEMPO (Mantivemos a mesma lógica robusta) ---
        agora = time.time()
        
        if rosto_encontrado:
            ultimo_visto = agora
            
        tempo_ausente = agora - ultimo_visto

        if tempo_ausente < BUFFER_SEGURANCA:
            status = "TRABALHANDO"
            cor = (0, 255, 0)
            tempo_sessao += INTERVALO_LOOP # Soma o tempo do loop
            
            # Barra de Buffer (Amarela)
            if not rosto_encontrado:
                pct = 1 - (tempo_ausente / BUFFER_SEGURANCA)
                cv2.rectangle(frame, (20, 80), (20 + int(200 * pct), 90), (0, 255, 255), -1)
                cv2.putText(frame, "Sem sinal visual...", (20, 75), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
        else:
            status = "AUSENTE"
            cor = (0, 0, 255)

        # Exibição
        hh = int(tempo_sessao // 3600)
        mm = int((tempo_sessao % 3600) // 60)
        ss = int(tempo_sessao % 60)
        tempo_fmt = f"{hh:02d}h {mm:02d}m {ss:02d}s"

        cv2.putText(frame, f"{status} | {tempo_fmt}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
        
        cv2.imshow("Monitor DNN", frame)
        print(f"\r[{status}] Tempo: {tempo_fmt} ", end="")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        time.sleep(INTERVALO_LOOP)

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n\n--- FIM ---\nTempo Total: {hh:02d}h {mm:02d}m {ss:02d}s")