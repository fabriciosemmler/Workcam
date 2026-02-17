import cv2
import time
import os
import urllib.request
import numpy as np

# --- SOLUÇÃO DO ERRO DE CAMINHO ---
# 1. Descobre onde o script está
DIRETORIO_BASE = os.path.dirname(os.path.abspath(__file__))
# 2. Força o Python a "entrar" nessa pasta
os.chdir(DIRETORIO_BASE)

# Nomes dos arquivos (agora sem caminho completo, pois já estamos na pasta)
ARQUIVO_PROTO = "deploy.prototxt"
ARQUIVO_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"

# --- CONFIGURAÇÃO ---
BUFFER_SEGURANCA = 5.0   # Tolerância (s) para falha na detecção
INTERVALO_LOOP = 0.5     # Intervalo para poupar CPU
CONFIANCA_MINIMA = 0.5   # 50% de certeza

# --- MÓDULO 1: SETUP AUTOMÁTICO ---
def verificar_modelos():
    arquivos = {
        ARQUIVO_PROTO: "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        ARQUIVO_MODEL: "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    }
    
    print(f"--- Verificando diretório: {os.getcwd()} ---")
    
    for nome_arquivo, url in arquivos.items():
        if not os.path.exists(nome_arquivo):
            print(f"Baixando {nome_arquivo}...")
            try:
                urllib.request.urlretrieve(url, nome_arquivo)
                print("Download concluído.")
            except Exception as e:
                print(f"Erro fatal ao baixar {nome_arquivo}: {e}")
                exit()
    print("Modelos carregados.")

verificar_modelos()

# --- MÓDULO 2: INICIALIZAÇÃO ---
print("Inicializando Rede Neural...")
net = cv2.dnn.readNetFromCaffe(ARQUIVO_PROTO, ARQUIVO_MODEL)
cap = cv2.VideoCapture(0)

# Variáveis de Estado
ultimo_visto = time.time()
tempo_sessao = 0
status = "AUSENTE"
hh, mm, ss = 0, 0, 0 # Inicialização segura

print(f"\n--- MONITOR DNN ATIVO ---")
print("Detector robusto (óculos/luz).")
print("Pressione 'q' na janela ou Ctrl+C aqui para encerrar.\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro ao ler câmera.")
            break

        # Prepara a imagem para a Rede Neural (Blob)
        (h, w) = frame.shape[:2]
        # O resize para 300x300 é exigência deste modelo específico
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))

        # Detecção
        net.setInput(blob)
        detections = net.forward()

        rosto_encontrado = False
        maior_confianca = 0
        
        # Analisa as detecções
        for i in range(0, detections.shape[2]):
            confianca = detections[0, 0, i, 2]

            if confianca > CONFIANCA_MINIMA:
                rosto_encontrado = True
                
                # Guarda a maior confiança para exibir
                if confianca > maior_confianca:
                    maior_confianca = confianca

                # Desenha quadrado visual
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Retângulo Verde
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                
                # Mostra % de certeza
                texto = f"{confianca * 100:.0f}%"
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(frame, texto, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        # --- LÓGICA DE TEMPO ---
        agora = time.time()
        
        if rosto_encontrado:
            ultimo_visto = agora
            
        tempo_ausente = agora - ultimo_visto

        if tempo_ausente < BUFFER_SEGURANCA:
            status = "TRABALHANDO"
            cor = (0, 255, 0) # Verde
            tempo_sessao += INTERVALO_LOOP
            
            # Barra Amarela (Buffer) - Só aparece se perder o rosto temporariamente
            if not rosto_encontrado:
                pct = 1 - (tempo_ausente / BUFFER_SEGURANCA)
                cv2.rectangle(frame, (20, 80), (20 + int(200 * pct), 90), (0, 255, 255), -1)
                cv2.putText(frame, "Sem sinal...", (20, 75), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
        else:
            status = "AUSENTE"
            cor = (0, 0, 255) # Vermelho

        # Exibição
        hh = int(tempo_sessao // 3600)
        mm = int((tempo_sessao % 3600) // 60)
        ss = int(tempo_sessao % 60)
        tempo_fmt = f"{hh:02d}h {mm:02d}m {ss:02d}s"

        cv2.putText(frame, f"{status} | {tempo_fmt}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
        
        cv2.imshow("Monitor Workcam", frame)
        print(f"\r[{status}] Tempo: {tempo_fmt} ", end="")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        time.sleep(INTERVALO_LOOP)

except KeyboardInterrupt:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n\n--- RELATORIO FINAL ---\nTempo Total: {hh:02d}h {mm:02d}m {ss:02d}s")