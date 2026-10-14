# -*- coding: utf-8 -*-
"""
Ir Além (opção 3.1) - Detecção em tempo real de xícaras e tesouras.

Lê quadros de uma webcam do PC, de uma câmera ESP32-CAM (stream MJPEG via Wi-Fi)
ou de um arquivo de vídeo, e aplica o modelo YOLOv5s treinado na Entrega 1
(arquivo de pesos .pt, o "best.pt" do treino de 60 épocas).

Exemplos de uso (rode a partir da pasta ir_alem/src):
    python detector_webcam.py                              # webcam padrão (índice 0)
    python detector_webcam.py --fonte 1                    # segunda webcam
    python detector_webcam.py --fonte http://192.168.0.50:81/stream   # ESP32-CAM
    python detector_webcam.py --fonte video.mp4 --sem-janela --saida resultado.mp4

Teclas na janela: Q ou ESC = sair | S = salvar o quadro atual em PNG.
"""
import argparse                      # leitura dos parâmetros de linha de comando
import time                          # medição de tempo (FPS e latência)
from pathlib import Path             # caminhos independentes de sistema operacional

import cv2                           # captura de vídeo e desenho das caixas
import torch                         # carrega e executa o modelo YOLOv5

# Commit do YOLOv5 usado no treino da Entrega 1 (garante a mesma versão do código)
COMMIT_YOLOV5 = "f9578796b8a64889aa9e66dc06914d19f40d338e"
# Pesos padrão: modelo de 60 épocas, escolhido na validação da Entrega 1
PESOS_PADRAO = Path(__file__).resolve().parent.parent.parent / "assets" / "modelos" / "yolo_60ep.pt"
# Cores das caixas em BGR (padrão do OpenCV), uma por classe
CORES = [(60, 160, 255), (80, 200, 120)]


def ler_argumentos():
    """Define e lê os argumentos de linha de comando."""
    p = argparse.ArgumentParser(description="Detector YOLOv5 em tempo real (webcam / ESP32-CAM / vídeo)")
    p.add_argument("--pesos", default=str(PESOS_PADRAO), help="arquivo .pt treinado na Entrega 1")
    p.add_argument("--fonte", default="0", help="índice da webcam (0,1..), URL do stream ESP32-CAM ou arquivo de vídeo")
    p.add_argument("--conf", type=float, default=0.5, help="confiança mínima para exibir uma detecção")
    p.add_argument("--imgsz", type=int, default=320, help="tamanho de entrada (o mesmo do treino: 320)")
    p.add_argument("--saida", default=None, help="se informado, grava o vídeo anotado neste arquivo .mp4")
    p.add_argument("--pasta-prints", default="prints", help="pasta onde a tecla S salva os quadros")
    p.add_argument("--sem-janela", action="store_true", help="não abre janela (útil em servidor / teste)")
    p.add_argument("--max-quadros", type=int, default=0, help="para após N quadros (0 = sem limite)")
    return p.parse_args()


def carregar_modelo(pesos, conf):
    """Carrega o YOLOv5 customizado via torch.hub, na versão exata usada no treino."""
    modelo = torch.hub.load(f"ultralytics/yolov5:{COMMIT_YOLOV5}", "custom", path=pesos)
    modelo.conf = conf               # limiar de confiança
    modelo.eval()                    # modo de inferência
    return modelo


def abrir_fonte(fonte):
    """Abre a câmera. Texto numérico vira índice de webcam; o resto (URL/arquivo) é passado como está."""
    return cv2.VideoCapture(int(fonte) if fonte.isdigit() else fonte)


def desenhar(quadro, deteccoes, nomes):
    """Desenha caixa, nome da classe e confiança de cada detecção no quadro (BGR)."""
    for x1, y1, x2, y2, conf, cls in deteccoes:
        cls = int(cls)
        cor = CORES[cls % len(CORES)]
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(quadro, p1, p2, cor, 2)
        rotulo = f"{nomes[cls]} {float(conf):.2f}"
        cv2.putText(quadro, rotulo, (p1[0], max(p1[1] - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
    return quadro


def main():
    args = ler_argumentos()
    modelo = carregar_modelo(args.pesos, args.conf)
    nomes = modelo.names             # {0: 'xicara', 1: 'tesoura'} conforme o treino
    cap = abrir_fonte(args.fonte)
    if not cap.isOpened():
        raise SystemExit(f"Não foi possível abrir a fonte de vídeo: {args.fonte}")

    gravador = None                  # criado sob demanda quando --saida é usado
    Path(args.pasta_prints).mkdir(exist_ok=True)
    n, t_ini, lat_ms = 0, time.time(), []

    while True:
        ok, quadro = cap.read()      # lê um quadro da câmera
        if not ok:                   # fim do vídeo ou falha de conexão
            break
        t0 = time.perf_counter()
        # O YOLOv5 espera RGB; o OpenCV entrega BGR, por isso a conversão
        res = modelo(cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB), size=args.imgsz)
        deteccoes = res.xyxy[0].cpu().numpy()          # colunas: x1, y1, x2, y2, conf, classe
        lat_ms.append((time.perf_counter() - t0) * 1000)

        quadro = desenhar(quadro, deteccoes, nomes)
        n += 1
        fps = n / (time.time() - t_ini)
        cv2.putText(quadro, f"{fps:.1f} FPS | {lat_ms[-1]:.0f} ms", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if args.saida:               # grava o vídeo anotado, se solicitado
            if gravador is None:
                h, w = quadro.shape[:2]
                gravador = cv2.VideoWriter(args.saida, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
            gravador.write(quadro)

        if not args.sem_janela:
            cv2.imshow("FarmTech - Ir Alem (Q sai, S salva)", quadro)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), 27):
                break
            if tecla == ord("s"):    # salva o quadro anotado para usar como evidência
                arq = Path(args.pasta_prints) / f"print_{int(time.time())}.png"
                cv2.imwrite(str(arq), quadro)
                print("Quadro salvo em", arq)

        if args.max_quadros and n >= args.max_quadros:
            break

    cap.release()
    if gravador:
        gravador.release()
    cv2.destroyAllWindows()
    if lat_ms:
        print(f"Quadros: {n} | latência média do modelo: {sum(lat_ms)/len(lat_ms):.1f} ms | FPS médio: {fps:.1f}")


if __name__ == "__main__":
    main()
