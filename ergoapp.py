#!/usr/bin/env python3
"""ErgoApp - Análise ergonômica automatizada (RULA) com MediaPipe Pose.

Uso:
  python ergoapp.py --image foto.jpg
  python ergoapp.py --video video.mp4 --out saida.mp4 --csv saida.csv
  python ergoapp.py --webcam            (tecla q para sair)
Opções RULA: --force 0-3 (carga/força)  --static (postura estática >1 min ou repetitiva)

Fluxo: imagem -> MediaPipe Pose (X,Y,Z) -> ângulos (NumPy) -> Tabelas A/B/C -> risco 1-7.
Os ângulos usam as coordenadas 3D (world landmarks), por isso funcionam em
fotos/vídeos de frente, costas e lado. A sobreposição é desenhada em 2D.
"""
import argparse, csv, math, time, urllib.request
from pathlib import Path
import cv2, numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

MODEL = Path(__file__).with_name("pose_landmarker_full.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/latest/pose_landmarker_full.task")
IDX = dict(nose=0, ear_l=7, ear_r=8, sh_l=11, sh_r=12, el_l=13, el_r=14, wr_l=15, wr_r=16,
           pk_l=17, pk_r=18, ix_l=19, ix_r=20, hip_l=23, hip_r=24, kn_l=25, kn_r=26,
           an_l=27, an_r=28)
BONES = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),
         (24,26),(26,28),(15,19),(16,20),(0,7),(0,8),(7,8)]

# Tabela A: [braço][antebraço] -> 8 valores (punho 1-4 x giro 1-2)
TA = [
 [[1,2,2,2,2,3,3,3],[2,2,2,2,3,3,3,3],[2,3,3,3,3,4,4,4]],
 [[2,3,3,3,3,4,4,4],[3,3,3,3,3,4,4,4],[3,4,4,4,4,4,5,5]],
 [[3,3,4,4,4,4,5,5],[3,4,4,4,4,4,5,5],[4,4,4,4,4,5,5,5]],
 [[4,4,4,4,4,5,5,5],[4,4,4,4,4,5,5,5],[4,4,4,5,5,5,6,6]],
 [[5,5,5,5,5,6,6,7],[5,6,6,6,6,6,7,7],[6,6,6,7,7,7,7,8]],
 [[7,7,7,7,7,8,8,9],[8,8,8,8,8,9,9,9],[9]*8]]
# Tabela B: [pescoço][tronco] -> (pernas 1, pernas 2)
TB = [
 [(1,3),(2,3),(3,4),(5,5),(6,6),(7,7)], [(2,3),(2,3),(4,5),(5,5),(6,7),(7,7)],
 [(3,3),(3,4),(4,5),(5,6),(6,7),(7,7)], [(5,5),(5,6),(6,7),(7,7),(7,7),(8,8)],
 [(7,7),(7,7),(7,8),(8,8),(8,8),(8,8)], [(8,8),(8,8),(8,8),(8,9),(9,9),(9,9)]]
# Tabela C: [A 1..8][B 1..7]
TC = [[1,2,3,3,4,5,5],[2,2,3,4,4,5,5],[3,3,3,4,4,5,6],[3,3,3,4,5,6,6],
      [4,4,4,5,6,7,7],[4,4,5,6,6,7,7],[5,5,6,6,7,7,7],[5,5,6,7,7,7,7]]
RISK = {1:"Aceitavel",2:"Aceitavel",3:"Investigar",4:"Investigar",
        5:"Mudar em breve",6:"Mudar em breve",7:"Mudar imediatamente"}
RISK_PT = {1:"Aceitável",2:"Aceitável",3:"Investigar mais",4:"Investigar mais",
           5:"Investigar e mudar em breve",6:"Investigar e mudar em breve",7:"Investigar e mudar já"}

def unit(v):
    n = np.linalg.norm(v); return v / n if n > 1e-9 else v
def ang(a, b, c):
    return math.degrees(math.acos(np.clip(unit(a-b) @ unit(c-b), -1, 1)))
def horiz(v):
    return unit(np.array([v[0], 0.0, v[2]]))
def yaw_between(a, b):
    return math.degrees(math.acos(np.clip(horiz(a) @ horiz(b), -1, 1)))

def detect_view(P, lm):
    torso = np.linalg.norm((P['sh_l']+P['sh_r'])/2 - (P['hip_l']+P['hip_r'])/2)
    if abs(P['sh_l'][0] - P['sh_r'][0]) / max(torso, 1e-6) < 0.35:
        return "lado"
    return "frente" if lm[11].x > lm[12].x else "costas"   # ombro esq. à direita da imagem = de frente

def body_frame(P):
    sh = (P['sh_l']+P['sh_r'])/2; hip = (P['hip_l']+P['hip_r'])/2
    up = unit(sh - hip)
    lat = P['sh_l'] - P['sh_r']; lat = unit(lat - (lat @ up) * up)
    fwd = unit(np.cross(lat, up))
    if fwd @ (P['nose'] - (P['ear_l']+P['ear_r'])/2) < 0: fwd = -fwd   # frente = lado do nariz
    lat = unit(np.cross(up, fwd))
    return dict(sh=sh, hip=hip, up=up, fwd=fwd, lat=lat)

def arm_scores(P, F, s, o):
    sh, el, wr = P['sh_'+s], P['el_'+s], P['wr_'+s]
    d = unit(el - sh)
    flex = math.degrees(math.atan2(d @ F['fwd'], -(d @ F['up'])))
    ua = 2 if flex < -20 else 1 if flex <= 20 else 2 if flex <= 45 else 3 if flex <= 90 else 4
    abd = math.degrees(math.atan2(abs(d @ F['lat']), -(d @ F['up'])))
    if abd > 30: ua += 1
    if o.get('raised'): ua += 1        # ombro elevado
    if o.get('supported'): ua -= 1     # braço apoiado
    ua = max(ua, 1)
    elbow = ang(sh, el, wr)
    la = 1 if 60 <= elbow <= 100 else 2
    out = F['lat'] * np.sign((sh - F['sh']) @ F['lat'])
    s_off = (wr - F['hip']) @ out; sw = np.linalg.norm(P['sh_l'] - P['sh_r'])
    if s_off < 0 or s_off > sw * 1.2: la = min(la + 1, 3)
    hand = (P['ix_'+s] + P['pk_'+s]) / 2
    dev = math.degrees(math.acos(np.clip(unit(hand - wr) @ unit(wr - el), -1, 1)))
    wrist = min((1 if dev < 5 else 2 if dev <= 15 else 3) + (1 if o.get('wdev') else 0), 4)  # +1 desvio radial/ulnar
    return dict(ua=ua, la=la, wr=wrist, tw=o.get('twist', 1)), dict(braco=flex, abducao=abd, cotovelo=elbow, punho=dev)

def cruzar_tabelas(ua, la, wr, tw, neck, trunk, legs, muscle=0, force=0):
    """Cruza as Tabelas A, B e C do RULA (passos da planilha oficial).
    ua 1-6 | la 1-3 | wr 1-4 | tw 1-2 | neck 1-6 | trunk 1-6 | legs 1-2 | muscle 0-1 | force 0-3"""
    pA = TA[min(ua, 6)-1][min(la, 3)-1][(min(wr, 4)-1)*2 + tw-1]   # Tabela A: braço x antebraço x punho x giro
    A = pA + muscle + force                                         # escore A (punho/braço)
    pB = TB[min(neck, 6)-1][min(trunk, 6)-1][legs-1]               # Tabela B: pescoço x tronco x pernas
    B = pB + muscle + force                                         # escore B (pescoço/tronco/pernas)
    C = TC[min(A, 8)-1][min(B, 7)-1]                                # Tabela C: A (1-8+) x B (1-7+)
    return dict(A=pA, B=pB, scoreA=A, scoreB=B, final=int(C), muscle=muscle, force=force)

def rula(P, lm, force=0, static=False, **o):
    view = detect_view(P, lm); F = body_frame(P)
    sides = ['l', 'r']
    if view == "lado": sides = ['l'] if lm[11].visibility >= lm[12].visibility else ['r']
    best = None
    for s in sides:
        sc, an = arm_scores(P, F, s, o)
        a = TA[min(sc['ua'],6)-1][sc['la']-1][(min(sc['wr'],4)-1)*2 + sc['tw']-1]
        if best is None or a > best[0]: best = (a, sc, an, s)
    A, sc, an, side = best
    # Pescoço
    h = unit((P['ear_l']+P['ear_r'])/2 - F['sh'])
    nflex = math.degrees(math.atan2(h @ F['fwd'], h @ F['up']))
    neck = 4 if nflex < -5 else 1 if nflex < 10 else 2 if nflex < 20 else 3
    if math.degrees(math.atan2(abs(h @ F['lat']), h @ F['up'])) > 15: neck += 1
    if yaw_between(P['ear_l']-P['ear_r'], P['sh_l']-P['sh_r']) > 20: neck += 1
    # Tronco
    t = F['up']; vup = np.array([0, -1.0, 0]); fh = horiz(F['fwd']); lh = horiz(F['lat'])
    tflex = math.degrees(math.atan2(t @ fh, t @ vup))
    trunk = 1 if abs(tflex) < 5 else 2 if abs(tflex) <= 20 else 3 if abs(tflex) <= 60 else 4
    if math.degrees(math.atan2(abs(t @ lh), t @ vup)) > 10: trunk += 1
    if yaw_between(P['sh_l']-P['sh_r'], P['hip_l']-P['hip_r']) > 20: trunk += 1
    # Pernas
    legs = 1 if abs(P['an_l'][1] - P['an_r'][1]) < 0.10 else 2
    neck, trunk = min(neck, 6), min(trunk, 6)
    cz = cruzar_tabelas(sc['ua'], sc['la'], sc['wr'], sc['tw'], neck, trunk, legs, 1 if static else 0, force)
    an.update(pescoco=nflex, tronco=tflex)
    return dict(view=view, side=side, **sc, neck=neck, trunk=trunk, legs=legs, **cz,
                risk=RISK_PT[cz['final']], angles={k: round(v, 1) for k, v in an.items()})

def extract(res):
    if not res.pose_landmarks: return None
    lm, wl = res.pose_landmarks[0], res.pose_world_landmarks[0]
    return {k: np.array([wl[i].x, wl[i].y, wl[i].z]) for k, i in IDX.items()}, lm

RISK_BGR = {1: (80, 200, 80), 2: (80, 200, 80), 3: (0, 200, 255), 4: (0, 200, 255), 5: (0, 140, 255), 6: (0, 140, 255), 7: (60, 60, 255)}

def draw(img, lm, r, skeleton=True):
    """Desenha SOMENTE o esqueleto (sem textos) sobre a imagem. skeleton=False deixa a imagem limpa."""
    if not skeleton: return img
    h, w = img.shape[:2]; col = RISK_BGR[r['final']]; t = max(2, w // 400)
    for a, b in BONES:
        cv2.line(img, (int(lm[a].x*w), int(lm[a].y*h)), (int(lm[b].x*w), int(lm[b].y*h)), col, t)
    for q in lm: cv2.circle(img, (int(q.x*w), int(q.y*h)), t + 1, (255, 255, 255), -1)
    return img

def with_panel(img, r):
    """Devolve a imagem COMPLETA com o painel RULA ao lado (à direita), sem nada por cima da imagem."""
    h, w = img.shape[:2]; s = max(h / 720, 1.0); pw = int(430 * s); lh = int(28 * s); H = max(h, lh * 20)
    canvas = np.full((H, w + pw, 3), 26, np.uint8); canvas[:h, :w] = img
    W, G = (255, 255, 255), (170, 170, 170)
    if r is None: lines = [("Nenhuma pessoa detectada", W)]
    else:
        c = RISK_BGR[r['final']]; an = r['angles']
        lines = [(f"RULA final: {r['final']}", c), (RISK[r['final']], c), (f"Vista: {r['view']}  Lado: {r['side'].upper()}", W), ("", W),
                 (f"Braco {r['ua']}  Antebraco {r['la']}", W), (f"Punho {r['wr']}  Giro {r['tw']}", W), (f"Tabela A: {r['A']} -> Escore A: {r['scoreA']}", W), ("", W),
                 (f"Pescoco {r['neck']}  Tronco {r['trunk']}  Pernas {r['legs']}", W), (f"Tabela B: {r['B']} -> Escore B: {r['scoreB']}", W), ("", W),
                 (f"Tabela C (A x B): {r['final']}", c), ("", W), ("Angulos (graus)", G),
                 (f"Braco {an['braco']}  Cotovelo {an['cotovelo']}", W), (f"Punho {an['punho']}  Pescoco {an['pescoco']}", W), (f"Tronco {an['tronco']}", W)]
    for k, (t, c) in enumerate(lines):
        cv2.putText(canvas, t, (w + int(16 * s), int(36 * s) + k * lh), cv2.FONT_HERSHEY_SIMPLEX, 0.62 * s, c, max(1, int(s)) + (1 if k == 0 else 0), cv2.LINE_AA)
    return canvas

def landmarker(mode):
    if not MODEL.exists():
        print("Baixando modelo do MediaPipe..."); urllib.request.urlretrieve(MODEL_URL, MODEL)
    opts = vision.PoseLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path=str(MODEL)),
                                        running_mode=mode, num_poses=1)
    return vision.PoseLandmarker.create_from_options(opts)

def opts(a):
    return dict(force=a.force, static=a.static, raised=a.raised, supported=a.supported, wdev=a.wrist_dev, twist=a.twist)

def run_image(a):
    img = cv2.imread(a.image)
    if img is None: raise SystemExit("Não foi possível abrir a imagem.")
    with landmarker(vision.RunningMode.IMAGE) as pl:
        res = pl.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    ex = extract(res)
    if not ex: raise SystemExit("Nenhuma pessoa detectada.")
    r = rula(*ex, **opts(a)); draw(img, ex[1], r); img = with_panel(img, r)
    out = a.out or str(Path(a.image).with_suffix("")) + "_rula.jpg"
    cv2.imwrite(out, img); print(r); print("Salvo em", out)
    if a.show: cv2.imshow("ErgoApp", img); cv2.waitKey(0)

def run_video(a):
    cap = cv2.VideoCapture(0 if a.webcam else a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30; writer = None; rows = []; n = 0; t0 = time.time()
    with landmarker(vision.RunningMode.VIDEO) as pl:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok: break
            ts = int((time.time()-t0)*1000) if a.webcam else int(n*1000/fps); n += 1
            res = pl.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts)
            ex = extract(res)
            if ex:
                r = rula(*ex, **opts(a)); draw(frame, ex[1], r)
                rows.append(dict(frame=n, t=round(ts/1000, 2), view=r['view'], final=r['final'], A=r['scoreA'], B=r['scoreB'],
                                 braco=r['ua'], antebraco=r['la'], punho=r['wr'], pescoco=r['neck'], tronco=r['trunk'], pernas=r['legs']))
            frame = with_panel(frame, r if ex else None)
            if a.out and not a.webcam:
                if writer is None:
                    writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0]))
                writer.write(frame)
            if a.show or a.webcam:
                cv2.imshow("ErgoApp (q = sair)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release(); writer and writer.release(); cv2.destroyAllWindows()
    if rows:
        finals = [x['final'] for x in rows]
        print(f"Frames analisados: {len(rows)} | RULA máx: {max(finals)} | médio: {np.mean(finals):.2f}")
        if a.csv:
            with open(a.csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
            print("CSV salvo em", a.csv)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ErgoApp - RULA com MediaPipe")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image"); g.add_argument("--video"); g.add_argument("--webcam", action="store_true")
    p.add_argument("--out"); p.add_argument("--csv"); p.add_argument("--show", action="store_true")
    p.add_argument("--force", type=int, default=0, choices=[0,1,2,3]); p.add_argument("--static", action="store_true")
    p.add_argument("--raised", action="store_true", help="ombro elevado"); p.add_argument("--supported", action="store_true", help="braço apoiado")
    p.add_argument("--wrist-dev", action="store_true", help="desvio radial/ulnar do punho"); p.add_argument("--twist", type=int, default=1, choices=[1,2], help="giro do punho")
    a = p.parse_args()
    run_image(a) if a.image else run_video(a)
