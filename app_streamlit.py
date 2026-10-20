"""ErgoApp com Streamlit + streamlit-webrtc.   Execução:  streamlit run app_streamlit.py"""
import tempfile, threading, time
import av, cv2, mediapipe as mp, numpy as np, pandas as pd, streamlit as st
from mediapipe.tasks.python import vision
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
from ergoapp import draw, extract, landmarker, rula
from exporters import COLS, RISCO, excel_bytes, linha_hist, passos, pdf_bytes

st.set_page_config(page_title="ErgoApp - RULA", layout="wide")
st.title("ErgoApp · Análise ergonômica RULA")
sb = st.sidebar; sb.header("Identificação")
IDENT = {k: sb.text_input(k) or "-" for k in ("Empresa", "Setor/posto", "Trabalhador avaliado", "Avaliador")}
sb.header("Parâmetros RULA")
OPTS = dict(force=sb.selectbox("Força/carga (0 a 3)", [0, 1, 2, 3]), static=sb.checkbox("Postura estática/repetitiva"),
            raised=sb.checkbox("Ombro elevado"), supported=sb.checkbox("Braço apoiado"),
            wdev=sb.checkbox("Desvio do punho"), twist=sb.selectbox("Giro do punho", [1, 2]))
SKEL = sb.checkbox("Mostrar esqueleto sobre a imagem", value=True)
RTC = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

class RulaProcessor(VideoProcessorBase):
    """Roda no servidor: recebe cada quadro da webcam, aplica Pose + RULA e devolve o quadro anotado."""
    def __init__(self):
        self.pl = landmarker(vision.RunningMode.VIDEO); self.lock = threading.Lock()
        self.opts, self.last, self.frame, self.hist, self.skel = {}, None, None, [], True; self.t0, self.last_ts = time.time(), -1
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        ts = max(int((time.time() - self.t0) * 1000), self.last_ts + 1); self.last_ts = ts
        ex = extract(self.pl.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), ts))
        if ex:
            with self.lock: opts, skel = dict(self.opts), self.skel
            r = rula(*ex, **opts); draw(img, ex[1], r, skel)
            with self.lock:
                self.last, self.frame = r, img.copy()
                if len(self.hist) < 5000: self.hist.append(linha_hist(ts / 1000, r))
        return av.VideoFrame.from_ndarray(img, format="bgr24")

@st.cache_resource
def img_lm(): return landmarker(vision.RunningMode.IMAGE)

def mostrar(r, hist):
    if r is None: st.info("Aguardando uma pessoa no enquadramento…"); return False
    a, b = st.columns(2); a.metric("RULA final", r["final"]); b.metric("Classificação", RISCO[r["final"]]); st.caption(f'Vista: {r["view"]} · lado avaliado: {r["side"].upper()}')
    st.dataframe(pd.DataFrame(passos(r), columns=["Etapa", "Pontos"]), hide_index=True)
    if len(hist) > 1: st.line_chart(hist.set_index("tempo_s")["rula"])
    return True

def exportar(r, hist, frame, key):
    if r is None: return
    if st.button("Gerar relatórios", key=f"g_{key}"):
        st.session_state[f"f_{key}"] = dict(xlsx=excel_bytes(r, IDENT, hist), pdf=pdf_bytes(r, IDENT, hist, frame),
                                            laudo=pdf_bytes(r, IDENT, hist, frame, laudo=True))
    f = st.session_state.get(f"f_{key}")
    if f:
        c1, c2, c3 = st.columns(3)
        c1.download_button("Excel (.xlsx)", f["xlsx"], "ergoapp_rula.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"x_{key}")
        c2.download_button("Relatório PDF", f["pdf"], "ergoapp_relatorio_rula.pdf", "application/pdf", key=f"p_{key}")
        c3.download_button("Laudo ergonômico (PDF)", f["laudo"], "ergoapp_laudo_ergonomico.pdf", "application/pdf", key=f"l_{key}")

@st.fragment(run_every=1.0)
def painel_ao_vivo(ctx):
    p = ctx.video_processor
    if not p: st.info("Clique em START para ligar a câmera."); return
    with p.lock:
        p.opts, p.skel = dict(OPTS), SKEL; r = p.last; frame = None if p.frame is None else p.frame.copy(); hist = pd.DataFrame(p.hist, columns=COLS)
    if mostrar(r, hist): exportar(r, hist, frame, "live")

tab_cam, tab_foto, tab_video = st.tabs(["Câmera ao vivo", "Foto", "Vídeo"])
with tab_cam:
    col_video, col_info = st.columns([3, 2])
    with col_video:   # vídeo completo, sem nenhum texto por cima
        ctx = webrtc_streamer(key="ergoapp", video_processor_factory=RulaProcessor, rtc_configuration=RTC, async_processing=True,
                              media_stream_constraints={"video": {"width": {"ideal": 1280}}, "audio": False})
    with col_info:    # informações RULA ao lado
        painel_ao_vivo(ctx)
with tab_foto:
    up = st.file_uploader("Foto (frente, costas ou lado)", type=["jpg", "jpeg", "png"])
    if up:
        img = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        ex = extract(img_lm().detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))))
        if not ex: st.warning("Nenhuma pessoa detectada.")
        else:
            r = rula(*ex, **OPTS); draw(img, ex[1], r, SKEL); hist = pd.DataFrame([linha_hist(0, r)], columns=COLS)
            c_img, c_info = st.columns([3, 2]); c_img.image(img, channels="BGR")
            with c_info: mostrar(r, hist); exportar(r, hist, img, f"foto_{up.name}_{up.size}")
with tab_video:
    upv = st.file_uploader("Vídeo", type=["mp4", "mov", "avi"]); passo = st.slider("Analisar 1 a cada N quadros", 1, 10, 3)
    if upv and st.button("Analisar vídeo"):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f: f.write(upv.getvalue())
        cap = cv2.VideoCapture(f.name); fps = cap.get(cv2.CAP_PROP_FPS) or 30; total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        rows, worst, i, bar = [], None, 0, st.progress(0.0)
        with landmarker(vision.RunningMode.VIDEO) as pl:
            while True:
                ok, fr = cap.read()
                if not ok: break
                if i % passo == 0:
                    ts = int(i * 1000 / fps)
                    ex = extract(pl.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)), ts))
                    if ex:
                        r = rula(*ex, **OPTS); draw(fr, ex[1], r, SKEL); rows.append(linha_hist(ts / 1000, r))
                        if worst is None or r["final"] > worst[0]["final"]: worst = (r, fr.copy())
                i += 1
                if i % 15 == 0: bar.progress(min(i / total, 1.0))
        cap.release(); bar.empty()
        st.session_state["vid"] = (worst, pd.DataFrame(rows, columns=COLS)) if worst else None
    v = st.session_state.get("vid")
    if v:
        (r, fr), hist = v; c_img, c_info = st.columns([3, 2]); c_img.image(fr, channels="BGR", caption="Quadro de maior risco")
        with c_info: mostrar(r, hist); exportar(r, hist, fr, "video")
