# ErgoApp – RULA com MediaPipe
pip install -r requirements.txt

## Streamlit + WebRTC (câmera no navegador, foto, vídeo e exportações)
python -m streamlit run app_streamlit.py     # use "python -m" se o comando "streamlit" não for encontrado
(ou execute ./iniciar.sh  /  iniciar.bat)     # a câmera exige localhost ou HTTPS

## Linha de comando (saída = imagem/vídeo completos + painel RULA ao lado)
python ergoapp.py --image foto.jpg | --video video.mp4 --out saida.mp4 --csv saida.csv | --webcam

## Web estático (sem servidor Python)
cd web && python -m http.server 8000
