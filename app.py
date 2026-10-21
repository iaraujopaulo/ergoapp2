import os
import streamlit.web.cli as stcli
import sys

if __name__ == "__main__":
    # Substitua pelo nome exato do seu arquivo principal do Streamlit se necessário
    sys.argv = ["streamlit", "run", "app_streamlit.py", "--server.port", "8000", "--server.address", "0.0.0.0"]
    sys.exit(stcli.main())
