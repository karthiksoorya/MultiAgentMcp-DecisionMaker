@echo off
echo Starting Multi-Agent PostgreSQL Analysis System - Web Interface
echo ================================================================
echo.
echo Opening Streamlit app in your browser...
echo Press Ctrl+C to stop the application
echo.
streamlit run streamlit_app.py --server.port=8501 --server.address=localhost
pause