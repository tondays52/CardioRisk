🪟 Window 1: Start Backend (FastAPI)



cd C:\Users\tonda\Desktop\hrp
venv\Scripts\activate
python backend\api\main.py


🪟 Window 2: Start HTTP Server for Quick Tools




cd C:\Users\tonda\Desktop\hrp
venv\Scripts\activate
python -m http.server 8502 --directory frontend
Keep this window open!



🪟 Window 3: Start Streamlit Dashboard



cd C:\Users\tonda\Desktop\hrp
venv\Scripts\activate
streamlit run frontend/app.py



🪟 Window 4: Start React Landing Page




cd "C:\Users\tonda\Desktop\replit landging pg\shorts-inspired-landing\artifacts"
$env:PORT = "5173"
$env:BASE_PATH = "/"
pnpm run dev
