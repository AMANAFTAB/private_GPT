# PrivateGPT

A private AI chat application built with Streamlit that allows users to:
- Chat with an AI assistant in private sessions
- Upload documents (PDF, DOCX, images) and ask questions about them
- Collaborate with teammates in shared Group Rooms

## How to Run
1. Install requirements: `pip install -r requirements.txt`
2. Add your secrets to `.streamlit/secrets.toml`
3. Run: `streamlit run app.py`

## Tech Stack
- Streamlit
- Azure AI (Llama 4 Maverick + Ada Embeddings)
- Firebase Realtime Database
- FAISS Vector Search

## Deployment
- Deployed on streamlit community cloud. URL: https://am-intelligence.streamlit.app/
```
