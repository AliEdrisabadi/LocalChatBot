import streamlit as st
import sqlite3
import os
import wave
import json
import tempfile
import cv2
import numpy as np

# LLM library for local inference
from llama_cpp import Llama  
# For offline speech recognition
from vosk import Model as VoskModel, KaldiRecognizer  
# For real-time video/audio processing
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, VideoProcessorBase

##############################################
# 1. SET UP THE DATABASE
##############################################
def setup_database():
    connection = sqlite3.connect("sample.db")
    cursor = connection.cursor()
    table_info = """
    CREATE TABLE IF NOT EXISTS MANUFACTURE (
        PRODUCT_ID INTEGER PRIMARY KEY AUTOINCREMENT,
        NAME TEXT,
        CLASS TEXT,
        SELECTION TEXT,
        SCORE INTEGER,
        DESCRIPTION TEXT
    );
    """
    cursor.execute(table_info)
    cursor.execute("SELECT COUNT(*) FROM MANUFACTURE")
    count = cursor.fetchone()[0]
    if count == 0:
        sample_data = [
            ("ProdA", "Class1", "Accepted", 85, "High quality product suitable for assembly line A"),
            ("ProdB", "Class2", "Rejected", 60, "Low quality product; failed quality inspection"),
            ("ProdC", "Class1", "Accepted", 90, "Premium product with high customer satisfaction")
        ]
        cursor.executemany(
            "INSERT INTO MANUFACTURE (NAME, CLASS, SELECTION, SCORE, DESCRIPTION) VALUES (?, ?, ?, ?, ?)",
            sample_data
        )
        connection.commit()
    cursor.close()
    connection.close()

##############################################
# 2. FUNCTION TO EXECUTE SQL QUERIES
##############################################
def read_sql_query(sql, db):
    try:
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return rows
    except Exception as e:
        st.error("Error in read_sql_query: " + str(e))
        return []

##############################################
# 3. LOAD THE LLM MODEL FOR VOICE-TO-SQL (SQLCoder-7B-2)
##############################################
@st.cache_resource
def load_llm_model(model_choice: str):
    """
    Loads the SQLCoder-7B-2 model for Voice-to-SQL.
    Adjust if you want to change the file path or model name.
    """
    model_paths = {
        "SQLCoder-7B-2": os.path.join("models", "sqlcoder-7b-2", "sqlcoder-7b-q5_k_m.gguf")
    }
    model_path = model_paths.get(model_choice)
    if not model_path or not os.path.exists(model_path):
        st.error(f"Model path for '{model_choice}' not found. Please check your file structure.")
        return None
    llm = Llama(
        model_path=model_path,
        n_threads=8,
        n_batch=8,
        verbose=True
    )
    return llm

##############################################
# 4. LOAD THE PDF-SPECIFIC LLM MODELS (for Document Chat)
##############################################
@st.cache_resource
def load_pdf_llm_model(model_choice: str):
    """
    Loads one of the three PDF chat models:
    - DeepSeek-R1-Distill-Llama-8B-GGUF
    - gemma-3-12b-it-GGUF
    - Meta-Llama-3-8B-Instruct-GGUF
    """
    model_paths = {
        "DeepSeek-R1-Distill-Llama-8B-GGUF": os.path.join("models", "DeepSeek-R1-Distill-Llama-8B-GGUF", "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"),
        "gemma-3-12b-it-GGUF": os.path.join("models", "gemma-3-12b-it-GGUF", "gemma-3-12b-it-Q4_K_M.gguf"),
        "Meta-Llama-3-8B-Instruct-GGUF": os.path.join("models", "Meta-Llama-3-8B-Instruct-GGUF", "Meta-Llama-3-8B-Instruct.Q4_K_S.gguf")
    }
    model_path = model_paths.get(model_choice)
    if not model_path or not os.path.exists(model_path):
        st.error(f"Model path for '{model_choice}' not found. Please check your file structure.")
        return None
    llm = Llama(
        model_path=model_path,
        n_threads=8,
        n_batch=8,
        verbose=True
    )
    return llm

##############################################
# 5. LOAD THE VOSK MODEL FOR VOICE TRANSCRIPTION
##############################################
@st.cache_resource
def load_vosk_model():
    vosk_model_path = os.path.join("models", "vosk-model-en-us-0.22-lgraph")
    if not os.path.exists(vosk_model_path):
        st.error(f"Vosk model not found at {vosk_model_path}")
        return None
    return VoskModel(vosk_model_path)

##############################################
# 6. TRANSCRIBE AUDIO USING VOSK
##############################################
def transcribe_audio(audio_bytes, vosk_model):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_file_path = tmp_file.name
    wf = wave.open(tmp_file_path, "rb")
    rec = KaldiRecognizer(vosk_model, wf.getframerate())
    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            results.append(result.get("text", ""))
    final_result = json.loads(rec.FinalResult())
    results.append(final_result.get("text", ""))
    wf.close()
    os.remove(tmp_file_path)
    return " ".join(results).strip()

##############################################
# 7. AUDIO PROCESSOR FOR streamlit-webrtc
##############################################
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        super().__init__()
        self.frames = []

    def recv(self, frame):
        self.frames.append(frame.to_bytes())
        return frame

    def get_audio_bytes(self):
        return b"".join(self.frames)

    def clear_audio_bytes(self):
        self.frames = []

##############################################
# 8. GENERATE SQL QUERY USING THE LLM MODEL (for Voice-to-SQL)
##############################################
def generate_sql_query(llm, user_question):
    table_schema = """
    CREATE TABLE MANUFACTURE (
      PRODUCT_ID INTEGER PRIMARY KEY,
      NAME TEXT,
      CLASS TEXT,
      SELECTION TEXT,
      SCORE INTEGER,
      DESCRIPTION TEXT
    );
    """
    prompt = f"""### Task
Generate a SQL query to answer [QUESTION]{user_question}[/QUESTION]

### Database Schema
{table_schema}

### Answer
Given the database schema, here is the SQL query that [QUESTION]{user_question}[/QUESTION]
[SQL]
"""
    response = llm(prompt, max_tokens=150, temperature=0.0)
    return response['choices'][0]['text'].strip()

##############################################
# 9. AR OBJECT RECOGNITION PROCESSOR (YOLO)
##############################################
class ARObjectRecognitionProcessor(VideoProcessorBase):
    def __init__(self):
        super().__init__()
        yolo_dir = os.path.join("models", "yolo")
        weights_path = os.path.join(yolo_dir, "yolov3.weights")
        config_path = os.path.join(yolo_dir, "yolov3.cfg")
        names_path = os.path.join(yolo_dir, "coco.names")

        self.net = cv2.dnn.readNet(weights_path, config_path)
        with open(names_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        height, width, _ = img.shape

        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids, confidences, boxes = [], [], []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = self.classes[class_ids[i]]
                confidence = confidences[i]
                color = (0, 255, 0)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame.from_ndarray(img, format="bgr24")

##############################################
# 10. VOICE-TO-SQL PAGE
##############################################
def voice_to_sql_page():
    st.title("SQLCoder-7B-2 Text-to-SQL Converter with Voice")
    st.markdown("Convert voice or text queries into SQL, then execute on sample.db.")

    setup_database()
    llm = load_llm_model("SQLCoder-7B-2")  # Using default model for SQL
    vosk_model = load_vosk_model()
    if vosk_model is None:
        st.error("Vosk model is missing. Check the path in load_vosk_model().")
        return

    if "query_text" not in st.session_state:
        st.session_state.query_text = ""
    if "recording" not in st.session_state:
        st.session_state.recording = False

    user_question = st.text_input("Your Query:", st.session_state.query_text)

    if st.button("Record Voice Query"):
        st.session_state.recording = True

    if st.session_state.recording:
        st.info("Recording... Speak now.")
        webrtc_ctx = webrtc_streamer(key="audio_recorder", audio_processor_factory=AudioProcessor)
        if webrtc_ctx.audio_receiver:
            audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            if audio_frames:
                audio_bytes = b"".join(frame.to_bytes() for frame in audio_frames)
                transcription = transcribe_audio(audio_bytes, vosk_model)
                if transcription:
                    st.session_state.query_text += (" " + transcription) if st.session_state.query_text else transcription
        st.session_state.recording = False

    if st.button("Generate SQL & Execute"):
        final_query = st.session_state.query_text if st.session_state.query_text else user_question
        if final_query.strip():
            generated_sql = generate_sql_query(llm, final_query)
            st.write("**Generated SQL Query:**")
            st.code(generated_sql, language="sql")
            if generated_sql.strip():
                data = read_sql_query(generated_sql, "sample.db")
                st.header("Query Results:")
                if data:
                    for row in data:
                        st.write(row)
                else:
                    st.write("No data returned.")
            else:
                st.error("No SQL query generated.")
        else:
            st.warning("Please enter or record a query first.")

##############################################
# 11. AR OBJECT RECOGNITION PAGE
##############################################
def ar_object_recognition_page():
    st.title("AR Object Recognition Demo")
    st.markdown("Real-time object detection using YOLOv3 with bounding box overlays.")
    webrtc_streamer(key="ar_object_recognition", video_processor_factory=ARObjectRecognitionProcessor)

##############################################
# 12. DOCUMENT CHAT PAGE (PDF Chat with your three models)
##############################################
def answer_question_from_document(llm, document_text, question):
    prompt = f"""You are an expert on the following document:
{document_text}

Based on the document, answer the following question:
{question}

Answer:"""
    response = llm(prompt, max_tokens=150, temperature=0.3)
    answer = response['choices'][0]['text'].strip()
    return answer

def document_chat_page():
    st.title("Document Chat for PDF")
    st.markdown("Upload a PDF document and chat with it using your local models.")
    
    # Use the three models specified for PDF chat.
    model_options = [
        "DeepSeek-R1-Distill-Llama-8B-GGUF",
        "gemma-3-12b-it-GGUF",
        "Meta-Llama-3-8B-Instruct-GGUF"
    ]
    model_choice = st.selectbox("Select Model", model_options, index=0)
    
    llm = load_pdf_llm_model(model_choice)
    if llm is None:
        st.stop()
    
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    if uploaded_file is not None:
        try:
            import fitz  # PyMuPDF: pip install PyMuPDF
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            document_text = ""
            for page in doc:
                document_text += page.get_text()
        except Exception as e:
            st.error("Error reading PDF: " + str(e))
            document_text = ""
        
        if document_text:
            st.subheader("Document Content")
            st.text_area("Document Text", document_text, height=300)
            
            if "doc_chat_history" not in st.session_state:
                st.session_state.doc_chat_history = []
            
            user_question = st.text_input("Ask a question about the document:")
            if st.button("Send Question"):
                if user_question:
                    answer = answer_question_from_document(llm, document_text, user_question)
                    st.session_state.doc_chat_history.append(("User", user_question))
                    st.session_state.doc_chat_history.append(("Bot", answer))
            
            st.subheader("Chat History")
            for speaker, message in st.session_state.get("doc_chat_history", []):
                if speaker == "User":
                    st.markdown(f"**User:** {message}")
                else:
                    st.markdown(f"**Bot:** {message}")

##############################################
# 13. MAIN APP ENTRY POINT
##############################################
def main():
    st.set_page_config(page_title="AI for Manufacturing", layout="wide")
    page = st.sidebar.selectbox("Select Feature", [
        "Voice-to-SQL", 
        "AR Object Recognition",
        "Document Chat"
    ])
    if page == "Voice-to-SQL":
        voice_to_sql_page()
    elif page == "AR Object Recognition":
        ar_object_recognition_page()
    elif page == "Document Chat":
        document_chat_page()

if __name__ == "__main__":
    main()
