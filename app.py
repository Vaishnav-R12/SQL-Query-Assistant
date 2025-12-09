import streamlit as st
import google.generativeai as genai
import mysql.connector
import pandas as pd
import io
from mysql.connector import Error
import networkx as nx
import plotly.express as px
import speech_recognition as sr
import tempfile
import os
from gtts import gTTS
import requests
import json
from datetime import datetime
import time
import plotly.graph_objects as go
import requests
import sqlite3   
import re
import pyrebase
import re
import json
from datetime import datetime

# ----------------- FIREBASE CONFIG -----------------
firebase_config = st.secrets["firebase"]
firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

# ----------------- SESSION STATE INIT -----------------
if 'user' not in st.session_state:
    st.session_state.user = None
if 'show_signup' not in st.session_state:
    st.session_state.show_signup = False
if 'signed_up_username' not in st.session_state:
    st.session_state.signed_up_username = None
if 'show_reset_password' not in st.session_state:
    st.session_state.show_reset_password = False

# Configure the API Key
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')
YOUTUBE_API_KEY = st.secrets["YOUTUBE_API_KEY"]

def delete_firebase_account(id_token):
    """Delete Firebase account for current user"""
    api_key = firebase_config["apiKey"]
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:delete?key={api_key}"
    payload = {"idToken": id_token}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return True
    else:
        raise Exception(response.json())

def validate_password_strength(password):
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r"[0-9]", password):
        return False, "Password must contain at least one number"
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain at least one special character"
    return True, ""

def show_auth_ui():
    st.title("🔐 SQL Query Assistant")

    # If already logged in
    if st.session_state.user:
        email = st.session_state.user['email']
        username = st.session_state.signed_up_username or email
        st.success(f"Welcome, {username}!")

        if not st.session_state.user.get("emailVerified", False):
            st.warning("⚠️ Please verify your email to unlock all features.")
            if st.button("Send Verification Email"):
                try:
                    auth.send_email_verification(st.session_state.user['idToken'])
                    st.success("Verification email sent!")
                except Exception as e:
                    st.error(f"Error: {e}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚪 Logout"):
                st.session_state.user = None
                st.session_state.signed_up_username = None
                st.success("Logged out successfully!")
                st.rerun()
        with col2:
            if st.button("🔑 Reset Password"):
                st.session_state.show_reset = True
                st.rerun()

    # Reset password flow
    elif st.session_state.get("show_reset", False):
        st.subheader("🔄 Reset Password")
        email = st.text_input("Enter your email")
        if st.button("Send Reset Link"):
            try:
                auth.send_password_reset_email(email)
                st.success("Password reset email sent! Check your inbox (and spam).")
                st.session_state.show_reset = False
            except Exception as e:
                try:
                    st.error(json.loads(e.args[1])["error"]["message"])
                except:
                    st.error(str(e))
        if st.button("⬅ Back to Login"):
            st.session_state.show_reset = False

    # Signup flow
    elif st.session_state.show_signup:
        st.subheader("📝 Sign Up")
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        confirm = st.text_input("Confirm Password", type="password")

        strong, msg = validate_password_strength(password)
        if password and not strong:
            st.warning(msg)

        if st.button("Create Account"):
            if password != confirm:
                st.error("Passwords do not match!")
            elif not strong:
                st.error(msg)
            else:
                try:
                    user = auth.create_user_with_email_and_password(email, password)
                    auth.send_email_verification(user["idToken"])
                    st.session_state.user = user
                    st.session_state.signed_up_username = username
                    st.success("✅ Account created! Please verify your email.")
                    st.session_state.show_signup = False
                    st.rerun()
                except Exception as e:
                    try:
                        st.error(json.loads(e.args[1])["error"]["message"])
                    except:
                        st.error(str(e))

        if st.button("⬅ Back to Login"):
            st.session_state.show_signup = False

    # Login flow
    else:
        st.subheader("🔑 Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Login"):
                try:
                    user = auth.sign_in_with_email_and_password(email, password)
                    st.session_state.user = user
                    st.success(f"✅ Welcome {email}!")
                    st.rerun()
                except Exception as e:
                    try:
                        st.error(json.loads(e.args[1])["error"]["message"])
                    except:
                        st.error(str(e))
        with col2:
            if st.button("Sign Up"):
                st.session_state.show_signup = True
        with col3:
            if st.button("Forgot Password?"):
                st.session_state.show_reset = True

def fetch_schema(db_params):
    """
    Fetch database schema (tables and columns) for MySQL or SQLite.
    db_params must include:
        - type: "mysql" or "sqlite"
        - database: db name or path
        - host, user, password (for MySQL)
    """
    schema = {}

    try:
        if db_params["type"] == "mysql":
            conn = mysql.connector.connect(
                host=db_params["host"],
                user=db_params["user"],
                password=db_params["password"],
                database=db_params["database"]
            )
            cursor = conn.cursor()
            cursor.execute("SHOW TABLES;")
            tables = cursor.fetchall()

            for (table_name,) in tables:
                cursor.execute(f"SHOW COLUMNS FROM {table_name};")
                columns = cursor.fetchall()
                schema[table_name] = [col[0] for col in columns]

            cursor.close()
            conn.close()

        elif db_params["type"] == "sqlite":
            conn = sqlite3.connect(db_params["database"])
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()

            for (table_name,) in tables:
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                schema[table_name] = [col[1] for col in columns]

            cursor.close()
            conn.close()

        else:
            raise ValueError("Unsupported database type.")

    except Exception as e:
        return {"error": str(e)}

    return schema

def get_db_files():
    db_dir = "databases"
    if not os.path.exists(db_dir):
        try:
            os.makedirs(db_dir)
        except OSError as e:
            st.error(f"Failed to create databases directory: {e}")
            return []
    return [f for f in os.listdir(db_dir) if f.endswith(".db")]

def create_sample_database():
    db_dir = "databases"
    if not os.path.exists(db_dir):
        try:
            os.makedirs(db_dir)
        except OSError as e:
            st.error(f"Failed to create databases directory: {e}")
            return
    
    db_path = os.path.join(db_dir, "sales_demo.db")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")
        
        # Create sample tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                customer_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS products (
                product_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                price REAL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id INTEGER PRIMARY KEY,
                customer_id INTEGER,
                product_id INTEGER,
                order_date TEXT,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
                FOREIGN KEY (product_id) REFERENCES products(product_id)
            )
        """)
        
        # Insert sample data
        cursor.execute("INSERT OR IGNORE INTO customers (customer_id, name, email) VALUES (1, 'Alice', 'alice@example.com')")
        cursor.execute("INSERT OR IGNORE INTO customers (customer_id, name, email) VALUES (2, 'Bob', 'bob@example.com')")
        cursor.execute("INSERT OR IGNORE INTO products (product_id, name, price) VALUES (1, 'Laptop', 999.99)")
        cursor.execute("INSERT OR IGNORE INTO products (product_id, name, price) VALUES (2, 'Phone', 499.99)")
        cursor.execute("INSERT OR IGNORE INTO orders (order_id, customer_id, product_id, order_date) VALUES (1, 1, 1, '2023-10-01')")
        cursor.execute("INSERT OR IGNORE INTO orders (order_id, customer_id, product_id, order_date) VALUES (2, 2, 2, '2023-10-02')")
        
        conn.commit()
        # Verify tables were created
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall() if table[0] != "sqlite_sequence"]
        conn.close()
        
        if set(tables) != {"customers", "products", "orders"}:
            st.error(f"Failed to create all expected tables in {db_path}. Created tables: {tables}")
    except sqlite3.Error as e:
        st.error(f"Failed to create sample database: {e}")

def render_download_buttons(report_data):
    download_format = st.selectbox(
        "Choose download format for the query report:",
        ["CSV", "JSON", "Text", "PDF"]
    )
    if download_format == "CSV":
        # Convert DataFrame results to string for CSV compatibility
        report_data_csv = report_data.copy()
        if "results" in report_data_csv.columns:
            report_data_csv["results"] = report_data_csv["results"].apply(
                lambda x: x.to_string(index=False) if isinstance(x, pd.DataFrame) else str(x) if pd.notna(x) else ""
            )
        st.download_button(
            label="Download CSV",
            data=report_data_csv.to_csv(index=False),
            file_name="sql_query_report.csv",
            mime="text/csv",
            key="csv_download"
        )
    elif download_format == "JSON":
        # Convert DataFrame results to list of dictionaries for JSON
        report_data_json = report_data.copy()
        if "results" in report_data_json.columns:
            report_data_json["results"] = report_data_json["results"].apply(
                lambda x: x.to_dict(orient="records") if isinstance(x, pd.DataFrame) else str(x) if pd.notna(x) else None
            )
        st.download_button(
            label="Download JSON",
            data=report_data_json.to_json(orient="records"),
            file_name="sql_query_report.json",
            mime="application/json",
            key="json_download"
        )
    elif download_format == "Text":
        text_content = ""
        for _, row in report_data.iterrows():
            text_content += f"Prompt:\n{row['prompt']}\n\n"
            text_content += f"Query:\n{row['query']}\n\n"
            text_content += f"Explanation:\n{row['explanation']}\n\n"
            text_content += f"Expected Output:\n{row['output']}\n\n"
            if "results" in row and row["results"] is not None:
                text_content += "Executed Query Results:\n"
                if isinstance(row["results"], pd.DataFrame):
                    text_content += row["results"].to_string(index=False) + "\n"
                else:
                    text_content += str(row["results"]) + "\n"
            text_content += "-" * 50 + "\n"
        st.download_button(
            label="Download Text",
            data=text_content,
            file_name="sql_query_report.txt",
            mime="text/plain",
            key="text_download"
        )
    elif download_format == "PDF":
        pdf_content = generate_pdf_report(report_data)
        st.download_button(
            label="Download PDF",
            data=pdf_content,
            file_name="sql_query_report.pdf",
            mime="application/pdf",
            key="pdf_download"
        )

def fetch_databases(host, user, password):
    # Note: This function will use demo_mode from main(), so it needs to be adjusted there
    # For now, we'll assume demo_mode is passed or accessed globally if needed
    try:
        conn = mysql.connector.connect(host=host, user=user, password=password)
        cursor = conn.cursor()
        cursor.execute("SHOW DATABASES")
        databases = [db[0] for db in cursor.fetchall()]
        conn.close()
        return databases
    except Error as e:
        return []

def get_sql_topic_from_gemini(query):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")  # Replace with the desired model
        response = model.generate_content(query)
        topic = response.text.strip()  # Get the SQL topic from Gemini's response
        return topic
    except Exception as e:
        st.error(f"Error getting topic from Gemini API: {e}")
        return None

    
# Function to search YouTube videos based on the query
def search_youtube_videos(api_key, search_query):
    youtube_search_url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": f"SQL {search_query} tutorial",  # More descriptive search query
        "type": "video",
        "key": api_key,
        "maxResults": 7,
    }

    try:
        response = requests.get(youtube_search_url, params=params)
        response.raise_for_status()
        response_data = response.json()

        # Extract video details
        video_results = []
        for video in response_data.get("items", []):
            video_id = video["id"]["videoId"]
            title = video["snippet"]["title"]
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            video_results.append({"title": title, "url": video_url})
        
        return video_results
    except requests.exceptions.RequestException as e:
        st.error(f"Error with YouTube API: {e}")
        return []
    except KeyError as e:
        st.error(f"Unexpected response format: {e}")
        return []


# Function to determine the modified table (can be enhanced based on your use case)
def get_modified_table_from_query(query):
    # Identify the modified table based on query keywords (e.g., INSERT INTO, UPDATE, DELETE FROM)
    query_lower = query.strip().lower()
    if query_lower.startswith("insert into"):
        return query.split()[2]  # Assuming table name follows 'INSERT INTO'
    elif query_lower.startswith("update"):
        return query.split()[1]  # Assuming table name follows 'UPDATE'
    elif query_lower.startswith("delete from"):
        return query.split()[2]  # Assuming table name follows 'DELETE FROM'
    elif query_lower.startswith(("create", "alter", "drop")):
        # For DDL queries, the table is usually specified after the command
        return query.split()[2] if len(query.split()) > 2 else None
    return None
    
def execute_query(query, db_params):
    try:
        if not db_params.get("database") and not query.strip().lower().startswith("create database"):
            return {"status": "error", "message": "Database connection details are missing."}

        if db_params.get("type") == "sqlite":
            conn = sqlite3.connect(db_params["database"])
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")  # Enable foreign keys for SQLite
            cursor.execute(query)
            
            if query.strip().lower().startswith(("create", "alter", "drop")):
                conn.commit()
                tables = fetch_tables(db_params)
                conn.close()
                return {"status": "success", "message": "DDL query executed.", "data": tables}
            elif query.strip().lower().startswith("select"):
                data = pd.read_sql_query(query, conn)
                conn.close()
                return {"status": "success", "data": data}
            else:  # INSERT, UPDATE, DELETE
                conn.commit()
                modified_table = get_modified_table_from_query(query)
                if modified_table:
                    table_data = fetch_table_data(db_params, modified_table)
                    conn.close()
                    return {"status": "success", "data": table_data}
                conn.close()
                return {"status": "success", "data": pd.DataFrame()}
        else:  # MySQL
            if query.strip().lower().startswith(("create database", "drop database")):
                conn = mysql.connector.connect(
                    host=db_params.get("host", "localhost"),
                    user=db_params.get("user", "root"),
                    password=db_params.get("password", "")
                )
                cursor = conn.cursor()
                cursor.execute(query)
                conn.close()
                return {"status": "success", "message": f"Database operation successful."}
            
            conn = mysql.connector.connect(**{k: v for k, v in db_params.items() if k != "type"})
            cursor = conn.cursor()
            cursor.execute(query)
            
            if query.strip().lower().startswith(("create", "alter", "drop")):
                conn.commit()
                tables = fetch_tables(db_params)
                conn.close()
                return {"status": "success", "message": "DDL query executed.", "data": tables}
            elif query.strip().lower().startswith("select"):
                columns = [col[0] for col in cursor.description]
                data = cursor.fetchall()
                conn.close()
                return {"status": "success", "data": pd.DataFrame(data, columns=columns)}
            else:
                conn.commit()
                modified_table = get_modified_table_from_query(query)
                if modified_table:
                    table_data = fetch_table_data(db_params, modified_table)
                    conn.close()
                    return {"status": "success", "data": table_data}
                conn.close()
                return {"status": "success", "data": pd.DataFrame()}
    except (Error, sqlite3.Error) as e:
        return {"status": "error", "message": f"Error executing query: {str(e)}"}

# Function to process SQL file content
def process_sql_file(file_content):
    queries = file_content.strip().split(";")
    return [query.strip() for query in queries if query.strip()]

def transcribe_audio_to_text():
    """
    Captures audio from the microphone and transcribes it into text.
    Returns the transcribed text or an error message.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Please speak into the microphone.")
        try:
            audio = recognizer.listen(source, timeout=5)
            st.info("Processing the audio...")
            text = recognizer.recognize_google(audio)  # Using Google Web Speech API
            return text
        except sr.WaitTimeoutError:
            return "Error: Timeout. Please speak louder or check your microphone."
        except sr.UnknownValueError:
            return "Error: Could not understand the audio."
        except sr.RequestError as e:
            return f"Error: Could not request results from the speech recognition service; {e}"

# Function to generate explanation for SQL query
def generate_explanation(sql_query):
    explanation_template = """
        Explain the SQL Query snippet:
        {sql_query}
        Please provide the simplest explanation:
    """
    explanation_formatted = explanation_template.format(sql_query=sql_query)
    explanation_response = model.generate_content(explanation_formatted)
    explanation = explanation_response.text.strip()

    return explanation

def text_to_speech(text):
    if not text.strip():
        st.warning("No text to convert to speech.")
        return None,None
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tts.save(tmp_file.name)
            tmp_file.seek(0)
            audio_bytes = tmp_file.read()
            tmp_file_path = tmp_file.name
        return audio_bytes, tmp_file_path  # Return bytes and path for cleanup
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return None, None

# Function to generate expected output for SQL query
def generate_expected_output(sql_query):
    expected_output_template = """
        What would be the expected output of the SQL Query snippet:
        {sql_query}
        Provide a sample tabular response with no explanation:
    """
    expected_output_formatted = expected_output_template.format(sql_query=sql_query)
    expected_output_response = model.generate_content(expected_output_formatted)
    expected_output = expected_output_response.text.strip()
    return expected_output

# Fetch tables from the database
def fetch_tables(db_params):
    try:
        if db_params.get("type") == "sqlite":
            if not db_params.get("database"):
                return {"status": "error", "message": "No SQLite database selected."}
            if not os.path.exists(db_params["database"]):
                return {"status": "error", "message": f"Database file {db_params['database']} does not exist."}
            # Retry connection up to 3 times with delay to ensure file is ready
            for attempt in range(3):
                try:
                    conn = sqlite3.connect(db_params["database"])
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = [table[0] for table in cursor.fetchall() if table[0] != "sqlite_sequence"]
                    conn.close()
                    return pd.DataFrame(tables, columns=["Table Name"])
                except sqlite3.Error as e:
                    if attempt < 2:
                        time.sleep(0.5)  # Wait before retrying
                        continue
                    return {"status": "error", "message": f"Error fetching tables: {str(e)} - Path: {db_params['database']}"}
        else:
            conn = mysql.connector.connect(**{k: v for k, v in db_params.items() if k != "type"})
            cursor = conn.cursor()
            cursor.execute("SHOW TABLES")
            tables = [table[0] for table in cursor.fetchall()]
            conn.close()
            return pd.DataFrame(tables, columns=["Table Name"])
    except (Error, sqlite3.Error) as e:
        return {"status": "error", "message": f"Error fetching tables: {str(e)} - Path: {db_params.get('database', 'N/A')}"}
def fetch_table_data(db_params, table_name):
    try:
        if db_params.get("type") == "sqlite":
            conn = sqlite3.connect(db_params["database"])
            query = f"SELECT * FROM {table_name}"
            data = pd.read_sql_query(query, conn)
            conn.close()
            return data
        else:
            conn = mysql.connector.connect(**{k: v for k, v in db_params.items() if k != "type"})
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM {table_name}")
            columns = [col[0] for col in cursor.description]
            data = cursor.fetchall()
            conn.close()
            return pd.DataFrame(data, columns=columns)
    except (Error, sqlite3.Error) as e:
        return {"status": "error", "message": f"Error fetching data from table {table_name}: {str(e)}"}

    # Function to generate a PDF report from the generated data
def generate_pdf_report(report_data):
    import io
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add title
    pdf.cell(200, 10, txt="SQL Query Report", ln=True, align="C")
    pdf.ln(10)

    # Iterate through the data and add it to the PDF
    for _, row in report_data.iterrows():
        pdf.set_font("Arial", style="B", size=12)
        pdf.multi_cell(0, 10, f"Prompt:")
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, str(row["prompt"]))

        pdf.set_font("Arial", style="B", size=12)
        pdf.multi_cell(0, 10, f"Query:")
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, str(row["query"]))

        pdf.set_font("Arial", style="B", size=12)
        pdf.multi_cell(0, 10, f"Explanation:")
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, str(row["explanation"]))

        pdf.set_font("Arial", style="B", size=12)
        pdf.multi_cell(0, 10, f"Expected Output:")
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, str(row["output"]))
        

        # Include executed results if available (fixed condition)
        if "results" in row and row["results"] is not None:
            pdf.set_font("Arial", style="B", size=12)
            pdf.multi_cell(0, 10, "Executed Query Results:")
            pdf.set_font("Arial", size=12)
            if isinstance(row["results"], pd.DataFrame):
                results_text = row["results"].to_string(index=False)
            else:
                results_text = str(row["results"])
            pdf.multi_cell(0, 10, results_text)

        pdf.ln(5)

    pdf_output = io.BytesIO()
    pdf_content = pdf.output(dest="S").encode("latin1")
    pdf_output.write(pdf_content)
    pdf_output.seek(0)
    return pdf_output   

def main():
    st.set_page_config(page_title="SQL Query Assistant", page_icon="🔍", layout="wide")
    # 🔐 Show auth first
    if not st.session_state.user:
        show_auth_ui()
        return  # ⛔ Stop until login
    # st.set_page_config(page_title="SQL Query Assistant", page_icon="🔍", layout="wide")
    
 
    st.sidebar.subheader("Account Settings")

    col1, col2 = st.sidebar.columns(2)

    with col1:
     if st.button("🚪 Logout"):
        st.session_state.user = None
        st.session_state.signed_up_username = None
        st.success("Logged out successfully!")
        st.rerun()

    with col2:
     if st.button("🗑️ Delete Account"):
        try:
            # Need to re-authenticate before delete
            user = st.session_state.user
            auth.delete_user_account(user['idToken'])
            st.session_state.user = None
            st.session_state.signed_up_username = None
            st.success("Account deleted successfully!")
            st.rerun()
        except Exception as e:
            try:
                st.error(json.loads(e.args[1])["error"]["message"])
            except:
                st.error(str(e))

   
   # Create sample database for demo mode
    create_sample_database()

    st.markdown(
        """
        <div style="text-align: center; padding: 20px;">
            <h1>SQL Query Assistant 🤖</h1>
            <h3>Generate SQL queries effortlessly ✨</h3>
            <h4>Get explanations, expected outputs, and optional execution 📚</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

   # Initialize session state at the very start
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {
            "Chat 1": [
                {"role": "agent", "content": "Hi there! How can I help you with SQL today? 😊", "timestamp": datetime.now().strftime("%H:%M:%S")}
            ]
        }
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    if "use_emojis" not in st.session_state:
        st.session_state.use_emojis = True
    if "active_chat" not in st.session_state:
        st.session_state.active_chat = "Chat 1"
    if "editing_message" not in st.session_state:
        st.session_state.editing_message = None  # Tracks which message is being edited
    if "generated_data" not in st.session_state:
        st.session_state.generated_data = []

    # Sidebar for Database Connection
    st.sidebar.header("Database Connection (Optional)")
    db_type = st.sidebar.radio("Database Type", ["Demo (SQLite)", "MySQL"], index=0)  # Default to Demo

    if db_type == "Demo (SQLite)":
        db_path = os.path.join("databases", "sales_demo.db")
        if not os.path.exists(db_path):
            st.sidebar.error("Sample database not found. Attempting to recreate...")
            create_sample_database()
        st.sidebar.info("Demo Mode: Using a sample SQLite sales database with 'orders', 'customers', and 'products' tables.")
        db_params = {"database": db_path, "type": "sqlite"}
  
    elif db_type == "MySQL":
        host = st.sidebar.text_input("Host", "localhost")
        user = st.sidebar.text_input("Username", "root")
        password = st.sidebar.text_input("Password", type="password")
        databases = fetch_databases(host, user, password) or []
        database = st.sidebar.selectbox(
            "Select Database",
            databases if databases else ["Enter credentials to fetch databases"],
            disabled=not bool(databases)
        )
        db_params = {
            "host": host,
            "user": user,
            "password": password,
            "database": database if database != "Enter credentials to fetch databases" else "",
            "type": "mysql"
        }

    if st.sidebar.button("Test Connection"):
        try:
            if db_params.get("type") == "sqlite" and db_params.get("database"):
                conn = sqlite3.connect(db_params["database"])
            elif db_params.get("type") == "mysql" and db_params.get("database"):
                conn = mysql.connector.connect(**{k: v for k, v in db_params.items() if k != "type"})
            else:
                st.sidebar.error("Please select or create a database first.")
                return
            conn.close()
            st.sidebar.success("Database connection successful!")
        except (Error, sqlite3.Error) as e:
            st.sidebar.error(f"Connection failed: {e}")

    if db_params.get("database"):  # Check if database is set instead of all(db_params.values())
        tables = fetch_tables(db_params)
        if isinstance(tables, pd.DataFrame):
            st.sidebar.subheader("Tables in Database")
            selected_table = st.sidebar.selectbox("Select a table to view:", tables["Table Name"].tolist())
            if selected_table:
                table_data = fetch_table_data(db_params, selected_table)
                if isinstance(table_data, pd.DataFrame):
                    st.sidebar.write(f"Contents of `{selected_table}`:")
                    st.sidebar.dataframe(table_data)
                else:
                    st.sidebar.error(table_data["message"])
        else:
            st.sidebar.error(tables["message"])


    st.sidebar.header("Upload SQL Files")
    uploaded_files = st.sidebar.file_uploader("Upload one or more .sql files", type=["sql"], accept_multiple_files=True)

    generated_data = []  # To store generated prompts, queries, and outputs
    # Use session state to persist generated_data across interactions
    if "generated_data" not in st.session_state:
        st.session_state.generated_data = []
    tabs = st.selectbox(
        "Choose a feature",
        ["Generate Query from English", "Upload SQL Files", "Visualize Data", "Database Schema", "Learn SQL","Query History","SQL Playground"]
    )

    if tabs == "Generate Query from English":
     st.header("Generate SQL Queries from Plain English")
     if "voice_input" not in st.session_state:
        st.session_state["voice_input"] = ""
     if "generated_data" not in st.session_state:
        st.session_state.generated_data = []

     st.markdown("Enter multiple queries in plain English (separate each query with a semicolon `;`):")
     text_input = st.text_area(
        "E.g., Show all customers; Count orders by customer; List products priced above 500",
        value=st.session_state.get("voice_input", ""),
        height=120,
        key="text_input_generate_query"
     )

     if st.button("Record Audio 🎙️", key="record_audio_generate_query"):
        voice_text = transcribe_audio_to_text()
        if "Error" not in voice_text:
            st.session_state["voice_input"] = voice_text
            st.success(f"Transcribed Text: {voice_text}")
            text_input = voice_text
        else:
            st.error(voice_text)

     if st.button("Generate and Execute SQL Queries", key="generate_execute_queries"):
        st.session_state.generated_data = []
        if not text_input.strip():
            st.warning("Please enter at least one valid plain English query.")
            return

        queries = [q.strip() for q in text_input.split(";") if q.strip()]
        if not queries:
            st.warning("No valid queries provided. Please use semicolons to separate queries.")
            return

        with st.spinner("Generating and executing SQL queries..."):
            try:
                for idx, query in enumerate(queries, 1):
                    template = """
                        Create a SQL Query snippet using the below text:
                        {query}
                        I just want a SQL Query.
                    """
                    formatted_template = template.format(query=query)
                    response = model.generate_content(formatted_template)
                    sql_query = response.text.strip().lstrip("```sql").rstrip("```")
                    
                    explanation = generate_explanation(sql_query)
                    expected_output = generate_expected_output(sql_query)
                    
                    executed_results = None
                    if db_params.get("database"):
                        result = execute_query(sql_query, db_params)
                        if result["status"] == "success":
                            if "data" in result and not result["data"].empty:
                                executed_results = result["data"]
                                st.session_state.query_history.append({"query": sql_query, "result": result["data"]})
                            else:
                                executed_results = result["message"]
                                st.session_state.query_history.append({"query": sql_query, "result": result["message"]})
                        else:
                            executed_results = result["message"]
                            st.session_state.query_history.append({"query": sql_query, "result": result["message"]})
                    
                    st.session_state.generated_data.append({
                        "prompt": query,
                        "query": sql_query,
                        "explanation": explanation,
                        "output": expected_output,
                        "results": executed_results
                    })
            
            except Exception as e:
                st.error(f"An error occurred while processing queries: {e}")

    # Render generated content from session state
     if st.session_state.generated_data:
        st.subheader("Generated Queries and Results")
        for idx, data in enumerate(st.session_state.generated_data, 1):
            st.markdown(f"### Query {idx}: {data['prompt']}")
            st.success(f"Generated SQL Query {idx}:")
            st.code(data["query"], language="sql")
            
            st.success("Explanation:")
            st.markdown(data["explanation"])
            audio_bytes, tmp_file_path = text_to_speech(data["explanation"])
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3")
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            
            st.success("Expected Output:")
            st.markdown(data["output"])
            
            if data["results"] is not None:
                if isinstance(data["results"], pd.DataFrame):
                    st.success(f"Query {idx} executed successfully! Displaying results:")
                    st.dataframe(data["results"])
                else:
                    st.success(data["results"])
            
            st.download_button(
                label=f"Download SQL Query {idx}",
                data=data["query"],
                file_name=f"generated_query_{idx}.sql",
                mime="text/sql",
                key=f"download_query_{idx}"
            )
            
            st.markdown("---")

    # Download Report Section
     if st.session_state.generated_data:
        st.subheader("Download Query Report")
        download_format = st.selectbox(
            "Choose download format for the query report:",
            ["CSV", "JSON", "Text", "PDF"],
            key="download_format_generate_query"
        )
        
        report_data = pd.DataFrame(st.session_state.generated_data)
        report_data_copy = report_data.copy()
        
        if download_format == "CSV":
            report_data_copy["results"] = report_data_copy["results"].apply(
                lambda x: x.to_string(index=False) if isinstance(x, pd.DataFrame) else str(x) if pd.notna(x) else ""
            )
            st.download_button(
                label="Download CSV",
                data=report_data_copy.to_csv(index=False),
                file_name="sql_query_report.csv",
                mime="text/csv",
                key="download_csv_generate_query"
            )
        elif download_format == "JSON":
            report_data_copy["results"] = report_data_copy["results"].apply(
                lambda x: x.to_dict(orient="records") if isinstance(x, pd.DataFrame) else str(x) if pd.notna(x) else None
            )
            st.download_button(
                label="Download JSON",
                data=report_data_copy.to_json(orient="records"),
                file_name="sql_query_report.json",
                mime="application/json",
                key="download_json_generate_query"
            )
        elif download_format == "Text":
            text_content = ""
            for _, row in report_data_copy.iterrows():
                text_content += f"Prompt:\n{row['prompt']}\n\n"
                text_content += f"Query:\n{row['query']}\n\n"
                text_content += f"Explanation:\n{row['explanation']}\n\n"
                text_content += f"Expected Output:\n{row['output']}\n\n"
                if "results" in row and row["results"] is not None:
                    text_content += "Executed Query Results:\n"
                    if isinstance(row["results"], pd.DataFrame):
                        text_content += row["results"].to_string(index=False) + "\n"
                    else:
                        text_content += str(row["results"]) + "\n"
                text_content += "-" * 50 + "\n"
            st.download_button(
                label="Download Text",
                data=text_content,
                file_name="sql_query_report.txt",
                mime="text/plain",
                key="download_text_generate_query"
            )
        elif download_format == "PDF":
            pdf_content = generate_pdf_report(report_data_copy)
            st.download_button(
                label="Download PDF",
                data=pdf_content,
                file_name="sql_query_report.pdf",
                mime="application/pdf",
                key="download_pdf_generate_query"
            )

    elif tabs == "Upload SQL Files":
        if uploaded_files:
            # Clear previous generated_data to avoid duplicates
            st.session_state.generated_data = []
            # st.info("Processing uploaded SQL files...")
            for uploaded_file in uploaded_files:
                st.markdown(f"## File: {uploaded_file.name}")
                file_content = uploaded_file.read().decode("utf-8")
                queries = process_sql_file(file_content)

                for idx, query in enumerate(queries, start=1):
                    st.markdown(f"### Query {idx}")
                    st.code(query, language="sql")

                    explanation = generate_explanation(query)
                    expected_output = generate_expected_output(query)

                    st.success("Explanation:")
                    st.markdown(explanation)
                    if explanation:
                        # st.audio(text_to_speech(explanation), format="audio/mp3")
                        audio_bytes, tmp_file_path = text_to_speech(explanation)
                        if audio_bytes:
                         st.audio(audio_bytes, format="audio/mp3")
                        os.unlink(tmp_file_path)  # Clean up
                    # st.audio(text_to_speech(explanation), format="audio/mp3")

                    st.success("Expected Output:")
                    st.markdown(expected_output)

                    # Execute query if database connection is provided
                    executed_results = None
                    if all(db_params.values()):
                        st.info("Executing query...")
                        result = execute_query(query, db_params)
                        if result["status"] == "success":
                            if "data" in result and not result["data"].empty:
                                st.success("Query executed successfully! Displaying results:")
                                st.dataframe(result["data"])
                                executed_results = result["data"]  # Store the DataFrame
                            else:
                                st.success(result["message"])
                                executed_results = result["message"]  # Store the message
                                st.session_state.query_history.append({"query": query, "result": result["message"]})
                        else:
                            st.error(f"Error executing query: {result['message']}")
                            executed_results = result["message"]  # Store
                            st.session_state.query_history.append({"query": query, "result": result["message"]})


                    st.session_state.generated_data.append({
                        "prompt": f"Query {idx} from file {uploaded_file.name}",
                        "query": query,
                        "explanation": explanation,
                        "output": expected_output,
                        "results": executed_results if executed_results is not None else "No results (no DB connection or query failed)"
                    })

            # st.write("Generated Data for Report:", st.session_state.generated_data)
        else:
            st.warning("Please upload a SQL file to get started.")

        # Download Report Section
        if st.session_state.generated_data:
            report_data = pd.DataFrame(st.session_state.generated_data)
            st.subheader("Download Query Report")
            render_download_buttons(report_data)

    elif tabs == "Visualize Data":
     st.header("Visualize Data")

    # Initialize session state variables
     if "viz_text_input" not in st.session_state:
        st.session_state.viz_text_input = ""
     if "viz_sql_query" not in st.session_state:
        st.session_state.viz_sql_query = ""
     if "viz_data" not in st.session_state:
        st.session_state.viz_data = None
     if "all_columns" not in st.session_state:
        st.session_state.all_columns = []
     if "numeric_columns" not in st.session_state:
        st.session_state.numeric_columns = []
     if "categorical_columns" not in st.session_state:
        st.session_state.categorical_columns = []
     if "x_axis" not in st.session_state:
        st.session_state.x_axis = None
     if "y_axis" not in st.session_state:
        st.session_state.y_axis = None
     if "z_axis" not in st.session_state:
        st.session_state.z_axis = None
     if "color_column" not in st.session_state:
        st.session_state.color_column = None
     if "size_column" not in st.session_state:
        st.session_state.size_column = None
     if "chart_type" not in st.session_state:
        st.session_state.chart_type = "Bar"
     if "chart_title" not in st.session_state:
        st.session_state.chart_title = "Data Visualization"
     if "aggregation" not in st.session_state:
        st.session_state.aggregation = "None"  # For charts that need aggregation
     if "viz_fig" not in st.session_state:
        st.session_state.viz_fig = None

    # Text input for plain English query
     text_input = st.text_area(
        "Enter your data request in plain English:",
        placeholder="E.g., Show all employees in the IT department; Add a new customer; Update order price",
        value=st.session_state.viz_text_input,
        key="viz_text_input_area"
     )

    # Update session state with new input and reset only if input changes
     if text_input != st.session_state.viz_text_input:
        st.session_state.viz_text_input = text_input
        st.session_state.viz_sql_query = ""
        st.session_state.viz_data = None
        st.session_state.all_columns = []
        st.session_state.numeric_columns = []
        st.session_state.categorical_columns = []
        st.session_state.x_axis = None
        st.session_state.y_axis = None
        st.session_state.z_axis = None
        st.session_state.color_column = None
        st.session_state.size_column = None
        st.session_state.viz_fig = None

     if st.button("Generate and Execute Query"):
        # Clear previous state for a fresh run
        st.session_state.viz_sql_query = ""
        st.session_state.viz_data = None
        st.session_state.all_columns = []
        st.session_state.numeric_columns = []
        st.session_state.categorical_columns = []
        st.session_state.x_axis = None
        st.session_state.y_axis = None
        st.session_state.z_axis = None
        st.session_state.color_column = None
        st.session_state.size_column = None
        st.session_state.viz_fig = None

        if not text_input.strip():
            st.warning("Please enter a valid plain English request.")
        else:
            with st.spinner("Generating SQL Query..."):
                try:
                    template = """
                        Create a SQL Query snippet using the below text:
                        {text_input}
                        I just want a SQL Query.
                    """
                    formatted_template = template.format(text_input=text_input)
                    response = model.generate_content(formatted_template)
                    sql_query = response.text.strip().lstrip("```sql").rstrip("```")
                    st.session_state.viz_sql_query = sql_query

                    if db_params.get("database"):
                        st.info("Executing query...")
                        result = execute_query(sql_query, db_params)
                        if result["status"] == "success":
                            if "data" in result and isinstance(result["data"], pd.DataFrame) and not result["data"].empty:
                                # SELECT query with data
                                st.session_state.viz_data = result["data"]
                                st.session_state.all_columns = result["data"].columns.tolist()
                                st.session_state.numeric_columns = result["data"].select_dtypes(include=["number"]).columns.tolist()
                                st.session_state.categorical_columns = result["data"].select_dtypes(include=["object", "category"]).columns.tolist()
                                if st.session_state.categorical_columns:
                                    st.session_state.x_axis = st.session_state.categorical_columns[0]
                                elif st.session_state.all_columns:
                                    st.session_state.x_axis = st.session_state.all_columns[0]
                                if st.session_state.numeric_columns:
                                    st.session_state.y_axis = st.session_state.numeric_columns[0]
                                st.session_state.query_history.append({
                                    "query": sql_query,
                                    "result": result["data"]
                                })
                            else:
                                # Non-SELECT query (INSERT, UPDATE, DELETE, DDL)
                                modified_table = get_modified_table_from_query(sql_query)
                                if modified_table:
                                    table_data = fetch_table_data(db_params, modified_table)
                                    if isinstance(table_data, pd.DataFrame) and not table_data.empty:
                                        st.session_state.viz_data = table_data
                                        st.session_state.all_columns = table_data.columns.tolist()
                                        st.session_state.numeric_columns = table_data.select_dtypes(include=["number"]).columns.tolist()
                                        st.session_state.categorical_columns = table_data.select_dtypes(include=["object", "category"]).columns.tolist()
                                        if st.session_state.categorical_columns:
                                            st.session_state.x_axis = st.session_state.categorical_columns[0]
                                        elif st.session_state.all_columns:
                                            st.session_state.x_axis = st.session_state.all_columns[0]
                                        if st.session_state.numeric_columns:
                                            st.session_state.y_axis = st.session_state.numeric_columns[0]
                                        st.session_state.query_history.append({
                                            "query": sql_query,
                                            "result": table_data
                                        })
                                    else:
                                        st.session_state.viz_data = None
                                        st.session_state.query_history.append({
                                            "query": sql_query,
                                            "result": result.get("message", "No data returned")
                                        })
                                else:
                                    # For DDL queries like CREATE/ALTER/DROP, fetch schema
                                    tables = fetch_tables(db_params)
                                    if isinstance(tables, pd.DataFrame):
                                        st.session_state.viz_data = tables
                                        st.session_state.all_columns = tables.columns.tolist()
                                        st.session_state.numeric_columns = []
                                        st.session_state.categorical_columns = tables.columns.tolist()
                                        if st.session_state.categorical_columns:
                                            st.session_state.x_axis = st.session_state.categorical_columns[0]
                                        st.session_state.query_history.append({
                                            "query": sql_query,
                                            "result": tables
                                        })
                                    else:
                                        st.session_state.viz_data = None
                                        st.session_state.query_history.append({
                                            "query": sql_query,
                                            "result": result.get("message", "No data returned")
                                        })
                        else:
                            st.error(f"Error executing query: {result['message']}")
                            st.session_state.viz_data = None
                            st.session_state.query_history.append({
                                "query": sql_query,
                                "result": result["message"]
                            })
                    else:
                        st.warning("No database connection provided. Visualization requires a connected database.")
                        st.session_state.viz_data = None

                except Exception as e:
                    st.error(f"Error generating or executing query: {e}")
                    st.session_state.viz_data = None

    # Display results and visualization options
     if st.session_state.viz_sql_query:
        st.success("Generated SQL Query:")
        st.code(st.session_state.viz_sql_query, language="sql")

        if st.session_state.viz_data is not None:
            if not st.session_state.viz_data.empty:
                st.success("Query Results:")
                st.dataframe(st.session_state.viz_data)
                # Download data option
                st.download_button(
                    "Download Data as CSV",
                    data=st.session_state.viz_data.to_csv(index=False).encode('utf-8'),
                    file_name="visualization_data.csv",
                    mime="text/csv",
                    key="download_viz_data"
                )

                # Chart customization options
                st.subheader("Customize Your Chart")
                chart_title = st.text_input("Chart Title", value=st.session_state.chart_title, key="chart_title_input")
                st.session_state.chart_title = chart_title

                # Chart type selection with more options
                chart_type = st.selectbox(
                    "Select chart type",
                    ["Bar", "Line", "Pie", "Donut", "Scatter", "Bubble", "Histogram", "Box", "Area", "Stacked Area", "Stacked Bar", "3D Scatter", "Heatmap", "Treemap", "Sunburst", "Funnel", "Radar", "Waterfall"],
                    index=["Bar", "Line", "Pie", "Donut", "Scatter", "Bubble", "Histogram", "Box", "Area", "Stacked Area", "Stacked Bar", "3D Scatter", "Heatmap", "Treemap", "Sunburst", "Funnel", "Radar", "Waterfall"].index(st.session_state.chart_type) if st.session_state.chart_type in ["Bar", "Line", "Pie", "Donut", "Scatter", "Bubble", "Histogram", "Box", "Area", "Stacked Area", "Stacked Bar", "3D Scatter", "Heatmap", "Treemap", "Sunburst", "Funnel", "Radar", "Waterfall"] else 0,
                    key="chart_type_select"
                )
                st.session_state.chart_type = chart_type

                # Aggregation option for grouped charts
                if chart_type in ["Bar", "Line", "Scatter", "Area", "Stacked Area", "Stacked Bar"]:
                    aggregation = st.selectbox(
                        "Aggregation for Y-axis",
                        ["None", "Count", "Sum", "Average", "Min", "Max"],
                        index=["None", "Count", "Sum", "Average", "Min", "Max"].index(st.session_state.aggregation),
                        key="aggregation_select"
                    )
                    st.session_state.aggregation = aggregation

                # X/Y/Z axis selection with better defaults
                if st.session_state.all_columns:
                    x_axis = st.selectbox(
                        "Select X-axis column (categorical preferred for bars/pies)",
                        st.session_state.all_columns,
                        index=st.session_state.all_columns.index(st.session_state.x_axis) if st.session_state.x_axis in st.session_state.all_columns else 0,
                        key="x_axis_select"
                    )
                    y_axis = st.selectbox(
                        "Select Y-axis column (numeric preferred)",
                        [""] + st.session_state.numeric_columns,
                        index=st.session_state.numeric_columns.index(st.session_state.y_axis) + 1 if st.session_state.y_axis in st.session_state.numeric_columns else 0,
                        key="y_axis_select"
                    ) if chart_type != "Histogram" else None
                    z_axis = None
                    if chart_type in ["3D Scatter", "Bubble"]:
                        z_axis = st.selectbox(
                            "Select Z-axis column (for 3D) or Size (for Bubble)",
                            [""] + st.session_state.numeric_columns,
                            index=st.session_state.numeric_columns.index(st.session_state.z_axis) + 1 if st.session_state.z_axis in st.session_state.numeric_columns else 0,
                            key="z_axis_select"
                        )
                        st.session_state.z_axis = z_axis if z_axis != "" else None

                    color_column = st.selectbox(
                        "Select Color column (categorical for grouping)",
                        ["None"] + st.session_state.categorical_columns,
                        index=st.session_state.categorical_columns.index(st.session_state.color_column) + 1 if st.session_state.color_column in st.session_state.categorical_columns else 0,
                        key="color_select"
                    )
                    st.session_state.color_column = color_column if color_column != "None" else None

                    size_column = None
                    if chart_type in ["Scatter", "Bubble"]:
                        size_column = st.selectbox(
                            "Select Size column (numeric for bubble size)",
                            ["None"] + st.session_state.numeric_columns,
                            index=st.session_state.numeric_columns.index(st.session_state.size_column) + 1 if st.session_state.size_column in st.session_state.numeric_columns else 0,
                            key="size_select"
                        )
                        st.session_state.size_column = size_column if size_column != "None" else None

                    # Update session state with selections
                    st.session_state.x_axis = x_axis
                    st.session_state.y_axis = y_axis if y_axis != "" else None

                    # Generate and display chart
                    if st.button("Generate Chart"):
                        if st.session_state.x_axis:
                            with st.spinner("Generating visualization..."):
                                try:
                                    data = st.session_state.viz_data.copy()  # Avoid modifying original data
                                    # Apply aggregation if selected
                                    if st.session_state.aggregation != "None" and st.session_state.y_axis:
                                        agg_func = {
                                            "Count": "count",
                                            "Sum": "sum",
                                            "Average": "mean",
                                            "Min": "min",
                                            "Max": "max"
                                        }[st.session_state.aggregation]
                                        group_cols = [st.session_state.x_axis]
                                        if st.session_state.color_column:
                                            group_cols.append(st.session_state.color_column)
                                        data = data.groupby(group_cols)[st.session_state.y_axis].agg(agg_func).reset_index()

                                    # Chart generation with enhancements
                                    color = st.session_state.color_column
                                    size = st.session_state.size_column

                                    fig = None
                                    if chart_type == "Bar":
                                        fig = px.bar(data, x=x_axis, y=y_axis, color=color, title=chart_title, barmode="group" if color else None)
                                    elif chart_type == "Stacked Bar":
                                        fig = px.bar(data, x=x_axis, y=y_axis, color=color, title=chart_title, barmode="stack")
                                    elif chart_type == "Line":
                                        fig = px.line(data, x=x_axis, y=y_axis, color=color, title=chart_title)
                                    elif chart_type == "Pie":
                                        fig = px.pie(data, names=x_axis, values=y_axis, color=color, title=chart_title)
                                    elif chart_type == "Donut":
                                        fig = px.pie(data, names=x_axis, values=y_axis, color=color, hole=0.4, title=chart_title)
                                    elif chart_type == "Scatter":
                                        fig = px.scatter(data, x=x_axis, y=y_axis, color=color, size=size, title=chart_title)
                                    elif chart_type == "Bubble":
                                        if z_axis:
                                            fig = px.scatter(data, x=x_axis, y=y_axis, size=z_axis, color=color, title=chart_title)
                                        else:
                                            st.warning("Bubble chart requires a Size (Z-axis) column.")
                                    elif chart_type == "Histogram":
                                        fig = px.histogram(data, x=x_axis, color=color, title=chart_title)
                                    elif chart_type == "Box":
                                        fig = px.box(data, x=x_axis, y=y_axis, color=color, title=chart_title)
                                    elif chart_type == "Area":
                                        fig = px.area(data, x=x_axis, y=y_axis, color=color, title=chart_title)
                                    elif chart_type == "Stacked Area":
                                        fig = px.area(data, x=x_axis, y=y_axis, color=color, title=chart_title)
                                    elif chart_type == "3D Scatter":
                                        if y_axis and z_axis:
                                            fig = px.scatter_3d(data, x=x_axis, y=y_axis, z=z_axis, color=color, size=size, title=chart_title)
                                        else:
                                            st.warning("3D Scatter plot requires Y-axis and Z-axis columns.")
                                    elif chart_type == "Heatmap":
                                        if st.session_state.numeric_columns:
                                            corr = data[st.session_state.numeric_columns].corr()
                                            fig = px.imshow(corr, text_auto=True, title=chart_title)
                                        else:
                                            st.warning("Heatmap requires numeric columns.")
                                    elif chart_type == "Treemap":
                                        fig = px.treemap(data, path=[x_axis, color] if color else [x_axis], values=y_axis, title=chart_title)
                                    elif chart_type == "Sunburst":
                                        fig = px.sunburst(data, path=[x_axis, color] if color else [x_axis], values=y_axis, title=chart_title)
                                    elif chart_type == "Funnel":
                                        fig = px.funnel(data, x=x_axis, y=y_axis, color=color, title=chart_title)
                                    elif chart_type == "Radar":
                                        if y_axis:
                                            fig = px.line_polar(data, r=y_axis, theta=x_axis, color=color, line_close=True, title=chart_title)
                                        else:
                                            st.warning("Radar chart requires a Y-axis column.")
                                    elif chart_type == "Waterfall":
                                        if y_axis:
                                            fig = go.Figure(go.Waterfall(
                                                x=data[x_axis],
                                                y=data[y_axis],
                                                connector={"line": {"color": "rgb(63, 63, 63)"}}
                                            ))
                                            fig.update_layout(title=chart_title)
                                        else:
                                            st.warning("Waterfall chart requires a Y-axis column.")

                                    if fig:
                                        st.session_state.viz_fig = fig
                                except Exception as e:
                                    st.error(f"Error generating chart: {e}")
                        else:
                            st.warning("Please select an X-axis column to visualize.")
                    else:
                        st.warning("No columns available to visualize.")
                else:
                    st.warning("Query returned no data to visualize.")
            else:
                st.warning("No data available to visualize from the last execution.")

    # Display the chart and download options persistently if fig exists
     if st.session_state.viz_fig:
        st.plotly_chart(st.session_state.viz_fig, use_container_width=True)

        # Download chart option
        download_format = st.selectbox("Download chart as:", ["PNG", "SVG", "PDF"], key="download_format_select")
        buffer = io.BytesIO()
        st.session_state.viz_fig.write_image(buffer, format=download_format.lower())
        buffer.seek(0)
        mime_types = {
            "png": "image/png",
            "svg": "image/svg+xml",
            "pdf": "application/pdf"
        }
        mime = mime_types.get(download_format.lower(), "application/octet-stream")
        st.download_button(
            label="Download Chart",
            data=buffer.getvalue(),
            file_name=f"chart.{download_format.lower()}",
            mime=mime,
            key="download_chart_button"
        )

    # Reset button to clear visualization
     if st.session_state.viz_fig and st.button("Reset Visualization"):
        st.session_state.viz_fig = None
        st.rerun()
     else:
        st.info("Enter a request and click 'Generate and Execute Query' to start.")


    elif tabs == "Query History":
     st.header("Query History 📜")
     if "query_history" not in st.session_state:
      st.session_state.query_history = []
     if "show_clear_all_confirm" not in st.session_state:
      st.session_state.show_clear_all_confirm = False


     if st.session_state.query_history:
        # 🔄 Sort order
        filtered_history = list(enumerate(st.session_state.query_history, 1))
        sort_order = st.radio("Sort by:", ["Newest First", "Oldest First"], horizontal=True)
        if sort_order == "Newest First":
            filtered_history = list(reversed(filtered_history))

        # 📥 Export options
        col_export, col_clear_all = st.columns([1, 1])
        with col_export:
            export_format = st.selectbox("Export format:", ["Text", "JSON", "CSV"], key="export_format")
            if st.button("Export History", key="export_history"):
                if export_format == "Text":
                    text_content = ""
                    for idx, entry in enumerate(st.session_state.query_history, 1):
                        text_content += f"Query {idx}:\n{entry['query']}\n\nResult:\n"
                        if isinstance(entry["result"], pd.DataFrame):
                            text_content += entry["result"].to_string(index=False) + "\n"
                        else:
                            text_content += str(entry["result"]) + "\n"
                        text_content += "-" * 50 + "\n\n"
                    st.download_button(
                        label="Download History (Text)",
                        data=text_content,
                        file_name="query_history.txt",
                        mime="text/plain",
                        key="download_history_text"
                    )
                elif export_format == "JSON":
                    json_content = []
                    for idx, entry in enumerate(st.session_state.query_history, 1):
                        result = entry["result"].to_dict(orient="records") if isinstance(entry["result"], pd.DataFrame) else str(entry["result"])
                        json_content.append({"query_number": idx, "query": entry["query"], "result": result})
                    st.download_button(
                        label="Download History (JSON)",
                        data=json.dumps(json_content, indent=2, default=str),  # Handle non-serializable objects
                        file_name="query_history.json",
                        mime="application/json",
                        key="download_history_json"
                    )
                elif export_format == "CSV":
                    csv_buffer = io.StringIO()
                    rows = []
                    for idx, entry in enumerate(st.session_state.query_history, 1):
                        result = (
                            entry["result"].to_dict(orient="records")
                            if isinstance(entry["result"], pd.DataFrame)
                            else [{"result": str(entry["result"])}]  # Ensure consistent structure
                        )
                        rows.append({"Query #": idx, "Query": entry["query"], "Result": str(result)})
                    pd.DataFrame(rows).to_csv(csv_buffer, index=False)
                    st.download_button(
                        label="Download History (CSV)",
                        data=csv_buffer.getvalue(),
                        file_name="query_history.csv",
                        mime="text/csv",
                        key="download_history_csv"
                    )

        # 🧹 Clear All History with Confirmation
        with col_clear_all:
            if st.button("Clear All History", key="clear_all_history"):
                st.session_state.show_clear_all_confirm = True

        if st.session_state.show_clear_all_confirm:
            st.warning("⚠️ Are you sure you want to clear all query history? This cannot be undone.")
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                if st.button("✅ Yes, Clear All", key="confirm_clear_all"):
                    st.session_state.query_history = []
                    st.session_state.show_clear_all_confirm = False
                    st.success("All query history cleared!")
                    st.rerun()
            with col_c2:
                if st.button("❌ Cancel", key="cancel_clear_all"):
                    st.session_state.show_clear_all_confirm = False
                    st.info("Cancelled clearing history.")
                    st.rerun()

        # 📜 Display Queries
        for idx, entry in filtered_history:
            with st.expander(f"Query {idx}: {entry['query'][:50]}...", expanded=False):
                st.code(entry["query"], language="sql")
                st.write("**Result:**")
                if isinstance(entry["result"], pd.DataFrame):
                    st.dataframe(entry["result"])
                else:
                    st.write(entry["result"])
                if st.button("🗑️ Remove this query", key=f"clear_query_{idx}"):
                    # Remove by matching the exact entry to handle filtered/reversed lists
                    st.session_state.query_history.remove(entry)
                    st.success(f"Query {idx} removed from history!")
                    st.rerun()
     else:
        st.info("No queries in history yet. Run some queries in the AI Agent or other tabs!")
    
    
    elif tabs == "Database Schema":
     st.header("Database Schema Explorer 🗃️")

     if not all(db_params.values()):
        st.warning("Please connect to a database to view the schema.")
     else:
        schema = fetch_tables(db_params)
        if isinstance(schema, pd.DataFrame) and not schema.empty:
            # Original Graph Displayed First
            st.subheader("Schema Graph")
            G = nx.DiGraph()
            for table in schema["Table Name"]:
                table_data = fetch_table_data(db_params, table)
                G.add_node(table)
                for col in table_data.columns:
                    G.add_edge(table, col)
            try:
                st.graphviz_chart(nx.nx_pydot.to_pydot(G).to_string())
            except OSError as e:
                st.error(f"Graph visualization failed: {e}. Ensure Graphviz is installed and 'dot' is in your PATH.")

            # Interactive Table-Based Schema Explorer
            st.subheader("Schema Overview")
            tables = schema["Table Name"].tolist()
            for table in tables:
                with st.expander(f"Table: {table}", expanded=False):
                    try:

                        if db_params.get("type") == "sqlite":
                            conn = sqlite3.connect(db_params["database"])
                            cursor = conn.cursor()
                            cursor.execute(f"PRAGMA table_info({table})")
                            columns_info = pd.DataFrame(cursor.fetchall(), columns=["cid", "name", "type", "notnull", "default", "pk"])
                            columns_info = columns_info[["name", "type", "notnull", "default", "pk"]]
                            columns_info.columns = ["Field", "Type", "Null", "Default", "Key"]
                            columns_info["Null"] = columns_info["Null"].apply(lambda x: "NO" if x else "YES")
                            columns_info["Key"] = columns_info["Key"].apply(lambda x: "PRI" if x else "")
                        else:
                            conn = mysql.connector.connect(**{k: v for k, v in db_params.items() if k != "type"})
                            cursor = conn.cursor()
                            cursor.execute(f"SHOW COLUMNS FROM {table}")
                            columns_info = pd.DataFrame(cursor.fetchall(), columns=["Field", "Type", "Null", "Key", "Default", "Extra"])
                        
                        st.write("Columns:")
                        st.dataframe(columns_info)
                      

                        # Show sample data
                        sample_data = fetch_table_data(db_params, table).head(5)
                        st.write("Sample Data:")
                        st.dataframe(sample_data)

                        # Detect relationships

                        if db_params.get("type") == "sqlite":
                            cursor.execute(f"PRAGMA foreign_key_list({table})")
                            fk_info = cursor.fetchall()
                            relationships = [f"{fk[3]} → {fk[2]}" for fk in fk_info]  # from_column → to_table
                        else:
                            cursor.execute(f"SHOW CREATE TABLE {table}")
                            create_stmt = cursor.fetchone()[1].lower()
                            relationships = []
                            for line in create_stmt.split("\n"):
                                if "foreign key" in line:
                                    fk_col = line.split("`")[1]
                                    ref_table = line.split("references")[1].split("`")[1]
                                    relationships.append(f"{fk_col} → {ref_table}")
                        
                        if relationships:
                            st.write("Relationships:")
                            for rel in relationships:
                                st.write(f"- {rel}")
                        else:
                            st.write("No foreign key relationships detected.")
                        conn.close()
                    except Error as e:
                        st.error(f"Error fetching details for {table}: {e}")

            # Feature: Schema Export
            st.subheader("Export Schema")
            export_format = st.selectbox("Choose export format:", ["SQL Script", "JSON"])
            if st.button("Export"):
                if db_params.get("type") == "sqlite":
                    conn = sqlite3.connect(db_params["database"])
                else:
                    conn = mysql.connector.connect(**{k: v for k, v in db_params.items() if k != "type"})
                cursor = conn.cursor()





                
                if export_format == "SQL Script":
                    sql_script = ""
                    for table in tables:
                        cursor.execute(f"SHOW CREATE TABLE {table}")
                        sql_script += cursor.fetchone()[1] + ";\n\n"
                    st.download_button(
                        label="Download SQL Script",
                        data=sql_script,
                        file_name="schema.sql",
                        mime="text/sql"
                    )
                elif export_format == "JSON":
                    schema_json = {}
                    for table in tables:
                        cursor.execute(f"DESCRIBE {table}")
                        schema_json[table] = [{"column": row[0], "type": row[1]} for row in cursor.fetchall()]
                    st.download_button(
                        label="Download JSON",
                        data=json.dumps(schema_json, indent=2),
                        file_name="schema.json",
                        mime="application/json"
                    )
                conn.close()

            # Feature: Interactive Schema Editor (Enhanced with Remove Options)
            st.subheader("Schema Editor")
            action = st.selectbox("Choose an action:", ["Add Table", "Add Column", "Remove Column", "Remove Table"])
            
            if action == "Add Table":
                table_name = st.text_input("New Table Name:", key="new_table_name")
                column_def = st.text_area("Column Definitions (e.g., id INT PRIMARY KEY, name VARCHAR(255)):", key="new_table_cols")
                if st.button("Generate SQL", key="generate_add_table"):
                    if not table_name or not column_def:
                        st.warning("Please provide a table name and column definitions.")
                    elif not table_name.isalnum():
                        st.warning("Table name should contain only alphanumeric characters (no spaces or special characters).")
                    else:
                        sql = f"CREATE TABLE `{table_name}` ({column_def});"
                        st.code(sql, language="sql")
                        st.session_state["generated_sql"] = sql
                
                if "generated_sql" in st.session_state and st.button("Execute", key="execute_add_table"):
                    with st.spinner("Executing query..."):
                        result = execute_query(st.session_state["generated_sql"], db_params)
                        if result["status"] == "success":
                            st.success("Table created successfully!")
                            del st.session_state["generated_sql"]
                            st.rerun()
                        else:
                            st.error(f"Error executing query: {result['message']}")

            elif action == "Add Column":
                table_to_modify = st.selectbox("Select Table:", tables, key="select_table_to_modify_add")
                column_name = st.text_input("New Column Name:", key="new_column_name")
                column_type = st.text_input("Column Type (e.g., VARCHAR(255), INT):", key="new_column_type")
                if st.button("Generate SQL", key="generate_add_column"):
                    if not table_to_modify or not column_name or not column_type:
                        st.warning("Please provide a table, column name, and column type.")
                    else:
                        safe_column_name = f"`{column_name}`" if any(c in column_name for c in "+-*/() ") else column_name
                        sql = f"ALTER TABLE `{table_to_modify}` ADD COLUMN {safe_column_name} {column_type};"
                        st.code(sql, language="sql")
                        st.session_state["generated_sql"] = sql
                
                if "generated_sql" in st.session_state and st.button("Execute", key="execute_add_column"):
                    with st.spinner("Executing query..."):
                        result = execute_query(st.session_state["generated_sql"], db_params)
                        if result["status"] == "success":
                            st.success("Column added successfully!")
                            del st.session_state["generated_sql"]
                            st.rerun()
                        else:
                            st.error(f"Error executing query: {result['message']}")

            elif action == "Remove Column":
             table_to_modify = st.selectbox("Select Table:", tables, key="select_table_to_modify_remove_col")
             if table_to_modify:
        # ✅ Check which database type is being used
              if db_params.get("type") == "sqlite":
               conn = sqlite3.connect(db_params["database"])
               cursor = conn.cursor()
               cursor.execute(f"PRAGMA table_info({table_to_modify})")
               columns = [row[1] for row in cursor.fetchall()]  # column names are in second position
               conn.close()
              else:  # MySQL
               conn = mysql.connector.connect(**{k: v for k, v in db_params.items() if k != "type"})
               cursor = conn.cursor()
               cursor.execute(f"SHOW COLUMNS FROM `{table_to_modify}`")
               columns = [row[0] for row in cursor.fetchall()]
               conn.close()

              column_to_remove = st.selectbox("Select Column to Remove:", columns, key="column_to_remove")
              if st.button("Generate SQL", key="generate_remove_column"):
               if not column_to_remove:
                st.warning("Please select a column to remove.")
               else:
                safe_column_name = f"`{column_to_remove}`" if any(c in column_to_remove for c in "+-*/() ") else column_to_remove
                sql = f"ALTER TABLE `{table_to_modify}` DROP COLUMN {safe_column_name};"
                st.code(sql, language="sql")
                st.session_state["generated_sql"] = sql

              if "generated_sql" in st.session_state and st.button("Execute", key="execute_remove_column"):
               with st.spinner("Executing query..."):
                result = execute_query(st.session_state["generated_sql"], db_params)
                if result["status"] == "success":
                    st.success("Column removed successfully!")
                    del st.session_state["generated_sql"]
                    st.rerun()
                else:
                    st.error(f"Error executing query: {result['message']}")


            elif action == "Remove Table":
                table_to_remove = st.selectbox("Select Table to Remove:", tables, key="select_table_to_remove")
                if st.button("Generate SQL", key="generate_remove_table"):
                    if not table_to_remove:
                        st.warning("Please select a table to remove.")
                    else:
                        sql = f"DROP TABLE `{table_to_remove}`;"
                        st.code(sql, language="sql")
                        st.session_state["generated_sql"] = sql
                
                if "generated_sql" in st.session_state and st.button("Execute", key="execute_remove_table"):
                    with st.spinner("Executing query..."):
                        result = execute_query(st.session_state["generated_sql"], db_params)
                        if result["status"] == "success":
                            st.success("Table removed successfully!")
                            del st.session_state["generated_sql"]
                            st.rerun()
                        else:
                            st.error(f"Error executing query: {result['message']}")

        else:
            st.error("Error fetching schema: " + schema.get("message", "Unknown error"))


    elif tabs == "Learn SQL":
     st.header("Learn SQL with AI, Videos, and Quizzes 🎥📝")
    
    # SQL Learning Roadmap
     st.subheader("SQL Learning Roadmap 🛤️")
     roadmap = [
        "1️⃣ **Introduction to Databases & SQL** – Learn about relational databases and SQL basics.",
        "2️⃣ **Basic Queries** – SELECT, WHERE, ORDER BY, LIMIT.",
        "3️⃣ **Filtering & Aggregation** – GROUP BY, HAVING, COUNT, AVG, SUM.",
        "4️⃣ **Joins & Relationships** – INNER JOIN, LEFT JOIN, RIGHT JOIN, FULL JOIN.",
        "5️⃣ **Subqueries & Nested Queries** – Writing efficient subqueries.",
        "6️⃣ **Advanced SQL Functions** – CASE, COALESCE, Common Table Expressions (CTEs).",
        "7️⃣ **Indexes & Performance Optimization** – Indexing, Query Optimization.",
        "8️⃣ **Stored Procedures & Triggers** – Automating SQL tasks.",
        "9️⃣ **SQL for Data Analysis** – Window functions, analytical queries.",
        "🔟 **Practice & Real-World Applications** – Work on projects & real datasets."
    ]
     for step in roadmap:
        st.markdown(step)

    # Tabs for Learning and Quiz
     learn_tab, quiz_tab = st.tabs(["Learn SQL", "Take a Quiz"])

    # Learn SQL Tab
     with learn_tab:
        st.markdown("<h4 style='color: #333333;'>Ask a question about SQL (e.g., 'Explain SQL JOIN')</h4>", unsafe_allow_html=True)
        query = st.text_input("Enter your SQL question here", key="learn_sql_input")
        want_voice_for_topic = st.checkbox("Generate voice explanation for SQL Topic Content", value=False)

        if st.button("Generate", key="learn_generate"):
            if query:
                with st.spinner("Fetching SQL Topic Content..."):
                    try:
                        # Generate SQL topic content
                        topic_template = """
                            Provide a detailed explanation of the SQL topic: {query}
                            Focus on the content and concepts related to this topic.
                        """
                        formatted_template = topic_template.format(query=query)
                        response = model.generate_content(formatted_template)
                        sql_topic_content = response.text.strip()

                        if sql_topic_content:
                            st.success("SQL Topic Content:")
                            st.markdown(sql_topic_content)
                            if want_voice_for_topic:
                                with st.spinner("Generating voice explanation..."):
                                    audio_bytes, tmp_file_path = text_to_speech(sql_topic_content)
                                    if audio_bytes:
                                        st.audio(audio_bytes, format="audio/mp3")
                                    if tmp_file_path and os.path.exists(tmp_file_path):
                                        os.unlink(tmp_file_path)

                        # Fetch YouTube videos
                        videos = search_youtube_videos(YOUTUBE_API_KEY, query)
                        if videos:
                            st.write("Here are some tutorial videos for you:")
                            for video in videos:
                                st.markdown(f"[{video['title']}]({video['url']})")
                        else:
                            st.write("No videos found. Try another query.")

                        # Generate additional explanation
                        explanation = generate_explanation(query)
                        if explanation:
                            st.success("Additional Explanation:")
                            st.markdown(explanation)
                            with st.spinner("Generating voice explanation..."):
                                audio_bytes, tmp_file_path = text_to_speech(explanation)
                                if audio_bytes:
                                    st.audio(audio_bytes, format="audio/mp3")
                                if tmp_file_path and os.path.exists(tmp_file_path):
                                    os.unlink(tmp_file_path)

                    except Exception as e:
                        st.error(f"Error processing your request: {e}")
            else:
                st.warning("Please enter an SQL topic or question.")

    # Quiz Tab
     with quiz_tab:
        st.subheader("SQL Quiz Time! 📝")
        quiz_topic = st.text_input("Enter an SQL topic for the quiz (e.g., 'SQL Joins')", key="quiz_topic_input")
        
        # Add option to select number of questions
        num_questions = st.slider(
            "How many questions would you like in your quiz?",
            min_value=1,
            max_value=10,
            value=5,  # Default value
            step=1,
            key="num_questions_slider"
        )

        # Initialize session state variables
        if "quiz_questions" not in st.session_state:
            st.session_state.quiz_questions = []
        if "user_answers" not in st.session_state:
            st.session_state.user_answers = {}
        if "quiz_submitted" not in st.session_state:
            st.session_state.quiz_submitted = False
        if "incorrect_explanations" not in st.session_state:
            st.session_state.incorrect_explanations = {}  # Store explanations for incorrect answers
        if "show_explanations" not in st.session_state:
            st.session_state.show_explanations = False  # Toggle for showing explanations

        if st.button("Generate Quiz", key="generate_quiz"):
            if quiz_topic:
                with st.spinner(f"Generating {num_questions} quiz questions..."):
                    try:
                        quiz_template = """
                            Create a quiz with {num_questions} multiple-choice questions about the SQL topic: {quiz_topic}.
                            For each question, provide:
                            - The question text
                            - Four answer options (labeled a, b, c, d)
                            - The correct answer (e.g., 'a', 'b', 'c', or 'd')
                            Format each question as follows:
                            QX: [Question text]
                            a) [Option a]
                            b) [Option b]
                            c) [Option c]
                            d) [Option d]
                            Correct Answer: [correct answer letter]
                        """
                        formatted_quiz_template = quiz_template.format(num_questions=num_questions, quiz_topic=quiz_topic)
                        response = model.generate_content(formatted_quiz_template)
                        quiz_content = response.text.strip()

                        # Parse quiz content into a structured format
                        questions = []
                        lines = quiz_content.split("\n")
                        current_question = {}
                        for line in lines:
                            line = line.strip()
                            if line.startswith("Q"):
                                if current_question:
                                    questions.append(current_question)
                                current_question = {"question": line[3:].strip(), "options": {}, "correct_answer": None}
                            elif line.startswith(("a)", "b)", "c)", "d)")):
                                letter = line[0]
                                current_question["options"][letter] = line[3:].strip()
                            elif line.startswith("Correct Answer:"):
                                current_question["correct_answer"] = line.split(":")[1].strip()
                        if current_question:
                            questions.append(current_question)

                        # Ensure we have the requested number of questions
                        if len(questions) < num_questions:
                            st.warning(f"Only {len(questions)} questions were generated instead of {num_questions} due to API limitations.")
                        st.session_state.quiz_questions = questions[:num_questions]  # Trim to requested number
                        st.session_state.user_answers = {}
                        st.session_state.quiz_submitted = False
                        st.session_state.incorrect_explanations = {}
                        st.session_state.show_explanations = False
                        st.success(f"Quiz with {len(st.session_state.quiz_questions)} questions generated for '{quiz_topic}'!")
                    except Exception as e:
                        st.error(f"Error generating quiz: {e}")
            else:
                st.warning("Please enter an SQL topic for the quiz.")

        # Display Quiz Questions
        if st.session_state.quiz_questions and not st.session_state.quiz_submitted:
            st.markdown("### Your Quiz")
            for idx, q in enumerate(st.session_state.quiz_questions, 1):
                st.write(f"**Q{idx}: {q['question']}**")
                selected_answer = st.radio(
                    f"Select your answer for Q{idx}",
                    options=list(q["options"].keys()),
                    format_func=lambda x: f"{x} {q['options'][x]}",
                    key=f"q_{idx}"
                )
                st.session_state.user_answers[idx] = selected_answer

            if st.button("Submit Answers", key="submit_quiz"):
                # Generate explanations for incorrect answers
                with st.spinner("Evaluating answers..."):
                    for idx, q in enumerate(st.session_state.quiz_questions, 1):
                        user_answer = st.session_state.user_answers.get(idx)
                        if user_answer != q["correct_answer"]:
                            explanation_template = """
                                The user answered a quiz question incorrectly. Provide a concise explanation of why the user's answer was wrong and clarify the correct answer.
                                Question: {question}
                                User's Answer: {user_answer}
                                Correct Answer: {correct_answer}
                                Format the response as:
                                **Why Your Answer Was Incorrect**:
                                [Explanation of why the user's answer is wrong]
                                **Correct Answer Explanation**:
                                [Explanation of why the correct answer is right]
                            """
                            formatted_explanation = explanation_template.format(
                                question=q["question"],
                                user_answer=f"{user_answer} {q['options'][user_answer]}",
                                correct_answer=f"{q['correct_answer']} {q['options'][q['correct_answer']]}"
                            )
                            try:
                                explanation_response = model.generate_content(formatted_explanation)
                                st.session_state.incorrect_explanations[idx] = explanation_response.text.strip()
                            except Exception as e:
                                st.session_state.incorrect_explanations[idx] = f"Error generating explanation: {str(e)}"
                st.session_state.quiz_submitted = True
                st.rerun()

        # Evaluate and Display Results
        if st.session_state.quiz_submitted and st.session_state.quiz_questions:
            st.markdown("### Quiz Results")
            score = 0
            total = len(st.session_state.quiz_questions)
            for idx, q in enumerate(st.session_state.quiz_questions, 1):
                user_answer = st.session_state.user_answers.get(idx)
                correct_answer = q["correct_answer"]
                is_correct = user_answer == correct_answer
                if is_correct:
                    score += 1
                st.write(f"**Q{idx}: {q['question']}**")
                st.write(f"Your Answer: {user_answer} {q['options'][user_answer]}")
                st.write(f"Correct Answer: {correct_answer} {q['options'][correct_answer]}")
                st.write(f"Result: {'✅ Correct' if is_correct else '❌ Incorrect'}")
                st.markdown("---")

            percentage = (score / total) * 100
            st.markdown(f"**Your Score: {score}/{total} ({percentage:.2f}%)**")
            if percentage == 100:
                st.success("Perfect score! You're an SQL master! 🎉")
            elif percentage >= 70:
                st.success("Great job! You're getting the hang of it! 😊")
            else:
                st.info("Nice try! Keep practicing to improve your SQL skills! 📚")

            # Option to toggle explanations for incorrect answers
            st.session_state.show_explanations = st.checkbox("Show explanations for incorrect answers", value=st.session_state.show_explanations)
            if st.session_state.show_explanations:
                for idx, q in enumerate(st.session_state.quiz_questions, 1):
                    if st.session_state.user_answers.get(idx) != q["correct_answer"] and idx in st.session_state.incorrect_explanations:
                        st.markdown(f"**Explanation for Q{idx}**:\n{st.session_state.incorrect_explanations[idx]}")
                        st.markdown("---")

            if st.button("Try Another Quiz", key="reset_quiz"):
                st.session_state.quiz_questions = []
                st.session_state.user_answers = {}
                st.session_state.quiz_submitted = False
                st.session_state.incorrect_explanations = {}
                st.session_state.show_explanations = False
                st.rerun()

        elif not st.session_state.quiz_questions:
            st.info("Enter a topic, choose the number of questions, and click 'Generate Quiz' to start!")


    elif tabs == "SQL Playground":
     st.header("SQL Playground 🛠️")

    # Create subtabs for SQL, English, and Query Builder
     sql_tab, english_tab, builder_tab = st.tabs(["SQL Queries", "Plain English Queries", "Interactive Query Builder"])

    # -------------------- SQL QUERIES SUBTAB --------------------
     with sql_tab:
        st.subheader("Write and Execute SQL Queries")
        st.markdown("Enter multiple SQL queries separated by semicolons (e.g., `SELECT * FROM customers; SELECT * FROM orders`).")

        user_query = st.text_area(
            "Write your SQL queries here:",
            height=200,
            placeholder="e.g., SELECT * FROM customers; SELECT * FROM orders",
            key="sql_playground_query"
        )
        
        if st.button("Run SQL Queries", key="run_sql_queries"):
            if not user_query.strip():
                st.warning("Please enter at least one valid SQL query.")
            elif not db_params.get("database"):
                st.warning("Please connect to a database to execute queries.")
            else:
                with st.spinner("Executing SQL queries..."):
                    queries = process_sql_file(user_query)  # Split queries by semicolon
                    if not queries:
                        st.warning("No valid queries provided. Please use semicolons to separate queries.")
                    else:
                        for idx, query in enumerate(queries, 1):
                            st.markdown(f"### Query {idx}")
                            st.code(query, language="sql")
                            result = execute_query(query, db_params)
                            if result["status"] == "success":
                                if "data" in result and not result["data"].empty:
                                    st.success(f"Query {idx} executed successfully! Displaying results:")
                                    st.dataframe(result["data"])
                                    st.session_state.query_history.append({"query": query, "result": result["data"]})
                                else:
                                    st.success(result["message"])
                                    st.session_state.query_history.append({"query": query, "result": result["message"]})
                            else:
                                st.error(f"Query {idx} failed: {result['message']}")
                                st.session_state.query_history.append({"query": query, "result": result["message"]})
                            st.markdown("---")

    # -------------------- PLAIN ENGLISH QUERIES SUBTAB --------------------
     with english_tab:
        st.subheader("Interact with Database Using Plain English")
        st.markdown("Enter multiple queries in plain English, separated by semicolons (e.g., `Show all customers; Count orders by customer`).")

        english_input = st.text_area(
            "Describe your queries in plain English:",
            height=200,
            placeholder="e.g., Show all customers; Count orders by customer",
            key="english_playground_query"
        )
        
        if st.button("Run English Queries", key="run_english_queries"):
            if not english_input.strip():
                st.warning("Please enter at least one valid plain English query.")
            elif not db_params.get("database"):
                st.warning("Please connect to a database to execute queries.")
            else:
                with st.spinner("Generating and executing queries..."):
                    queries = [q.strip() for q in english_input.split(";") if q.strip()]
                    if not queries:
                        st.warning("No valid queries provided. Please use semicolons to separate queries.")
                    else:
                        for idx, query in enumerate(queries, 1):
                            st.markdown(f"### Query {idx}: {query}")
                            try:
                                # Generate SQL from plain English
                                template = """
                                    Create a SQL Query snippet using the below text:
                                    {english_input}
                                    I just want a SQL Query.
                                """
                                formatted_template = template.format(english_input=query)
                                response = model.generate_content(formatted_template)
                                sql_query = response.text.strip().lstrip("```sql").rstrip("```")
                                
                                # Display the generated SQL
                                st.success(f"Generated SQL Query {idx}:")
                                st.code(sql_query, language="sql")
                                
                                # Execute the query
                                result = execute_query(sql_query, db_params)
                                if result["status"] == "success":
                                    if "data" in result and not result["data"].empty:
                                        st.success(f"Query {idx} executed successfully! Displaying results:")
                                        st.dataframe(result["data"])
                                        st.session_state.query_history.append({"query": sql_query, "result": result["data"]})
                                    else:
                                        st.success(result["message"])
                                        st.session_state.query_history.append({"query": sql_query, "result": result["message"]})
                                else:
                                    st.error(f"Query {idx} failed: {result['message']}")
                                    st.session_state.query_history.append({"query": sql_query, "result": result["message"]})
                            except Exception as e:
                                st.error(f"Error generating or executing Query {idx}: {e}")
                            st.markdown("---")
     with builder_tab:
      st.subheader("🛠️ Interactive Query Builder")
      st.markdown("Build and execute queries visually without writing SQL.")

      query_type = st.radio(
        "Choose Query Type:",
        ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE TABLE", "ALTER TABLE", "DROP TABLE"],
        horizontal=True
      )

    # -------------------- SELECT QUERY BUILDER --------------------
      if query_type == "SELECT":
        tables = fetch_tables(db_params)
        if isinstance(tables, pd.DataFrame) and not tables.empty:
            table_choice = st.selectbox("Choose Main Table", tables["Table Name"].tolist())

            if table_choice:
                table_data = fetch_table_data(db_params, table_choice)
                if isinstance(table_data, pd.DataFrame):
                    columns = table_data.columns.tolist()
                    selected_columns = st.multiselect("Select Columns", columns, default=columns[:2])

                    # JOIN support
                    st.markdown("### 🔗 Add Join (Optional)")
                    join_table = st.selectbox("Join with Table", ["None"] + tables["Table Name"].tolist())
                    join_condition = ""
                    if join_table != "None":
                        join_type = st.selectbox("Join Type", ["INNER JOIN", "LEFT JOIN", "RIGHT JOIN"])
                        join_col1 = st.selectbox(f"{table_choice} Column", columns)
                        join_col2 = st.selectbox(f"{join_table} Column", fetch_table_data(db_params, join_table).columns.tolist())
                        join_condition = f" {join_type} {join_table} ON {table_choice}.{join_col1} = {join_table}.{join_col2}"

                    # WHERE clause
                    st.markdown("### ➕ Add Filter")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        filter_col = st.selectbox("Column", ["None"] + columns)
                    with col2:
                        operator = st.selectbox("Operator", ["=", "!=", ">", "<", ">=", "<=", "LIKE"])
                    with col3:
                        filter_val = st.text_input("Value")
                    where_clause = ""
                    if filter_col != "None" and filter_val:
                        where_clause = f" WHERE {table_choice}.{filter_col} {operator} '{filter_val}'"

                    # ORDER BY & LIMIT
                    st.markdown("### 🔽 Order & Limit")
                    order_col = st.selectbox("Order By", ["None"] + columns)
                    order_dir = st.radio("Order", ["ASC", "DESC"], horizontal=True)
                    limit = st.number_input("Limit Rows", min_value=1, value=10)

                    if st.button("⚡ Generate & Run Query", key="run_select_query"):
                        sql_query = f"SELECT {', '.join(selected_columns) if selected_columns else '*'} FROM {table_choice}"
                        if join_condition:
                            sql_query += join_condition
                        if where_clause:
                            sql_query += where_clause
                        if order_col != "None":
                            sql_query += f" ORDER BY {order_col} {order_dir}"
                        if limit:
                            sql_query += f" LIMIT {limit}"

                        st.code(sql_query, language="sql")
                        result = execute_query(sql_query, db_params)
                        if result["status"] == "success":
                            if isinstance(result["data"], pd.DataFrame) and not result["data"].empty:
                                st.success("✅ Query executed successfully!")
                                st.dataframe(result["data"])
                            else:
                                st.success(result.get("message", "Query executed"))
                        else:
                            st.error(result["message"])
        else:
            st.warning("No tables found in the database.")

    # -------------------- INSERT QUERY BUILDER --------------------
      elif query_type == "INSERT":
        tables = fetch_tables(db_params)
        if isinstance(tables, pd.DataFrame) and not tables.empty:
            table_choice = st.selectbox("Choose Table", tables["Table Name"].tolist())
            if table_choice:
                cols = fetch_table_data(db_params, table_choice).columns.tolist()
                values = {}
                for col in cols:
                    values[col] = st.text_input(f"Value for {col}")

                if st.button("Insert Row"):
                    cols_str = ", ".join(values.keys())
                    vals_str = ", ".join([f"'{v}'" for v in values.values()])
                    sql_query = f"INSERT INTO {table_choice} ({cols_str}) VALUES ({vals_str});"
                    st.code(sql_query, language="sql")
                    result = execute_query(sql_query, db_params)
                    if result["status"] == "success":
                        st.success("✅ Row inserted successfully!")
                    else:
                        st.error(result["message"])

    # -------------------- UPDATE QUERY BUILDER --------------------
      elif query_type == "UPDATE":
        tables = fetch_tables(db_params)
        if isinstance(tables, pd.DataFrame) and not tables.empty:
            table_choice = st.selectbox("Choose Table", tables["Table Name"].tolist())
            if table_choice:
                cols = fetch_table_data(db_params, table_choice).columns.tolist()
                col_to_update = st.selectbox("Column to Update", cols)
                new_value = st.text_input("New Value")
                condition = st.text_input("WHERE condition (optional)", placeholder="e.g., id=5")

                if st.button("Update Rows"):
                    sql_query = f"UPDATE {table_choice} SET {col_to_update} = '{new_value}'"
                    if condition:
                        sql_query += f" WHERE {condition}"
                    st.code(sql_query, language="sql")
                    result = execute_query(sql_query, db_params)
                    if result["status"] == "success":
                        st.success("✅ Rows updated successfully!")
                    else:
                        st.error(result["message"])

    # -------------------- DELETE QUERY BUILDER --------------------
      elif query_type == "DELETE":
       tables = fetch_tables(db_params)
       if isinstance(tables, pd.DataFrame) and not tables.empty:
        table_choice = st.selectbox("Choose Table", tables["Table Name"].tolist())

        if table_choice:
            table_data = fetch_table_data(db_params, table_choice)
            if isinstance(table_data, pd.DataFrame) and not table_data.empty:
                st.markdown("### Preview Table Data")
                st.dataframe(table_data)

                # Select column used for identifying rows (primary key / unique column)
                cols = table_data.columns.tolist()
                id_col = st.selectbox("Select Identifier Column", cols)

                # Pick a row to delete
                row_ids = table_data[id_col].unique().tolist()
                selected_id = st.selectbox(f"Select {id_col} to Delete", row_ids)

                # Generate DELETE query for selected row
                if st.button("🗑️ Delete Selected Row"):
                    sql_query = f"DELETE FROM {table_choice} WHERE {id_col} = '{selected_id}'"
                    st.code(sql_query, language="sql")

                    result = execute_query(sql_query, db_params)
                    if result["status"] == "success":
                        st.success(f"✅ Row with {id_col} = {selected_id} deleted successfully!")
                    else:
                        st.error(result["message"])

            else:
                st.warning("No data available in this table to delete.")


    # -------------------- CREATE TABLE BUILDER --------------------
      elif query_type == "CREATE TABLE":
        table_name = st.text_input("New Table Name")
        num_cols = st.number_input("Number of Columns", min_value=1, value=2)

        cols = []
        for i in range(num_cols):
            col_name = st.text_input(f"Column {i+1} Name", key=f"col_name_{i}")
            col_type = st.selectbox(
                f"Column {i+1} Type",
                ["INT", "VARCHAR(255)", "TEXT", "DATE", "FLOAT", "BOOLEAN"],
                key=f"col_type_{i}"
            )
            cols.append(f"{col_name} {col_type}")

        if st.button("Create Table"):
            if table_name and all(cols):
                sql_query = f"CREATE TABLE {table_name} ({', '.join(cols)});"
                st.code(sql_query, language="sql")
                result = execute_query(sql_query, db_params)
                if result["status"] == "success":
                    st.success(f"✅ Table `{table_name}` created successfully!")
                else:
                    st.error(result["message"])
            else:
                st.error("Please provide a table name and column definitions.")

    # -------------------- ALTER TABLE BUILDER --------------------
      elif query_type == "ALTER TABLE":
        tables = fetch_tables(db_params)
        if isinstance(tables, pd.DataFrame) and not tables.empty:
            table_choice = st.selectbox("Choose Table", tables["Table Name"].tolist())
            action = st.selectbox("Action", ["ADD COLUMN", "MODIFY COLUMN", "DROP COLUMN"])

            if action == "ADD COLUMN":
                new_col = st.text_input("New Column Name")
                new_type = st.selectbox("Data Type", ["INT", "VARCHAR(255)", "TEXT", "DATE", "FLOAT", "BOOLEAN"])
                if st.button("Alter Table"):
                    sql_query = f"ALTER TABLE {table_choice} ADD COLUMN {new_col} {new_type};"
                    st.code(sql_query, language="sql")
                    result = execute_query(sql_query, db_params)
                    if result["status"] == "success":
                        st.success("✅ Column added successfully!")
                    else:
                        st.error(result["message"])

            elif action == "MODIFY COLUMN":
                cols = fetch_table_data(db_params, table_choice).columns.tolist()
                col_to_modify = st.selectbox("Select Column", cols)
                new_type = st.selectbox("New Data Type", ["INT", "VARCHAR(255)", "TEXT", "DATE", "FLOAT", "BOOLEAN"])
                if st.button("Alter Table"):
                    sql_query = f"ALTER TABLE {table_choice} MODIFY COLUMN {col_to_modify} {new_type};"
                    st.code(sql_query, language="sql")
                    result = execute_query(sql_query, db_params)
                    if result["status"] == "success":
                        st.success("✅ Column modified successfully!")
                    else:
                        st.error(result["message"])

            elif action == "DROP COLUMN":
                cols = fetch_table_data(db_params, table_choice).columns.tolist()
                col_to_drop = st.selectbox("Select Column to Drop", cols)
                if st.button("Alter Table"):
                    sql_query = f"ALTER TABLE {table_choice} DROP COLUMN {col_to_drop};"
                    st.code(sql_query, language="sql")
                    result = execute_query(sql_query, db_params)
                    if result["status"] == "success":
                        st.success("✅ Column dropped successfully!")
                    else:
                        st.error(result["message"])

    # -------------------- DROP TABLE BUILDER --------------------
      elif query_type == "DROP TABLE":
        tables = fetch_tables(db_params)
        if isinstance(tables, pd.DataFrame) and not tables.empty:
            table_choice = st.selectbox("Choose Table to Drop", tables["Table Name"].tolist())
            confirm = st.checkbox("⚠️ I understand this will permanently delete the table")
            if st.button("Drop Table") and confirm:
                sql_query = f"DROP TABLE {table_choice};"
                st.code(sql_query, language="sql")
                result = execute_query(sql_query, db_params)
                if result["status"] == "success":
                    st.success(f"✅ Table `{table_choice}` dropped successfully!")
                else:
                    st.error(result["message"])


      else:
            st.warning("No tables found in the database.")

    



if __name__ == "__main__":
    main()









