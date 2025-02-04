import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import pandas as pd
import math
import json
import time
import random
from typing import List, Dict, Any
import threading
import queue
from datetime import datetime
import logging
import re
import os
from fpdf import FPDF  # For PDF generation
import requests
import xml.etree.ElementTree as ET
from flask import Flask, request, jsonify, send_from_directory
import urllib.parse

# ----------------------- Configuration ------------------------
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"  # Replace with your actual API key
GEMINI_MODEL_NAME = "models/embedding-001"     # Specify the embedding model to use
GEMINI_CHAT_MODEL_NAME = "models/gemini-pro"     # Specify the chat model
INDEX_FILE = "legal_index.json"
LOG_FILE = "legal_expert_system.log"
SCENARIO_DIRECTORY = "scenarios"  # Directory to store scenario JSON files

SIMULATE_EMBEDDINGS = False      # Set to False to use the Gemini API
SIMULATION_VECTOR_LENGTH = 384

# API Configuration
ENABLE_FLASK_API = False         # Set to True to enable the Flask API
API_PORT = 5000

# Multithreading / Rate Limiting
MAX_THREADS = 5
REQUEST_DELAY_SECONDS = 1
RANDOMIZE_REQUEST_DELAY = True

# AI Engine Configuration
DEFAULT_AI_ENGINE = "Gemini"

# User Files Location
USER_FILES_DIR = os.path.expanduser("~/.legal_expert_system")

# Credentials and Connectors
CREDENTIALS_FILE = os.path.join(USER_FILES_DIR, "credentials.json")
CONNECTORS_FILE = os.path.join(USER_FILES_DIR, "connectors.json")

# ----------------------- File System Setup ------------------------
os.makedirs(SCENARIO_DIRECTORY, exist_ok=True)
os.makedirs(USER_FILES_DIR, exist_ok=True)

# ----------------------- License and Heading ------------------------
LICENSE_TEXT = """
MIT License

Copyright (c) 2023 Travis Michael O'Dell

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

HEADING = "# #theirleft"
SUMMARY = "This is an AI-powered legal research and analysis application. It allows users to search legal databases, analyze scenarios, generate reports, and more."

# ----------------------- Logging ------------------------
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------- Rate Limiting ------------------------
last_request_time = {}  # Track last request per domain

# ----------------------- Task Management ------------------------
task_queue = queue.Queue()
task_results = {}
task_counter = 0
task_lock = threading.Lock()

def task_worker():
    while True:
        task_function, args, task_id = task_queue.get()
        try:
            if task_function == "index_website":
                url, category = args
                result = process_website_wrapper(url, category)
                task_results[task_id] = result
            elif task_function == "gemini_search":
                query = args[0]
                result = gemini_search(query)
                task_results[task_id] = result
        except Exception as e:
            logging.error(f"Task {task_id} failed: {e}")
            task_results[task_id] = {"status": "failed", "error": str(e)}
        finally:
            task_queue.task_done()

worker_thread = threading.Thread(target=task_worker, daemon=True)
worker_thread.start()

def add_task(task_function, *args):
    global task_counter
    with task_lock:
        task_counter += 1
        task_id = task_counter
    task_queue.put((task_function, args, task_id))
    task_results[task_id] = {"status": "pending"}
    return task_id

def get_task_status(task_id):
    return task_results.get(task_id, {"status": "unknown"})

# ----------------------- AI Engine Management ------------------------
ai_engines = {}  # Loaded from CONNECTORS_FILE
current_ai_engine = DEFAULT_AI_ENGINE

def load_ai_engine_configs():
    global ai_engines
    try:
        with open(CONNECTORS_FILE, "r") as f:
            ai_engines = json.load(f)
    except FileNotFoundError:
        logging.warning(f"AI engine config file not found: {CONNECTORS_FILE}")
        ai_engines = {}

def save_ai_engine_configs():
    try:
        with open(CONNECTORS_FILE, "w") as f:
            json.dump(ai_engines, f, indent=4)
    except Exception as e:
        logging.error(f"Error saving AI engine configurations: {e}")

def set_current_ai_engine(engine_name: str):
    global current_ai_engine
    if engine_name in ai_engines:
        current_ai_engine = engine_name
        print(f"AI engine set to: {current_ai_engine}")
    else:
        raise ValueError(f"Invalid AI engine name: {engine_name}")

load_ai_engine_configs()

# ----------------------- Gemini Modules -----------------------
class GeminiEmbedder:
    def __init__(self, model_name: str = GEMINI_MODEL_NAME, api_key: str = GEMINI_API_KEY,
                 simulate: bool = SIMULATE_EMBEDDINGS):
        self.model_name = model_name
        self.api_key = api_key
        self.simulate = simulate
        self.vector_length = SIMULATION_VECTOR_LENGTH

    def embed(self, text: str) -> List[float]:
        if self.simulate:
            print(f"Simulating embedding for: {text[:50]}...")
            return self._simulate_embedding()
        else:
            print(f"Calling Gemini API to embed: {text[:50]}...")
            return self._call_gemini_api(text)

    def _call_gemini_api(self, text: str) -> List[float]:
        time.sleep(0.5)
        print("Calling Gemini API...")
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model_name)
            response = model.embed_content(content=text)
            print("Gemini API call successful.")
            return response['embedding']
        except Exception as e:
            print(f"Error calling Gemini API: {e}. Falling back to simulation.")
            return self._simulate_embedding()

    def _simulate_embedding(self) -> List[float]:
        return [random.uniform(-0.1, 0.1) for _ in range(self.vector_length)]

class GeminiChat:
    def __init__(self, model_name: str = GEMINI_CHAT_MODEL_NAME, api_key: str = GEMINI_API_KEY):
        self.model_name = model_name
        self.api_key = api_key
        self.model = None

    def _initialize_model(self):
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
        except Exception as e:
            logging.error(f"Error initializing Gemini model: {e}")
            raise

    def generate_report(self, prompt: str, context: List[str] = None) -> str:
        if self.model is None:
            self._initialize_model()
        full_prompt = ""
        if context:
            full_prompt += "Here is some relevant information:\n"
            for item in context:
                full_prompt += item + "\n"
            full_prompt += "\n"
        full_prompt += prompt
        try:
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            logging.error(f"Error generating report with Gemini: {e}")
            return f"Error generating report: {e}"

    def generate_questions(self, scenario_data: dict) -> List[str]:
        if self.model is None:
            self._initialize_model()
        prompt = f"""
Based on the following legal scenario, please generate a list of follow-up questions that could help clarify the situation:

Scenario:
{json.dumps(scenario_data, indent=2)}

Questions:
"""
        try:
            response = self.model.generate_content(prompt)
            questions_text = response.text
            questions = re.findall(r'\d+\.\s*(.*)', questions_text)
            return questions
        except Exception as e:
            logging.error(f"Error generating questions: {e}")
            return [f"Error generating questions: {e}"]

    def analyze_scenario(self, scenario_data: dict) -> dict:
        if self.model is None:
            self._initialize_model()
        questions = self.generate_questions(scenario_data)
        prompt = f"""
Analyze the following legal scenario and provide a detailed assessment including:
- A general assessment of the situation.
- Possible outcomes.
- Likely outcomes based on similar cases.
- Objectives of the prosecution and defense.
- Potential strategies.
- Analysis of law enforcement actions.
- Follow-up questions.

Scenario:
{json.dumps(scenario_data, indent=2)}

Analysis:
"""
        try:
            response = self.model.generate_content(prompt)
            analysis_text = response.text
            assessment = re.search(r"Assessment:\s*(.*?)Possible Outcomes:", analysis_text, re.DOTALL)
            possible_outcomes = re.search(r"Possible Outcomes:\s*(.*?)Likely Outcomes:", analysis_text, re.DOTALL)
            likely_outcomes = re.search(r"Likely Outcomes:\s*(.*?)Objectives of the Prosecution:", analysis_text, re.DOTALL)
            prosecution_objectives = re.search(r"Objectives of the Prosecution:\s*(.*?)Objectives of the Defense:", analysis_text, re.DOTALL)
            defense_objectives = re.search(r"Objectives of the Defense:\s*(.*?)Prosecution Strategies:", analysis_text, re.DOTALL)
            prosecution_strategies = re.search(r"Prosecution Strategies:\s*(.*?)Defense Strategies:", analysis_text, re.DOTALL)
            defense_strategies = re.search(r"Defense Strategies:\s*(.*?)Law Enforcement Actions:", analysis_text, re.DOTALL)
            law_enforcement_actions = re.search(r"Law Enforcement Actions:\s*(.*?)Summary:", analysis_text, re.DOTALL)
            summary = re.search(r"Summary:\s*(.*?)Relevant Laws:", analysis_text, re.DOTALL)
            relevant_laws = re.search(r"Relevant Laws:\s*(.*)", analysis_text, re.DOTALL)
            return {
                "assessment": assessment.group(1).strip() if assessment else "Not Available",
                "questions": questions,
                "possible_outcomes": possible_outcomes.group(1).strip() if possible_outcomes else "Not Available",
                "likely_outcomes": likely_outcomes.group(1).strip() if likely_outcomes else "Not Available",
                "prosecution_objectives": prosecution_objectives.group(1).strip() if prosecution_objectives else "Not Applicable",
                "defense_objectives": defense_objectives.group(1).strip() if defense_objectives else "Not Applicable",
                "prosecution_strategies": prosecution_strategies.group(1).strip() if prosecution_strategies else "Not Applicable",
                "defense_strategies": defense_strategies.group(1).strip() if defense_strategies else "Not Applicable",
                "law_enforcement_actions": law_enforcement_actions.group(1).strip() if law_enforcement_actions else "Not Available",
                "summary": summary.group(1).strip() if summary else "Not Available",
                "relevant_laws": relevant_laws.group(1).strip() if relevant_laws else "Not Available"
            }
        except Exception as e:
            logging.error(f"Error analyzing scenario: {e}")
            return {"error": f"Error analyzing scenario: {e}", "questions": []}

# ----------------------- Data Processing and Indexing -----------------------
class VectorIndexEngine:
    def __init__(self, index_file: str = INDEX_FILE):
        self.index = []
        self.index_file = index_file
        self.load_index()

    def index_document(self, doc_id: str, embedding: List[float], metadata: Dict[str, Any], full_doc: Dict[str, Any]):
        record = {
            "id": doc_id,
            "embedding": embedding,
            "metadata": metadata,
            "document": full_doc,
        }
        self.index.append(record)
        print(f"Indexed document {doc_id}")
        self.save_index()

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        return 0.0 if norm1 == 0 or norm2 == 0 else dot_product / (norm1 * norm2)

    def search(self, query_embedding: List[float], top_k: int = 5, metadata_filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        results = []
        for record in self.index:
            if metadata_filters:
                match = all(record["metadata"].get(key) == value for key, value in metadata_filters.items())
                if not match:
                    continue
            similarity = self.cosine_similarity(query_embedding, record["embedding"])
            results.append((similarity, record))
        results.sort(key=lambda x: x[0], reverse=True)
        return [{"similarity": sim, "document": rec["document"]} for sim, rec in results[:top_k]]

    def save_index(self):
        try:
            with open(self.index_file, "w") as f:
                json.dump(self.index, f, indent=4)
            print(f"Index saved to {self.index_file}")
        except Exception as e:
            print(f"Error saving index: {e}")

    def load_index(self):
        try:
            with open(self.index_file, "r") as f:
                self.index = json.load(f)
            print(f"Index loaded from {self.index_file} with {len(self.index)} documents.")
        except FileNotFoundError:
            print("Index file not found. Starting with an empty index.")
            self.index = []
        except Exception as e:
            print(f"Error loading index: {e}. Starting with an empty index.")
            self.index = []

def gemini_search(query: str):
    print(f"Performing Gemini search for: {query}")
    time.sleep(1)
    return [
        {"url": f"https://example.com/search-result-1?q={query}", "title": "Search Result 1"},
        {"url": f"https://example.com/search-result-2?q={query}", "title": "Search Result 2"},
        {"url": f"https://example.com/search-result-3?q={query}", "title": "Search Result 3"},
    ]

# ----------------------- Website Processing -----------------------
class WebsiteCrawler:
    def __init__(self, url):
        self.url = url
        self.domain = urllib.parse.urlparse(url).netloc

    def fetch_site(self):
        print(f"Fetching site: {self.url}")
        try:
            delay = REQUEST_DELAY_SECONDS
            if RANDOMIZE_REQUEST_DELAY:
                delay += random.uniform(0, delay / 2)
            time.sleep(delay)
            raw_html = f"<html><body><h1>Fetched Site from {self.url}</h1><p>Content from {self.url}</p></body></html>"
            raw_css = "body { font-family: sans-serif; }"
            return raw_html, raw_css
        except Exception as e:
            print(f"Error fetching: {e}")
            return "<html><body>Error fetching site</body></html>", ""

    def render_page_and_capture(self):
        print(f"Rendering and capturing screenshot for: {self.url}")
        time.sleep(1)
        return "screenshot_data.png"

class DOMParser:
    def __init__(self):
        self.metadata = {}

    def parse(self, html: str, css: str):
        print("Parsing HTML and CSS...")
        time.sleep(0.5)
        dom_tree = {
            "tag": "html",
            "children": [{
                "tag": "body",
                "children": [
                    {"tag": "h1", "text": "Example Site", "id": "main-heading"},
                    {"tag": "p", "text": "Some content", "id": "paragraph-1"}
                ]
            }]
        }
        self.metadata = {
            "title": "Example Site Title",
            "publication_date": "2023-10-27"
        }
        return dom_tree

    def to_dict(self):
        return {"tag": "html", "children": []}

    def extract_links(self):
        return [self.metadata.get("url", "http://example.com/page2"), "http://example.com/about"]

    def get_text(self):
        return "This is the main text content of the page."

class VisionEngine:
    def identify_elements(self, screenshot_data):
        print("Processing screenshot with vision engine...")
        time.sleep(1)
        return {
            "header": {"x": 0, "y": 0, "width": 100, "height": 20},
            "main_content": {"x": 0, "y": 20, "width": 100, "height": 80},
            "footer": {"x": 0, "y": 100, "width": 100, "height": 20}
        }

class CSSVectorizer:
    def generate_vectors(self, raw_css, dom_tree):
        print("Generating vector shapes from CSS...")
        time.sleep(0.5)
        return {
            "main-heading": {"shape": "rectangle", "position": [10, 20], "color": "blue"},
            "paragraph-1": {"shape": "text", "position": [10, 50], "font": "sans-serif"}
        }

class SVGGenerator:
    def create_layers(self, dom_tree, css_vector_map, visual_elements):
        print("Generating SVG layers...")
        time.sleep(0.5)
        return "<svg>...</svg>"

class SourceRatingEngine:
    def compute_rating(self, url, metadata):
        print(f"Computing rating for: {url}")
        time.sleep(0.5)
        return random.randint(70, 100)

class RelevanceEvaluator:
    def evaluate_content(self, text_content, metadata):
        print("Evaluating content relevance...")
        time.sleep(0.5)
        return random.randint(60, 100), ["tag1", "tag2", "tag3"]

def process_website_wrapper(url, category=None):
    try:
        return process_website(url, category)
    except Exception as e:
        logging.error(f"Error processing {url}: {e}")
        return {"status": "failed", "error": str(e)}

def process_website(url, category=None):
    print(f"Processing {url} ...")
    crawler = WebsiteCrawler(url)
    raw_html, raw_css = crawler.fetch_site()
    time.sleep(1)
    dom_parser = DOMParser()
    dom_tree = dom_parser.parse(raw_html, raw_css)
    metadata = dom_parser.metadata
    if category:
        metadata["category"] = category
    screenshot = crawler.render_page_and_capture()
    vision = VisionEngine()
    visual_elements = vision.identify_elements(screenshot)
    vector_mapper = CSSVectorizer()
    css_vector_map = vector_mapper.generate_vectors(raw_css, dom_tree)
    svg_generator = SVGGenerator()
    svg_layers = svg_generator.create_layers(dom_tree, css_vector_map, visual_elements)
    links = dom_parser.extract_links()
    rating_engine = SourceRatingEngine()
    source_rating = rating_engine.compute_rating(url, metadata)
    relevance_evaluator = RelevanceEvaluator()
    try:
        relevance_score, tags = relevance_evaluator.evaluate_content(dom_parser.get_text(), metadata)
    except Exception as e:
        print(f"Error with relevancy score: {e}")
        relevance_score, tags = "", []
    document = {
        "url": url,
        "timestamp": time.time(),
        "text": dom_parser.get_text(),
        "dom": dom_parser.to_dict(),
        "css_vector_map": css_vector_map,
        "visual_elements": visual_elements,
        "svg_layers": svg_layers,
        "links": links,
        "ratings": {
            "source_rating": source_rating,
            "relevance_score": relevance_score,
            "tags": tags
        },
        "metadata": metadata
    }
    gemini_embedder = GeminiEmbedder()
    embedding = gemini_embedder.embed(document["text"])
    document["embedding"] = embedding
    vector_index.index_document(doc_id=url, embedding=embedding, metadata=document["ratings"], full_doc=document)
    print(f"Processed and saved data for {url}")
    return {"status": "success"}

# ----------------------- PDF Report Generation -----------------------
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, f'{HEADING} - Legal Scenario Analysis Report', 0, 1, 'C')
        self.set_font('Arial', '', 8)
        self.multi_cell(0, 5, LICENSE_TEXT)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln(5)

    def add_sources(self, sources):
        if sources:
            self.chapter_title("Sources")
            for i, source in enumerate(sources):
                self.chapter_body(f"{i+1}. {source}")

def create_pdf_report(scenario_data: dict, analysis_results: dict):
    pdf = PDF()
    pdf.add_page()
    pdf.chapter_title("Scenario Details")
    pdf.chapter_body(f"Scenario: {scenario_data.get('scenario', 'N/A')}")
    pdf.chapter_body(f"Subject: {scenario_data.get('subject', 'N/A')}")
    pdf.chapter_body(f"Assailant: {scenario_data.get('assailant', 'N/A')}")
    pdf.chapter_body(f"Location: {scenario_data.get('location', 'N/A')}")
    pdf.chapter_body(f"Details: {scenario_data.get('details', 'N/A')}")
    pdf.chapter_title("Executive Summary")
    pdf.chapter_body(analysis_results.get("summary", "No summary available."))
    pdf.chapter_title("General Assessment")
    pdf.chapter_body(analysis_results.get("assessment", "Assessment not available."))
    pdf.chapter_title("Follow-up Questions")
    questions = analysis_results.get("questions", [])
    if questions:
        for q in questions:
            pdf.chapter_body(f"- {q}")
    else:
        pdf.chapter_body("No follow-up questions generated.")
    pdf.chapter_title("Possible Outcomes")
    pdf.chapter_body(analysis_results.get("possible_outcomes", "Not Available"))
    pdf.chapter_title("Likely Outcomes")
    pdf.chapter_body(analysis_results.get("likely_outcomes", "Not Available"))
    if analysis_results.get("prosecution_objectives", "Not Applicable") != "Not Applicable":
        pdf.chapter_title("Likely Objectives of the Prosecution")
        pdf.chapter_body(analysis_results.get("prosecution_objectives"))
    if analysis_results.get("defense_objectives", "Not Applicable") != "Not Applicable":
        pdf.chapter_title("Likely Objectives of the Defense")
        pdf.chapter_body(analysis_results.get("defense_objectives"))
    if analysis_results.get("prosecution_strategies", "Not Applicable") != "Not Applicable":
        pdf.chapter_title("Potential Prosecution Strategies")
        pdf.chapter_body(analysis_results.get("prosecution_strategies"))
    if analysis_results.get("defense_strategies", "Not Applicable") != "Not Applicable":
        pdf.chapter_title("Potential Defense Strategies")
        pdf.chapter_body(analysis_results.get("defense_strategies"))
    if analysis_results.get("law_enforcement_actions", "Not Available") != "Not Available":
        pdf.chapter_title("Law Enforcement Actions and Legality")
        pdf.chapter_body(analysis_results.get("law_enforcement_actions"))
    pdf.chapter_title("Relevant Laws")
    pdf.chapter_body(analysis_results.get("relevant_laws", "Not Available"))
    pdf.add_sources(analysis_results.get("sources", []))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    pdf_filename = os.path.join(SCENARIO_DIRECTORY, f"analysis_{timestamp}.pdf")
    pdf.output(pdf_filename)
    return pdf_filename

# ----------------------- Helper Formatters -----------------------
def format_laws(laws_list):
    if laws_list:
        return "\n".join(f"{i+1}. {law}" for i, law in enumerate(laws_list))
    return "None"

def format_actions(action_list):
    if action_list:
        return "\n".join(f"- {action.strip()}" for action in action_list)
    return "None"

def format_objectives(objective_list):
    if objective_list:
        return "\n".join(f"- {objective.strip()}" for objective in objective_list)
    return "None"

def format_strategies(strategy_list):
    if strategy_list:
        return "\n".join(f"- {strategy.strip()}" for strategy in strategy_list)
    return "None"

def format_string(text, max_line_length=100):
    words = text.split()
    formatted_lines = []
    current_line = ""
    for word in words:
        if len(current_line + word) + 1 <= max_line_length:
            current_line += " " + word if current_line else word
        else:
            formatted_lines.append(current_line)
            current_line = word
    if current_line:
        formatted_lines.append(current_line)
    return "\n".join(formatted_lines)

# ----------------------- Flask API Definitions -----------------------
api = Flask(__name__)

@api.route('/scenarios/<path:filename>')
def serve_scenario(filename):
    return send_from_directory(SCENARIO_DIRECTORY, filename, as_attachment=True)

@api.route('/analyze', methods=['POST'])
def analyze_legal_scenario():
    scenario_data = request.json
    try:
        analysis_results = gemini_chat.analyze_scenario(scenario_data)
        formatted_results = {
            "assessment": analysis_results.get("assessment", "Not Available"),
            "questions": analysis_results.get("questions", []),
            "possible_outcomes": analysis_results.get("possible_outcomes", "Not Available"),
            "likely_outcomes": analysis_results.get("likely_outcomes", "Not Available"),
            "prosecution_objectives": analysis_results.get("prosecution_objectives", "Not Applicable"),
            "defense_objectives": analysis_results.get("defense_objectives", "Not Applicable"),
            "prosecution_strategies": analysis_results.get("prosecution_strategies", "Not Applicable"),
            "defense_strategies": analysis_results.get("defense_strategies", "Not Applicable"),
            "law_enforcement_actions": analysis_results.get("law_enforcement_actions", "Not Available"),
            "summary": analysis_results.get("summary", "Not Available"),
            "relevant_laws": analysis_results.get("relevant_laws", "Not Available")
        }
        return jsonify(formatted_results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@api.route('/law_detail', methods=['GET'])
def get_law_information():
    law_query = request.args.get('lawName')
    logging.info(law_query)
    try:
        api_raw_law_data = gemini_search(query=law_query + " details")
        for item in api_raw_law_data:
            fetch_summary(item['url'])
        formatted_data = {"law_details": api_raw_law_data}
        return jsonify(formatted_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def fetch_summary(url: str) -> str:
    print(f"Fetching summary from URL {url}")
    try:
        with requests.get(url, timeout=5) as response:
            if response.status_code == 200:
                return f"URL: {url}\n"
            else:
                return f"ERROR: {response.status_code}\n"
    except requests.exceptions.RequestException as e:
        return f"Error during fetch: {e}\n"

if ENABLE_FLASK_API:
    api_thread = threading.Thread(target=api.run, kwargs={'debug': False, 'port': API_PORT})
    api_thread.daemon = True
    api_thread.start()
    print(f"Flask API enabled and running on port {API_PORT}")

# ----------------------- UI Components (ipywidgets) -----------------------
# Initialize engines
vector_index = VectorIndexEngine()
gemini_chat = GeminiChat()

def create_title_and_license():
    title_widget = widgets.HTML(f"<h1>{HEADING}</h1><h2>A Travis Michael O'Dell Project</h2>")
    license_widget = widgets.Textarea(value=LICENSE_TEXT, description='License:', disabled=True,
                                      layout={'height': '300px', 'width': '70%'})
    return widgets.VBox([title_widget, license_widget])

title_and_license = create_title_and_license()

# Settings Tab
creds_out = widgets.Output()
creds_pass1 = widgets.Text(placeholder='Password for API', description="API Password:")
creds_user = widgets.Text(placeholder="User Name for API", description="API User:")
creds_button = widgets.Button(description="Save Creds")

def handle_get_creds(button):
    with creds_out:
        clear_output()
        cred_data = {creds_user.value: creds_pass1.value}
        try:
            with open(CREDENTIALS_FILE, "w") as f:
                json.dump(cred_data, f, indent=4)
            print(f"Credentials saved to {CREDENTIALS_FILE}")
        except Exception as e:
            print(f"Error saving credentials: {e}")

creds_button.on_click(handle_get_creds)

def create_settings_tab():
    enable_api_toggle = widgets.ToggleButton(value=ENABLE_FLASK_API, description='Enable Flask API',
                                             tooltip='Toggle Flask API', icon='power-off')
    def on_api_toggle_value_change(change):
        global ENABLE_FLASK_API
        ENABLE_FLASK_API = change['new']
        print("Flask API enabled. Please restart." if ENABLE_FLASK_API else "Flask API disabled.")
    enable_api_toggle.observe(on_api_toggle_value_change, names='value')
    settings_output = widgets.Output()
    with settings_output:
        clear_output()
        print("Settings Tab Content")
        print(f"User Files Directory: {USER_FILES_DIR}")
        print(f"Credentials File: {CREDENTIALS_FILE}")
        print(f"Connectors File: {CONNECTORS_FILE}")
        show_settings_button = widgets.Button(description="Show Settings", button_style='info')
        display(show_settings_button, creds_user, creds_pass1, creds_button, creds_out)
        def display_current_settings(b):
            with settings_output:
                print(f"Current AI Engine: {current_ai_engine}")
        show_settings_button.on_click(display_current_settings)
    settings_box = widgets.VBox([enable_api_toggle, settings_output])
    return settings_box

settings_tab = create_settings_tab()

# Search Tab
search_query = widgets.Text(placeholder='Enter your legal query here', description='Query:')
search_button = widgets.Button(description="Search")
search_output = widgets.Output()
search_context_output = widgets.Output()
chat_output = widgets.Output()

def handle_search(button):
    with search_output:
        clear_output()
        print("Searching...")
    with chat_output:
        clear_output()
        print("Generating report...")
    query = search_query.value
    query_embedding = GeminiEmbedder().embed(query)
    search_results = vector_index.search(query_embedding, top_k=5)
    context_documents = []
    with search_output:
        clear_output()
        if not search_results:
            print("No results found.")
            return
        print(f"Found {len(search_results)} results:")
        for result in search_results:
            sim = result["similarity"]
            doc = result["document"]
            print(f"URL: {doc.get('url')}, Similarity: {sim:.3f}")
            print(f"Ratings: {doc.get('ratings')}")
            snippet = doc.get('text', '')[:200]
            print(f"Text Snippet: {snippet}...")
            print("----")
            if len(context_documents) < 3:
                context_documents.append(f"URL: {doc.get('url')}\nText: {doc.get('text', '')[:500]}")
    with search_context_output:
        clear_output()
        print("Relevant document excerpts:")
        for i, doc_excerpt in enumerate(context_documents):
            print(f"{i+1}. {doc_excerpt}\n")
    report = gemini_chat.generate_report(
        f"Generate a detailed legal report for the query: '{query}'.", context_documents)
    with chat_output:
        clear_output()
        display(HTML(f"<pre>{report}</pre>"))

search_button.on_click(handle_search)

# Browse Tab
browse_output = widgets.Output()

def create_browse_display():
    with browse_output:
        clear_output()
        print("Browse Tab Content")
        print("Index file:", INDEX_FILE)
        print("Number of indexed documents:", len(vector_index.index))
        if not vector_index.index:
            print("No documents in the index yet.")
            return
        doc_options = [doc["document"]["url"] for doc in vector_index.index]
        doc_dropdown = widgets.Dropdown(options=doc_options, description="Select Document:")
        doc_details_output = widgets.Output()
        def display_document_details(change):
            selected_url = change.new
            with doc_details_output:
                clear_output()
                for rec in vector_index.index:
                    if rec["document"]["url"] == selected_url:
                        doc = rec["document"]
                        print(f"URL: {doc.get('url')}")
                        print(f"Timestamp: {doc.get('timestamp')}")
                        print(f"Ratings: {doc.get('ratings')}")
                        print(f"Metadata: {doc.get('metadata')}")
                        print(f"Text: {doc.get('text')}")
                        break
        doc_dropdown.observe(display_document_details, names='value')
        display(doc_dropdown, doc_details_output)

# Train Tab
train_url_text = widgets.Text(description="URL:")
train_category_dropdown = widgets.Dropdown(options=["Contract Law", "Criminal Law", "Constitutional Law", "International Law", "Other"],
                                          description="Category:")
train_add_button = widgets.Button(description="Add Task")
train_output = widgets.Output()
train_status_output = widgets.Output()

def handle_add_task(button):
    url = train_url_text.value
    category = train_category_dropdown.value
    task_id = add_task("index_website", url, category)
    with train_output:
        print(f"Added task {task_id} for URL: {url} (Category: {category})")
    update_task_status()

def update_task_status():
    with train_status_output:
        clear_output()
        for task_id, status in task_results.items():
            print(f"Task {task_id}: {status.get('status', 'unknown')}")
            if status.get("status") == "failed":
                print(f"  Error: {status.get('error')}")

train_add_button.on_click(handle_add_task)

# Postulate Tab
postulate_area = widgets.Text(description="Area of Law:")
postulate_button = widgets.Button(description="Postulate")
postulate_output = widgets.Output()

def handle_postulate(button):
    with postulate_output:
        clear_output()
        print("Gathering relevant data and laws...")
    area = postulate_area.value
    search_results = gemini_search(f"Relevant laws and data on {area}")
    for result in search_results:
        add_task("index_website", result["url"], area)
    with postulate_output:
        print(f"Added {len(search_results)} websites related to '{area}' to the indexing queue.")
        update_task_status()

postulate_button.on_click(handle_postulate)

# Scenarios Tab
scenario_input = widgets.Textarea(placeholder='Describe the legal scenario', description='Scenario:',
                                  layout=widgets.Layout(width='70%'))
subject_details = widgets.Text(placeholder='Enter subject details', description='Subject:')
assailant_details = widgets.Text(placeholder='Enter assailant details', description='Assailant:')
location_details = widgets.Text(placeholder='Enter location', description='Location:')
case_details = widgets.Textarea(placeholder='Enter case details', description='Details:')

gather_info_button = widgets.Button(description="Gather Information")
analyze_button = widgets.Button(description="Analyze Scenario")
save_button = widgets.Button(description="Save Scenario")
load_button = widgets.Button(description="Load Scenario")
scenario_output = widgets.Output()

def generate_suggested_questions(scenario_data: dict) -> List[str]:
    questions = []
    if not scenario_data.get("subject"):
        questions.append("Could you provide more details about the subject involved?")
    if not scenario_data.get("assailant"):
        questions.append("Do you have any information about the assailant?")
    if not scenario_data.get("location"):
        questions.append("Where did the event take place?")
    if not scenario_data.get("details"):
        questions.append("Could you please describe the incident in more detail?")
    questions.append("Is there any chance that evidence of this event exists elsewhere?")
    questions.append("Were there any witnesses or additional parties involved?")
    return questions

def handle_gather_info(button):
    with scenario_output:
        clear_output()
        scenario_data = {
            "scenario": scenario_input.value,
            "subject": subject_details.value,
            "assailant": assailant_details.value,
            "location": location_details.value,
            "details": case_details.value
        }
        followup_questions = gemini_chat.generate_questions(scenario_data)
        additional_questions = generate_suggested_questions(scenario_data)
        print("Follow-up Questions from Gemini:")
        for q in followup_questions:
            print(f"- {q}")
        print("\nAdditional Suggested Questions:")
        for q in additional_questions:
            print(f"- {q}")

def handle_analyze_scenario(button):
    with scenario_output:
        clear_output()
        print("Analyzing scenario...")
        scenario_data = {
            "scenario": scenario_input.value,
            "subject": subject_details.value,
            "assailant": assailant_details.value,
            "location": location_details.value,
            "details": case_details.value
        }
        analysis_results = gemini_chat.analyze_scenario(scenario_data)
        if "error" in analysis_results:
            print(analysis_results["error"])
            return
        print("Scenario Analysis:")
        print(f"\nGeneral Assessment:\n{format_string(analysis_results['assessment'])}")
        print(f"\nFollow-up Questions from Gemini:")
        for q in analysis_results['questions']:
            print(f"- {q}")
        print(f"\nPossible Outcomes:\n{format_string(analysis_results['possible_outcomes'])}")
        print(f"\nLikely Outcomes:\n{format_string(analysis_results['likely_outcomes'])}")
        print(f"\nProsecution Objectives:\n{format_objectives([analysis_results.get('prosecution_objectives', 'N/A')])}")
        print(f"\nDefense Objectives:\n{format_objectives([analysis_results.get('defense_objectives', 'N/A')])}")
        print(f"\nProsecution Strategies:\n{format_strategies([analysis_results.get('prosecution_strategies', 'N/A')])}")
        print(f"\nDefense Strategies:\n{format_strategies([analysis_results.get('defense_strategies', 'N/A')])}")
        print(f"\nLaw Enforcement Actions:\n{format_actions([analysis_results.get('law_enforcement_actions', 'N/A')])}")
        print(f"\nSummary:\n{format_string(analysis_results['summary'])}")
        print(f"\nRelevant Laws:\n{format_laws([analysis_results.get('relevant_laws', 'N/A')])}")
        pdf_filename = create_pdf_report(scenario_data, analysis_results)
        print(f"\nPDF report generated: {pdf_filename}")
        display(HTML(f'<a href="/scenarios/{os.path.basename(pdf_filename)}" target="_blank">Download PDF Report</a>'))

def handle_save_scenario(button):
    with scenario_output:
        clear_output()
        scenario_data = {
            "scenario": scenario_input.value,
            "subject": subject_details.value,
            "assailant": assailant_details.value,
            "location": location_details.value,
            "details": case_details.value
        }
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(SCENARIO_DIRECTORY, f"scenario_{timestamp}.json")
        try:
            with open(filename, "w") as f:
                json.dump(scenario_data, f, indent=4)
            print(f"Scenario saved to {filename}")
        except Exception as e:
            print(f"Error saving scenario: {e}")

def handle_load_scenario(button):
    with scenario_output:
        clear_output()
        scenario_files = [f for f in os.listdir(SCENARIO_DIRECTORY) if f.endswith(".json")]
        if not scenario_files:
            print("No saved scenarios found.")
            return
        scenario_dropdown = widgets.Dropdown(options=scenario_files, description="Select Scenario:")
        load_output = widgets.Output()
        def load_selected_scenario(change):
            with load_output:
                clear_output()
                filepath = os.path.join(SCENARIO_DIRECTORY, change.new)
                try:
                    with open(filepath, "r") as f:
                        scenario_data = json.load(f)
                    scenario_input.value = scenario_data.get("scenario", "")
                    subject_details.value = scenario_data.get("subject", "")
                    assailant_details.value = scenario_data.get("assailant", "")
                    location_details.value = scenario_data.get("location", "")
                    case_details.value = scenario_data.get("details", "")
                    print(f"Scenario loaded from {filepath}")
                except Exception as e:
                    print(f"Error loading scenario: {e}")
        scenario_dropdown.observe(load_selected_scenario, names='value')
        display(scenario_dropdown, load_output)

gather_info_button.on_click(handle_gather_info)
analyze_button.on_click(handle_analyze_scenario)
save_button.on_click(handle_save_scenario)
load_button.on_click(handle_load_scenario)

scenarios_tab = widgets.VBox([
    scenario_input,
    subject_details,
    assailant_details,
    location_details,
    case_details,
    widgets.HBox([gather_info_button, analyze_button, save_button, load_button]),
    scenario_output
])

# ----------------------- Main Layout -----------------------
tab = widgets.Tab()
tab.children = [
    widgets.VBox([search_query, search_button, search_output, search_context_output, chat_output]),
    widgets.VBox([browse_output]),
    widgets.VBox([train_url_text, train_category_dropdown, train_add_button, train_output, train_status_output]),
    widgets.VBox([postulate_area, postulate_button, postulate_output]),
    scenarios_tab,
    settings_tab
]
tab.set_title(0, "Search")
tab.set_title(1, "Browse")
tab.set_title(2, "Train")
tab.set_title(3, "Postulate")
tab.set_title(4, "Scenarios")
tab.set_title(5, "Settings")

# Display the UI
create_browse_display()  # Populate Browse tab initially
update_task_status()     # Show any pending tasks

display(title_and_license)
display(tab)
