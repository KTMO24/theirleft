import ipywidgets as widgets
from IPython.display import display, clear_output, HTML, FileLink
import pandas as pd
import math
import json
import time
import random
from typing import List, Dict, Any
import concurrent.futures
import queue
import threading
from datetime import datetime
import logging
import re
import os
from fpdf import FPDF  # For PDF generation
import requests
import urllib.parse
import xml.etree.ElementTree as ET
from flask import Flask, request, jsonify, send_from_directory
import flask_cors

# ----------------------- Configuration ------------------------
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"  # TODO: Replace with your actual API key
GEMINI_MODEL_NAME = "models/embedding-001"  # Specify the embedding model to use
GEMINI_CHAT_MODEL_NAME = "models/gemini-pro"  # Specify the chat model
INDEX_FILE = "legal_index.json"
LOG_FILE = "legal_expert_system.log"
SCENARIO_DIRECTORY = "scenarios"  # Directory to store scenario JSON files

# Whether to simulate embeddings instead of calling the Gemini API
SIMULATE_EMBEDDINGS = True  # Set to False to use the Gemini API
SIMULATION_VECTOR_LENGTH = 384  # Length of the dummy vectors (adjust if necessary)

# --- API Configuration ---
ENABLE_FLASK_API = False  # Set to True to enable the Flask API  <-- VERY IMPORTANT
API_PORT = 5000

# --- Multithreading, Rate Limiting, Throttling ---
MAX_THREADS = 5  # Maximum number of threads for website processing
REQUEST_DELAY_SECONDS = 1  # Minimum delay between requests to the same domain
RANDOMIZE_REQUEST_DELAY = True  # Add random variation to the request delay

# --- AI Engine Configuration ---
DEFAULT_AI_ENGINE = "Gemini"

# --- User Files Location ---
USER_FILES_DIR = os.path.expanduser("~/.legal_expert_system")  # You can customize this

# --- Credentials and Connectors ---
CREDENTIALS_FILE = os.path.join(USER_FILES_DIR, "credentials.json")
CONNECTORS_FILE = os.path.join(USER_FILES_DIR, "connectors.json")

# -----------------------  End Configuration --------------------

# --- File System Setup ---
# Create directories if they don't exist
os.makedirs(SCENARIO_DIRECTORY, exist_ok=True)
os.makedirs(USER_FILES_DIR, exist_ok=True)

# --- License and Heading ---
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

# --- Logging ---
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- Rate Limiting and Throttling ---
last_request_time = {}  # Dictionary to track the last request time for each domain

# ----------------------- Task Management -----------------------
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
            # Add more task types as needed
        except Exception as e:
            logging.error(f"Task {task_id} failed: {e}")
            task_results[task_id] = {"status": "failed", "error": str(e)}
        finally:
            task_queue.task_done()

# Start the task worker thread
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

# ----------------------- AI Engine Management -----------------------

# Placeholder for AI engine configurations (loaded from connectors.json)
ai_engines = {}  # {engine_name: {config}}

# Currently selected AI engine
current_ai_engine = DEFAULT_AI_ENGINE

def load_ai_engine_configs():
    """Loads AI engine configurations from connectors.json."""
    global ai_engines
    try:
        with open(CONNECTORS_FILE, "r") as f:
            ai_engines = json.load(f)
    except FileNotFoundError:
        logging.warning(f"AI engine config file not found: {CONNECTORS_FILE}")
        ai_engines = {}
    except Exception as e: #Catch other errors
        logging.exception(f"Error loading AI engine configs: {e}")
        ai_engines = {}


def save_ai_engine_configs():
    """Saves AI engine configurations to connectors.json."""
    try:
        with open(CONNECTORS_FILE, "w") as f:
            json.dump(ai_engines, f, indent=4)
    except Exception as e:
        logging.error(f"Error saving AI engine configurations: {e}")

def set_current_ai_engine(engine_name: str):
    """Sets the currently active AI engine."""
    global current_ai_engine
    if engine_name in ai_engines:
        current_ai_engine = engine_name
        print(f"AI engine set to: {current_ai_engine}")
    else:
        raise ValueError(f"Invalid AI engine name: {engine_name}")

# Load AI engine configurations on startup
load_ai_engine_configs()

# ----------------------- Gemini Modules -----------------------
# Placeholder class, replace with actual implementation.
class GeminiEmbedder:
  def __init__(self, model_name = GEMINI_MODEL_NAME,api_key = GEMINI_API_KEY, simulate = SIMULATE_EMBEDDINGS):
      pass
  def embed(self, text: str):
      return [random.uniform(-0.1, 0.1) for _ in range(384)]
class GeminiChat:
   def __init__(self, model_name: str = GEMINI_CHAT_MODEL_NAME, api_key: str = GEMINI_API_KEY):
        self.model_name = model_name
        self.api_key = api_key
        self.model = None

   def generate_report(self, prompt: str, context: List[str] = None) -> str:
      return "Placeholder report"

   def generate_questions(self, scenario_data: dict) -> List[str]:
      return ["Placeholder question 1", "Placeholder question 2"]

   def analyze_scenario(self, scenario_data: dict) -> dict:
      return {
                "assessment": "Placeholder assessment",
                "questions":  ["Placeholder question 1", "Placeholder question 2"],
                "possible_outcomes": "Placeholder outcomes",
                "likely_outcomes": "Placeholder likely outcomes",
                "prosecution_objectives": "Not Applicable",
                "defense_objectives": "Not Applicable",
                "prosecution_strategies": "Not Applicable",
                "defense_strategies": "Not Applicable",
                "law_enforcement_actions": "Not Available",
                "summary": "Placeholder Summary",
                "relevant_laws": "Placeholder Relevant Laws",
                "sources": []
            }
# ----------------------- Data Processing and Indexing -----------------------

class VectorIndexEngine:
    def __init__(self, index_file: str = INDEX_FILE):
        self.index = []  # List of records
        self.index_file = index_file
        self.load_index()  # Load from file on initialization

    def index_document(self, doc_id: str, embedding: List[float], metadata: Dict[str, Any], full_doc: Dict[str, Any]):
        """Indexes a document by storing its ID, embedding, metadata, and full document data."""
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
        """Calculates the cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def search(self, query_embedding: List[float], top_k: int = 5, metadata_filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Searches the index for documents similar to the query embedding, with optional metadata filters."""
        results = []
        for record in self.index:
            if metadata_filters:
                match = True
                for key, value in metadata_filters.items():
                    if key not in record["metadata"] or record["metadata"][key] != value:
                        match = False
                        break
                if not match:
                    continue

            similarity = self.cosine_similarity(query_embedding, record["embedding"])
            results.append((similarity, record))

        results.sort(key=lambda x: x[0], reverse=True)
        return [{"similarity": sim, "document": rec["document"]} for sim, rec in results[:top_k]]

    def save_index(self):
        """Saves the index to a JSON file."""
        try:
            with open(self.index_file, "w") as f:
                json.dump(self.index, f, indent=4)
            print(f"Index saved to {self.index_file}")
        except Exception as e:
            print(f"Error saving index: {e}")

    def load_index(self):
        """Loads the index from a JSON file."""
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
    """Placeholder for a function that uses the Gemini API to perform a web search."""
    print(f"Performing Gemini search for: {query}")
    time.sleep(1)  # Simulate search latency
    return [
        {"url": f"https://example.com/search-result-1?q={query}", "title": "Search Result 1"},
        {"url": f"https://example.com/search-result-2?q={query}", "title": "Search Result 2"},
        {"url": f"https://example.com/search-result-3?q={query}", "title": "Search Result 3"},
    ]

class WebsiteCrawler:
    """Placeholder for a website crawling module."""
    def __init__(self, url):
        self.url = url
        self.domain = urllib.parse.urlparse(url).netloc

    def fetch_site(self):
        """Simulates fetching the HTML and CSS content of a website."""
        print(f"Fetching site: {self.url}")
        try:
          time_since_last = time.time() - last_request_time.get(self.domain, 0)

          delay = REQUEST_DELAY_SECONDS
          if RANDOMIZE_REQUEST_DELAY:
            delay += random.uniform(0, delay / 2) # Add some randomness
            time.sleep(delay)
          
          raw_html = f"<html><body><h1>Fetched Site from {self.url}</h1><p>Some content from {self.url}</p></body></html>"
          raw_css = "body { font-family: sans-serif; }"
          return raw_html, raw_css
        except Exception as e:
          print(f"Error Fetching: {e}")
          return "<html><body> Error  Fetching Site </body></html>", ""
    def render_page_and_capture(self):
        """Simulates rendering the page and capturing a screenshot."""
        print(f"Rendering and capturing screenshot for: {self.url}")
        time.sleep(1)
        return "screenshot_data.png"

    def get_domain(self):
        from urllib.parse import urlparse
        return urlparse(self.url).netloc

class DOMParser:
    """Placeholder for a module to parse HTML/CSS into a DOM tree."""
    def __init__(self):
        self.metadata = {}

    def parse(self, html: str, css: str):
        """Parses HTML and CSS content and returns a simplified DOM tree representation."""
        print("Parsing HTML and CSS...")
        time.sleep(0.5)
        dom_tree = {
            "tag": "html",
            "children": [
                {"tag": "body",
                 "children": [
                     {"tag": "h1", "text": "Example Site", "id": "main-heading"},
                     {"tag": "p", "text": "Some content", "id": "paragraph-1"}
                 ]}
            ]
        }
        self.metadata = {
            "title": "Example Site Title",
            "publication_date": "2023-10-27"
        }
        return dom_tree

    def to_dict(self):
        return {"tag": "html", "children": []}  # Simplified example

    def extract_links(self):
        return [self.url + "/page2", self.url + "/about"]

    def get_text(self):
        return "This is the main text content of the page. "

class VisionEngine:
    """Placeholder for a module to process screenshots and identify visual elements."""
    def identify_elements(self, screenshot_data):
        """Simulates identifying visual elements in a screenshot."""
        print("Processing screenshot with vision engine...")
        time.sleep(1)
        return {
            "header": {"x": 0, "y": 0, "width": 100, "height": 20},
            "main_content": {"x": 0, "y": 20, "width": 100, "height": 80},
            "footer": {"x": 0, "y": 100, "width": 100, "height": 20}
        }

class CSSVectorizer:
    """Placeholder for a module to convert CSS rules to vector shapes."""
    def generate_vectors(self, raw_css, dom_tree):
        """Simulates converting CSS rules to vector shapes."""
        print("Generating vector shapes from CSS...")
        time.sleep(0.5)
        return {
            "main-heading": {"shape": "rectangle", "position": [10, 20], "color": "blue"},
            "paragraph-1": {"shape": "text", "position": [10, 50], "font": "sans-serif"}
        }

class SVGGenerator:
    """Placeholder for a module to build layered SVGs from DOM and visual data."""
    def create_layers(self, dom_tree, css_vector_map, visual_elements):
        """Simulates creating layered SVG representations."""
        print("Generating SVG layers...")
        time.sleep(0.5)
        return "<svg>...</svg>"  # Simplified example

class SourceRatingEngine:
    """Placeholder for a module to evaluate source quality."""
    def compute_rating(self, url, metadata):
        """Simulates computing a source rating based on URL and metadata."""
        print(f"Computing rating for: {url}")
        time.sleep(0.5)
        return random.randint(70, 100)

class RelevanceEvaluator:
    """Placeholder for a module to evaluate content relevance."""
    def evaluate_content(self, text_content, metadata):
        """Simulates evaluating content relevance and extracting tags."""
        print("Evaluating content relevance...")
        time.sleep(0.5)
        return random.randint(60, 100), ["tag1", "tag2", "tag3"]

def process_website_wrapper(url, category=None):
    """Wrapper function to handle exceptions during website processing."""
    try:
        return process_website(url, category)
    except Exception as e:
        logging.error(f"Error processing {url}: {e}")
        return {"status": "failed", "error": str(e)}

def process_website(url, category=None):
    """
    Processes a single website: crawls, parses, vectorizes, generates SVG, and evaluates.
    """
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
    source_rating = rating_engine.compute_rating(url, metadata=dom_parser.metadata)

    relevance_evaluator = RelevanceEvaluator()
    relevance_score = ""
    tags =[]
    try:
      relevance_score, tags = relevance_evaluator.evaluate_content(
          text_content=dom_parser.get_text(),
          metadata=dom_parser.metadata
      )
    except Exception as e:
      print(f"Error With relevancy score: {e}")

    document = {
        "url": url,
        "timestamp": time.time(),
        "text": dom_parser.get_text(),
        "dom": dom_tree.to_dict(),
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

    vector_index.index_document(
        doc_id=url,
        embedding=embedding,
        metadata=document["ratings"],
        full_doc=document
    )

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
        self.ln()

    def add_sources(self, sources):
        if sources:
            self.chapter_title("Sources")
            for i, source in enumerate(sources):
                self.chapter_body(f"{i+1}. {source}")

def create_pdf_report(scenario_data: dict, analysis_results: dict):
    """
    Generates a PDF report of the scenario analysis.
    """
    pdf = PDF()
    pdf.add_page()

    # Scenario Details
    pdf.chapter_title("Scenario Details")
    pdf.chapter_body(f"Scenario: {scenario_data.get('scenario', 'N/A')}")
    pdf.chapter_body(f"Subject: {scenario_data.get('subject', 'N/A')}")
    pdf.chapter_body(f"Assailant: {scenario_data.get('assailant', 'N/A')}")
    pdf.chapter_body(f"Location: {scenario_data.get('location', 'N/A')}")
    pdf.chapter_body(f"Details: {scenario_data.get('details', 'N/A')}")

    # Analysis Summary
    pdf.chapter_title("Executive Summary")
    pdf.chapter_body(analysis_results.get("summary", "No summary available."))

    # General Assessment
    pdf.chapter_title("General Assessment")
    pdf.chapter_body(analysis_results.get("assessment", "Assessment not available."))

    # Follow-up Questions
    pdf.chapter_title("Follow-up Questions")
    questions = analysis_results.get("questions", [])
    if questions:
        for q in questions:
            pdf.chapter_body(f"- {q}")
    else:
        pdf.chapter_body("No follow-up questions generated.")

    # Possible Outcomes
    pdf.chapter_title("Possible Outcomes")
    pdf.chapter_body(analysis_results.get("possible_outcomes", "Not Available"))

    # Likely Outcomes
    pdf.chapter_title("Likely Outcomes")
    pdf.chapter_body(analysis_results.get("likely_outcomes", "Not Available"))

    # Objectives and Strategies
    if analysis_results['prosecution_objectives'] != "Not Applicable":
        pdf.chapter_title("Likely Objectives of the Prosecution")
        pdf.chapter_body(analysis_results['prosecution_objectives'])
    if analysis_results['defense_objectives'] != "Not Applicable":
        pdf.chapter_title("Likely Objectives of the Defense")
        pdf.chapter_body(analysis_results['defense_objectives'])
    if analysis_results['prosecution_strategies'] != "Not Applicable":
        pdf.chapter_title("Potential Prosecution Strategies")
        pdf.chapter_body(analysis_results['prosecution_strategies'])
    if analysis_results['defense_strategies'] != "Not Applicable":
        pdf.chapter_title("Potential Defense Strategies")
        pdf.chapter_body(analysis_results['defense_strategies'])

    # Law Enforcement Actions
    if analysis_results['law_enforcement_actions'] != "Not Available":
        pdf.chapter_title("Law Enforcement Actions and Legality")
        pdf.chapter_body(analysis_results['law_enforcement_actions'])

    # Relevant Laws
    pdf.chapter_title("Relevant Laws")
    pdf.chapter_body(analysis_results.get("relevant_laws", "Not Available"))

    # Add sources if available
    pdf.add_sources(analysis_results.get("sources", []))

    # Save the PDF
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    pdf_filename = os.path.join(SCENARIO_DIRECTORY, f"analysis_{timestamp}.pdf")
    pdf.output(pdf_filename)
    return pdf_filename

# ----------------------- Helper Text-Based Formatters  and Functions -----------------------
def format_laws(laws_list):
  """Formats a list of laws (strings) into a numbered list or returns 'None' if empty."""
  if laws_list:
    return "\n".join(f"{i+1}. {law}" for i, law in enumerate(laws_list))
  else:
    return "None"

def format_actions(action_list):
  """Formats a list of law enforcement actions into a numbered list or returns 'None' if empty."""
  if action_list:
    return "\n".join(f"- {action.strip()}" for action in action_list)
  return "None"

def format_objectives(objective_list):
    """Formats a list of objectives into a numbered list or returns 'None' if empty."""
    if objective_list:
        return "\n".join([f"- {objective.strip()}" for objective in objective_list])
    else:
        return "None"

def format_strategies(strategy_list):
  """Formats a list of strategies into a numbered list or returns 'None' if empty."""
  if strategy_list:
    return "\n".join([f"- {strategy.strip()}" for strategy in strategy_list])
  return "None"
# ----------------------- API Definitions (Flask) -----------------------

# --- Helper Functions for Formatting and Display  ---

import urllib.parse
def fetch_summary(url: str) -> str:
    """
    Fetches the summary from a given URL.
    """
    print(f"Fetching summary from URL {url}")
    try:
        with requests.get(url, timeout=5) as response:
          if(response.status_code == 200):
            return f"URL: {url}\n"
          else:
            return f"ERROR: {response.status_code}\n"
    except requests.exceptions.RequestException as e:
        return f"Error during fetch: {e}\n"

def extract_urls(text: str):
    """Extract all URLs from a given text"""
    url = re.findall(r'(https?://\S+)', text)
    return url

def find_law_by_name(law_name):
    """ Placeholder: Finds law details by name in database or search """
    return f"Details of: {law_name} Placeholder"

def format_for_api(analysis_results: dict):
    """Formats the analysis results for the API response."""
    return {
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

def get_law_details(law_name):
      try:
          logging.info("Loading Law data from " + str(law_name))
          law_url = ""
          api_law_data = gemini_search(query=law_name + " details")
          
          for item in api_law_data:
            if "law" in item['url']:
              law_url = item['url']
              break
          if(law_url):
            return format_string(str(law_url)+ " "+ fetch_summary(law_url))
          
          # TODO: Implement actual retrieval logic for the specific law
          return "Details for " + law_name + " Not found"
      except Exception as e:
        return "Can't load Law Details"

def extract_laws(raw_text):
        """Extract all Laws from a given text"""
        extractedLaws = re.findall(r"[A-Z][a-z]+\s+Law", raw_text)
        return extractedLaws

def format_string(text, max_line_length=100):
        """Formats a given text to new lines based on spaces"""
        formatted_lines = []
        words = text.split()
        current_line = ""

        for word in words:
            if len(current_line + word) + 1 <= max_line_length:  # +1 for space
                current_line += " " + word if current_line else word
            else:
                formatted_lines.append(current_line)
                current_line = word  # start new line

        if current_line:  # append last incomplete line
            formatted_lines.append(current_line)

        return "\n".join(formatted_lines)

# --- Create a settings tab ---
def create_settings_tab():
    """Creates the settings tab interface."""

    # Toggle for enabling/disabling the Flask API
    enable_api_toggle = widgets.ToggleButton(
        value=ENABLE_FLASK_API,  # Initial value from configuration
        description='Enable Flask API',
        disabled=False,
        button_style='',  # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Check to enable the Flask API',
        icon='power-off'
    )

    def on_api_toggle_value_change(change):
        global ENABLE_FLASK_API
        ENABLE_FLASK_API = change['new']
        if ENABLE_FLASK_API:
            print("Flask API enabled. Please restart")
        else:
            print("Flask API disabled.")

    enable_api_toggle.observe(on_api_toggle_value_change, names='value')

    settings_output = widgets.Output()

    with settings_output:
        clear_output()
        print("Settings Tab Content")
        print(f"User Files Directory: {USER_FILES_DIR}")
        print(f"Credentials File: {CREDENTIALS_FILE}")
        print(f"Connectors File: {CONNECTORS_FILE}")

        # Create a button to show current AI Engine configuration
        show_settings_button = widgets.Button(description="Show Settings", button_style='info')
        display(show_settings_button,add_text,   creds_pass1, creds_user, creds_button, creds_out  ) #Add Text Creds

        def display_current_settings(b):
          with settings_output:
            print(f"Current AI Engine: {current_ai_engine}")
        show_settings_button.on_click(display_current_settings)

    settings_box = widgets.VBox([
        enable_api_toggle,  # Include the enable API toggle
        settings_output  # Include the settings output
    ])
    # Button action
    creds_button.on_click(handle_get_creds)

    return settings_box

def create_connectors_tab():
        connnector_button = widgets.Button(
        description='Connector Settings',
        button_style='success'
        )
        connnector_output = widgets.Output()

def show_ai_engines(area: str):
          return area

# - Add Credentials or API Keys
load_button.on_click(handle_load_scenario)
gather_info_button.on_click(handle_gather_info)
analyze_button.on_click(handle_analyze_scenario)
save_button.on_click(handle_save_scenario)
# Layout of timeline
 # load the  html string
 """,width="400px")

def handle_analyze_scenario(button):
    """Analyzes the scenario using Gemini and displays the results."""

    #what happens when button is clicked
 with timeline_output:
   clear_output()
   print("loading from "+filename.value) # just say this
with scenario_output:
    clear_output()
    display(HTML(f"<h1>{HEADING}</h1>"))

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
    print("Generating timeline... ")
def handle_save_scenario(button):

    """Saves the scenario data to a JSON file."""
 #what happens when button is clicked
load_button.on_click(handle_load_scenario)
gather_info_button.on_click(handle_gather_info)
analyze_button.on_click(handle_analyze_scenario)
save_button.on_click(handle_save_scenario)

# --- Timeline ---

# --- Layout ---

add_event_button = widgets.Button(description="Add Event to Timeline")

""Text Editor:"",
        button,add_text
 """ # set what to put
def display_current_connectors(e):
    display (HEADING , area, license, new_row.children)
 # load the  html string

# Action when button create
def on_load(e):
   print("Now Please write and load the data  here")

# Layout of timeline
def handle_browse_display(press): # call browseDisplay,
  print ("reLoading")
  browseDisplay()
# call function for creating tab

  "Text Editor:"",
        button,add_text
 """,width="400px")

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
    """Loads a scenario from a JSON file."""
    with scenario_output:
        clear_output()
        # Input for the user to provide the filename
        filename_input = widgets.Text(placeholder='Enter the filename to load (e.g., scenario_20240125-123456.json)', description='Filename:')

        # Load Button for Timeline
        load_from_file = widgets.Button(description="Load", button_style='success', layout=widgets.Layout(width="100px"))
        display(filename_input,load_from_file)

# Action to load from the file
        def on_load(e):

            try:
              new_row.children = []
            except:
                print("Error: Check Text Area")
        # Action to load from the file
        load_from_file.on_click(on_load)
        print("Now Please load and write the text")

settings_output = widgets.Output()
        def handle_browse_display(press): # call browseDisplay,

 browseDisplay()# function for creating tab

# --- Main Layout ---
        #self.license = lic
items_auto = [

    widgets.Label(value="""Legal Assistant System 
                  This is an AI-powered legal research and analysis application."""),

    add_event_button

]
box_layout = Layout(display='flex',
                    flex_flow='row',
                    align_items='stretch',
                    border='solid 1px blue',
                    width='100%')
with scenario_output:
  timeline_events_container = widgets.Box(children=items_auto, layout=box_layout)  # Container for Timeline events
    clear_output()
    create_timeline()# initial create
display_timeline()

 def build_timeline(): # display the UI
 new_row_event = []
 load_button # load button

with load_output:
# clear the window to avoid re-render and bugs

# This function create the UI elements
def create_timeline():

 def update_timeline(): # function for rendering that data
 with timeline_output:

 df = pd.DataFrame(timeline_events)
 df_html = HTML(df.to_html(index=False)) # pass false
 display(df_html) # write our data here

# Load the display
display(tab)
display(title_w) # Show license and credit!
 #what happens when button is clicked
   print("Now Please write and load the data  here")
       # What append the info.
# Layout of timeline

with timeline_output:
      print ('Updating Timeline Events')
# load what ever the data has
  with layout_auto.align_items:

display (add_text,  self =widgets.Box)# set what to put
tab.observe(on_tab_select, names=['selected_index'])# observe for tab change (new) to tab 3 will result call create_time()
update_timeline_table()
  with load_output:

#Now write to the  screen: 
"""
Code section will set layout with tab
"" "

# ---  Set UI  ---

creds_user =widgets.Text(placeholder="User Name for API", value="" , description="Enter User/Key"  )
creds_pass1 = widgets.Text(placeholder='Password for OpenAI Key', value="", description="Add Details"
        text = widgets.HBox([   layout = Layout(padding='8px', border='1px solid black')]) # set what to put
        display (  text, creds_user, creds_pass1) # load credentials data data
  "Text Editor:" "",width="400px")

# call function for creating the UI elements for times lines

# observe the current settings

"""Creates UI elements for adding a new event to the timeline."""
def create_timeline():
 def update_timeline():

# This function update the table
      code = "Now please write and load the time and actions here"

  license = widgets.Label(
 layout.align_self="center",
        description='License:',
        disabled=True,
        value=LICENSE_TEXT,
        layout={'height': '170px', 'width': '80%'}
# Timeline tab to 3 for now
        )
# observe for tab change (new) to tab 3 will result call create_time()

add_connector = widgets.Text(
 description='Add :',
  disabled=False
 #Layout that will create the row

 #Layout that will create the row
        with creds_out:
         global new_row
         new_row = ""
        # Add to table
        list2d_as_html = HTML(timeline_events.to_html(index=False))# create all data inside one function call
# Update what load inside time Line
        else:
# load on all
 # Set UI  ---
  #  with timeline_output:
          clear_output()
          # List to hold timeline rows
          timeline_items = [
 # Add license + title
def format_action(url: str) -> str:

 print(f"This is an action url for now: {url}") #what is will do on this function/action and what the data is
       # action_url =f"Fetching {action} info"  #Placeholder or call to API or something here . We donre have that, for now!
  return

def show_license(txt : str =LICENSE_TEXT ):
  """ Show all information with Licencse """
  return f"This project License is: { txt }" # return what it does/did

if ENABLE_FLASK_API:

    # Start the Flask API in a separate thread
    api_thread = threading.Thread(target=api.run, kwargs={'debug': False, 'port': API_PORT})

    api_thread.daemon = True
    api_thread.start()

    print(f"Flask API enabled and running on port {API_PORT}")

def create_title_and_license():
    """Creates a widget to display title and license information."""
    title_widget = widgets.HTML(f"<h1>{HEADING}</h1><h2>A Travis Michael O'Dell Project</h2>")
    license_widget = widgets.Textarea(
        value=LICENSE_TEXT,
        description='License:',
        disabled=True,
        layout={'height': '300px', 'width': '70%'} #Set a default width and height
    )
    return widgets.VBox([title_widget, license_widget])
# ----------------------- UI Components -----------------------

# Initialize engines
vector_index = VectorIndexEngine()
gemini_chat = GeminiChat()

# Add heading and title information
title_and_license = create_title_and_license()

# Add settings Tab
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
    with search_context_output:
        clear_output()

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
            similarity = result["similarity"]
            doc = result["document"]
            print(f"  URL: {doc.get('url')}, Similarity: {similarity:.3f}")
            print(f"  Ratings: {doc.get('ratings')}")
            print(f"  Text Snippet: {doc.get('text')[:200]}...")
            print("  ----")

            if len(context_documents) < 3:
                context_documents.append(f"URL: {doc.get('url')}\nText: {doc.get('text')[:500]}")

    with search_context_output:
        clear_output()
        print("Relevant document excerpts (used as context for report generation):\n")
        for i, doc_excerpt in enumerate(context_documents):
            print(f"{i+1}. {doc_excerpt}\n")

    report = gemini_chat.generate_report(
        f"Based on the provided information and the user's query: '{query}', generate a detailed report that addresses the query. Include relevant legal information, resources, potential contacts, and any other helpful information.",
        context_documents
    )

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
            selected_doc_url = change.new
            with doc_details_output:
                clear_output()
                for doc_record in vector_index.index:
                    if doc_record["document"]["url"] == selected_doc_url:
                        doc = doc_record["document"]
                        print(f"URL: {doc.get('url')}")
                        print(f"Timestamp: {doc.get('timestamp')}")
                        print(f"Ratings: {doc.get('ratings')}")
                        print(f"Metadata: {doc.get('metadata')}")
                        print(f"Text: {doc.get('text')}")
                        break

        doc_dropdown.observe(display_document_details, names='value')

        display(doc_dropdown)
        display(doc_details_output)

# Train Tab
train_url_text = widgets.Text(description="URL:")
train_category_dropdown = widgets.Dropdown(
    options=["Contract Law", "Criminal Law", "Constitutional Law", "International Law", "Other"],
    description="Category:"
)
train_add_button = widgets.Button(description="Add Task")
train_output = widgets.Output()
train_status_output = widgets.Output()

def handle_add_task(button):
    url = train_url_text.value
    category = train_category_dropdown.value
    task_id = add_task("index_website", url, category)
    with train_output:
        print(f"Added task {task_id} to index: {url} (Category: {category})")
    update_task_status()

train_add_button.on_click(handle_add_task)

def update_task_status():
    with train_status_output:
        clear_output()
        for task_id, status in task_results.items():
            print(f"Task {task_id}: {status.get('status', 'unknown')}")
            if status.get("status") == "failed":
                print(f"  Error: {status.get('error')}")

# Postulate Tab
postulate_area = widgets.Text(description="Area of Law:")
postulate_button = widgets.Button(description="Postulate")
postulate_output = widgets.Output()

def handle_postulate(button):
    with postulate_output:
        clear_output()
        print("Gathering relevant data and laws...")
    area_of_law = postulate_area.value

    search_results = gemini_search(f"Relevant laws and data on {area_of_law}")

    for result in search_results:
        task_id = add_task("index_website", result["url"], area_of_law)

    with postulate_output:
        print(f"Added {len(search_results)} websites related to '{area_of_law}' to the indexing queue.")
        update_task_status()

postulate_button.on_click(handle_postulate)

# Scenarios Tab
scenario_input = widgets.Textarea(
    placeholder='Describe the legal scenario',
    description='Scenario:',
    layout=widgets.Layout(width='70%')
)
subject_details = widgets.Text(placeholder='Enter subject details', description='Subject:')
assailant_details = widgets.Text(placeholder='Enter assailant details', description='Assailant:')
location_details = widgets.Text(placeholder='Enter location', description='Location:')
case_details = widgets.Textarea(placeholder='Enter case details', description='Details:')

gather_info_button = widgets.Button(description="Gather Information")
analyze_button = widgets.Button(description="Analyze Scenario")
save_button = widgets.Button(description="Save Scenario")
load_button = widgets.Button(description="Load Scenario")
scenario_output = widgets.Output()

# Timeline
timeline_events = []
timeline_table = widgets.Output()

def handle_gather_info(button):
    """
    Gathers additional information using Gemini and our own suggested questions.
    """
    with scenario_output:
        clear_output()
        print("Gathering additional information using Gemini...")
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
        if additional_questions:
            print("\nAdditional Suggested Questions:")
            for q in additional_questions:
                print(f"- {q}")

def handle_analyze_scenario(button):
    """Analyzes the scenario using Gemini, displays the results, and generates a PDF report."""
    with scenario_output:
        clear_output()
        print("Analyzing scenario...")

        scenario_data = {
            "scenario": scenario_input.value,
            "subject": subject_details.value,
            "assailant": assailant_details.value,
            "location": location_details.value,
            "details": case_details.value,
            "timeline": timeline_events  # Include the timeline data
        }

        analysis_results = gemini_chat.analyze_scenario(scenario_data)

        if "error" in analysis_results:
            print(analysis_results["error"])
            return

        print("Scenario Analysis:")
        print(f"\nGeneral Assessment:\n{analysis_results['assessment']}")

        print(f"\nFollow-up Questions from Gemini:")
        for q in analysis_results['questions']:
            print(f"- {q}")
        print(f"\nPossible Outcomes:\n{analysis_results['possible_outcomes']}")
        print(f"\nLikely Outcomes:\n{analysis_results['likely_outcomes']}")

        if analysis_results['prosecution_objectives'] != "Not Applicable":
            print(f"\nLikely Objectives of the Prosecution:\n{analysis_results['prosecution_objectives']}")
        if analysis_results['defense_objectives'] != "Not Applicable":
            print(f"\nLikely Objectives of the Defense:\n{analysis_results['defense_objectives']}")
        if analysis_results['prosecution_strategies'] != "Not Applicable":
            print(f"\nPotential Prosecution Strategies:\n{analysis_results['prosecution_strategies']}")
        if analysis_results['defense_strategies'] != "Not Applicable":
            print(f"\nPotential Defense Strategies:\n{analysis_results['defense_strategies']}")
        if analysis_results['law_enforcement_actions'] != "Not Available":
            print(f"\nLaw Enforcement Actions and Legality:\n{analysis_results['law_enforcement_actions']}")

        # Summary and Relevant Laws
        print(f"\nSummary:\n{analysis_results['summary']}")
        print(f"\nRelevant Laws:\n{analysis_results['relevant_laws']}")

        # Generate and offer PDF report
        pdf_filename = create_pdf_report(scenario_data, analysis_results)
        print(f"\nPDF report generated: {pdf_filename}")
        display(HTML(f'<a href="/scenarios/{os.path.basename(pdf_filename)}" target="_blank">Download PDF Report</a>'))  # Corrected link


def handle_save_scenario(button):
    """Saves the scenario data to a JSON file."""
    with scenario_output:
        clear_output()
        scenario_data = {
            "scenario": scenario_input.value,
            "subject": subject_details.value,
            "assailant": assailant_details.value,
            "location": location_details.value,
            "details": case_details.value,
            "timeline": timeline_events, #Save the timeline
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
      """Loads a scenario from a JSON file."""
    with scenario_output:
        clear_output()
        scenario_files = [f for f in os.listdir(SCENARIO_DIRECTORY) if f.endswith(".json")]

        if not scenario_files:
            print("No saved scenarios found.")
            return

        scenario_dropdown = widgets.Dropdown(
            options=scenario_files,
            description="Select Scenario:"
        )
        display(scenario_dropdown)  # Display *before* the observe call.
        load_output = widgets.Output()
        display(load_output) #Display output

        def load_selected_scenario(change):
            global timeline_events #So we modify the global
            selected_scenario = change.new
            with load_output:
                clear_output()
                filepath = os.path.join(SCENARIO_DIRECTORY, selected_scenario)
                try:
                    with open(filepath, "r") as f:
                        scenario_data = json.load(f)
                    #Populate fields
                    scenario_input.value = scenario_data.get("scenario", "")
                    subject_details.value = scenario_data.get("subject", "")
                    assailant_details.value = scenario_data.get("assailant", "")
                    location_details.value = scenario_data.get("location", "")
                    case_details.value = scenario_data.get("details", "")
                    #Load the timeline
                    timeline_events = scenario_data.get("timeline", [])
                    update_timeline_table()

                    print(f"Scenario loaded from {filepath}")
                except Exception as e:
                    print(f"Error loading scenario: {e}")

        scenario_dropdown.observe(load_selected_scenario, names='value')
        display(scenario_dropdown)
        display(load_output)

# --- Timeline Tab ---
timeline_events_container = widgets.VBox()
def add_timeline_event_ui():
    """Creates UI elements for adding a new event to the timeline."""
    event_time = widgets.Text(description="Time (Approx.):", layout=widgets.Layout(width='30%'))
    event_description = widgets.Textarea(description="Event:", layout=widgets.Layout(width='70%'))
    add_event_button = widgets.Button(description="Add Event")

    new_row = widgets.HBox([event_time, event_description, add_event_button])
    timeline_events_container.children += (new_row,)

    def handle_add_event(button):
        global timeline_events
        timeline_events.append({"time": event_time.value, "description": event_description.value})
        event_time.value = ""
        event_description.value = ""
        update_timeline_table()

    add_event_button.on_click(handle_add_event)

def update_timeline_table():
    """Updates the timeline table display."""
    with timeline_table:
        clear_output()
        if not timeline_events:
            print("No events added to the timeline yet.")
            return
        df = pd.DataFrame(timeline_events)
        display(HTML(df.to_html(index=False, classes='table table-striped')))


# ----------------------- Suggested Questions Helper -----------------------
def generate_suggested_questions(scenario_data: dict) -> List[str]:
    """
    Generates additional suggested questions based on missing or ambiguous scenario details.
    """
    questions = []
    if not scenario_data.get("subject"):
        questions.append("Could you provide more details about the subject involved?")
    if not scenario_data.get("assailant"):
        questions.append("Do you have any information about the assailant? For example, did Party A act in a specific manner?")
    if not scenario_data.get("location"):
        questions.append("Where did the event take place?")
    if not scenario_data.get("details"):
        questions.append("Could you please describe the incident in more detail?")
    # Additional probing questions
    questions.append("Is there any chance that evidence of this event exists elsewhere?")
    questions.append("Were there any witnesses or additional parties involved?")
    return questions

# Button event handlers
load_button.on_click(handle_load_scenario)
gather_info_button.on_click(handle_gather_info)
analyze_button.on_click(handle_analyze_scenario)
save_button.on_click(handle_save_scenario)

# Add event button for timeline (outside the functions)
add_event_button = widgets.Button(description="Add Event to Timeline")
add_event_button.on_click(lambda b: add_timeline_event_ui()) # Use lambda for click handler.
timeline_table = widgets.Output() #Define before usage.

# --- Scenarios Tab Layout ---
scenarios_tab = widgets.VBox([
    scenario_input,
    subject_details,
    assailant_details,
    location_details,
    case_details,
    widgets.HBox([gather_info_button, analyze_button, save_button, load_button]),
    scenario_output,
    widgets.Label("Timeline of Events:"),
    add_event_button, # Add to layout *before* the table.
    timeline_table,
    timeline_events_container  # Add the container
])

# ----------------------- Main Layout -----------------------
# Tab creation and children are correct.
tab = widgets.Tab()
tab.children = [
    widgets.VBox([search_query, search_button, search_output, search_context_output, chat_output]),  # Search
    widgets.VBox([browse_output]),  # Browse
    widgets.VBox([train_url_text, train_category_dropdown, train_add_button, train_output, train_status_output]),  # Train
    widgets.VBox([postulate_area, postulate_button, postulate_output]),  # Postulate
    scenarios_tab,  # Scenarios
    settings_tab,  # Settings
]
tab.set_title(0, "Search")
tab.set_title(1, "Browse")
tab.set_title(2, "Train")
tab.set_title(3, "Postulate")
tab.set_title(4, "Scenarios")
tab.set_title(5, "Settings")

# ----------------------- Initialization -----------------------

# Initial display setup.  Important for dynamic updates.
create_browse_display()
update_task_status()
update_timeline_table()


# --- Tab selection handler ---
def on_tab_select(change):
    if change['new'] == 1:  # Browse tab
        create_browse_display()  # Update the browse tab content when selected
    elif change['new'] == 5:
        #Update the settings and connectors here
        pass


tab.observe(on_tab_select, names='selected_index')

# --- Display the Main UI ---
main_ui = widgets.VBox([title_and_license, tab, active_learning_monitor])
display(main_ui)
# --- Flask API Setup (Optional)
api = Flask(__name__)
flask_cors.CORS(api) #Enable CORS

@api.route('/scenarios/<path:filename>')
def serve_scenario(filename):
    return send_from_directory(SCENARIO_DIRECTORY, filename, as_attachment=True)

@api.route('/analyze', methods=['POST'])
def analyze_legal_scenario_api():
  scenario_data = request.json
  try:
    analysis_results = gemini_chat.analyze_scenario(scenario_data)
    return jsonify(analysis_results), 200
  except Exception as e:
    return jsonify({"error": str(e)}), 500

@api.route('/law_detail', methods=['GET'])
def get_law_information_api():
    law_query = request.args.get('lawName')
    # Placeholder - replace with actual implementation using Gemini
    return jsonify({"law_details": f"Details for {law_query} (Placeholder)"}), 200

if ENABLE_FLASK_API:
    # Start the Flask API in a separate thread
    api_thread = threading.Thread(target=api.run, kwargs={'debug': False, 'port': API_PORT, 'use_reloader': False})
    api_thread.daemon = True
    api_thread.start()
    print(f"Flask API enabled and running on port {API_PORT}")
# ----------------------- Background Tasks (Optional) -----------------------

def periodic_task_update():
    while True:
        update_task_status_ui(train_status_output)  # Make sure this output is displayed in your UI.
        time.sleep(5)

update_thread = threading.Thread(target=periodic_task_update, daemon=True)
update_thread.start()
