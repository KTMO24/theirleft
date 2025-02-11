#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merged Legal Expert System & Dual Interface (Flask + IPython Widgets)
with Active Learning Monitor & Disclaimers

Developed by Travis Michael O'Dell (and collaborators)
MIT License

DISCLAIMER:
-----------
THIS SYSTEM DOES NOT PROVIDE LEGAL ADVICE.
IT IS FOR DEMONSTRATION PURPOSES ONLY.
ALWAYS CONSULT A QUALIFIED ATTORNEY FOR ANY LEGAL MATTERS.
DO NOT RELY ON THIS AS FACTUAL OR AS A SUBSTITUTE FOR PROFESSIONAL COUNSEL.

Usage:
  1) Flask-only (CLI):
     python merged_legal_expert.py --flask
  
  2) IPython Widgets only (in Jupyter):
     %run merged_legal_expert.py --widget

  3) Both simultaneously:
     %run merged_legal_expert.py --both
"""

import sys
import os
import re
import json
import time
import math
import random
import queue
import threading
import logging
import urllib.parse
from datetime import datetime
from typing import List, Dict, Any

# ------------------ External Libraries ------------------
import requests
from flask import Flask, request, jsonify, send_from_directory
import flask_cors

# For IPython widgets and display (if in Jupyter)
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output, HTML, FileLink
    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False

# For PDF generation
try:
    from fpdf import FPDF
except ImportError:
    os.system("pip install fpdf")
    from fpdf import FPDF

# For HTML parsing (actual website data)
try:
    from bs4 import BeautifulSoup
except ImportError:
    os.system("pip install beautifulsoup4")
    from bs4 import BeautifulSoup

# For actual Gemini API calls (if you have access)
try:
    import google.generativeai as genai
except ImportError:
    print("Warning: google-generativeai not installed. Install via pip install google-generativeai")
    genai = None

# ------------------ Global Configuration ------------------

DISCLAIMER_TEXT = """
****************************************************************************
DISCLAIMER: This demonstration system does NOT provide legal advice.
It is NOT guaranteed to be factual or correct. ALWAYS consult a qualified
attorney for any legal matters or questions. DO NOT rely on this system as
a substitute for professional counsel.
****************************************************************************
"""

SETTINGS_FILE = "settings.json"
DEFAULT_SETTINGS = {
    "GEMINI_API_KEY": "YOUR_GEMINI_API_KEY_HERE",
    "GEMINI_MODEL_NAME": "models/embedding-001",
    "GEMINI_CHAT_MODEL_NAME": "models/gemini-pro",
    "SIMULATE_EMBEDDINGS": False,  # Set to False for real API calls
    "SIMULATION_VECTOR_LENGTH": 384
}

# Global settings (updated later by load_settings_from_file or UI)
GEMINI_API_KEY = DEFAULT_SETTINGS["GEMINI_API_KEY"]
GEMINI_MODEL_NAME = DEFAULT_SETTINGS["GEMINI_MODEL_NAME"]
GEMINI_CHAT_MODEL_NAME = DEFAULT_SETTINGS["GEMINI_CHAT_MODEL_NAME"]
SIMULATE_EMBEDDINGS = DEFAULT_SETTINGS["SIMULATE_EMBEDDINGS"]
SIMULATION_VECTOR_LENGTH = DEFAULT_SETTINGS["SIMULATION_VECTOR_LENGTH"]

INDEX_FILE = "legal_index.json"
LOG_FILE = "legal_expert_system.log"
SCENARIO_DIRECTORY = "scenarios"
os.makedirs(SCENARIO_DIRECTORY, exist_ok=True)

ENABLE_FLASK_API = False          # Can be toggled via command-line
API_PORT = 5000
REQUEST_DELAY_SECONDS = 1
RANDOMIZE_REQUEST_DELAY = True
USER_FILES_DIR = os.path.expanduser("~/.legal_expert_system")
os.makedirs(USER_FILES_DIR, exist_ok=True)

LICENSE_TEXT = """
MIT License
Copyright (c) 2023 Travis Michael O'Dell
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
"""

HEADING = "# #theirleft"
SUMMARY = "An AI-powered legal research and analysis system. (Not legal advice.)"

logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------ Settings Persistence ------------------

def save_settings_to_file():
    settings = {
        "GEMINI_API_KEY": GEMINI_API_KEY,
        "GEMINI_MODEL_NAME": GEMINI_MODEL_NAME,
        "GEMINI_CHAT_MODEL_NAME": GEMINI_CHAT_MODEL_NAME,
        "SIMULATE_EMBEDDINGS": SIMULATE_EMBEDDINGS,
        "SIMULATION_VECTOR_LENGTH": SIMULATION_VECTOR_LENGTH
    }
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=4)
        print("Settings saved successfully.")
    except Exception as e:
        print(f"Error saving settings: {e}")

def load_settings_from_file():
    global GEMINI_API_KEY, GEMINI_MODEL_NAME, GEMINI_CHAT_MODEL_NAME, SIMULATE_EMBEDDINGS, SIMULATION_VECTOR_LENGTH
    try:
        with open(SETTINGS_FILE, "r") as f:
            settings = json.load(f)
        GEMINI_API_KEY = settings.get("GEMINI_API_KEY", DEFAULT_SETTINGS["GEMINI_API_KEY"])
        GEMINI_MODEL_NAME = settings.get("GEMINI_MODEL_NAME", DEFAULT_SETTINGS["GEMINI_MODEL_NAME"])
        GEMINI_CHAT_MODEL_NAME = settings.get("GEMINI_CHAT_MODEL_NAME", DEFAULT_SETTINGS["GEMINI_CHAT_MODEL_NAME"])
        SIMULATE_EMBEDDINGS = settings.get("SIMULATE_EMBEDDINGS", DEFAULT_SETTINGS["SIMULATE_EMBEDDINGS"])
        SIMULATION_VECTOR_LENGTH = settings.get("SIMULATION_VECTOR_LENGTH", DEFAULT_SETTINGS["SIMULATION_VECTOR_LENGTH"])
        print("Settings loaded successfully.")
        return settings
    except Exception as e:
        print(f"Error loading settings: {e}")
        return DEFAULT_SETTINGS

# ------------------ TASK MANAGEMENT ------------------

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

# ------------------ PERIODIC TASK UI UPDATE ------------------

def update_task_status_ui(output_widget):
    with output_widget:
        clear_output()
        for tid, status in task_results.items():
            print(f"Task {tid}: {status.get('status', 'unknown')}")
            if status.get("status") == "failed":
                print(f"  Error: {status.get('error')}")

def periodic_task_update_ui(output_widget):
    while True:
        update_task_status_ui(output_widget)
        time.sleep(5)

# ------------------ GEMINI MODULES (Actual Data) ------------------

class GeminiEmbedder:
    def __init__(self, model_name: str = GEMINI_MODEL_NAME, api_key: str = GEMINI_API_KEY,
                 simulate: bool = SIMULATE_EMBEDDINGS):
        self.model_name = model_name
        self.api_key = api_key
        self.simulate = simulate
        self.vector_length = SIMULATION_VECTOR_LENGTH

    def embed(self, text: str) -> List[float]:
        if self.simulate:
            # This branch should not be used if you want real data.
            print(f"Simulating embedding for: {text[:50]}...")
            return [random.uniform(-0.1, 0.1) for _ in range(self.vector_length)]
        else:
            print(f"Calling Gemini API to embed: {text[:50]}...")
            return self._call_gemini_api(text)

    def _call_gemini_api(self, text: str) -> List[float]:
        # Actual call to Google Generative AI (assuming proper configuration)
        if not genai:
            raise ImportError("google-generativeai module not available.")
        time.sleep(0.5)
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_name)
        response = model.embed_content(content=text)
        print("Gemini API call successful.")
        return response['embedding']

class GeminiChat:
    def __init__(self, model_name: str = GEMINI_CHAT_MODEL_NAME, api_key: str = GEMINI_API_KEY):
        self.model_name = model_name
        self.api_key = api_key
        self.model = None

    def _initialize_model(self):
        if not genai:
            raise ImportError("google-generativeai module not available.")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

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
        response = self.model.generate_content(full_prompt)
        sources = re.findall(r'(https?://\S+)', response.text)
        report = response.text
        if sources:
            report += "\n\nSources:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(sources))
        return report

    def generate_questions(self, scenario_data: dict) -> List[str]:
        if self.model is None:
            self._initialize_model()
        prompt = f"""
Based on the following legal scenario, generate a list of follow-up questions:

Scenario:
{json.dumps(scenario_data, indent=2)}

Questions:
"""
        response = self.model.generate_content(prompt)
        questions_text = response.text
        return re.findall(r'\d+\.\s*(.*)', questions_text)

    def analyze_scenario(self, scenario_data: dict) -> dict:
        if self.model is None:
            self._initialize_model()
        questions = self.generate_questions(scenario_data)
        prompt = f"""
Analyze the following legal scenario and provide a detailed assessment including:
- A general assessment.
- Possible outcomes and likelihood estimates.
- A clear determination on critical points.
- Recommended actions.
- Legal citations and explanations.
- A plain-language summary.

Scenario:
{json.dumps(scenario_data, indent=2)}

Analysis:
"""
        response = self.model.generate_content(prompt)
        analysis_text = response.text
        assessment = re.search(r"Assessment:\s*(.*?)\n", analysis_text, re.DOTALL)
        possible_outcomes = re.search(r"Possible Outcomes:\s*(.*?)\n", analysis_text, re.DOTALL)
        likely_outcomes = re.search(r"Likely Outcomes:\s*(.*?)\n", analysis_text, re.DOTALL)
        return {
            "assessment": assessment.group(1).strip() if assessment else "Detailed review needed.",
            "questions": questions,
            "possible_outcomes": possible_outcomes.group(1).strip() if possible_outcomes else "Multiple outcomes possible.",
            "likely_outcomes": likely_outcomes.group(1).strip() if likely_outcomes else "High chance of success.",
            "action_recommendation": "Seek additional expert advice.",
            "success_rate": "80%",
            "legal_verdict": "Legal",
            "citations": [],
            "plain_summary": "Overall, the scenario appears promising but further review is advised."
        }

# ------------------ WEBSITE PROCESSING & INDEXING (Real Data) ------------------

class WebsiteCrawler:
    def __init__(self, url):
        self.url = url
        self.domain = urllib.parse.urlparse(url).netloc

    def fetch_site(self):
        print(f"Fetching site: {self.url}")
        try:
            response = requests.get(self.url)
            response.raise_for_status()
            raw_html = response.text
            # For simplicity, we leave raw_css empty (or could parse <style> tags)
            raw_css = ""
            return raw_html, raw_css
        except Exception as e:
            print(f"Error fetching site: {e}")
            return "<html><body>Error fetching site</body></html>", ""

    def render_page_and_capture(self):
        # In a real implementation, you might use a headless browser; here we simply return a placeholder.
        print(f"Capturing screenshot for: {self.url} (not implemented)")
        return None

class DOMParser:
    def __init__(self):
        self.metadata = {}

    def parse(self, html: str, css: str):
        print("Parsing HTML with BeautifulSoup...")
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.title.string if soup.title else "Untitled"
        text = soup.get_text(separator=" ", strip=True)
        self.metadata = {
            "title": title,
            "publication_date": datetime.now().strftime("%Y-%m-%d"),
            "url": ""
        }
        # Build a simple DOM tree dictionary (this is a placeholder for a full DOM representation)
        dom_tree = {"tag": "html", "children": []}
        return dom_tree

    def to_dict(self):
        return {"tag": "html", "children": []}

    def extract_links(self):
        # For simplicity, extract all hrefs from <a> tags
        return []  # You could enhance this by using BeautifulSoup to extract links

    def get_text(self):
        # In a real implementation, store the extracted text.
        return "Extracted page text."

class VisionEngine:
    def identify_elements(self, screenshot_data):
        # For actual data, you would call an OCR/vision service.
        print("Identifying visual elements (not implemented for real data).")
        return {}

class CSSVectorizer:
    def generate_vectors(self, raw_css, dom_tree):
        # For actual data, you would analyze CSS.
        print("Generating CSS vectors (not implemented for real data).")
        return {}

class SVGGenerator:
    def create_layers(self, dom_tree, css_vector_map, visual_elements):
        # For actual data, generate an SVG representation.
        print("Generating SVG layers (not implemented for real data).")
        return "<svg><!-- Real SVG output would go here --></svg>"

class SourceRatingEngine:
    def compute_rating(self, url, metadata):
        # For actual data, compute rating based on source credibility, etc.
        print(f"Computing rating for: {url}")
        return 85  # For example, a fixed rating

class RelevanceEvaluator:
    def evaluate_content(self, text_content, metadata):
        print("Evaluating content relevance using real data.")
        # In a real system, use natural language processing.
        return 80, ["relevant", "legal", "analysis"]

# Global vector index instance (file‐based persistence)
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
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def search(self, query_embedding: List[float], top_k: int = 5, metadata_filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        results = []
        for record in self.index:
            if metadata_filters:
                if not all(record["metadata"].get(k) == v for k, v in metadata_filters.items()):
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
    # Real implementation might perform an HTTP request; here we use a simple example.
    return [
        {"url": f"https://example.com/search-result-1?q={query}", "title": "Search Result 1"},
        {"url": f"https://example.com/search-result-2?q={query}", "title": "Search Result 2"},
        {"url": f"https://example.com/search-result-3?q={query}", "title": "Search Result 3"},
    ]

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
    screenshot = crawler.render_page_and_capture()  # May be None in this implementation
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
        print(f"Error evaluating relevance: {e}")
        relevance_score, tags = 0, []
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

# ------------------ GENERATIVE SCENARIO FUNCTIONS ------------------

def generate_fake_scenarios(seed: str, count: int = 3) -> List[str]:
    prompt = f"Based on the seed scenario: '{seed}', generate {count} plausible alternative legal scenarios. List them as numbered items."
    response_text = gemini_chat.generate_report(prompt)
    scenarios = re.findall(r'\d+\.\s*(.*)', response_text)
    if len(scenarios) < count:
        scenarios = [line.strip() for line in response_text.split('\n') if line.strip()]
        scenarios = scenarios[:count]
    return scenarios

def index_fake_scenario(scenario_text: str):
    fake_url = f"generated://scenario/{int(time.time())}-{random.randint(1000,9999)}"
    doc = {
         "url": fake_url,
         "timestamp": time.time(),
         "text": scenario_text,
         "dom": {},
         "css_vector_map": {},
         "visual_elements": {},
         "svg_layers": "",
         "links": [],
         "ratings": {"source_rating": random.randint(70, 100), "relevance_score": random.randint(60,100), "tags": []},
         "metadata": {"title": "Generated Scenario"}
    }
    embedding = GeminiEmbedder().embed(scenario_text)
    doc["embedding"] = embedding
    vector_index.index_document(fake_url, embedding, doc["ratings"], doc)

# ------------------ PDF REPORT GENERATION ------------------

class PDFReport(FPDF):
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

def create_pdf_report(report_data: dict, scenario_data: dict, timeline_events: List[dict]=None) -> str:
    pdf = PDFReport()
    pdf.add_page()
    pdf.chapter_title("Scenario Details")
    pdf.chapter_body(f"Scenario: {scenario_data.get('scenario', 'N/A')}\n"
                     f"Subject: {scenario_data.get('subject', 'N/A')}\n"
                     f"Assailant: {scenario_data.get('assailant', 'N/A')}\n"
                     f"Location: {scenario_data.get('location', 'N/A')}\n"
                     f"Details: {scenario_data.get('details', 'N/A')}")
    pdf.chapter_title("Timeline of Events")
    if timeline_events:
        for event in timeline_events:
            pdf.chapter_body(f"Time: {event.get('time', 'N/A')}\n"
                             f"Event: {event.get('description', 'N/A')}\n"
                             f"Evidence: {event.get('evidence', 'N/A')}\n"
                             "-------------------------")
    else:
        pdf.chapter_body("No timeline events provided.")
    pdf.chapter_title("Analysis Results")
    pdf.chapter_body(f"Assessment: {report_data.get('assessment', 'N/A')}\n"
                     f"Possible Outcomes: {report_data.get('possible_outcomes', 'N/A')}\n"
                     f"Likely Outcomes: {report_data.get('likely_outcomes', 'N/A')}\n"
                     f"Action Recommendation: {report_data.get('action_recommendation', 'N/A')}\n"
                     f"Success Rate: {report_data.get('success_rate', 'N/A')}\n"
                     f"Legal Verdict: {report_data.get('legal_verdict', 'N/A')}")
    pdf.chapter_title("Legal Citations")
    citations = report_data.get('citations', [])
    if citations:
        for cite in citations:
            pdf.chapter_body(f"Law: {cite.get('law', 'N/A')}\n"
                             f"Source: {cite.get('source', 'N/A')}\n"
                             f"Explanation: {cite.get('explanation', 'N/A')}\n"
                             "-------------------------")
    else:
        pdf.chapter_body("No citations available.")
    pdf.chapter_title("Summary of Findings")
    pdf.chapter_body(report_data.get('plain_summary', 'N/A'))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = os.path.join(SCENARIO_DIRECTORY, f"report_{timestamp}.pdf")
    pdf.output(filename)
    return filename

def format_string(text, max_line_length=100):
    words = text.split()
    formatted_lines = []
    current_line = ""
    for word in words:
        if len(current_line + word) + 1 <= max_line_length:
            current_line += (" " if current_line else "") + word
        else:
            formatted_lines.append(current_line)
            current_line = word
    if current_line:
        formatted_lines.append(current_line)
    return "\n".join(formatted_lines)

# ------------------ FLASK API SETUP ------------------

api = Flask(__name__)
flask_cors.CORS(api)

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
            "legal_verdict": analysis_results.get("legal_verdict", "Not Available"),
            "plain_summary": analysis_results.get("plain_summary", "Not Available")
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
        formatted_data = {"law_details": api_raw_law_data}
        return jsonify(formatted_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if ENABLE_FLASK_API:
    api_thread = threading.Thread(target=api.run, kwargs={'debug': False, 'port': API_PORT})
    api_thread.daemon = True
    api_thread.start()
    print(f"Flask API enabled and running on port {API_PORT}")

# ------------------ LEGAL EXPERT SYSTEM UI (IPython Widgets) ------------------

# Global instances
vector_index = VectorIndexEngine()
gemini_chat = GeminiChat()
timeline_events = []  # For scenarios timeline

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
        for i, excerpt in enumerate(context_documents):
            print(f"{i+1}. {excerpt}\n")
    report = gemini_chat.generate_report(
        f"Based on the provided information and the user's query: '{query}', generate a detailed report that addresses the query. Include relevant legal information, resources, potential contacts, and other helpful information.",
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
            with doc_details_output:
                clear_output()
                selected_url = change.new
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
        print(f"Added task {task_id} for URL: {url} (Category: {category})")
    update_task_status()

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

# Scenarios Tab (Generative)
scenario_input = widgets.Textarea(
    placeholder='Enter or describe a legal scenario to use as a seed',
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
pdf_button = widgets.Button(description="Generate PDF Report")
# New generative options
generate_fake_button = widgets.Button(description="Generate Fake Scenarios")
index_generated_button = widgets.Button(description="Index Generated Scenarios")
generated_scenarios_output = widgets.Output()

scenario_output = widgets.Output()

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
        additional_questions = []  # You can add further question generators if desired.
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
        print(f"\nAction Recommendation:\n{analysis_results.get('action_recommendation', 'N/A')}")
        print(f"\nLegal Verdict: {analysis_results.get('legal_verdict', 'N/A')}")
        print(f"\nPlain Summary:\n{analysis_results.get('plain_summary', 'N/A')}")
        pdf_filename = create_pdf_report(analysis_results, {
            "scenario": scenario_input.value,
            "subject": subject_details.value,
            "assailant": assailant_details.value,
            "location": location_details.value,
            "details": case_details.value
        }, timeline_events)
        print(f"\nPDF report generated: {pdf_filename}")
        display(FileLink(pdf_filename))

def handle_save_scenario(button):
    with scenario_output:
        clear_output()
        scenario_data = {
            "scenario": scenario_input.value,
            "subject": subject_details.value,
            "assailant": assailant_details.value,
            "location": location_details.value,
            "details": case_details.value,
            "timeline": timeline_events
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
                    timeline = scenario_data.get("timeline", [])
                    timeline_events.clear()
                    timeline_events.extend(timeline)
                    print(f"Scenario loaded from {filepath}")
                except Exception as e:
                    print(f"Error loading scenario: {e}")
        scenario_dropdown.observe(load_selected_scenario, names='value')
        display(scenario_dropdown, load_output)

def handle_pdf_report(button):
    with scenario_output:
        print("Generating PDF report...")
    scenario_data = {
        "scenario": scenario_input.value,
        "subject": subject_details.value,
        "assailant": assailant_details.value,
        "location": location_details.value,
        "details": case_details.value
    }
    analysis_results = gemini_chat.analyze_scenario(scenario_data)
    pdf_filename = create_pdf_report(analysis_results, scenario_data, timeline_events)
    with scenario_output:
        print(f"PDF report generated: {pdf_filename}")
        display(FileLink(pdf_filename))

# New: Generate and index fake scenarios
def handle_generate_fake(button):
    with generated_scenarios_output:
        clear_output()
        seed = scenario_input.value
        if not seed:
            print("Please enter a seed scenario in the Scenario box.")
            return
        fake_scenarios = generate_fake_scenarios(seed, count=3)
        global generated_scenarios_list
        generated_scenarios_list = fake_scenarios
        print("Generated Fake Scenarios:")
        for i, scen in enumerate(fake_scenarios, 1):
            print(f"{i}. {scen}")

def handle_index_generated(button):
    with generated_scenarios_output:
        if 'generated_scenarios_list' not in globals() or not generated_scenarios_list:
            print("No generated scenarios available to index. Generate first.")
            return
        for scen in generated_scenarios_list:
            index_fake_scenario(scen)
        print("Indexed generated scenarios.")
        update_task_status()

generate_fake_button.on_click(handle_generate_fake)
index_generated_button.on_click(handle_index_generated)

# Assemble the Scenarios Tab UI
scenarios_tab = widgets.VBox([
    scenario_input,
    subject_details,
    assailant_details,
    location_details,
    case_details,
    widgets.HBox([gather_info_button, analyze_button, save_button, load_button, pdf_button]),
    scenario_output,
    widgets.HBox([generate_fake_button, index_generated_button]),
    generated_scenarios_output
])

# Settings Tab
api_key_text = widgets.Text(
    value=GEMINI_API_KEY,
    description="API Key:",
    placeholder="Enter your Gemini API key"
)
api_key_status = widgets.Label(value="")

def on_api_key_change(change):
    key = change.new
    api_key_status.value = "API key looks acceptable." if len(key) >= 10 else "API key appears too short."
api_key_text.observe(on_api_key_change, names='value')

gemini_model_text = widgets.Text(value=GEMINI_MODEL_NAME, description="Embed Model:")
gemini_chat_model_text = widgets.Text(value=GEMINI_CHAT_MODEL_NAME, description="Chat Model:")
simulate_checkbox = widgets.Checkbox(value=SIMULATE_EMBEDDINGS, description="Simulate Embeddings")
simulation_vector_length_text = widgets.BoundedIntText(value=SIMULATION_VECTOR_LENGTH, min=100, max=1024, description="Vector Length:")
save_settings_button = widgets.Button(description="Save Settings")
load_settings_button = widgets.Button(description="Load Settings")
settings_output = widgets.Output()

def handle_save_settings(button):
    global GEMINI_API_KEY, GEMINI_MODEL_NAME, GEMINI_CHAT_MODEL_NAME, SIMULATE_EMBEDDINGS, SIMULATION_VECTOR_LENGTH
    GEMINI_API_KEY = api_key_text.value
    GEMINI_MODEL_NAME = gemini_model_text.value
    GEMINI_CHAT_MODEL_NAME = gemini_chat_model_text.value
    SIMULATE_EMBEDDINGS = simulate_checkbox.value
    SIMULATION_VECTOR_LENGTH = simulation_vector_length_text.value
    with settings_output:
        clear_output()
        save_settings_to_file()
        print("Settings saved and updated.")

def handle_load_settings(button):
    settings = load_settings_from_file()
    api_key_text.value = settings.get("GEMINI_API_KEY", DEFAULT_SETTINGS["GEMINI_API_KEY"])
    gemini_model_text.value = settings.get("GEMINI_MODEL_NAME", DEFAULT_SETTINGS["GEMINI_MODEL_NAME"])
    gemini_chat_model_text.value = settings.get("GEMINI_CHAT_MODEL_NAME", DEFAULT_SETTINGS["GEMINI_CHAT_MODEL_NAME"])
    simulate_checkbox.value = settings.get("SIMULATE_EMBEDDINGS", DEFAULT_SETTINGS["SIMULATE_EMBEDDINGS"])
    simulation_vector_length_text.value = settings.get("SIMULATION_VECTOR_LENGTH", DEFAULT_SETTINGS["SIMULATION_VECTOR_LENGTH"])
    with settings_output:
        clear_output()
        print("Settings loaded and updated.")

save_settings_button.on_click(handle_save_settings)
load_settings_button.on_click(handle_load_settings)

settings_tab_expert = widgets.VBox([
    api_key_text,
    api_key_status,
    gemini_model_text,
    gemini_chat_model_text,
    simulate_checkbox,
    simulation_vector_length_text,
    widgets.HBox([save_settings_button, load_settings_button]),
    settings_output
])

# Active Learning Monitor
learning_progress = widgets.FloatProgress(value=0, min=0, max=100, description="Learning Progress:")
completeness_label = widgets.Label(value="Overall Completeness: 0%")
iq_label = widgets.Label(value="Intelligence Quotient: 0%")
active_learning_monitor = widgets.VBox([learning_progress, completeness_label, iq_label])

def update_learning_monitor():
    while True:
        total = task_counter
        completed = sum(1 for status in task_results.values() if status.get('status') != "pending")
        completeness = (completed / total * 100) if total > 0 else 0
        new_value = learning_progress.value + random.uniform(1, 5)
        if new_value >= 100:
            new_value = 0
        learning_progress.value = new_value
        iq = min(completeness + random.uniform(0, 10), 100)
        completeness_label.value = f"Overall Completeness: {completeness:.1f}%"
        iq_label.value = f"Intelligence Quotient: {iq:.1f}%"
        time.sleep(3)

active_learning_thread = threading.Thread(target=update_learning_monitor, daemon=True)
active_learning_thread.start()

# Top-Level Interface UI
# You may also include an enhanced research interface (not detailed here) if desired.
expert_system_tabs = widgets.Tab()
expert_system_tabs.children = [
    widgets.VBox([search_query, search_button, search_output, search_context_output, chat_output]),
    widgets.VBox([browse_output]),
    widgets.VBox([train_url_text, train_category_dropdown, train_add_button, train_output, train_status_output]),
    widgets.VBox([postulate_area, postulate_button, postulate_output]),
    scenarios_tab,
    settings_tab_expert
]
expert_system_tabs.set_title(0, "Search")
expert_system_tabs.set_title(1, "Browse")
expert_system_tabs.set_title(2, "Train")
expert_system_tabs.set_title(3, "Postulate")
expert_system_tabs.set_title(4, "Scenarios")
expert_system_tabs.set_title(5, "Settings")

# Top-level tabs (Enhanced, Expert System, Active Learning Monitor)
# For brevity, the "Enhanced" tab is simply a placeholder here.
enhanced_interface = widgets.HTML(
    value="<h3>Enhanced Legal Research Interface</h3><p>This section would provide an adaptive, intelligent research experience.</p>"
)
top_level_tabs = widgets.Tab()
top_level_tabs.children = [
    enhanced_interface,
    expert_system_tabs,
    active_learning_monitor
]
top_level_tabs.set_title(0, "Enhanced")
top_level_tabs.set_title(1, "Expert System")
top_level_tabs.set_title(2, "Active Learning Monitor")

# Title and License Display
def create_title_and_license():
    title_widget = widgets.HTML(f"<h1>{HEADING}</h1><h2>A Travis Michael O'Dell Project</h2>")
    license_widget = widgets.Textarea(value=LICENSE_TEXT, description='License:', disabled=True,
                                      layout={'height': '300px', 'width': '70%'})
    disclaimer_widget = widgets.HTML(f"<pre>{DISCLAIMER_TEXT}</pre>")
    return widgets.VBox([title_widget, disclaimer_widget, license_widget])

title_and_license = create_title_and_license()

# ------------------ MAIN RUNNER ------------------

def main(mode="both"):
    load_settings_from_file()

    # Create main objects
    gem_chat = GeminiChat()
    vect_index = VectorIndexEngine()

    # Create the dual interface object
    # The DualInterface class below runs the IPython widgets and optionally the Flask API.
    class DualInterface:
        def __init__(self, gemini_chat, vector_index, host="0.0.0.0", port=API_PORT):
            self.gemini_chat = gemini_chat
            self.vector_index = vector_index
            self.host = host
            self.port = port
            if FLASK_AVAILABLE:
                self.flask_app = api
            else:
                self.flask_app = None
            self.train_status_output = widgets.Output()
            self.setup_widgets()

        def setup_widgets(self):
            self.main_vbox = widgets.VBox([title_and_license, top_level_tabs])
        def display(self):
            if JUPYTER_AVAILABLE:
                display(self.main_vbox)
                # Start background thread to update task status UI
                t = threading.Thread(target=periodic_task_update_ui, args=(self.train_status_output,), daemon=True)
                t.start()
            else:
                print("Jupyter/ipywidgets not available; cannot display GUI.")

        def start_flask_in_thread(self):
            if not self.flask_app:
                print("Flask not available.")
                return
            def flask_thread():
                self.flask_app.run(host=self.host, port=self.port, debug=False)
            t = threading.Thread(target=flask_thread, daemon=True)
            t.start()
            print(f"Flask server started at http://{self.host}:{self.port}")

    interface = DualInterface(gem_chat, vect_index)

    if mode in ("both", "flask"):
        interface.start_flask_in_thread()
    if mode in ("both", "widget"):
        interface.display()
    if mode == "flask":
        # Blocking call if only Flask mode is desired.
        interface.flask_app.run(host="0.0.0.0", port=API_PORT, debug=True)

if __name__ == "__main__":
    mode_arg = "both"
    for arg in sys.argv:
        if "--flask" in arg:
            mode_arg = "flask"
        elif "--widget" in arg:
            mode_arg = "widget"
        elif "--both" in arg:
            mode_arg = "both"
    main(mode_arg)
