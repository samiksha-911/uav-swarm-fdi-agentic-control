#!/usr/bin/env python3
"""
generate_project_documentation.py
==================================
Generates a comprehensive, professional PDF documentation for the
UAV Swarm Fault Detection & Isolation (FDI) Project.

Usage:
    /usr/bin/python3 generate_project_documentation.py

Output:
    UAV_Swarm_FDI_Project_Documentation.pdf
"""

import os
import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, mm, cm
from reportlab.lib.colors import (
    HexColor, black, white, gray, darkgray, lightgrey,
    Color, red, green, blue
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, ListFlowable, ListItem, KeepTogether,
    Image as RLImage
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle
from reportlab.graphics import renderPDF

# ─── Constants ────────────────────────────────────────────────────────────────
OUTPUT_PDF = os.path.join(os.path.dirname(__file__), "UAV_Swarm_FDI_Project_Documentation.pdf")
PAGE_W, PAGE_H = A4
MARGIN = 1 * inch

# ─── Color Palette ────────────────────────────────────────────────────────────
PRIMARY = HexColor("#1a237e")       # Deep Indigo
SECONDARY = HexColor("#0d47a1")    # Blue
ACCENT = HexColor("#00bcd4")       # Cyan
ACCENT2 = HexColor("#ff6f00")      # Amber
LIGHT_BG = HexColor("#e8eaf6")     # Light lavender
TABLE_HEADER = HexColor("#283593")
TABLE_STRIPE = HexColor("#f5f5f5")
DANGER = HexColor("#c62828")
SUCCESS = HexColor("#2e7d32")

# ─── Styles ───────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

# Title page styles
styles.add(ParagraphStyle(
    'DocTitle', parent=styles['Title'],
    fontSize=28, leading=34, textColor=PRIMARY,
    spaceAfter=6, alignment=TA_CENTER, fontName='Helvetica-Bold',
))
styles.add(ParagraphStyle(
    'DocSubtitle', parent=styles['Normal'],
    fontSize=16, leading=20, textColor=SECONDARY,
    spaceAfter=20, alignment=TA_CENTER, fontName='Helvetica',
))
styles.add(ParagraphStyle(
    'DocAuthor', parent=styles['Normal'],
    fontSize=12, leading=16, textColor=darkgray,
    spaceAfter=4, alignment=TA_CENTER, fontName='Helvetica-Oblique',
))

# Section heading styles
styles.add(ParagraphStyle(
    'CH', parent=styles['Heading1'],
    fontSize=22, leading=26, textColor=PRIMARY,
    spaceBefore=20, spaceAfter=10, fontName='Helvetica-Bold',
    borderWidth=0, borderColor=ACCENT, borderPadding=4,
))
styles.add(ParagraphStyle(
    'SH', parent=styles['Heading2'],
    fontSize=16, leading=20, textColor=SECONDARY,
    spaceBefore=14, spaceAfter=6, fontName='Helvetica-Bold',
))
styles.add(ParagraphStyle(
    'SSH', parent=styles['Heading3'],
    fontSize=13, leading=17, textColor=HexColor("#37474f"),
    spaceBefore=10, spaceAfter=4, fontName='Helvetica-Bold',
))

# Body text
styles.add(ParagraphStyle(
    'B', parent=styles['Normal'],
    fontSize=10.5, leading=15, textColor=black,
    spaceBefore=2, spaceAfter=6, alignment=TA_JUSTIFY,
    fontName='Helvetica',
))
styles.add(ParagraphStyle(
    'BI', parent=styles['Normal'],
    fontSize=10.5, leading=15, textColor=black,
    spaceBefore=2, spaceAfter=4, alignment=TA_JUSTIFY,
    fontName='Helvetica-Oblique',
))

# Code / monospaced
styles.add(ParagraphStyle(
    'CodeBlock', parent=styles['Normal'],
    fontSize=9, leading=12, textColor=HexColor("#263238"),
    fontName='Courier', backColor=HexColor("#eceff1"),
    leftIndent=12, rightIndent=12,
    spaceBefore=4, spaceAfter=4, borderPadding=6,
))

# Bullet / list
styles.add(ParagraphStyle(
    'BulletItem', parent=styles['Normal'],
    fontSize=10.5, leading=14.5, textColor=black,
    leftIndent=24, bulletIndent=12,
    spaceBefore=1, spaceAfter=2, fontName='Helvetica',
))

# Callout / note
styles.add(ParagraphStyle(
    'Note', parent=styles['Normal'],
    fontSize=10, leading=14, textColor=HexColor("#1b5e20"),
    fontName='Helvetica-Oblique',
    leftIndent=16, borderWidth=1, borderColor=SUCCESS,
    borderPadding=8, backColor=HexColor("#e8f5e9"),
    spaceBefore=8, spaceAfter=8,
))
styles.add(ParagraphStyle(
    'Warning', parent=styles['Normal'],
    fontSize=10, leading=14, textColor=DANGER,
    fontName='Helvetica-Bold',
    leftIndent=16, borderWidth=1, borderColor=DANGER,
    borderPadding=8, backColor=HexColor("#ffebee"),
    spaceBefore=8, spaceAfter=8,
))

# Table caption
styles.add(ParagraphStyle(
    'Caption', parent=styles['Normal'],
    fontSize=9, leading=12, textColor=darkgray,
    fontName='Helvetica-Oblique', alignment=TA_CENTER,
    spaceBefore=4, spaceAfter=8,
))


# ─── Helper functions ────────────────────────────────────────────────────────

def ch(text):
    """Chapter heading"""
    return Paragraph(text, styles['CH'])

def sh(text):
    """Section heading"""
    return Paragraph(text, styles['SH'])

def ssh(text):
    """Sub-section heading"""
    return Paragraph(text, styles['SSH'])

def p(text):
    """Body paragraph"""
    return Paragraph(text, styles['B'])

def pi(text):
    """Italic paragraph"""
    return Paragraph(text, styles['BI'])

def code(text):
    """Code block"""
    return Paragraph(text.replace('\n', '<br/>').replace(' ', '&nbsp;'), styles['CodeBlock'])

def note(text):
    """Green note box"""
    return Paragraph(f"📌 {text}", styles['Note'])

def warn(text):
    """Red warning box"""
    return Paragraph(f"⚠️ {text}", styles['Warning'])

def bullet_list(items):
    """Create a bullet list"""
    return ListFlowable(
        [ListItem(Paragraph(item, styles['BulletItem']), bulletColor=ACCENT)
         for item in items],
        bulletType='bullet', start='•',
        leftIndent=16, bulletFontSize=10,
    )

def spacer(h=12):
    return Spacer(1, h)

def hr():
    return HRFlowable(width="100%", thickness=1, color=lightgrey, spaceBefore=8, spaceAfter=8)

def make_table(data, col_widths=None, has_header=True):
    """Create a styled table."""
    style_cmds = [
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('LEADING', (0, 0), (-1, -1), 13),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, lightgrey),
    ]
    if has_header:
        style_cmds += [
            ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
        ]
    # Alternate row shading
    for i in range(1, len(data)):
        if i % 2 == 0:
            style_cmds.append(('BACKGROUND', (0, i), (-1, i), TABLE_STRIPE))
    
    t = Table(data, colWidths=col_widths, repeatRows=1 if has_header else 0)
    t.setStyle(TableStyle(style_cmds))
    return t


# ─── Header/Footer ───────────────────────────────────────────────────────────

def header_footer(canvas, doc):
    """Draw page header and footer on every page (except title page)."""
    canvas.saveState()
    page_num = doc.page
    if page_num > 1:
        # Header line
        canvas.setStrokeColor(ACCENT)
        canvas.setLineWidth(1.5)
        canvas.line(MARGIN, PAGE_H - 0.6*inch, PAGE_W - MARGIN, PAGE_H - 0.6*inch)
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(darkgray)
        canvas.drawString(MARGIN, PAGE_H - 0.55*inch,
                          "UAV Swarm FDI — Autonomous Multi-Agent Fault Detection & Isolation System")
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 0.55*inch, "Project Documentation")
        
        # Footer
        canvas.setStrokeColor(lightgrey)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, 0.6*inch, PAGE_W - MARGIN, 0.6*inch)
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(gray)
        canvas.drawString(MARGIN, 0.42*inch, "Confidential — UAV Swarm FDI Research Project")
        canvas.drawRightString(PAGE_W - MARGIN, 0.42*inch, f"Page {page_num}")
    canvas.restoreState()


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD THE DOCUMENT
# ═══════════════════════════════════════════════════════════════════════════════

def build_doc():
    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=0.9*inch, bottomMargin=0.8*inch,
        title="UAV Swarm FDI — Complete Project Documentation",
        author="UAV FDI Research Team",
    )
    
    story = []
    
    # ═══════════════════════════════════════════════════════════════════════
    # TITLE PAGE
    # ═══════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 1.5*inch))
    
    # Decorative top line
    story.append(HRFlowable(width="80%", thickness=3, color=PRIMARY, spaceBefore=0, spaceAfter=20))
    
    story.append(Paragraph("UAV Swarm Fault Detection<br/>&amp; Isolation System", styles['DocTitle']))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Autonomous Multi-Agent Agentic AI Architecture<br/>"
        "with Real-Time Mission Control Dashboard",
        styles['DocSubtitle']
    ))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="60%", thickness=1.5, color=ACCENT, spaceBefore=10, spaceAfter=20))
    
    story.append(Spacer(1, 0.5*inch))
    
    # Info table on title page
    today = datetime.datetime.now().strftime("%B %d, %Y")
    info_data = [
        ["Project", "UAV Swarm FDI — Multi-Agent Autonomous Safety System"],
        ["Domain", "Unmanned Aerial Vehicles (UAVs) / Autonomous Swarm Robotics"],
        ["Core Technology", "LangGraph Agentic AI, Control Barrier Functions (CBF), ML-FDI"],
        ["Version", "1.0 (Production-Ready Simulation)"],
        ["Date", today],
        ["Platform", "Python 3.x / Tornado / Three.js / ROS 2"],
    ]
    info_table = Table(info_data, colWidths=[1.8*inch, 4.2*inch])
    info_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 0), (0, -1), SECONDARY),
        ('TEXTCOLOR', (1, 0), (1, -1), black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LINEBELOW', (0, 0), (-1, -2), 0.5, HexColor("#e0e0e0")),
        ('LINEBELOW', (0, -1), (-1, -1), 1, ACCENT),
    ]))
    story.append(info_table)
    
    story.append(PageBreak())
    
    # ═══════════════════════════════════════════════════════════════════════
    # TABLE OF CONTENTS
    # ═══════════════════════════════════════════════════════════════════════
    story.append(Paragraph("Table of Contents", styles['DocTitle']))
    story.append(hr())
    story.append(spacer(8))
    
    toc_items = [
        ("1.", "Abstract & Project Overview"),
        ("2.", "Problem Statement & Motivation"),
        ("3.", "System Architecture"),
        ("  3.1", "High-Level Architecture Overview"),
        ("  3.2", "10-Node LangGraph Agent Pipeline"),
        ("  3.3", "SwarmBus — Decentralized Communication"),
        ("  3.4", "Control Barrier Functions (CBF-QP)"),
        ("  3.5", "Machine Learning FDI Model"),
        ("4.", "No-Fly Zone Geofencing & Dynamic Rerouting"),
        ("  4.1", "NFZ Configuration"),
        ("  4.2", "Multi-Layer Avoidance Strategy"),
        ("  4.3", "Dynamic Bypass Waypoint Algorithm"),
        ("5.", "Two-Layer Guardrail Safety System"),
        ("  5.1", "Layer 1 — Soft Policy Guardrail"),
        ("  5.2", "Layer 2 — CBF-QP Hard Safety Filter"),
        ("6.", "Swarm Coordination & Intelligence"),
        ("  6.1", "Leader Election Protocol"),
        ("  6.2", "Task Delegation & Handoff"),
        ("  6.3", "Inter-Agent Collision Avoidance"),
        ("7.", "Energy & Battery Management"),
        ("8.", "Risk Assessment & Agentic Reasoning"),
        ("9.", "Mission Control Web Dashboard"),
        ("  9.1", "3D Three.js Visualization"),
        ("  9.2", "2D Radar Map with Zoom/Pan"),
        ("  9.3", "Telemetry Cards & Live Status"),
        ("  9.4", "Fault Injection Console"),
        ("10.", "Terminology & Glossary"),
        ("  10.1", "Flight Actions"),
        ("  10.2", "FDI Fault Labels"),
        ("  10.3", "Energy Alerts"),
        ("  10.4", "Guardrail Statuses"),
        ("  10.5", "Swarm Roles"),
        ("11.", "Technology Stack"),
        ("12.", "File Structure & Codebase"),
        ("13.", "How to Run the System"),
        ("14.", "Future Enhancements"),
        ("15.", "Conclusion"),
    ]
    
    toc_data = []
    for num, title in toc_items:
        is_sub = num.startswith("  ")
        style = 'Helvetica' if is_sub else 'Helvetica-Bold'
        sz = 10 if is_sub else 11
        toc_data.append([
            Paragraph(f'<font name="{style}" size="{sz}" color="#{SECONDARY.hexval()[2:]}">{num.strip()}</font>', styles['B']),
            Paragraph(f'<font name="{style}" size="{sz}">{title}</font>', styles['B']),
        ])
    
    toc_table = Table(toc_data, colWidths=[0.8*inch, 5.2*inch])
    toc_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(toc_table)
    story.append(PageBreak())
    
    # ═══════════════════════════════════════════════════════════════════════
    # CHAPTER 1: ABSTRACT & PROJECT OVERVIEW
    # ═══════════════════════════════════════════════════════════════════════
    story.append(ch("1. Abstract &amp; Project Overview"))
    story.append(hr())
    story.append(p(
        "This project implements a <b>fully autonomous multi-agent UAV swarm system</b> "
        "capable of real-time <b>Fault Detection and Isolation (FDI)</b> during autonomous flight missions. "
        "The system deploys a fleet of 4 (or more, via horizontal scaling) UAV agents, each operating as an "
        "independent <b>LangGraph-based agentic AI pipeline</b> that processes sensor telemetry, detects faults "
        "using machine learning, reasons about risk, coordinates with peer agents, and executes safety-critical "
        "decisions — all without human intervention."
    ))
    story.append(p(
        "Each UAV agent runs a <b>10-node state graph pipeline</b> every simulation tick (~1 second), providing "
        "continuous monitoring, fault detection (6 fault classes), energy management, No-Fly Zone geofencing, "
        "and swarm coordination. The system features a <b>two-layer guardrail safety architecture</b>: a soft "
        "policy guardrail (Layer 1) that enforces operational rules, and a <b>Control Barrier Function (CBF) "
        "quadratic programming safety filter</b> (Layer 2) that mathematically guarantees collision avoidance "
        "and geofence compliance."
    ))
    story.append(p(
        "A <b>web-based Mission Control Dashboard</b> provides real-time visualization through an interactive "
        "3D Three.js scene and a 2D radar map with zoom/pan controls, live telemetry cards for each UAV, "
        "fault injection console, and swarm status overview. The entire system is designed to simulate what a "
        "real-world autonomous drone fleet management and safety assurance system would look like."
    ))
    story.append(note(
        "<b>Key Innovation:</b> This project combines Agentic AI (LangGraph state machines), formal safety "
        "verification (Control Barrier Functions), and Machine Learning fault detection into a unified, "
        "production-ready autonomous swarm architecture."
    ))
    
    # ═══════════════════════════════════════════════════════════════════════
    # CHAPTER 2: PROBLEM STATEMENT & MOTIVATION
    # ═══════════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(ch("2. Problem Statement &amp; Motivation"))
    story.append(hr())
    story.append(p(
        "As autonomous UAV operations scale from single-drone missions to multi-agent swarm deployments, "
        "several critical challenges emerge:"
    ))
    story.append(bullet_list([
        "<b>Sensor Fault Detection:</b> UAVs rely on GPS, IMU (accelerometer, gyroscope), barometer, and compass "
        "sensors. Hardware failures, electromagnetic interference, or environmental factors can corrupt sensor "
        "readings. Without real-time detection, a faulty sensor can cause crashes, navigation errors, or mission failure.",
        "<b>Autonomous Decision Making:</b> In remote or communication-degraded environments, UAVs must make "
        "safety-critical decisions independently — when to abort, reroute, land, or hand off tasks to peers.",
        "<b>No-Fly Zone Compliance:</b> UAVs must never enter restricted airspace (hospitals, airports, military "
        "zones). Violations can result in legal consequences, safety hazards, or drone destruction. The system "
        "must proactively detect and avoid NFZ boundaries, not just react after entry.",
        "<b>Swarm Coordination:</b> Multiple UAVs must avoid colliding with each other, dynamically reassign tasks "
        "when a peer becomes degraded, and maintain formation without centralized control.",
        "<b>Formal Safety Guarantees:</b> Simple rule-based geofencing is insufficient. The system needs "
        "mathematically provable safety guarantees — Control Barrier Functions provide this by ensuring the "
        "system state remains within a defined safe set at all times.",
    ]))
    story.append(p(
        "This project addresses all these challenges in a unified architecture, demonstrating how <b>Agentic AI</b>, "
        "<b>Machine Learning</b>, and <b>Formal Safety Methods</b> can work together to create a robust, "
        "autonomous UAV swarm system."
    ))
    
    # ═══════════════════════════════════════════════════════════════════════
    # CHAPTER 3: SYSTEM ARCHITECTURE
    # ═══════════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(ch("3. System Architecture"))
    story.append(hr())
    
    story.append(sh("3.1 High-Level Architecture Overview"))
    story.append(p(
        "The system consists of three major layers that work together to create a fully autonomous swarm:"
    ))
    
    arch_data = [
        ["Layer", "Component", "Purpose"],
        ["Intelligence\nLayer", "LangGraph 10-Node\nAgent Pipeline", 
         "Each UAV runs an independent AI agent that processes telemetry, detects faults, "
         "reasons about risk, and decides actions autonomously."],
        ["Safety\nLayer", "Two-Layer Guardrail\nSystem (L1 + L2 CBF)", 
         "Layer 1 enforces soft policy rules. Layer 2 uses CBF-QP optimization to "
         "mathematically guarantee collision avoidance and geofence compliance."],
        ["Communication\nLayer", "SwarmBus +\nROS 2 + MAVLink", 
         "Decentralized peer-to-peer message broker for heartbeats, task handoff, "
         "and leader election. Optional ROS 2 and MAVLink integration for real hardware."],
        ["Visualization\nLayer", "Mission Control\nWeb Dashboard", 
         "Tornado web server serving a single-page app with 3D Three.js scene, "
         "2D radar, live telemetry cards, and fault injection console."],
        ["ML Detection\nLayer", "ExtraTrees FDI\nClassifier", 
         "Pre-trained 6-class fault classifier (Normal, GPS, Compass, Accelerometer, "
         "Barometer, Gyroscope) loaded from joblib model files."],
    ]
    story.append(make_table(arch_data, col_widths=[1.1*inch, 1.3*inch, 3.6*inch]))
    story.append(Paragraph("Table 1: System Architecture Layers", styles['Caption']))
    
    # 3.2 LangGraph Pipeline
    story.append(sh("3.2 The 10-Node LangGraph Agent Pipeline"))
    story.append(p(
        "Every UAV agent executes a <b>LangGraph StateGraph</b> pipeline on every simulation tick "
        "(approximately once per second). The pipeline is a directed acyclic graph (DAG) of 10 processing "
        "nodes, where the output state of each node flows into the next. This design ensures that every "
        "decision is made through a structured, auditable, multi-stage reasoning process."
    ))
    
    pipeline_data = [
        ["#", "Node Name", "Function", "Description"],
        ["1", "Telemetry", "node_collect_telemetry",
         "Reads sensor data (position, velocity, altitude, battery), advances waypoint navigation, "
         "performs NFZ proximity detection, applies bypass waypoint rerouting, and updates flight position."],
        ["2", "Energy\nMonitor", "node_monitor_energy",
         "Tracks battery percentage, computes drain rate, classifies energy state "
         "(NOMINAL / WARNING / CRITICAL / EMERGENCY), triggers RTL (Return To Launch) if critical."],
        ["3", "FDI\nInference", "node_run_fdi",
         "Runs the ExtraTrees ML model on current sensor features to detect faults. "
         "Classifies into 6 categories: Normal, GPS Failure, Compass Failure, "
         "Accelerometer Failure, Barometer Failure, Gyroscope Failure."],
        ["4", "Risk\nAssessment", "node_compute_risk",
         "Calculates composite risk score (0.0–1.0) based on fault detection, battery level, "
         "speed anomalies, altitude, and NFZ proximity. Maps to risk tier: LOW / MODERATE / HIGH / CRITICAL."],
        ["5", "Agentic\nReasoning", "node_agentic_reasoning",
         "Core AI decision engine. Based on risk, faults, energy, and NFZ state, decides action: "
         "NOMINAL_CONTINUE, REDUCE_SPEED, ALTITUDE_HOLD, RETURN_TO_HOME, EMERGENCY_LAND, or DELEGATE_TO_SWARM."],
        ["6", "Swarm\nCoordination", "node_swarm_coordination",
         "Handles inter-agent communication: processes handoff requests from degraded peers, "
         "submits bids for task takeover, manages dynamic leader election."],
        ["7", "Guardrail\nL1", "node_guardrail_l1",
         "Soft policy guardrail: validates altitude bounds (2m–120m), speed limits (≤5 m/s), "
         "battery minimums. Can override or limit proposed actions if they violate operational policies."],
        ["8", "Guardrail\nL2", "node_guardrail_l2",
         "Hard safety filter using Control Barrier Functions (CBF) with Quadratic Programming (QP). "
         "Mathematically guarantees ground clearance, speed limits, inter-UAV separation, and NFZ geofencing."],
        ["9", "Execute\nAction", "node_execute_action",
         "Applies the finalized (guardrail-filtered) action. In simulation, updates the state object. "
         "With ROS 2 / MAVLink integration, this would send velocity commands to the flight controller."],
        ["10", "Finalize", "node_finalize",
         "Broadcasts heartbeat to SwarmBus with current state summary (position, velocity, battery, "
         "fault status, role, task progress). Makes UAV state available to all peers."],
    ]
    story.append(make_table(pipeline_data, col_widths=[0.3*inch, 0.7*inch, 1.4*inch, 3.6*inch]))
    story.append(Paragraph("Table 2: LangGraph 10-Node Pipeline — Each UAV Agent", styles['Caption']))
    
    story.append(note(
        "The pipeline is compiled via <font face='Courier'>build_uav_graph()</font> using LangGraph's "
        "<font face='Courier'>StateGraph</font> API. Edges connect START → telemetry → energy → fdi → "
        "risk → reasoning → swarm → guardrail_l1 → guardrail_l2 → execute → finalize → END."
    ))
    
    # 3.3 SwarmBus
    story.append(PageBreak())
    story.append(sh("3.3 SwarmBus — Decentralized Communication Protocol"))
    story.append(p(
        "The <b>SwarmBus</b> is a thread-safe, decentralized message broker that enables peer-to-peer "
        "communication between all UAV agents without a central server. It supports four message types:"
    ))
    
    msg_data = [
        ["Message Type", "Sender", "Purpose"],
        ["HEARTBEAT", "Every UAV\n(every tick)", 
         "Broadcasts current state (position, velocity, battery, fault, availability, role) "
         "to all peers. Used for situational awareness and collision avoidance."],
        ["HANDOFF_REQUEST", "Degraded UAV",
         "Sent when a UAV detects it can no longer complete its mission (low battery, fault). "
         "Includes task details and current waypoint for potential takeover."],
        ["HANDOFF_BID", "Healthy Peer",
         "Response to a HANDOFF_REQUEST. Contains bidding UAV's battery level, distance to "
         "task, and health status. Closest healthy peer with highest battery wins."],
        ["HANDOFF_ACCEPT", "Original UAV",
         "Confirms task transfer to the winning bidder. Includes remaining waypoints, "
         "task context, and authorization for the peer to take over the mission."],
    ]
    story.append(make_table(msg_data, col_widths=[1.3*inch, 1.0*inch, 3.7*inch]))
    story.append(Paragraph("Table 3: SwarmBus Message Types", styles['Caption']))
    
    story.append(p(
        "The SwarmBus also manages: <b>dynamic UAV registration</b> for horizontal scaling (adding new "
        "drones at runtime), <b>active fault injection</b> tracking (for testing), <b>ROS 2 publisher "
        "initialization</b> for live hardware integration, and <b>MAVLink connection management</b> for "
        "QGroundControl (QGC) compatibility."
    ))
    
    # 3.4 CBF
    story.append(sh("3.4 Control Barrier Functions (CBF-QP Safety Filter)"))
    story.append(p(
        "Control Barrier Functions provide <b>mathematically provable safety guarantees</b>. The CBF module "
        "(<font face='Courier'>step76_control_barrier_function.py</font>) implements a QP-based safety filter "
        "that takes the agent's proposed velocity command and projects it onto the nearest safe action."
    ))
    story.append(p("<b>The CBF enforces four barrier constraints simultaneously:</b>"))
    
    cbf_data = [
        ["Barrier", "Mathematical Formulation", "Purpose"],
        ["Ground Impact", "h_alt(z, vz) = (z − z_min) − (vz² / 2a_max) ≥ 0",
         "Ensures the UAV always has enough braking distance before hitting the ground."],
        ["Speed Limit", "h_speed(v) = v_max² − (vx² + vy² + vz²) ≥ 0",
         "Prevents the UAV from exceeding maximum kinetic speed."],
        ["Swarm\nSeparation", "h_swarm(p_i, p_j) = ||p_i − p_j||² − d_min² ≥ 0",
         "Maintains minimum separation distance (2.5m) between all UAV pairs."],
        ["NFZ\nGeofence", "h_nfz(p, c) = ||p − c||² − r_nfz² ≥ 0",
         "Repels UAV from No-Fly Zone centers. Uses repulsive barrier derivative."],
    ]
    story.append(make_table(cbf_data, col_widths=[0.9*inch, 2.5*inch, 2.6*inch]))
    story.append(Paragraph("Table 4: CBF Barrier Constraints", styles['Caption']))
    
    story.append(p(
        "The CBF-QP solver minimizes <b>||u − u_nom||²</b> (deviation from desired velocity) subject to all "
        "barrier constraints, using SciPy's quadratic programming solver. This means the UAV follows its "
        "intended path as closely as possible while being mathematically guaranteed to never violate safety constraints."
    ))
    
    # 3.5 ML FDI Model
    story.append(sh("3.5 Machine Learning FDI Model"))
    story.append(p(
        "The Fault Detection and Isolation (FDI) component uses a pre-trained <b>ExtraTrees (Extremely "
        "Randomized Trees) Classifier</b> to detect sensor faults from telemetry features in real-time."
    ))
    
    fdi_model_data = [
        ["Property", "Value"],
        ["Algorithm", "ExtraTrees Classifier (Ensemble of Randomized Decision Trees)"],
        ["Training Data", "PX4 SITL flight recordings — normal and fault-injected runs"],
        ["Feature Set", "Statistical features from windowed sensor streams (mean, std, range, etc.)"],
        ["Classes (6)", "Normal, GPS Failure, Compass Failure, Accelerometer Failure, "
                        "Barometer Failure, Gyroscope Failure"],
        ["Model File", "step70_clean_fdi_model.joblib"],
        ["Feature Names", "step70_feature_names.joblib"],
        ["Inference", "Real-time: features extracted every tick, classified in <1ms"],
    ]
    story.append(make_table(fdi_model_data, col_widths=[1.4*inch, 4.6*inch]))
    story.append(Paragraph("Table 5: FDI ML Model Specifications", styles['Caption']))
    
    # ═══════════════════════════════════════════════════════════════════════
    # CHAPTER 4: NO-FLY ZONE GEOFENCING
    # ═══════════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(ch("4. No-Fly Zone Geofencing &amp; Dynamic Rerouting"))
    story.append(hr())
    
    story.append(sh("4.1 NFZ Configuration"))
    story.append(p(
        "The system defines two No-Fly Zones representing restricted airspace over sensitive areas. "
        "These zones are circular with defined center coordinates and radii:"
    ))
    
    nfz_data = [
        ["Property", "NFZ-Alpha (Hospital)", "NFZ-Bravo (Airport)"],
        ["Center (x, y)", "(50.0, 60.0) meters", "(−15.0, 90.0) meters"],
        ["Core Radius", "18 meters", "22 meters"],
        ["Buffer Radius\n(Guardrail)", "28 meters (core + 10m)", "32 meters (core + 10m)"],
        ["Hard Wall\nClamp Radius", "23 meters (core + 5m)", "27 meters (core + 5m)"],
        ["Altitude Ceiling", "25 meters", "30 meters"],
        ["Visual Color", "Red (2D) / Red Cylinder (3D)", "Red (2D) / Red Cylinder (3D)"],
        ["Buffer Visual", "Dashed orange circle (2D)\nWireframe cylinder (3D)", 
         "Dashed orange circle (2D)\nWireframe cylinder (3D)"],
    ]
    story.append(make_table(nfz_data, col_widths=[1.4*inch, 2.3*inch, 2.3*inch]))
    story.append(Paragraph("Table 6: No-Fly Zone Configuration", styles['Caption']))
    
    story.append(sh("4.2 Multi-Layer Avoidance Strategy"))
    story.append(p(
        "The system uses a <b>three-tier defense-in-depth strategy</b> to ensure UAVs never enter No-Fly Zones. "
        "These layers work together, with each layer catching what the previous layer might miss:"
    ))
    
    avoid_data = [
        ["Layer", "Trigger Distance", "Action"],
        ["Predictive\nDetection", "Buffer radius\n(28m / 32m from center)", 
         "When a UAV's projected flight path intersects the buffer zone, the system "
         "calculates a tangent bypass waypoint at radius + 14m from NFZ center. "
         "The UAV latches onto this bypass target and navigates around the NFZ."],
        ["Hard Wall\nClamp", "Core + 5m\n(23m / 27m from center)", 
         "If a UAV somehow gets within 5m of the NFZ core boundary, its position is "
         "forcibly clamped to the safe radius. This is an emergency backstop."],
        ["CBF-QP\nBarrier", "Core radius\n(18m / 22m from center)", 
         "The Layer 2 CBF-QP solver includes NFZ geofence barriers that generate "
         "repulsive velocity corrections, making it mathematically impossible for "
         "the UAV to enter the NFZ core."],
    ]
    story.append(make_table(avoid_data, col_widths=[1.0*inch, 1.2*inch, 3.8*inch]))
    story.append(Paragraph("Table 7: Three-Tier NFZ Avoidance Strategy", styles['Caption']))
    
    story.append(sh("4.3 Dynamic Bypass Waypoint Algorithm"))
    story.append(p("When the predictive detection layer triggers, the bypass algorithm works as follows:"))
    story.append(bullet_list([
        "<b>Step 1 — Proximity Check:</b> Calculate distance from UAV to each NFZ center. If distance < buffer radius "
        "(28m for Alpha, 32m for Bravo), activate reroute mode.",
        "<b>Step 2 — Tangent Vector:</b> Compute the perpendicular tangent vector to the line connecting the UAV "
        "to the NFZ center. This gives the direction to fly 'around' the NFZ.",
        "<b>Step 3 — Bypass Waypoint:</b> Place a temporary bypass waypoint at NFZ center + (normalized_offset × "
        "(radius + 14m)) in the tangent direction.",
        "<b>Step 4 — Latch & Track:</b> Once a bypass target is set, the UAV locks onto it (latches) and doesn't "
        "recalculate every tick (prevents oscillation). Only releases when within 3m of the bypass point.",
        "<b>Step 5 — Resume:</b> After clearing the bypass waypoint, the UAV resumes navigation toward its "
        "original assigned waypoint.",
    ]))
    story.append(p(
        "The reroute is visually shown on the dashboard as a <b>cyan dashed line</b> from the UAV to its "
        "bypass target, with a <b>pulsing cyan halo</b> around the rerouting drone and a <b>diamond marker</b> "
        "at the bypass waypoint location."
    ))
    
    # ═══════════════════════════════════════════════════════════════════════
    # CHAPTER 5: TWO-LAYER GUARDRAIL SAFETY SYSTEM
    # ═══════════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(ch("5. Two-Layer Guardrail Safety System"))
    story.append(hr())
    story.append(p(
        "The guardrail system acts as a <b>two-step verification mechanism</b> — every action proposed by "
        "the Agentic Reasoning engine must pass through both layers before execution. This ensures that "
        "even if the AI makes an incorrect decision, the safety layers will catch and correct it."
    ))
    
    story.append(sh("5.1 Layer 1 — Soft Policy Guardrail"))
    story.append(p(
        "Layer 1 is a rule-based policy engine that checks proposed actions against operational constraints:"
    ))
    story.append(bullet_list([
        "<b>Altitude Bounds:</b> UAV altitude must remain within 2m – 120m AGL (Above Ground Level).",
        "<b>Speed Limits:</b> Horizontal speed must not exceed 5 m/s.",
        "<b>Battery Minimums:</b> If battery < 15%, force RETURN_TO_HOME regardless of current task.",
        "<b>NFZ Proximity:</b> If UAV is within NFZ buffer zone, action must include rerouting.",
        "<b>Action Validation:</b> Proposed action must be in the allowed set for current fault/energy state.",
    ]))
    story.append(p(
        "Layer 1 can <b>modify or override</b> the proposed action. For example, if the reasoning engine says "
        "'NOMINAL_CONTINUE' but battery is at 12%, Layer 1 will override to 'RETURN_TO_HOME'. The output "
        "status is either <b>PASSED</b> (action approved) or <b>MODIFIED</b> (action was changed)."
    ))
    
    story.append(sh("5.2 Layer 2 — CBF-QP Hard Safety Filter"))
    story.append(p(
        "Layer 2 is the mathematically rigorous safety filter. It takes the Layer 1-approved velocity command "
        "and solves a Quadratic Programming (QP) optimization problem to find the closest safe velocity:"
    ))
    story.append(code(
        "minimize    ||u − u_nom||²\n"
        "subject to  A_cbf · u ≤ b_cbf\n\n"
        "Where u_nom is the desired velocity, and A_cbf/b_cbf encode\n"
        "all barrier constraints (ground, speed, swarm, NFZ)."
    ))
    story.append(p(
        "If the proposed velocity is already safe, the CBF returns it unchanged (status: <b>PASSED</b>). "
        "If it would violate any barrier, the CBF computes the <b>minimum modification</b> needed to make it "
        "safe (status: <b>INTERVENED</b>). The UAV's trajectory is always the closest feasible path to its "
        "desired course."
    ))
    
    # ═══════════════════════════════════════════════════════════════════════
    # CHAPTER 6: SWARM COORDINATION
    # ═══════════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(ch("6. Swarm Coordination &amp; Intelligence"))
    story.append(hr())
    
    story.append(sh("6.1 Leader Election Protocol"))
    story.append(p(
        "The swarm uses a <b>dynamic leader election algorithm</b> based on a composite fitness score. "
        "Every tick, each UAV's fitness is evaluated:"
    ))
    story.append(code(
        "fitness_score = battery_pct × 0.4\n"
        "             + (1.0 − risk_score) × 100 × 0.3\n"
        "             + link_quality_pct × 0.2\n"
        "             + (1.0 if no_fault else 0.0) × 10 × 0.1"
    ))
    story.append(p(
        "The UAV with the highest fitness score becomes the <b>Swarm Leader</b> (marked with 👑). "
        "Leadership transfers seamlessly — if the current leader develops a fault or low battery, "
        "a healthier peer automatically takes over. The leader coordinates task assignments and "
        "serves as the reference node for swarm formation."
    ))
    
    story.append(sh("6.2 Task Delegation &amp; Handoff Protocol"))
    story.append(p(
        "When a UAV becomes degraded (fault detected, low battery, high risk), the <b>Handoff Protocol</b> "
        "enables graceful task transfer to a healthy peer:"
    ))
    story.append(bullet_list([
        "<b>Step 1 — HandoffRequest:</b> Degraded UAV broadcasts its current task, remaining waypoints, "
        "and reason for handoff to all peers via SwarmBus.",
        "<b>Step 2 — HandoffBid:</b> Each healthy, available peer evaluates its own capacity (battery, "
        "distance to task, fault status) and submits a bid score.",
        "<b>Step 3 — HandoffAccept:</b> The UAV with the best bid (closest + most battery) wins the task. "
        "The original UAV transfers task context and authorization.",
        "<b>Step 4 — Beacon Mode:</b> The degraded UAV enters 'Beacon Active (Awaiting Retrieval)' state — "
        "it hovers in place, broadcasting its location for physical retrieval while the winning peer "
        "continues the mission.",
    ]))
    
    story.append(sh("6.3 Inter-Agent Collision Avoidance"))
    story.append(p(
        "Inter-agent collision avoidance is handled at two levels: the SwarmBus shares position data between "
        "all agents every tick, and the CBF-QP Layer 2 safety filter includes <b>swarm separation barriers</b> "
        "that maintain a minimum distance of <b>2.5 meters</b> between any two UAVs. If two UAVs are on a "
        "collision course, the CBF automatically adjusts their velocities to diverge."
    ))
    
    # ═══════════════════════════════════════════════════════════════════════
    # CHAPTER 7: ENERGY & BATTERY MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(ch("7. Energy &amp; Battery Management"))
    story.append(hr())
    story.append(p(
        "The Energy Monitor node continuously tracks each UAV's battery state and classifies it into "
        "four alert levels:"
    ))
    
    energy_data = [
        ["Alert Level", "Battery Range", "Color", "System Response"],
        ["NOMINAL", "> 30%", "Green", "Normal operations continue. No intervention."],
        ["WARNING", "15% – 30%", "Yellow", "Reduced speed recommended. Swarm leader notified. "
         "Mission continues with conservative flight profile."],
        ["CRITICAL", "5% – 15%", "Orange/Red", "Forced RETURN_TO_HOME. Mission is aborted. "
         "Task handoff initiated to healthy peers."],
        ["EMERGENCY", "< 5%", "Red (Flashing)", "Immediate EMERGENCY_LAND at current position. "
         "UAV enters beacon mode for retrieval."],
    ]
    story.append(make_table(energy_data, col_widths=[1.0*inch, 1.0*inch, 0.7*inch, 3.3*inch]))
    story.append(Paragraph("Table 8: Energy Alert Levels", styles['Caption']))
    
    story.append(p(
        "Battery drain is simulated realistically: base drain of ~0.02% per tick plus speed-dependent "
        "drain. Fault conditions accelerate drain. The system also detects <b>anomalous battery drain</b> "
        "patterns that could indicate a faulty battery or power system."
    ))
    
    # ═══════════════════════════════════════════════════════════════════════
    # CHAPTER 8: RISK ASSESSMENT & AGENTIC REASONING
    # ═══════════════════════════════════════════════════════════════════════
    story.append(ch("8. Risk Assessment &amp; Agentic Reasoning"))
    story.append(hr())
    story.append(p(
        "The Risk Assessment node computes a <b>composite risk score</b> (0.0 to 1.0) by combining "
        "multiple risk factors with weighted contributions:"
    ))
    story.append(bullet_list([
        "<b>Fault Detection Risk (40%):</b> Non-normal FDI classification adds 0.4 to risk score.",
        "<b>Battery Risk (25%):</b> Scales inversely with battery percentage. Below 20% adds significant risk.",
        "<b>Speed Anomaly (15%):</b> Excessive speed (> 4 m/s) or very low speed (< 0.5 m/s) adds risk.",
        "<b>Altitude Risk (10%):</b> Very low altitude (< 3m) or very high altitude (> 100m) adds risk.",
        "<b>NFZ Proximity (10%):</b> Being near a No-Fly Zone buffer boundary adds risk.",
    ]))
    story.append(p(
        "The Agentic Reasoning engine maps the composite risk score to <b>flight actions</b>:"
    ))
    
    risk_action_data = [
        ["Risk Tier", "Risk Score", "Typical Action"],
        ["LOW", "0.0 – 0.3", "NOMINAL_CONTINUE — Continue mission normally."],
        ["MODERATE", "0.3 – 0.5", "REDUCE_SPEED — Slow down, increase monitoring frequency."],
        ["HIGH", "0.5 – 0.7", "ALTITUDE_HOLD — Hold position, attempt recovery."],
        ["CRITICAL", "0.7 – 1.0", "RETURN_TO_HOME or EMERGENCY_LAND — Abort mission."],
    ]
    story.append(make_table(risk_action_data, col_widths=[1.0*inch, 1.0*inch, 4.0*inch]))
    story.append(Paragraph("Table 9: Risk-to-Action Mapping", styles['Caption']))
    
    # ═══════════════════════════════════════════════════════════════════════
    # CHAPTER 9: MISSION CONTROL DASHBOARD
    # ═══════════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(ch("9. Mission Control Web Dashboard"))
    story.append(hr())
    story.append(p(
        "The Mission Control Dashboard is a <b>single-page web application</b> served by a Tornado web server "
        "(<font face='Courier'>mission_control.py</font>). It provides real-time visualization and control "
        "of the entire swarm mission."
    ))
    
    story.append(sh("9.1 3D Three.js Visualization"))
    story.append(bullet_list([
        "Full 3D scene rendered with Three.js and OrbitControls for free rotation/zoom.",
        "Each UAV is shown as a colored sphere with a glowing halo indicating its status.",
        "Flight paths are drawn as colored lines connecting waypoints.",
        "No-Fly Zones are rendered as semi-transparent red cylinders with wireframe buffer zones.",
        "Ground grid provides spatial reference for the 200m × 200m operational area.",
        "UAV labels with ID and altitude float above each drone.",
    ]))
    
    story.append(sh("9.2 2D Radar Map with Zoom/Pan"))
    story.append(bullet_list([
        "Top-down 2D Canvas-based radar view for precise position monitoring.",
        "<b>Mouse wheel zoom</b> — zooms in/out centered on cursor position. Everything scales proportionally "
        "(UAV markers, NFZ circles, buffer zones, waypoints, labels, routes).",
        "<b>Click-and-drag pan</b> — moves the viewport to explore different areas of the operational zone.",
        "NFZ-Alpha and NFZ-Bravo shown as filled red circles with labeled names.",
        "Outer buffer zones shown as dashed orange rings with appropriate radii.",
        "Planned routes drawn as colored dashed lines connecting waypoints.",
        "Dynamic reroute vectors shown as cyan dashed lines when a UAV is actively bypassing an NFZ.",
        "UAV callsigns and altitude labels rendered next to each drone marker.",
    ]))
    
    story.append(sh("9.3 Telemetry Cards &amp; Live Status"))
    story.append(p(
        "Each UAV has a dedicated telemetry card showing:"
    ))
    story.append(bullet_list([
        "<b>Position:</b> Current (x, y, z) coordinates in meters.",
        "<b>Battery:</b> Percentage with color-coded indicator (green/yellow/red).",
        "<b>FDI Status:</b> Current fault classification from the ML model.",
        "<b>Risk Score:</b> Composite risk value with tier label (LOW/MODERATE/HIGH/CRITICAL).",
        "<b>Flight Action:</b> Current action being executed (NOMINAL_CONTINUE, REROUTING, RTL, etc.).",
        "<b>Guardrail Status:</b> L1 and L2 guardrail results (PASSED/MODIFIED/INTERVENED).",
        "<b>Swarm Role:</b> Current role (Swarm Leader 👑, Relay/Escort, Survey, Reserve).",
        "<b>Task Progress:</b> Waypoints completed / total waypoints.",
        "<b>Reroute Badge:</b> Shows '⟲ REROUTING' when UAV is actively bypassing an NFZ.",
    ]))
    
    story.append(sh("9.4 Fault Injection Console"))
    story.append(p(
        "The dashboard includes a command console for injecting faults in real-time during a mission. "
        "This allows testing the system's response to various failure scenarios:"
    ))
    story.append(bullet_list([
        "<b>Inject Fault:</b> <font face='Courier'>fault UAV-2 GPS</font> — Injects a GPS failure on UAV-2.",
        "<b>Clear Fault:</b> <font face='Courier'>clear UAV-2</font> — Clears injected fault on UAV-2.",
        "<b>Clear All:</b> <font face='Courier'>clear all</font> — Clears all injected faults.",
        "<b>Set Battery:</b> <font face='Courier'>batt UAV-2 14</font> — Sets UAV-2 battery to 14%.",
        "Available fault types: GPS, Compass, Accelerometer, Barometer, Gyroscope.",
    ]))
    
    # ═══════════════════════════════════════════════════════════════════════
    # CHAPTER 10: TERMINOLOGY & GLOSSARY
    # ═══════════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(ch("10. Terminology &amp; Glossary"))
    story.append(hr())
    
    story.append(sh("10.1 Flight Actions"))
    actions_data = [
        ["Action", "Description"],
        ["NOMINAL_CONTINUE", "Normal flight — UAV is following its planned route to the next waypoint. "
         "All systems healthy, no intervention needed."],
        ["REDUCE_SPEED", "Speed reduced due to moderate risk (sensor anomaly or approaching NFZ buffer). "
         "Mission continues at lower velocity."],
        ["ALTITUDE_HOLD", "UAV holds current altitude and position while the system diagnoses a fault "
         "or waits for conditions to improve."],
        ["RETURN_TO_HOME (RTL)", "Mission aborted. UAV navigates back to its launch/home position "
         "due to critical battery, persistent fault, or mission commander order."],
        ["EMERGENCY_LAND", "Immediate landing at current position. Used when battery is critically low "
         "(< 5%) or a catastrophic fault makes continued flight dangerous."],
        ["DELEGATE_TO_SWARM", "Task is being transferred to a peer UAV via the Handoff Protocol. "
         "This UAV will enter beacon mode after successful handoff."],
        ["REROUTING", "UAV is actively navigating around a No-Fly Zone using a computed bypass "
         "waypoint. Shown as cyan dashed line on the radar."],
        ["Beacon Active\n(Awaiting Retrieval)", "UAV has completed its handoff, has very low battery, or has "
         "landed. It hovers/sits in place broadcasting its GPS location for physical recovery by ground crew."],
    ]
    story.append(make_table(actions_data, col_widths=[1.5*inch, 4.5*inch]))
    story.append(Paragraph("Table 10: Flight Action Definitions", styles['Caption']))
    
    story.append(sh("10.2 FDI Fault Labels"))
    fdi_labels_data = [
        ["Label", "Description"],
        ["Normal (No Failure)", "All sensors operating within expected parameters. No fault detected."],
        ["GPS Failure", "GPS receiver malfunction — position/velocity data unreliable. Could be jamming, "
         "signal loss, or hardware failure."],
        ["Compass Failure", "Magnetometer fault — heading data corrupted. Can cause navigation drift. "
         "Often caused by electromagnetic interference."],
        ["Accelerometer Failure", "IMU accelerometer fault — acceleration measurements unreliable. "
         "Affects attitude estimation and velocity integration."],
        ["Barometer Failure", "Pressure sensor fault — altitude readings inaccurate. Critical for "
         "altitude hold and terrain following."],
        ["Gyroscope Failure", "IMU gyroscope fault — angular velocity measurements corrupted. "
         "Affects attitude stabilization and rotation control."],
    ]
    story.append(make_table(fdi_labels_data, col_widths=[1.5*inch, 4.5*inch]))
    story.append(Paragraph("Table 11: FDI Fault Classification Labels", styles['Caption']))
    
    story.append(sh("10.3 Energy Alert Levels"))
    story.append(p(
        "See Table 8 in Chapter 7 for the complete energy alert level definitions "
        "(NOMINAL, WARNING, CRITICAL, EMERGENCY)."
    ))
    
    story.append(sh("10.4 Guardrail Status Codes"))
    guard_data = [
        ["Status", "Layer", "Meaning"],
        ["PASSED", "L1 or L2", "The proposed action/velocity passed safety checks without modification."],
        ["MODIFIED", "L1", "The soft policy guardrail changed the proposed action to meet operational rules "
         "(e.g., forced RTL due to low battery)."],
        ["INTERVENED", "L2 (CBF)", "The CBF-QP safety filter modified the velocity vector to prevent a barrier "
         "violation (collision, NFZ entry, ground impact, speed limit)."],
    ]
    story.append(make_table(guard_data, col_widths=[1.0*inch, 0.8*inch, 4.2*inch]))
    story.append(Paragraph("Table 12: Guardrail Status Codes", styles['Caption']))
    
    story.append(sh("10.5 Swarm Roles"))
    roles_data = [
        ["Role", "Description"],
        ["👑 Swarm Leader", "UAV with highest fitness score. Coordinates task assignments, serves as "
         "reference node for formation, and manages handoff arbitration."],
        ["Relay / Escort", "Communication relay node. Extends mesh network range and escorts other UAVs "
         "through contested or complex terrain."],
        ["Survey Bravo", "Dedicated survey role for Sector Bravo. Performs reconnaissance and data "
         "collection along its assigned route."],
        ["Swarm Reserve", "Standby UAV available for emergency task pickup. Performs perimeter patrol "
         "while ready to accept handoff tasks from degraded peers."],
    ]
    story.append(make_table(roles_data, col_widths=[1.3*inch, 4.7*inch]))
    story.append(Paragraph("Table 13: Swarm Role Definitions", styles['Caption']))
    
    # ═══════════════════════════════════════════════════════════════════════
    # CHAPTER 11: TECHNOLOGY STACK
    # ═══════════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(ch("11. Technology Stack"))
    story.append(hr())
    
    tech_data = [
        ["Technology", "Purpose", "Details"],
        ["Python 3.x", "Core Language", "All backend simulation, ML inference, and agent logic."],
        ["LangGraph", "Agentic AI Framework", "State graph pipeline for multi-node agent processing. "
         "Each UAV runs an independent StateGraph."],
        ["NumPy", "Numerical Computing", "Vector math for position updates, distance calculations, "
         "barrier function evaluation."],
        ["SciPy", "Optimization", "QP solver for Control Barrier Function safety filtering."],
        ["scikit-learn", "Machine Learning", "ExtraTrees classifier for FDI. Model trained offline, "
         "loaded via joblib for real-time inference."],
        ["joblib", "Model Serialization", "Loads pre-trained FDI model and feature names from .joblib files."],
        ["Tornado", "Web Server", "Asynchronous HTTP/WebSocket server for Mission Control dashboard. "
         "Handles REST API and real-time telemetry streaming."],
        ["Three.js", "3D Visualization", "WebGL-based 3D scene rendering in the browser. Shows UAVs, "
         "flight paths, NFZ cylinders, and ground grid."],
        ["HTML5 Canvas", "2D Radar", "Canvas-based 2D top-down radar with zoom/pan controls."],
        ["JavaScript (ES6)", "Frontend", "Dashboard UI logic, WebSocket client, telemetry rendering."],
        ["ROS 2 (Optional)", "Robot Framework", "Publishes UAV state to ROS 2 topics for integration "
         "with real PX4 autopilot and sensor hardware."],
        ["MAVLink (Optional)", "Protocol", "Standard UAV communication protocol. Enables integration "
         "with QGroundControl (QGC) for monitoring."],
        ["asyncio", "Async Runtime", "Python async/await for concurrent swarm tick execution."],
        ["CSS3", "Styling", "Dashboard styling with dark theme, responsive layout, and animations."],
    ]
    story.append(make_table(tech_data, col_widths=[1.2*inch, 1.1*inch, 3.7*inch]))
    story.append(Paragraph("Table 14: Technology Stack", styles['Caption']))
    
    # ═══════════════════════════════════════════════════════════════════════
    # CHAPTER 12: FILE STRUCTURE
    # ═══════════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(ch("12. File Structure &amp; Codebase"))
    story.append(hr())
    
    files_data = [
        ["File / Directory", "Size", "Description"],
        ["step80_multiagent_swarm.py", "~2500 lines", 
         "Core simulation engine. Contains SwarmBus, SwarmAgentState, all 10 pipeline nodes, "
         "build_uav_graph(), run_swarm(), terminal dashboard, and JSON telemetry export."],
        ["step76_control_barrier_function.py", "~220 lines", 
         "CBF-QP safety filter module. Implements ground, speed, swarm separation, "
         "and NFZ geofence barriers with QP optimization."],
        ["step70_build_and_train_clean.py", "—", 
         "Training script for the ExtraTrees FDI classifier. Processes PX4 flight "
         "recordings into windowed features and trains the 6-class model."],
        ["step70_clean_fdi_model.joblib", "~2 MB", 
         "Serialized pre-trained ExtraTrees classifier model."],
        ["step70_feature_names.joblib", "< 1 KB", 
         "List of feature names expected by the FDI model."],
        ["mission_control.py", "~300 lines", 
         "Tornado web server. Serves dashboard HTML, REST API endpoints "
         "(/api/telemetry, /api/start, /api/stop, /api/command), and WebSocket."],
        ["web_gui/index.html", "~1750 lines", 
         "Single-page dashboard: 3D Three.js scene, 2D Canvas radar, "
         "telemetry cards, fault injection console, zoom/pan controls."],
        ["run_mission_control.sh", "~43 lines", 
         "One-click launcher script. Kills stale processes, starts mission "
         "control server, opens browser."],
        ["master_system_report.json", "Dynamic", 
         "Live telemetry export (updated every tick). Contains all agent "
         "states, swarm status, NFZ data, and event log."],
        ["step71_live_px4_fdi.py", "—", 
         "Live PX4 SITL FDI integration script for real-time fault detection."],
        ["step72_generate_evaluation_report.py", "—", 
         "Generates evaluation plots and confusion matrices for the FDI model."],
        ["step73_autonomous_safety_governor.py", "—", 
         "Standalone autonomous safety governor prototype."],
        ["step75_agentic_safety_governor.py", "—", 
         "Agentic AI safety governor with LangGraph integration."],
        ["step_auto_*.csv / step*.csv", "Multiple", 
         "PX4 flight recording datasets for normal and fault-injected flights. "
         "Used for FDI model training and evaluation."],
        ["fdi_env/", "Directory", 
         "Python virtual environment with all project dependencies installed."],
    ]
    story.append(make_table(files_data, col_widths=[2.0*inch, 0.8*inch, 3.2*inch]))
    story.append(Paragraph("Table 15: Project File Structure", styles['Caption']))
    
    # ═══════════════════════════════════════════════════════════════════════
    # CHAPTER 13: HOW TO RUN
    # ═══════════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(ch("13. How to Run the System"))
    story.append(hr())
    
    story.append(sh("13.1 Prerequisites"))
    story.append(bullet_list([
        "<b>Python 3.8+</b> with pip.",
        "Virtual environment at <font face='Courier'>fdi_env/</font> with dependencies: "
        "langgraph, numpy, scipy, scikit-learn, joblib, tornado.",
        "A modern web browser (Chrome, Firefox, Edge) for the dashboard.",
        "Optional: ROS 2 Humble/Iron for live hardware integration.",
    ]))
    
    story.append(sh("13.2 Quick Start (One Command)"))
    story.append(p("Navigate to the project directory and run:"))
    story.append(code("cd /home/samiksha/uav_fdi_data\n./run_mission_control.sh"))
    story.append(p(
        "This will: (1) kill any stale processes, (2) start the Tornado web server on port 8080 "
        "(falls back to 8081 if 8080 is occupied), and (3) open the Mission Control Dashboard in your browser."
    ))
    
    story.append(sh("13.3 Manual Start"))
    story.append(p("If you prefer to start components individually:"))
    story.append(code(
        "# Step 1: Activate the virtual environment\n"
        "source fdi_env/bin/activate\n\n"
        "# Step 2: Start Mission Control web server\n"
        "python mission_control.py &\n\n"
        "# Step 3: Open browser to http://localhost:8080\n\n"
        "# Step 4: Click 'Launch Mission' in the dashboard"
    ))
    
    story.append(sh("13.4 Terminal-Only Mode (No Dashboard)"))
    story.append(p(
        "You can also run the swarm simulation directly in the terminal with the ANSI dashboard:"
    ))
    story.append(code(
        "source fdi_env/bin/activate\n"
        "python step80_multiagent_swarm.py"
    ))
    story.append(p(
        "This launches the rich terminal dashboard showing side-by-side agent status cards, "
        "swarm overview, and inter-agent communication log with color-coded outputs."
    ))
    
    story.append(sh("13.5 Fault Injection During Flight"))
    story.append(p(
        "While the mission is running, use the <b>Fault Injection Console</b> in the dashboard or "
        "the terminal to test system responses:"
    ))
    story.append(code(
        "# In terminal (stdin commands):\n"
        "fault UAV-2 GPS      # Inject GPS failure on UAV-2\n"
        "fault UAV-3 Compass  # Inject Compass failure on UAV-3\n"
        "batt UAV-1 12        # Set UAV-1 battery to 12%\n"
        "clear UAV-2          # Clear fault on UAV-2\n"
        "clear all            # Clear all injected faults"
    ))
    
    # ═══════════════════════════════════════════════════════════════════════
    # CHAPTER 14: FUTURE ENHANCEMENTS
    # ═══════════════════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(ch("14. Future Enhancements"))
    story.append(hr())
    story.append(bullet_list([
        "<b>Real Hardware Integration:</b> Deploy on actual PX4-powered drones via ROS 2 Micro-XRCE-DDS bridge. "
        "The agent pipeline is designed to swap in real sensor data seamlessly.",
        "<b>Dynamic NFZ Updates:</b> Support for receiving real-time NOTAM (Notice to Airmen) updates "
        "and dynamically adding/removing No-Fly Zones during flight.",
        "<b>Multi-Fault Detection:</b> Extend the FDI model to detect simultaneous faults on multiple "
        "sensors (currently detects single-fault scenarios).",
        "<b>Path Planning Optimization:</b> Integrate A* or RRT* path planning for optimal NFZ bypass "
        "routes instead of tangent-based bypass waypoints.",
        "<b>Weather Integration:</b> Factor in real-time wind and weather data for more realistic "
        "battery drain simulation and flight planning.",
        "<b>Swarm Formation Control:</b> Implement formation flying (V-formation, line abreast) "
        "with distributed consensus algorithms.",
        "<b>Secure Communication:</b> Add encryption and authentication to SwarmBus messages "
        "for cybersecurity in real-world deployments.",
        "<b>3D NFZ Volumes:</b> Extend NFZ from 2D circles to full 3D volumes with altitude ceilings "
        "and irregular shapes (polygon geofences).",
        "<b>Mission Replay:</b> Add mission replay capability using recorded telemetry CSV data "
        "for post-mission analysis and training.",
        "<b>LLM-Enhanced Reasoning:</b> Integrate large language model reasoning in the Agentic Reasoning "
        "node for natural language mission reports and adaptive decision-making.",
    ]))
    
    # ═══════════════════════════════════════════════════════════════════════
    # CHAPTER 15: CONCLUSION
    # ═══════════════════════════════════════════════════════════════════════
    story.append(ch("15. Conclusion"))
    story.append(hr())
    story.append(p(
        "The UAV Swarm Fault Detection &amp; Isolation system demonstrates a comprehensive, "
        "production-ready architecture for autonomous multi-agent drone operations. By combining "
        "<b>LangGraph-based Agentic AI</b> for intelligent decision-making, <b>Control Barrier Functions</b> "
        "for mathematically guaranteed safety, and <b>Machine Learning</b> for real-time fault detection, "
        "the system achieves a level of autonomy and safety assurance that meets the demands of "
        "real-world UAV swarm deployments."
    ))
    story.append(p(
        "Key achievements of this project include:"
    ))
    story.append(bullet_list([
        "A <b>10-node LangGraph pipeline</b> executing independently on each UAV agent every tick.",
        "<b>6-class sensor fault detection</b> using ExtraTrees ML with sub-millisecond inference.",
        "<b>Three-tier No-Fly Zone avoidance</b> (predictive rerouting, hard wall clamp, CBF barrier).",
        "<b>Two-layer guardrail safety system</b> providing both policy enforcement and mathematical safety guarantees.",
        "<b>Dynamic swarm coordination</b> with leader election, task handoff, and peer collision avoidance.",
        "<b>Real-time Mission Control dashboard</b> with 3D visualization, 2D radar, and live telemetry.",
        "<b>Horizontal scaling</b> — dynamically add or remove UAVs from the swarm at runtime.",
        "<b>ROS 2 and MAVLink integration readiness</b> for deployment on real PX4-powered drones.",
    ]))
    story.append(p(
        "The system is designed to be extensible, with clear separation between the intelligence layer "
        "(LangGraph agents), safety layer (CBF guardrails), communication layer (SwarmBus), and "
        "visualization layer (web dashboard). This modular architecture makes it straightforward to "
        "add new capabilities, integrate with real hardware, or adapt the system for different "
        "autonomous vehicle domains beyond UAVs."
    ))
    
    story.append(spacer(30))
    story.append(HRFlowable(width="60%", thickness=2, color=PRIMARY, spaceBefore=20, spaceAfter=10))
    story.append(Paragraph(
        "— End of Documentation —",
        ParagraphStyle('EndMark', parent=styles['Normal'],
                       fontSize=12, alignment=TA_CENTER, textColor=darkgray,
                       fontName='Helvetica-Oblique')
    ))
    story.append(Paragraph(
        f"Generated on {datetime.datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        ParagraphStyle('DateMark', parent=styles['Normal'],
                       fontSize=9, alignment=TA_CENTER, textColor=gray,
                       fontName='Helvetica')
    ))
    
    # ═══════════════════════════════════════════════════════════════════════
    # BUILD PDF
    # ═══════════════════════════════════════════════════════════════════════
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(f"\n✅ Documentation generated successfully!")
    print(f"📄 Output: {OUTPUT_PDF}")
    print(f"📐 Pages: {doc.page}")
    return OUTPUT_PDF


if __name__ == "__main__":
    build_doc()
