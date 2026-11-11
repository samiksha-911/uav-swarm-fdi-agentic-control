#!/usr/bin/env python3
"""
generate_copyright_submission_pdf.py
====================================
Generates a comprehensive, formal Software Copyright Filing Dossier in PDF format.
Includes:
1. Official Copyright Filing Declarations & Detailed Technical Statements of Novelty.
2. Form-ready descriptions for Copyright Office submission (India Form XIV / US Form TX).
3. Detailed algorithmic analysis of the 4 core proprietary innovations.
4. Complete, verbatim source code deposits for the exact 4 referenced files:
   - step80_multiagent_swarm.py
   - step76_control_barrier_function.py
   - mission_control.py
   - web_gui/index.html
"""

import os
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, white, gray, darkgray, lightgrey
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, ListFlowable, ListItem, Preformatted
)

BASE_DIR = "/home/samiksha/uav_fdi_data"
OUTPUT_PDF = os.path.join(BASE_DIR, "UAV_Swarm_FDI_Copyright_Filing_Dossier.pdf")

PAGE_W, PAGE_H = A4
MARGIN = 0.5 * inch

PRIMARY = HexColor("#1a237e")      # Deep Indigo
SECONDARY = HexColor("#0d47a1")    # Blue
ACCENT = HexColor("#00838f")       # Teal/Cyan
TABLE_HEADER = HexColor("#283593")
TABLE_STRIPE = HexColor("#f8fafc")
BORDER_COLOR = HexColor("#cbd5e1")

styles = getSampleStyleSheet()

styles.add(ParagraphStyle(
    'DocTitle', parent=styles['Title'],
    fontSize=24, leading=30, textColor=PRIMARY,
    spaceAfter=6, alignment=TA_CENTER, fontName='Helvetica-Bold'
))
styles.add(ParagraphStyle(
    'DocSubtitle', parent=styles['Normal'],
    fontSize=13, leading=18, textColor=SECONDARY,
    spaceAfter=15, alignment=TA_CENTER, fontName='Helvetica'
))
styles.add(ParagraphStyle(
    'SecHeading', parent=styles['Heading1'],
    fontSize=18, leading=22, textColor=PRIMARY,
    spaceBefore=14, spaceAfter=8, fontName='Helvetica-Bold'
))
styles.add(ParagraphStyle(
    'SubHeading', parent=styles['Heading2'],
    fontSize=13, leading=17, textColor=SECONDARY,
    spaceBefore=10, spaceAfter=4, fontName='Helvetica-Bold'
))
styles.add(ParagraphStyle(
    'SubSubHeading', parent=styles['Heading3'],
    fontSize=11, leading=15, textColor=HexColor("#334155"),
    spaceBefore=8, spaceAfter=3, fontName='Helvetica-Bold'
))
styles.add(ParagraphStyle(
    'BodyJustify', parent=styles['Normal'],
    fontSize=9.5, leading=14, textColor=black,
    spaceBefore=2, spaceAfter=5, alignment=TA_JUSTIFY, fontName='Helvetica'
))
styles.add(ParagraphStyle(
    'BulletText', parent=styles['Normal'],
    fontSize=9.5, leading=13.5, textColor=black,
    leftIndent=18, bulletIndent=8, spaceBefore=1, spaceAfter=2, fontName='Helvetica'
))
styles.add(ParagraphStyle(
    'FormalBox', parent=styles['Normal'],
    fontSize=9, leading=13.5, textColor=HexColor("#0f172a"),
    fontName='Helvetica-Oblique',
    leftIndent=12, borderWidth=1, borderColor=ACCENT,
    borderPadding=7, backColor=HexColor("#f0fdfa"),
    spaceBefore=6, spaceAfter=8
))
styles.add(ParagraphStyle(
    'SourceCodeStyle', parent=styles['Normal'],
    fontSize=6.0, leading=7.2, textColor=HexColor("#1e293b"),
    fontName='Courier', spaceBefore=0, spaceAfter=0
))

def hr():
    return HRFlowable(width="100%", thickness=1, color=lightgrey, spaceBefore=6, spaceAfter=6)

def make_table(data, colWidths=None, col_widths=None):
    widths = colWidths if colWidths is not None else col_widths
    style_cmds = [
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('LEADING', (0, 0), (-1, -1), 12),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style_cmds.append(('BACKGROUND', (0, i), (-1, i), TABLE_STRIPE))
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle(style_cmds))
    return t

def header_footer(canvas, doc):
    canvas.saveState()
    page_num = doc.page
    if page_num > 1:
        canvas.setStrokeColor(ACCENT)
        canvas.setLineWidth(1.2)
        canvas.line(MARGIN, PAGE_H - 0.35 * inch, PAGE_W - MARGIN, PAGE_H - 0.35 * inch)
        canvas.setFont('Helvetica-Bold', 7.5)
        canvas.setFillColor(HexColor("#334155"))
        canvas.drawString(MARGIN, PAGE_H - 0.30 * inch, "OFFICIAL SOFTWARE COPYRIGHT FILING DOSSIER")
        canvas.setFont('Helvetica', 7.5)
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 0.30 * inch, "UAV Swarm FDI & Dual-Layer Guardrail System")
        
        canvas.setStrokeColor(lightgrey)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN, 0.35 * inch, PAGE_W - MARGIN, 0.35 * inch)
        canvas.setFont('Helvetica', 7.5)
        canvas.setFillColor(gray)
        canvas.drawString(MARGIN, 0.22 * inch, "CONFIDENTIAL & PROPRIETARY — STATUTORY COPYRIGHT DEPOSIT")
        canvas.drawRightString(PAGE_W - MARGIN, 0.22 * inch, f"Deposit Page {page_num}")
    canvas.restoreState()

def wrap_and_chunk_code(file_path, chunk_lines=160):
    """Reads source file, normalizes tabs, and yields chunks of formatted code."""
    chunks = []
    if not os.path.exists(file_path):
        return ["# Error: File not found: " + file_path]
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    
    current_chunk = []
    for idx, line in enumerate(lines, start=1):
        clean_line = line.replace('\t', '    ').rstrip('\r\n')
        # Wrap extremely long lines so they do not clip in the PDF
        if len(clean_line) > 115:
            part1 = clean_line[:115]
            part2 = "    ↪ " + clean_line[115:]
            current_chunk.append(f"{idx:4d} | {part1}")
            current_chunk.append(f"     | {part2}")
        else:
            current_chunk.append(f"{idx:4d} | {clean_line}")
        
        if len(current_chunk) >= chunk_lines:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    return chunks

def build_pdf():
    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=0.45 * inch, bottomMargin=0.45 * inch,
        title="UAV Swarm FDI Software Copyright Filing Dossier",
        author="UAV FDI Research Team"
    )

    story = []

    # ─────────────────────────────────────────────────────────────────────────
    # COVER / TITLE SECTION
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.4 * inch))
    story.append(HRFlowable(width="100%", thickness=3, color=PRIMARY, spaceBefore=0, spaceAfter=15))
    story.append(Paragraph("STATUTORY SOFTWARE COPYRIGHT FILING DOSSIER", styles['DocTitle']))
    story.append(Paragraph("Autonomous Multi-Agent UAV Swarm Fault Detection, Isolation (FDI),<br/>and Two-Layer Guardrail Control Architecture", styles['DocSubtitle']))
    story.append(HRFlowable(width="70%", thickness=1.5, color=ACCENT, spaceBefore=5, spaceAfter=15))

    today_str = datetime.datetime.now().strftime("%B %d, %Y")
    meta_table_data = [
        ["Document Type", "Official Statutory Software Copyright Deposit & Technical Description"],
        ["Subject Matter", "Computer Software Work / Literary Work under Copyright Act"],
        ["Project Title", "Multi-Agent UAV Swarm FDI with Agentic AI & Formal CBF Guardrails"],
        ["Filing Date", today_str],
        ["Target Jurisdiction", "National Copyright Offices (Form XIV / Form TX compliant)"],
        ["Total Deposit Source Files", "4 Registered Source Modules (Complete Verbatim Code)"],
        ["Core Languages / Stack", "Python 3, LangGraph StateGraph, NumPy, SciPy, Tornado, Three.js, Canvas"],
    ]
    meta_table = Table(meta_table_data, colWidths=[2.2 * inch, 4.6 * inch])
    meta_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('TEXTCOLOR', (0, 0), (0, -1), SECONDARY),
        ('TEXTCOLOR', (1, 0), (1, -1), black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('BACKGROUND', (0, 0), (-1, -1), HexColor("#f1f5f9")),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 15))

    story.append(Paragraph("<b>CONFIDENTIALITY NOTICE & COPYRIGHT STATEMENT:</b><br/>"
                           "This document contains proprietary source code, algorithmic formulations, and design specifications. "
                           "Copyright © 2026. All rights reserved. Unlicensed reproduction, distribution, or decompilation of this "
                           "software without explicit written authorization is strictly prohibited by law.", styles['FormalBox']))

    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1: OFFICIAL APPLICATION FORM STATEMENTS (FORM XIV / FORM TX)
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("SECTION 1: STATUTORY DESCRIPTIONS FOR COPYRIGHT FILING", styles['SecHeading']))
    story.append(hr())
    story.append(Paragraph(
        "When filing an application for registration of copyright of computer software with government intellectual property offices, "
        "precise, formal technical descriptions are mandatory. The following subsections contain the ready-to-copy fields required "
        "for statutory submission.", styles['BodyJustify']
    ))

    story.append(Paragraph("1.1 Registration Package A: Core Swarm Autonomy & Safety Engine", styles['SubHeading']))
    reg_a_data = [
        ["Field Name", "Statutory Entry Content"],
        ["Title of Work", "Autonomous Multi-Agent UAV Swarm Fault Detection, Isolation, and Dual-Layer Safety Guardrail Control Engine (Version 1.0)"],
        ["Class of Work", "Computer Software / Literary Work"],
        ["Author(s)", "UAV FDI Research Team / Development Lead"],
        ["Date of Creation", "October 2026"],
        ["Primary Files", "step80_multiagent_swarm.py, step76_control_barrier_function.py"],
        ["Brief Statement of Functionality", 
         "Executes a decentralized 10-node state-graph agentic processing cycle on multiple independent unmanned aerial vehicles. "
         "Features real-time sensor fault classification across six fault types (Normal, GPS, Compass, Accel, Baro, Gyro), composite dynamic "
         "risk evaluation, two-layer runtime verification (L1 Semantic Policy + L2 Control Barrier Function QP solver), fault-disqualified leader "
         "re-election, dynamic latched tangent boundary rerouting around No-Fly Zones, and peer-to-peer task handoff with in-flight route splicing."]
    ]
    story.append(make_table(reg_a_data, colWidths=[1.8 * inch, 5.0 * inch]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("1.2 Registration Package B: Unified Ground Control & Mission Dashboard", styles['SubHeading']))
    reg_b_data = [
        ["Field Name", "Statutory Entry Content"],
        ["Title of Work", "Unified Web Ground Control and 3D/2D Multi-Agent Swarm Mission Command Interface (Mission Control v1.0)"],
        ["Class of Work", "Computer Software / Graphic & Literary Work"],
        ["Author(s)", "UAV FDI Research Team / Development Lead"],
        ["Date of Creation", "October 2026"],
        ["Primary Files", "mission_control.py, web_gui/index.html"],
        ["Brief Statement of Functionality",
         "A full-stack asynchronous ground control interface combining a Python Tornado REST/WebSocket server with a single-page frontend. "
         "Provides synchronized 3D WebGL spatial rendering via Three.js alongside a 2D canvas radar map featuring scale-preserving zoom and pan "
         "mechanics. Displays real-time telemetry cards, color-coded FDI alerts, live dynamic bypass corridors, and interactive fault injection controls."]
    ]
    story.append(make_table(reg_b_data, colWidths=[1.8 * inch, 5.0 * inch]))

    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2: THE 4 PROPRIETARY ALGORITHMS EXPLAINED
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("SECTION 2: PROPRIETARY ALGORITHMIC INVENTIONS", styles['SecHeading']))
    story.append(hr())
    story.append(Paragraph(
        "This section details the exact algorithmic logic and technical formulations invented and embodied in the registered source code. "
        "These four algorithms constitute the primary intellectual contribution of this project.", styles['BodyJustify']
    ))

    # Algo 1
    story.append(Paragraph("2.1 Algorithm 1: Two-Layer Guardrail Architecture (Semantic L1 + Kinematic L2 CBF-QP)", styles['SubHeading']))
    story.append(Paragraph(
        "<b>Problem Solved:</b> Traditional autonomous agents either rely purely on discrete AI rules (which cannot guarantee physical safety against collisions) "
        "or purely on low-level control theory (which cannot reason about mission goals, battery life, or sensor degradation).<br/>"
        "<b>Our Innovation:</b> A sequential, two-step verification mechanism where every flight decision passes through two independent filters:", styles['BodyJustify']
    ))
    story.append(ListFlowable([
        ListItem(Paragraph("<b>Layer 1 (Semantic Policy Guardrail):</b> Validates regulatory constraints including absolute altitude ceilings (2m to 120m), "
                           "maximum allowed nominal velocity (5 m/s), low-battery triggers (<15%), and enforces mission overrides such as Return-To-Home (RTL).", styles['BulletText'])),
        ListItem(Paragraph("<b>Layer 2 (Control Barrier Function QP Filter):</b> Takes the Layer 1-approved velocity command $\\mathbf{u}_{\\text{nom}} = [v_x, v_y, v_z]$ "
                           "and projects it onto a forward-invariant safe set $\\mathcal{C}$ by solving a Quadratic Program in real time:<br/>"
                           "$$\\min_{\\mathbf{u}} \\frac{1}{2} \\|\\mathbf{u} - \\mathbf{u}_{\\text{nom}}\\|^2 \\quad \\text{s.t.} \\quad A_{\\text{cbf}} \\mathbf{u} \\le \\mathbf{b}_{\\text{cbf}}$$<br/>"
                           "Simultaneously enforcing ground clearance ($h_{\\text{alt}}$), kinetic speed envelope ($h_{\\text{speed}}$), minimum swarm separation ($h_{\\text{swarm}} \\ge 2.5\\text{m}$), "
                           "and repulsive geofence barriers ($h_{\\text{nfz}}$).", styles['BulletText'])),
    ], bulletType='bullet', start='•', leftIndent=12))

    # Algo 2
    story.append(Paragraph("2.2 Algorithm 2: Dynamic Latching Tangent Bypass for Proactive NFZ Avoidance", styles['SubHeading']))
    story.append(Paragraph(
        "<b>Problem Solved:</b> When drones approach an obstacle boundary, standard potential field or barrier algorithms often experience 'limit-cycle chattering' "
        "(rapidly oscillating back and forth along the boundary surface) or fail to detect intersections between waypoints.<br/>"
        "<b>Our Innovation:</b> A predictive line-segment intersection detector coupled with a latched orthogonal tangent bypass generator:", styles['BodyJustify']
    ))
    story.append(ListFlowable([
        ListItem(Paragraph("<b>Line-Segment Projection:</b> Projects vector segment from current UAV position to the assigned waypoint against all No-Fly Zone circles. "
                           "Detects obstruction if the minimum distance to the center $c$ is less than $(R_{\\text{core}} + 10\\text{m})$.", styles['BulletText'])),
        ListItem(Paragraph("<b>Orthogonal Tangent Calculation:</b> Determines the normal unit vector $\\hat{n}$ outward from the NFZ center, derives the orthogonal tangent $\\hat{t}$, "
                           "and picks the tangent pointing toward the destination waypoint ($(\\hat{t} \\cdot \\mathbf{d}_{\\text{target}}) > 0$).", styles['BulletText'])),
        ListItem(Paragraph("<b>State Latching:</b> Places a bypass waypoint at $c + \\hat{n}(R + 14\\text{m}) + 10\\hat{t}$ and <b>latches</b> this coordinate. "
                           "The UAV tracks the latched point until it comes within a 3.0-meter radius, eliminating jitter and oscillation before smoothly resuming the primary mission path.", styles['BulletText'])),
    ], bulletType='bullet', start='•', leftIndent=12))

    # Algo 3
    story.append(Paragraph("2.3 Algorithm 3: Fault-Disqualified Decentralized Leader Election Protocol", styles['SubHeading']))
    story.append(Paragraph(
        "<b>Problem Solved:</b> In multi-agent swarms, if the leader experiences an in-flight sensor failure (such as GPS drift or compass degradation), "
        "it will propagate erroneous navigational cues to all follower drones, risking whole-swarm catastrophe.<br/>"
        "<b>Our Innovation:</b> A decentralized consensus election protocol in <font face='Courier'>SwarmBus</font> with strict health pre-filtering:", styles['BodyJustify']
    ))
    story.append(ListFlowable([
        ListItem(Paragraph("<b>Strict Health Exclusion:</b> Before computing fitness scores, any UAV with <i>any</i> detected sensor anomaly "
                           "(<font face='Courier'>fault_label != 'No Failure'</font>) or critical battery is immediately and unconditionally excluded from candidacy.", styles['BulletText'])),
        ListItem(Paragraph("<b>Composite Fitness Evaluation:</b> Eligible healthy candidates are scored according to:<br/>"
                           "$$\\text{Fitness} = 0.40 \\cdot \\text{Batt} + 0.30 \\cdot (1.0 - \\text{Risk}) + 0.20 \\cdot \\text{GPS}_{\\text{valid}} + 0.10 \\cdot \\text{Sensor}_{\\text{health}}$$<br/>"
                           "The highest-scoring healthy agent is democratically elected as Swarm Leader (👑), guaranteeing that a degraded drone can never command peers.", styles['BulletText'])),
    ], bulletType='bullet', start='•', leftIndent=12))

    # Algo 4
    story.append(Paragraph("2.4 Algorithm 4: In-Flight Trajectory Splicing & Dynamic Task Handoff Protocol", styles['SubHeading']))
    story.append(Paragraph(
        "<b>Problem Solved:</b> When an individual drone suffers a hardware failure, conventional systems simply abort its mission, leaving search grids or inspection routes incomplete.<br/>"
        "<b>Our Innovation:</b> An autonomous, seamless peer-to-peer workload handoff mechanism:", styles['BodyJustify']
    ))
    story.append(ListFlowable([
        ListItem(Paragraph("<b>Dynamic Handoff Request:</b> Upon detecting critical battery or sensor failure, the degraded UAV extracts all unvisited remaining route coordinates "
                           "(<font face='Courier'>remaining_route = route_coords[current_wp_idx:]</font>) and broadcasts a handover request across the <font face='Courier'>SwarmBus</font>.", styles['BulletText'])),
        ListItem(Paragraph("<b>Optimal Peer Bidding:</b> Healthy peers calculate a fitness bid considering battery capacity and Euclidean proximity to the abandoned task.", styles['BulletText'])),
        ListItem(Paragraph("<b>In-Flight Trajectory Splicing:</b> The winning peer dynamically splices the transferred waypoints directly into its active flight route immediately after its current target "
                           "(<font face='Courier'>peer.route_coords.insert(insert_pos, wp)</font>), expanding its task queue while the degraded UAV safely descends into Beacon Retrieval Mode.", styles['BulletText'])),
    ], bulletType='bullet', start='•', leftIndent=12))

    story.append(PageBreak())

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3: VERBATIM SOURCE CODE DEPOSIT INDEX
    # ─────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("SECTION 3: VERBATIM SOURCE CODE DEPOSIT INDEX", styles['SecHeading']))
    story.append(hr())
    story.append(Paragraph(
        "The following sections contain the complete, verbatim source code of the four primary project files designated for copyright deposit. "
        "Line numbers, formatting, and implementation logic are preserved for forensic intellectual property verification.", styles['BodyJustify']
    ))

    source_index_data = [
        ["Deposit Part", "Target File Path", "Language", "Lines", "Key Modules / Classes Embodied"],
        ["Part 3.1", "step80_multiagent_swarm.py", "Python 3", "2,539", "SwarmBus, SwarmAgentState, 10-Node Pipeline, Handoff Protocol, NFZ Bypass"],
        ["Part 3.2", "step76_control_barrier_function.py", "Python 3", "218", "UAVControlBarrierFunction, QP Solver, Multi-Barrier Formulations"],
        ["Part 3.3", "mission_control.py", "Python 3", "295", "Tornado Web Server, REST APIs, Telemetry WebSockets, Swarm Command Bridge"],
        ["Part 3.4", "web_gui/index.html", "HTML5/JS/CSS", "1,754", "Three.js 3D Scene, 2D Canvas Radar, Zoom/Pan Engine, Telemetry UI"],
        ["Total", "Complete System Core", "Multi", "4,806", "Full Autonomous Swarm & Mission Control Architecture"]
    ]
    story.append(make_table(source_index_data, colWidths=[0.8 * inch, 2.2 * inch, 0.8 * inch, 0.6 * inch, 2.4 * inch]))
    story.append(Spacer(1, 15))

    target_files = [
        ("Part 3.1: step80_multiagent_swarm.py (Core Swarm Autonomy & Pipeline)", 
         os.path.join(BASE_DIR, "step80_multiagent_swarm.py")),
        ("Part 3.2: step76_control_barrier_function.py (CBF-QP Safety Filter)", 
         os.path.join(BASE_DIR, "step76_control_barrier_function.py")),
        ("Part 3.3: mission_control.py (Tornado Asynchronous Web Server)", 
         os.path.join(BASE_DIR, "mission_control.py")),
        ("Part 3.4: web_gui/index.html (3D/2D Mission Control Dashboard)", 
         os.path.join(BASE_DIR, "web_gui", "index.html")),
    ]

    for section_title, fpath in target_files:
        story.append(PageBreak())
        story.append(Paragraph(section_title, styles['SecHeading']))
        story.append(hr())
        story.append(Paragraph(f"<b>Source File Path:</b> <font face='Courier'>{fpath}</font><br/>"
                               f"<b>Character Encoding:</b> UTF-8 | <b>Preservation:</b> Verbatim line-numbered software deposit", styles['BodyJustify']))
        story.append(Spacer(1, 6))

        code_chunks = wrap_and_chunk_code(fpath, chunk_lines=140)
        for chunk in code_chunks:
            p_code = Preformatted(chunk, styles['SourceCodeStyle'])
            story.append(p_code)

    # ─────────────────────────────────────────────────────────────────────────
    # BUILD PDF DOCUMENT
    # ─────────────────────────────────────────────────────────────────────────
    print(f"Building Copyright Dossier PDF at {OUTPUT_PDF}...")
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    print(f"✅ Copyright Dossier generated successfully! Total pages: {doc.page}")

if __name__ == "__main__":
    build_pdf()
