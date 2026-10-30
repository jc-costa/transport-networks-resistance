# ============================================================================
# PDF GENERATOR
# Generates individual experiment PDFs and a cumulative general PDF.
#
# Individual PDF: saved inside each exp_N folder, contains:
#   - Experiment parameters
#   - Network images (PNG files in the folder)
#   - Resistances table (edges)
#   - Impedance calculation results
#
# General PDF: saved in the experiments/ base folder, accumulates all
#   experiments. It is rebuilt from scratch each time using all
#   experiment_summary.json files found in exp_* sub-folders.
# ============================================================================

import os
import re
import json

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether
)


# -- colour palette ----------------------------------------------------------
COL_DARK_BG   = colors.HexColor("#1A1A2E")
COL_ACCENT    = colors.HexColor("#0F3460")
COL_HIGHLIGHT = colors.HexColor("#E94560")
COL_LIGHT     = colors.HexColor("#F5F5F5")
COL_WHITE     = colors.white
COL_TEXT      = colors.HexColor("#2C2C2C")
COL_MUTED     = colors.HexColor("#666666")
COL_ROW_ALT   = colors.HexColor("#EEF2F7")


# -- helpers -----------------------------------------------------------------

def _styles():
    """Return a dict of custom paragraph styles."""
    base = getSampleStyleSheet()

    s = {}

    s["title"] = ParagraphStyle(
        "title",
        parent=base["Title"],
        fontSize=22,
        textColor=COL_WHITE,
        leading=28,
        spaceAfter=4,
        alignment=TA_CENTER,
    )
    s["subtitle"] = ParagraphStyle(
        "subtitle",
        parent=base["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#AAAACC"),
        leading=16,
        alignment=TA_CENTER,
        spaceAfter=2,
    )
    s["section"] = ParagraphStyle(
        "section",
        parent=base["Heading2"],
        fontSize=13,
        textColor=COL_WHITE,
        backColor=COL_ACCENT,
        leading=18,
        spaceBefore=14,
        spaceAfter=6,
        leftIndent=-6,
        rightIndent=-6,
        borderPad=6,
    )
    s["body"] = ParagraphStyle(
        "body",
        parent=base["Normal"],
        fontSize=9,
        textColor=COL_TEXT,
        leading=13,
        spaceAfter=2,
    )
    s["caption"] = ParagraphStyle(
        "caption",
        parent=base["Normal"],
        fontSize=8,
        textColor=COL_MUTED,
        leading=11,
        alignment=TA_CENTER,
        spaceAfter=6,
    )
    s["mono"] = ParagraphStyle(
        "mono",
        parent=base["Code"],
        fontSize=8,
        textColor=COL_TEXT,
        leading=11,
    )
    s["no_perc"] = ParagraphStyle(
        "no_perc",
        parent=base["Normal"],
        fontSize=10,
        textColor=COL_HIGHLIGHT,
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    s["partial"] = ParagraphStyle(
        "partial",
        parent=base["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#AA6600"),
        alignment=TA_CENTER,
        spaceAfter=6,
        leading=14,
    )
    return s


def _header_block(story, exp_number, timestamp, environment, st):
    """Dark banner header with experiment title."""
    # Title table with dark background
    header_data = [[
        Paragraph(f"EXPERIMENT #{exp_number}", st["title"]),
    ]]
    sub_data = [[
        Paragraph(f"Timestamp: {timestamp}  |  Environment: {environment}", st["subtitle"]),
    ]]

    tbl = Table(header_data, colWidths=[17 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), COL_DARK_BG),
        ("TOPPADDING",    (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(tbl)

    sub = Table(sub_data, colWidths=[17 * cm])
    sub.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), COL_ACCENT),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(sub)
    story.append(Spacer(1, 10))


def _section_header(story, title, st):
    story.append(Paragraph(f"  {title}", st["section"]))


def _params_table(story, summary, st):
    """Render a parameters summary table."""
    params = summary.get("parameters", {})
    net    = summary.get("network_generation", {})
    perc   = summary.get("percolation_analysis", {})
    prun   = summary.get("pruning", {})

    rows = [
        ["Parameter", "Value"],
        ["Number of sticks (N)", str(params.get("N", "--"))],
        ["Stick length (a)", f"{params.get('a', '--')}"],
        ["Domain size", "1.0 x 1.0"],
        ["Source region", params.get("source_region", "--")],
        ["Drain region",  params.get("drain_region", "--")],
        ["Total intersections", str(net.get("total_intersections", "--"))],
        ["Has percolation", "[OK] Yes" if perc.get("has_percolation") else "[FAIL] No"],
        ["# Percolating paths", str(perc.get("num_percolating_paths", "--"))],
        ["Nodes before pruning", str(prun.get("nodes_before", "--"))],
        ["Nodes after pruning",  str(prun.get("nodes_after",  "--"))],
        ["Edges before pruning", str(prun.get("edges_before", "--"))],
        ["Edges after pruning",  str(prun.get("edges_after",  "--"))],
    ]

    col_w = [8 * cm, 9 * cm]
    tbl = Table(rows, colWidths=col_w, repeatRows=1)
    style = [
        # Header row
        ("BACKGROUND", (0, 0), (-1, 0), COL_DARK_BG),
        ("TEXTCOLOR",  (0, 0), (-1, 0), COL_WHITE),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0), 9),
        ("ALIGN",      (0, 0), (-1, 0), "CENTER"),
        # Body
        ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 1), (-1, -1), 8),
        ("TEXTCOLOR",  (0, 1), (-1, -1), COL_TEXT),
        ("ALIGN",      (0, 1), (0, -1), "LEFT"),
        ("ALIGN",      (1, 1), (1, -1), "CENTER"),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [COL_WHITE, COL_ROW_ALT]),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]
    tbl.setStyle(TableStyle(style))
    story.append(tbl)
    story.append(Spacer(1, 8))


def _images_section(story, exp_dir, st):
    """Embed all PNG images found in the experiment folder."""
    pngs = sorted([f for f in os.listdir(exp_dir) if f.lower().endswith(".png")])
    if not pngs:
        story.append(Paragraph("No images found in this experiment folder.", st["body"]))
        return

    page_w = 17 * cm  # usable width on A4 with 2 cm margins each side

    for fname in pngs:
        img_path = os.path.join(exp_dir, fname)
        try:
            img = Image(img_path)
            # Scale proportionally to fit page width
            ratio = img.imageWidth / img.imageHeight
            img.drawWidth  = page_w
            img.drawHeight = page_w / ratio
            # If image is taller than 10 cm, cap height
            if img.drawHeight > 12 * cm:
                img.drawHeight = 12 * cm
                img.drawWidth  = 12 * cm * ratio

            label = fname.replace("_", " ").replace(".png", "")
            story.append(KeepTogether([
                img,
                Paragraph(f"Fig. -- {label}", st["caption"]),
                Spacer(1, 4),
            ]))
        except Exception as e:
            story.append(Paragraph(f"[Could not load image: {fname} -- {e}]", st["body"]))


def _resistances_table(story, summary, st):
    """Render a table of edge resistances from edges_detailed."""
    edges = summary.get("network_pruned", {}).get("edges", [])
    if not edges:
        story.append(Paragraph("No edge data available.", st["body"]))
        return

    rows = [["Edge #", "Node 1", "Node 2", "Conductance (S)", "Resistance (Ohm)", "Voltage Drop (V)"]]
    for e in edges:
        res = e.get("resistance_ohms", None)
        res_str = f"{res:.4f}" if isinstance(res, (int, float)) and res != float('inf') else "inf"
        vdrop = e.get("voltage_drop_V", None)
        vdrop_str = f"{vdrop:.6f}" if isinstance(vdrop, (int, float)) else "--"
        cond = e.get("conductance_S", None)
        cond_str = f"{cond:.4f}" if isinstance(cond, (int, float)) else "--"

        rows.append([
            str(e.get("edge_index", "--")),
            str(e.get("node1", "--")),
            str(e.get("node2", "--")),
            cond_str,
            res_str,
            vdrop_str,
        ])

    col_w = [2 * cm, 2 * cm, 2 * cm, 3.5 * cm, 3.5 * cm, 4 * cm]
    tbl = Table(rows, colWidths=col_w, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), COL_DARK_BG),
        ("TEXTCOLOR",  (0, 0), (-1, 0), COL_WHITE),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0), 8),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 1), (-1, -1), 7),
        ("TEXTCOLOR",  (0, 1), (-1, -1), COL_TEXT),
        ("GRID",       (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [COL_WHITE, COL_ROW_ALT]),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    tbl.setStyle(TableStyle(style))
    story.append(tbl)
    story.append(Spacer(1, 8))


def _impedance_section(story, summary, st):
    """Render impedance calculation results."""
    imp_calc = summary.get("impedance_calculation", {})
    imp_vs_k = summary.get("impedance_vs_k", {})

    # Main result
    R_val = imp_calc.get("calculated_impedance_ohms", None)
    I_val = imp_calc.get("total_current_A", None)
    k_used = imp_calc.get("k_used_ohms", "--")

    main_rows = [
        ["Quantity", "Value"],
        ["Applied voltage (V)", f"{imp_calc.get('applied_voltage_V', 1.0):.2f} V"],
        ["Resistance per segment k", f"{k_used} Ohm"],
        ["Calculated impedance R_eq",
            f"{R_val:.6f} Ohm" if isinstance(R_val, (int, float)) else "--"],
        ["Total current I",
            f"{I_val:.6e} A" if isinstance(I_val, (int, float)) else "--"],
    ]

    col_w = [9 * cm, 8 * cm]
    tbl = Table(main_rows, colWidths=col_w)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), COL_DARK_BG),
        ("TEXTCOLOR",  (0, 0), (-1, 0), COL_WHITE),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0), 9),
        ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#D6EAF8")),  # highlight R_eq row
        ("FONTNAME",   (0, 3), (-1, 3), "Helvetica-Bold"),
        ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 1), (-1, -1), 8),
        ("TEXTCOLOR",  (0, 1), (-1, -1), COL_TEXT),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("GRID",       (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [COL_WHITE, COL_ROW_ALT]),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    tbl.setStyle(TableStyle(style))
    story.append(tbl)
    story.append(Spacer(1, 10))

    # Impedance vs k table
    if imp_vs_k:
        story.append(Paragraph("  Impedance for different k (resistance per segment) values", st["section"]))
        k_rows = [["k (Ohm/segment)", "R_equivalent (Ohm)"]]
        for kv, data in sorted(imp_vs_k.items(), key=lambda x: float(x[0])):
            r_eq = data.get("R_equivalent_ohms", None)
            k_rows.append([
                f"{float(kv):.0f}",
                f"{r_eq:.6f}" if isinstance(r_eq, (int, float)) else "--",
            ])
        col_w2 = [8.5 * cm, 8.5 * cm]
        tbl2 = Table(k_rows, colWidths=col_w2, repeatRows=1)
        style2 = [
            ("BACKGROUND", (0, 0), (-1, 0), COL_ACCENT),
            ("TEXTCOLOR",  (0, 0), (-1, 0), COL_WHITE),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, 0), 9),
            ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE",   (0, 1), (-1, -1), 8),
            ("TEXTCOLOR",  (0, 1), (-1, -1), COL_TEXT),
            ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
            ("GRID",       (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [COL_WHITE, COL_ROW_ALT]),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]
        tbl2.setStyle(TableStyle(style2))
        story.append(tbl2)
        story.append(Spacer(1, 8))


def _partial_exp_section(story, exp_num, exp_dir, st):
    """
    Render a section for an experiment that has no experiment_summary.json.
    Shows available PNG images and an informative status message.
    """
    pngs = sorted([f for f in os.listdir(exp_dir) if f.lower().endswith(".png")])

    # Header (sem timestamp/environment disponíveis)
    header_data = [[Paragraph(f"EXPERIMENT #{exp_num}", st["title"])]]
    sub_data    = [[Paragraph("Status: Partial data -- experiment_summary.json not found",
                              st["subtitle"])]]

    tbl = Table(header_data, colWidths=[17 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#2E1A00")),
        ("TOPPADDING",    (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(tbl)

    sub = Table(sub_data, colWidths=[17 * cm])
    sub.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#7A4000")),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(sub)
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "[WARNING]  Este experimento foi iniciado mas não foi concluído -- "
        "o arquivo experiment_summary.json não foi gerado. "
        "Apenas as imagens disponíveis são exibidas abaixo.",
        st["partial"]
    ))

    if pngs:
        _section_header(story, "Network Visualizations (available)", st)
        _images_section(story, exp_dir, st)
    else:
        story.append(Paragraph("Nenhuma imagem ou dado encontrado nesta pasta.", st["body"]))


# -- public API ---------------------------------------------------------------

def generate_experiment_pdf(exp_dir, summary=None):
    """
    Generate an individual PDF report for a single experiment.

    Args:
        exp_dir (str): Path to the experiment folder (e.g. experiments/exp_3)
        summary (dict): Pre-loaded summary dict. If None, reads
                        experiment_summary.json from exp_dir.

    Returns:
        str: Absolute path to the saved PDF file.
    """
    if summary is None:
        summary_path = os.path.join(exp_dir, "experiment_summary.json")
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"No experiment_summary.json found in {exp_dir}")
        with open(summary_path, "r") as f:
            summary = json.load(f)

    # Derive experiment number from folder name
    folder_name = os.path.basename(exp_dir)
    match = re.search(r"(\d+)$", folder_name)
    exp_number = match.group(1) if match else folder_name

    timestamp   = summary.get("timestamp", "--")
    environment = summary.get("environment", "--")
    has_perc    = summary.get("percolation_analysis", {}).get("has_percolation", False)

    pdf_path = os.path.join(exp_dir, f"experiment_{exp_number}_report.pdf")

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title=f"Experiment {exp_number} Report",
        author="Random Network Simulator",
    )

    st = _styles()
    story = []

    # -- Header ---------------------------------------------------------------
    _header_block(story, exp_number, timestamp, environment, st)

    # -- Parameters -----------------------------------------------------------
    _section_header(story, "1. Experiment Parameters & Network Statistics", st)
    _params_table(story, summary, st)

    # -- Images ---------------------------------------------------------------
    _section_header(story, "2. Network Visualizations", st)
    _images_section(story, exp_dir, st)

    # -- Resistances ----------------------------------------------------------
    _section_header(story, "3. Edge Resistances (Pruned Network)", st)
    if not has_perc:
        story.append(Paragraph("[WARNING]  No percolating path was found -- no resistance data.", st["no_perc"]))
    else:
        _resistances_table(story, summary, st)

    # -- Impedance ------------------------------------------------------------
    _section_header(story, "4. Impedance Calculation", st)
    if not has_perc:
        story.append(Paragraph("[WARNING]  Impedance calculation was skipped (no percolation).", st["no_perc"]))
    else:
        _impedance_section(story, summary, st)

    doc.build(story)
    print(f"  [OK] Individual PDF saved: {pdf_path}")
    return pdf_path


def generate_general_pdf(experiments_base_dir):
    """
    Generate (or update) the general PDF containing ALL experiments.
    Experiments with experiment_summary.json are shown in full detail.
    Experiments without it are shown with available images and a partial-data notice.

    Sorted numerically across all exp_* sub-folders.

    Args:
        experiments_base_dir (str): Path to the experiments/ folder.

    Returns:
        str: Absolute path to the saved PDF file.
    """
    # -- Collect all experiment folders, sorted numerically --------------------
    pattern = re.compile(r"^exp_(\d+)$")
    exp_entries = []
    for entry in os.listdir(experiments_base_dir):
        m = pattern.match(entry)
        if m:
            exp_entries.append((int(m.group(1)), entry))
    exp_entries.sort()

    if not exp_entries:
        print("  [WARNING]  No experiment folders found -- general PDF not created.")
        return None

    # Count complete vs partial
    n_complete = sum(
        1 for _, folder in exp_entries
        if os.path.exists(os.path.join(experiments_base_dir, folder, "experiment_summary.json"))
    )
    n_partial = len(exp_entries) - n_complete

    pdf_path = os.path.join(experiments_base_dir, "all_experiments_report.pdf")

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="All Experiments -- General Report",
        author="Random Network Simulator",
    )

    st = _styles()
    story = []

    # -- Cover page -----------------------------------------------------------
    cover_data = [[Paragraph("GENERAL EXPERIMENTS REPORT", st["title"])]]
    cover_sub  = [[Paragraph(
        f"Random Percolation Network Simulator  |  "
        f"{len(exp_entries)} experiment(s)  "
        f"({n_complete} complete, {n_partial} partial)",
        st["subtitle"]
    )]]

    cov = Table(cover_data, colWidths=[17 * cm])
    cov.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), COL_DARK_BG),
        ("TOPPADDING",    (0, 0), (-1, -1), 20),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(cov)

    sub = Table(cover_sub, colWidths=[17 * cm])
    sub.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), COL_ACCENT),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(sub)
    story.append(Spacer(1, 20))

    # -- Overview table (all experiments) -------------------------------------
    overview_rows = [
        ["Exp #", "Timestamp", "N", "a", "Percolação", "R_eq (k=1 Ohm)", "Intersecções", "Status"]
    ]
    # Track which rows are partial to colour them differently
    partial_row_indices = []

    for row_i, (exp_num, folder) in enumerate(exp_entries, start=1):
        exp_dir      = os.path.join(experiments_base_dir, folder)
        summary_path = os.path.join(exp_dir, "experiment_summary.json")

        if not os.path.exists(summary_path):
            # Count available PNGs so the user has some info
            n_imgs = len([f for f in os.listdir(exp_dir) if f.lower().endswith(".png")])
            overview_rows.append([
                str(exp_num), "--", "--", "--", "--", "--",
                f"{n_imgs} img(s)",
                "Parcial",
            ])
            partial_row_indices.append(row_i)  # 1-indexed within body rows
            continue

        with open(summary_path, "r") as f:
            s = json.load(f)
        params   = s.get("parameters", {})
        perc     = s.get("percolation_analysis", {})
        net      = s.get("network_generation", {})
        imp_vs_k = s.get("impedance_vs_k", {})

        has_p  = "[OK]" if perc.get("has_percolation") else "[FAIL]"
        r_eq_1 = imp_vs_k.get(1, imp_vs_k.get("1", {}))
        r_val  = r_eq_1.get("R_equivalent_ohms", None) if r_eq_1 else None
        r_str  = f"{r_val:.4f}" if isinstance(r_val, (int, float)) else "--"

        overview_rows.append([
            str(exp_num),
            str(s.get("timestamp", "--")),
            str(params.get("N", "--")),
            str(params.get("a", "--")),
            has_p,
            r_str,
            str(net.get("total_intersections", "--")),
            "Completo",
        ])

    col_w = [1.2*cm, 3.5*cm, 1.2*cm, 1.2*cm, 2.3*cm, 3.0*cm, 2.3*cm, 2.3*cm]
    ov_tbl = Table(overview_rows, colWidths=col_w, repeatRows=1)
    ov_style = [
        ("BACKGROUND", (0, 0), (-1, 0), COL_DARK_BG),
        ("TEXTCOLOR",  (0, 0), (-1, 0), COL_WHITE),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, 0), 7),
        ("FONTNAME",   (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",   (0, 1), (-1, -1), 6),
        ("TEXTCOLOR",  (0, 1), (-1, -1), COL_TEXT),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("GRID",       (0, 0), (-1, -1), 0.3, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [COL_WHITE, COL_ROW_ALT]),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    # Highlight partial rows in amber
    for ri in partial_row_indices:
        ov_style.append(("BACKGROUND",  (0, ri), (-1, ri), colors.HexColor("#FFF3CD")))
        ov_style.append(("TEXTCOLOR",   (7, ri), (7,  ri), colors.HexColor("#AA6600")))
        ov_style.append(("FONTNAME",    (7, ri), (7,  ri), "Helvetica-Bold"))
    ov_tbl.setStyle(TableStyle(ov_style))
    story.append(ov_tbl)
    story.append(PageBreak())

    # -- One section per experiment --------------------------------------------
    for exp_num, folder in exp_entries:
        exp_dir      = os.path.join(experiments_base_dir, folder)
        summary_path = os.path.join(exp_dir, "experiment_summary.json")

        if not os.path.exists(summary_path):
            # Partial experiment -- show whatever images are available
            pngs = sorted([f for f in os.listdir(exp_dir) if f.lower().endswith(".png")])
            if not pngs:
                # Completely empty folder -- skip page break to save space
                continue
            _partial_exp_section(story, exp_num, exp_dir, st)
            story.append(PageBreak())
            continue

        with open(summary_path, "r") as f:
            summary = json.load(f)

        timestamp   = summary.get("timestamp", "--")
        environment = summary.get("environment", "--")
        has_perc    = summary.get("percolation_analysis", {}).get("has_percolation", False)

        _header_block(story, exp_num, timestamp, environment, st)

        _section_header(story, "Parameters & Network Statistics", st)
        _params_table(story, summary, st)

        _section_header(story, "Network Visualizations", st)
        _images_section(story, exp_dir, st)

        _section_header(story, "Edge Resistances (Pruned Network)", st)
        if not has_perc:
            story.append(Paragraph("[WARNING]  No percolating path -- no resistance data.", st["no_perc"]))
        else:
            _resistances_table(story, summary, st)

        _section_header(story, "Impedance Calculation", st)
        if not has_perc:
            story.append(Paragraph("[WARNING]  Impedance calculation skipped (no percolation).", st["no_perc"]))
        else:
            _impedance_section(story, summary, st)

        story.append(PageBreak())

    doc.build(story)
    print(f"  [OK] General PDF saved: {pdf_path}  "
          f"({n_complete} complete + {n_partial} partial experiments)")
    return pdf_path


def generate_all_pdfs(exp_dir, experiments_base_dir, summary=None):
    """
    Convenience function: generate both the individual experiment PDF
    and update the general PDF.

    Args:
        exp_dir (str): Path to the current experiment folder.
        experiments_base_dir (str): Path to the experiments/ base folder.
        summary (dict): Optional pre-loaded summary dict.

    Returns:
        tuple: (individual_pdf_path, general_pdf_path)
    """
    print("\nGenerating PDFs...")
    ind = generate_experiment_pdf(exp_dir, summary=summary)
    gen = generate_general_pdf(experiments_base_dir)
    print("PDF generation complete.\n")
    return ind, gen
