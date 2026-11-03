"""Generate tests/fixtures/sample_policy.pdf (run once; binary is committed)."""
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "sample_policy.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

styles = getSampleStyleSheet()
h1 = ParagraphStyle("H1x", parent=styles["Heading1"], fontSize=16, spaceAfter=10)
h2 = ParagraphStyle("H2x", parent=styles["Heading2"], fontSize=12.5, spaceAfter=6)
body = ParagraphStyle("Bodyx", parent=styles["BodyText"], fontSize=10, leading=13)

doc = SimpleDocTemplate(str(OUT), pagesize=letter,
                        leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                        topMargin=0.8 * inch, bottomMargin=0.8 * inch)
story = [
    Paragraph("Acme Corp Remote Work Allowance Policy", h1),
    Paragraph("Policy Number: RW-2025-014. Effective Date: March 1, 2025.", body),
    Spacer(1, 8),
    Paragraph("1. Internet Stipend", h2),
    Paragraph("Employees working remotely for at least three days per week receive a monthly "
              "internet stipend of $50, paid through payroll. The stipend covers the primary "
              "household connection only and requires no receipts.", body),
    Paragraph("2. Equipment Budget", h2),
    Paragraph("Each remote employee is eligible for a one-time equipment budget of $600. "
              "Requests must be approved by the direct manager before purchase. Equipment "
              "remains company property and must be returned upon separation.", body),
    Spacer(1, 6),
    Paragraph("2.1 Reimbursable Equipment", h2),
    Paragraph("Monitors, keyboards, docking stations, ergonomic chairs, and webcams are "
              "reimbursable under the equipment budget. Personal items such as home gym "
              "equipment, coffee machines, and furniture for non-work rooms are excluded.", body),
    Paragraph("3. Tier Summary", h2),
]
table = Table(
    [["Tier", "Days/week remote", "Stipend", "Equipment budget"],
     ["Bronze", "3", "$50/mo", "$400"],
     ["Silver", "4", "$60/mo", "$600"],
     ["Gold", "5", "$75/mo", "$800"]],
    hAlign="LEFT",
)
table.setStyle(TableStyle([
    ("GRID", (0, 0), (-1, -1), 0.5, "grey"),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("BACKGROUND", (0, 0), (-1, 0), "#e8eef7"),
]))
story.append(table)
story.append(Spacer(1, 10))
Paragraph("4. Governance", h2)
story.append(Paragraph("This policy is owned by the Workplace Experience team and is reviewed "
                       "every January. Exceptions require written approval from the COO.", body))
doc.build(story)
print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
