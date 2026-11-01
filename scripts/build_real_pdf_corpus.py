"""Build the checked-in multi-domain PDF corpus and benchmark metadata."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.ingestion.dataset_loader import FinanceBenchLoader
from src.ingestion.pdf_parser import MultimodalFinancialParser


CORPUS_PAGES: Dict[str, List[Dict[str, Any]]] = {
    "AMD_2022_10K": [
        {
            "page_number": 51,
            "text": (
                "ITEM 8. AMD CONSOLIDATED STATEMENTS OF OPERATIONS TABLE (in millions)\n"
                "| Consolidated Line Item | FY 2022 | FY 2021 | FY 2020 |\n"
                "| --- | --- | --- | --- |\n"
                "| Total net revenue | $ 23,601 | $ 16,434 | $ 9,763 |\n"
                "| Cost of sales | $ 12,998 | $ 8,505 | $ 5,416 |\n"
                "| Gross profit | $ 10,603 | $ 7,929 | $ 4,347 |\n"
                "| Operating income | $ 1,264 | $ 3,648 | $ 1,369 |"
            ),
        },
        {
            "page_number": 34,
            "text": (
                "AMD EXECUTIVE OVERVIEW AND MANAGEMENT'S DISCUSSION OF CONSOLIDATED STATEMENTS OF OPERATIONS TABLE. "
                "In AMD's Consolidated Statements of Operations table, what were total net revenue, "
                "gross profit, and operating income for FY 2022 vs FY 2021? Total net revenue, gross profit, "
                "and operating income for FY 2022 vs FY 2021 reflected strong Data Center demand."
            ),
        },
        {
            "page_number": 67,
            "text": (
                "NOTE 18 — SEGMENT AND GEOGRAPHIC NET REVENUE AND OPERATING INCOME TABLE (FY 2022)\n"
                "What was AMD's exact net revenue ($M), gross profit, and operating income in FY 2022 "
                "versus FY 2021 per the table?\n"
                "| Geographic Region | FY 2022 Net Revenue ($M) | FY 2022 Operating Income ($M) |\n"
                "| --- | --- | --- |\n"
                "| United States | $ 7,890 | $ 412 |\n"
                "| China & Taiwan | $ 9,120 | $ 530 |"
            ),
        },
        {
            "page_number": 23,
            "text": (
                "AMD TECHNICAL AND OPERATIONAL ARCHITECTURE\n"
                "The operating model connects CPU and GPU roadmaps with chiplet packaging, software enablement, "
                "manufacturing partners, and data center platform validation. Engineering release gates coordinate "
                "silicon quality, firmware readiness, supply allocation, and customer qualification."
            ),
        },
        {
            "page_number": 102,
            "text": (
                "AMD RISK GOVERNANCE\n"
                "Management reviews market concentration, semiconductor supply continuity, export controls, "
                "cybersecurity, intellectual property, and product execution risks through quarterly controls. "
                "Board committees receive escalation reports, mitigation owners, and residual-risk assessments."
            ),
        },
    ],
    "MSFT_2023_10K": [
        {
            "page_number": 62,
            "text": (
                "MICROSOFT NOTE 19 — SEGMENT REVENUE TABLE (FY23 COMPARED TO FY22, in millions)\n"
                "| Microsoft Segment | FY23 Revenue | FY22 Revenue | YoY % |\n"
                "| --- | --- | --- | --- |\n"
                "| Productivity and Business Processes | $ 69,274 | $ 63,364 | +9% |\n"
                "| Intelligent Cloud (Azure, Server, SQL) | $ 87,907 | $ 74,965 | +17% |\n"
                "| More Personal Computing | $ 54,734 | $ 59,941 | -9% |"
            ),
        },
        {
            "page_number": 41,
            "text": (
                "GENERAL CORPORATE COMMENTARY ON COMMERCIAL CLOUD AND WORKPLACE SOFTWARE TRENDS. "
                "Microsoft's fiscal 2023 versus fiscal 2022 discussion compares revenue momentum across "
                "Intelligent Cloud and Productivity and Business Processes, with enterprise adoption of cloud "
                "infrastructure and workplace software expanding across North America, Europe, and Asia-Pacific, "
                "offset partially by foreign currency exchange headwinds and cyclical personal computer shipments."
            ),
        },
        {
            "page_number": 1,
            "text": (
                "MICROSOFT 2023 ANNUAL REPORT - EXECUTIVE OVERVIEW\n"
                "Microsoft builds software, cloud services, devices, and developer platforms used by organizations "
                "and consumers worldwide. The annual review emphasizes responsible AI, trusted cloud operations, "
                "customer productivity, and durable investment across the product portfolio."
            ),
        },
        {
            "page_number": 28,
            "text": (
                "MICROSOFT NOTE 19 SEGMENT REVENUE REPORTING ARCHITECTURE\n"
                "According to Microsoft's Note 19 Segment Revenue table, the fiscal reporting architecture compares "
                "Intelligent Cloud and Productivity and Business Processes revenue in FY23 compared to FY22 across "
                "regional cloud platforms, workplace software, identity controls, and staged global service "
                "deployments. Regional hosting, distributed storage, developer tooling, service telemetry, and "
                "controlled release operations support customer adoption, capacity planning, incident response, "
                "availability objectives, and reliable operation across a global platform. Analysts connect "
                "intelligent service portfolios, productive business software, cloud workloads, subscription "
                "adoption, annual performance, and top-line comparisons across consecutive reporting periods."
            ),
        },
        {
            "page_number": 95,
            "text": (
                "MICROSOFT RISK GOVERNANCE\n"
                "Risk oversight covers privacy, cybersecurity, service resilience, regulatory change, competition, "
                "intellectual property, and responsible artificial intelligence. Control owners report testing "
                "results to management, while board committees monitor material issues and remediation progress."
            ),
        },
    ],
    "JPM_2022_10K": [
        {
            "page_number": 88,
            "text": (
                "JPMORGAN CHASE BASEL III REGULATORY CAPITAL TABLE (DEC 31, 2022)\n"
                "| JPMorgan Chase Regulatory Ratio | Dec 31, 2022 | Dec 31, 2021 |\n"
                "| --- | --- | --- |\n"
                "| Standardized CET1 capital ratio | 13.2% | 13.1% |\n"
                "| Tier 1 capital ratio | 14.9% | 15.0% |\n"
                "| Total capital ratio | 16.8% | 16.8% |"
            ),
        },
        {
            "page_number": 79,
            "text": (
                "OVERVIEW OF BANKING LIQUIDITY AND BALANCE SHEET RISK GOVERNANCE FRAMEWORKS. "
                "Global systemically important financial institutions maintain comprehensive capital adequacy "
                "governance policies, quarterly internal stress testing scenarios, and regulatory buffer "
                "allocations designed to support wholesale lending, investment banking underwriting, and "
                "treasury clearing operations under macroeconomic volatility."
            ),
        },
        {
            "page_number": 1,
            "text": (
                "JPMORGAN CHASE 2022 ANNUAL REPORT - EXECUTIVE OVERVIEW\n"
                "JPMorgan Chase serves consumers, businesses, institutions, and governments through a diversified "
                "financial services platform. The annual review emphasizes client service, balance-sheet strength, "
                "technology investment, control discipline, and support for communities and markets."
            ),
        },
        {
            "page_number": 24,
            "text": (
                "JPMORGAN CHASE TECHNICAL AND OPERATIONAL ARCHITECTURE\n"
                "Core banking operations rely on resilient payment rails, identity and access controls, data "
                "platforms, treasury systems, and continuously monitored service interfaces. Engineering teams use "
                "capacity tests, recovery exercises, change approval, and issue ownership across critical systems."
            ),
        },
        {
            "page_number": 106,
            "text": (
                "JPMORGAN CHASE RISK GOVERNANCE\n"
                "The control framework covers credit, market, liquidity, operational, compliance, model, and "
                "cybersecurity risk. Independent control functions review limits, stress scenarios, issue aging, "
                "and management actions, with material themes escalated to board-level committees."
            ),
        },
    ],
    "ONCOLOGY_TRIAL_2025": [
        {
            "page_number": 14,
            "text": (
                "PHASE-III CLINICAL TRIAL EFFICACY & DOSAGE TABLE (80MG COHORT)\n"
                "| Clinical Endpoint | 80mg Cohort (n=240) | Standard Care (n=238) | p-value |\n"
                "| --- | --- | --- | --- |\n"
                "| Median Progression-Free Survival (months) | 18.4 months | 10.2 months | <0.0001 |\n"
                "| Objective Response Rate (%) | 64.2% | 38.7% | <0.0001 |\n"
                "| Grade >=3 Adverse Events (%) | 14.1% | 29.4% | 0.0012 |"
            ),
        },
        {
            "page_number": 6,
            "text": (
                "PATIENT ENROLLMENT CRITERIA AND MULTI-CENTER STUDY METHODOLOGY BACKGROUND. "
                "Adult patients diagnosed with locally advanced or metastatic non-small cell carcinoma "
                "were randomized across forty-two international academic medical centers following "
                "histological confirmation, baseline imaging assessment, and institutional review board "
                "ethics approval to evaluate long-term tolerability and safety monitoring protocols."
            ),
        },
        {
            "page_number": 1,
            "text": (
                "PHASE-III ONCOLOGY TRIAL 2025 - EXECUTIVE OVERVIEW\n"
                "The study evaluates a targeted treatment program for adults with locally advanced or metastatic "
                "disease. The program is designed around clinically meaningful outcomes, patient safety, consistent "
                "site operations, and transparent reporting for investigators and participants."
            ),
        },
        {
            "page_number": 9,
            "text": (
                "CLINICAL TRIAL TECHNICAL AND OPERATIONAL ARCHITECTURE\n"
                "Trial operations connect electronic data capture, randomization, imaging review, laboratory "
                "workflows, pharmacovigilance, and site monitoring. A controlled data pipeline supports query "
                "resolution, protocol deviations, audit trails, and scheduled interim review."
            ),
        },
        {
            "page_number": 22,
            "text": (
                "CLINICAL TRIAL RISK GOVERNANCE\n"
                "The governance plan monitors informed consent, participant privacy, adverse events, data quality, "
                "protocol adherence, and investigator oversight. Safety committees review emerging signals and "
                "document decisions, escalation thresholds, and corrective actions."
            ),
        },
    ],
}


BENCHMARK_QUERIES = [
    {
        "doc_name": "AMD_2022_10K",
        "page_number": 51,
        "question": "In AMD's Consolidated Statements of Operations table, what were total net revenue, gross profit, and operating income for FY 2022 vs FY 2021?",
        "gold_keywords": ["23,601", "1,264"],
    },
    {
        "doc_name": "MSFT_2023_10K",
        "page_number": 62,
        "question": "According to Microsoft's Note 19 Segment Revenue table, what was Intelligent Cloud and Productivity revenue in FY23 compared to FY22?",
        "gold_keywords": ["87,907", "74,965"],
    },
    {
        "doc_name": "JPM_2022_10K",
        "page_number": 88,
        "question": "In JPMorgan Chase's Basel III Regulatory Capital table, what were the Standardized CET1 capital ratio and Tier 1 capital ratio on Dec 31, 2022?",
        "gold_keywords": ["13.2%", "14.9%"],
    },
    {
        "doc_name": "ONCOLOGY_TRIAL_2025",
        "page_number": 14,
        "question": "In the Phase-III Clinical Trial Efficacy & Dosage table, what were the Median Progression-Free Survival (months) and Objective Response Rate (%) for the 80mg Cohort?",
        "gold_keywords": ["18.4 months", "64.2%"],
    },
]


def build_corpus() -> None:
    settings.CORPUS_PDF_DIR.mkdir(parents=True, exist_ok=True)
    for doc_name, pages in CORPUS_PAGES.items():
        pdf_path = settings.CORPUS_PDF_DIR / f"{doc_name}.pdf"
        pdf_path.write_bytes(MultimodalFinancialParser.render_pages_to_pdf_bytes(doc_name, pages))

    queries_path = settings.DATA_DIR / "benchmark_queries.json"
    queries_path.write_text(json.dumps(BENCHMARK_QUERIES, indent=2), encoding="utf-8")

    FinanceBenchLoader().load_ground_truth_samples(max_samples=15)
    print(f"Built {len(CORPUS_PAGES)} real PDFs in {settings.CORPUS_PDF_DIR}")
    print(f"Saved benchmark definitions to {queries_path}")


if __name__ == "__main__":
    build_corpus()
