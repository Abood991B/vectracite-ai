WORKBENCH_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>VectraCite-AI — Enterprise Document & Table Intelligence</title>
  <link rel="icon" href="data:," />
  <style>
    :root {
      --page: #f5f7fa;
      --surface: #ffffff;
      --surface-raised: #edf1f5;
      --surface-ink: #102033;
      --border: #d7dee8;
      --border-strong: #aebccc;
      --text: #102033;
      --text-secondary: #4e6073;
      --text-muted: #748396;
      --accent: #0f5c8d;
      --accent-hover: #0b4a72;
      --accent-soft: #eaf2f7;
      --success-bg: #edf7f1;
      --success-text: #167349;
      --success-border: #a9d6bb;
      --danger-bg: #fbefef;
      --danger-text: #a43838;
      --danger-border: #e4b8b8;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      --sans: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
      --ease: cubic-bezier(0.22, 1, 0.36, 1);
    }
    * { box-sizing: border-box; }
    button, input, textarea { font: inherit; }
    body {
      margin: 0;
      background: var(--page);
      color: var(--text);
      font-family: var(--sans);
      font-size: 14px;
      line-height: 1.5;
    }
    .header {
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 0 32px;
      display: flex;
      align-items: center;
      min-height: 64px;
      position: sticky;
      top: 0;
      z-index: 20;
    }
    .brand-group { display: flex; align-items: center; gap: 12px; flex-shrink: 0; }
    .brand-logo {
      width: 32px;
      height: 32px;
      border-radius: 8px;
      background: var(--surface-raised);
      border: 1px solid var(--border-strong);
      color: var(--accent);
      font-family: var(--mono);
      font-weight: 600;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 13px;
      letter-spacing: -0.04em;
    }
    .brand-title { font-weight: 600; font-size: 16px; color: var(--text); line-height: 20px; }
    .brand-sub { font-size: 12px; color: var(--text-muted); line-height: 16px; }
    .tabs { display: flex; gap: 4px; height: 64px; align-items: center; margin-left: auto; margin-right: 24px; }
    .tab-btn {
      height: 40px;
      background: transparent;
      border: 0;
      border-bottom: 2px solid transparent;
      padding: 0 12px;
      border-radius: 8px 8px 0 0;
      font-size: 13px;
      font-weight: 500;
      color: var(--text-secondary);
      cursor: pointer;
      transition: color 160ms var(--ease), background-color 160ms var(--ease), border-color 160ms var(--ease);
    }
    .tab-btn:hover { background: var(--surface-raised); color: var(--text); }
    .tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); }
    .status-pill {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
      font-family: var(--mono);
      color: var(--text-secondary);
      white-space: nowrap;
    }
    .status-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--success-text);
    }
    .main { max-width: 1440px; margin: 0 auto; padding: 32px 32px 64px; }
    .tab-view { display: none; }
    .tab-view.active { display: block; }
    .step-banner {
      border-top: 1px solid var(--border);
      border-bottom: 1px solid var(--border);
      padding: 16px 0;
      margin-bottom: 32px;
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 24px;
    }
    .step-item { display: flex; gap: 12px; align-items: flex-start; }
    .step-num {
      width: 32px;
      height: 32px;
      border: 1px solid var(--border-strong);
      border-radius: 8px;
      color: var(--accent);
      font-family: var(--mono);
      font-variant-numeric: tabular-nums;
      font-size: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }
    .step-title { font-weight: 500; font-size: 14px; line-height: 20px; }
    .step-desc { font-size: 12px; line-height: 20px; color: var(--text-secondary); }
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 24px;
      margin-bottom: 24px;
      box-shadow: 0 1px 2px rgba(16, 32, 51, 0.04);
    }
    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      margin-bottom: 16px;
    }
    .card-title { font-size: 16px; font-weight: 600; line-height: 24px; margin: 0; letter-spacing: -0.01em; }
    .card-subtitle { font-size: 13px; line-height: 20px; color: var(--text-secondary); margin-top: 4px; }
    .section-label { font-size: 13px; line-height: 20px; font-weight: 500; margin: 0 0 8px; color: var(--text); }
    .scenario-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }
    .scenario-card {
      min-height: 120px;
      text-align: left;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      cursor: pointer;
      transition: background-color 160ms var(--ease), border-color 160ms var(--ease), transform 160ms var(--ease);
    }
    .scenario-card:hover { border-color: var(--accent); background: var(--accent-soft); }
    .scenario-card:active { transform: scale(0.98); }
    .scenario-tag {
      font-family: var(--mono);
      font-size: 11px;
      font-weight: 500;
      color: var(--text-muted);
      margin-bottom: 8px;
      display: block;
    }
    .scenario-tag.danger { color: var(--danger-text); }
    .scenario-name { font-weight: 500; font-size: 14px; line-height: 20px; margin-bottom: 4px; color: var(--text); }
    .scenario-q { font-size: 12px; line-height: 18px; color: var(--text-secondary); display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
    .search-box { display: flex; gap: 12px; }
    .search-input {
      flex: 1;
      min-height: 48px;
      padding: 12px 16px;
      font-size: 14px;
      border: 1px solid var(--border-strong);
      border-radius: 8px;
      outline: none;
      color: var(--text);
      background: var(--surface);
    }
    .search-input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(15, 92, 141, 0.14); }
    .btn {
      min-height: 40px;
      padding: 0 16px;
      border-radius: 8px;
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
      border: 1px solid transparent;
      transition: color 160ms var(--ease), background-color 160ms var(--ease), border-color 160ms var(--ease), transform 160ms var(--ease);
    }
    .btn:active { transform: scale(0.98); }
    .btn:focus-visible, .tab-btn:focus-visible, .scenario-card:focus-visible, .doc-row:focus-visible { outline: 3px solid rgba(15, 92, 141, 0.26); outline-offset: 2px; }
    .btn-primary { background: var(--accent); color: #fff; }
    .btn-primary:hover { background: var(--accent-hover); }
    .btn-secondary { background: var(--surface-raised); color: var(--text); border-color: var(--border-strong); }
    .btn-secondary:hover { background: var(--surface); border-color: var(--accent); }
    .btn-danger { background: var(--danger-bg); color: var(--danger-text); border-color: var(--danger-border); }
    .metrics-strip {
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      margin-bottom: 20px;
    }
    .metric-pill {
      font-family: var(--mono);
      font-size: 12px;
      line-height: 20px;
      padding: 0 16px 0 0;
      color: var(--text-secondary);
      border: 0;
      border-right: 1px solid var(--border);
      border-radius: 0;
      background: transparent;
    }
    .metric-pill:last-child { border-right: 0; padding-right: 0; }
    .metric-pill.verified { color: var(--success-text); font-weight: 500; }
    .metric-pill.blocked { color: var(--danger-text); font-weight: 500; }
    .answer-banner {
      background: var(--accent-soft);
      border: 1px solid #b9d2df;
      border-left: 4px solid var(--accent);
      padding: 20px 24px;
      border-radius: 12px;
      font-size: 16px;
      line-height: 24px;
      font-weight: 500;
      margin-bottom: 24px;
    }
    .answer-banner.blocked {
      background: var(--danger-bg);
      border-color: var(--danger-border);
      border-left-color: var(--danger-text);
      color: var(--danger-text);
    }
    .grid-2 { display: grid; grid-template-columns: 1.25fr 0.95fr; gap: 32px; }
    table.data-table { width: 100%; border-collapse: collapse; font-size: 13px; }
    table.data-table th, table.data-table td { border-bottom: 1px solid var(--border); padding: 12px 8px; text-align: left; vertical-align: top; }
    table.data-table th { border-bottom: 2px solid var(--border-strong); font-weight: 500; color: var(--text-secondary); }
    table.data-table tr:last-child td { border-bottom: 0; }
    .mono { font-family: var(--mono); font-variant-numeric: tabular-nums; }
    .delta-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-top: 16px; }
    .delta-card { background: var(--surface-raised); border: 1px solid var(--border); border-left: 3px solid var(--accent); border-radius: 8px; padding: 12px 16px; }
    .delta-title { font-size: 12px; line-height: 18px; color: var(--text-secondary); font-weight: 500; }
    .delta-val { font-family: var(--mono); font-size: 14px; line-height: 20px; font-weight: 500; margin-top: 4px; color: var(--text); }
    .source-card { border: 1px solid var(--border); border-left: 3px solid var(--accent); border-radius: 8px; padding: 16px; margin-bottom: 12px; background: var(--surface-raised); }
    .source-header { display: flex; justify-content: space-between; align-items: center; gap: 16px; margin-bottom: 12px; font-family: var(--mono); font-size: 12px; color: var(--text-secondary); }
    .kb-layout { display: grid; grid-template-columns: 320px minmax(0, 1fr); gap: 32px; }
    .doc-row { border: 1px solid var(--border); border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; background: var(--surface); display: flex; justify-content: space-between; align-items: center; gap: 12px; cursor: pointer; transition: background-color 160ms var(--ease), border-color 160ms var(--ease); }
    .doc-row:hover { background: var(--surface-raised); border-color: var(--border-strong); }
    .doc-row.active { border-color: var(--accent); background: var(--accent-soft); box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.72); }
    .doc-name { font-weight: 600; }
    .doc-meta { font-size: 11px; color: var(--text-secondary); }
    .form-group { margin-bottom: 16px; }
    .form-label { display: block; font-size: 12px; font-weight: 500; line-height: 16px; margin-bottom: 8px; }
    .input-field { width: 100%; min-height: 40px; padding: 10px 12px; border: 1px solid var(--border-strong); border-radius: 8px; font-size: 13px; line-height: 20px; font-family: var(--sans); color: var(--text); background: var(--surface); }
    .input-field:focus { outline: 3px solid rgba(15, 92, 141, 0.14); border-color: var(--accent); }
    .full-width { width: 100%; }
    .status-text { margin-top: 8px; color: var(--text-muted); font-size: 12px; line-height: 20px; font-family: var(--mono); }
    .template-actions, .action-row { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
    .template-btn, .btn-compact, .api-link { min-height: 36px; padding: 0 12px; font-size: 12px; }
    .flex-button { flex: 1; }
    .raw-response { margin-top: 16px; }
    .raw-response summary { cursor: pointer; color: var(--text-secondary); font-family: var(--mono); font-size: 12px; }
    .benchmark-bars { margin-top: 20px; }
    .benchmark-summary { margin-top: 24px; }
    .benchmark-table-wrap { margin-top: 24px; overflow-x: auto; }
    .bar-row { margin-bottom: 20px; }
    .bar-label { display: flex; justify-content: space-between; gap: 16px; font-size: 13px; line-height: 20px; font-weight: 500; margin-bottom: 8px; }
    .bar-track { height: 8px; background: var(--surface-raised); border: 1px solid var(--border); border-radius: 4px; overflow: hidden; }
    .bar-fill { height: 100%; background: var(--accent); border-radius: 3px; transition: width 180ms var(--ease); }
    pre.code-block { background: var(--surface-ink); color: #eef3f7; padding: 16px; border-radius: 8px; overflow-x: auto; font-family: var(--mono); font-size: 12px; line-height: 20px; }
    .workbench-footer { max-width: 1440px; margin: 0 auto; padding: 0 32px 24px; color: var(--text-muted); font-size: 11px; line-height: 16px; text-align: right; }
    @media (max-width: 900px) {
      .header { padding: 16px 20px; flex-wrap: wrap; gap: 16px; }
      .tabs { order: 3; width: 100%; height: 40px; overflow-x: auto; margin: 0; }
      .step-banner, .grid-2, .kb-layout { grid-template-columns: 1fr; }
      .main { padding: 24px 20px 48px; }
      .workbench-footer { padding: 0 20px 24px; }
    }
    @media (max-width: 600px) {
      .brand-sub { display: none; }
      .status-pill { margin-left: auto; }
      .search-box, .card-header { flex-direction: column; align-items: stretch; }
      .search-box .btn { width: 100%; }
      .bar-label { flex-direction: column; gap: 0; }
      .card { padding: 20px; }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { scroll-behavior: auto !important; transition-duration: 0.01ms !important; animation-duration: 0.01ms !important; animation-iteration-count: 1 !important; }
    }
  </style>
</head>
<body>
  <header class="header">
    <div class="brand-group">
      <div class="brand-logo">VC</div>
      <div>
        <div class="brand-title">VectraCite-AI</div>
        <div class="brand-sub">Retrieval &amp; Table Coordinate Inspector</div>
      </div>
    </div>
    <nav class="tabs">
      <button class="tab-btn active" data-tab="tab-query">Ask &amp; Verify</button>
      <button class="tab-btn" data-tab="tab-kb">Knowledge Base &amp; PDF Upload</button>
      <button class="tab-btn" data-tab="tab-bench">Benchmarks &amp; System Health</button>
    </nav>
    <div class="status-pill" id="headerStatus">
      <span class="status-dot"></span>
      <span id="headerStatusText">Engine online</span>
    </div>
  </header>

  <main class="main">
    <!-- TAB 1: INTERACTIVE Q&A & TABLE COORDINATE VERIFICATION -->
    <section id="tab-query" class="tab-view active">
      <div class="step-banner">
        <div class="step-item">
          <div class="step-num">01</div>
          <div>
            <div class="step-title">Choose a question</div>
            <div class="step-desc">Start with a benchmark question or ask about an indexed filing</div>
          </div>
        </div>
        <div class="step-item">
          <div class="step-num">02</div>
          <div>
            <div class="step-title">Retrieve evidence</div>
            <div class="step-desc">Dense, lexical, and SCAR ranking find the relevant page</div>
          </div>
        </div>
        <div class="step-item">
          <div class="step-num">03</div>
          <div>
            <div class="step-title">Verify the figures</div>
            <div class="step-desc">Cells, citations, and cross-column changes stay tied to the source</div>
          </div>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <div>
            <h2 class="card-title">Ask a question</h2>
            <div class="card-subtitle">Choose a document question or write your own</div>
          </div>
        </div>

        <div class="scenario-grid" id="scenarioGrid"></div>

        <div class="search-box">
          <input id="qInput" class="search-input" type="text" placeholder="Ask about a document or table" />
          <button id="runQueryBtn" class="btn btn-primary">Run query</button>
        </div>
      </div>

      <div id="resultsContainer" style="display:none;">
        <div class="card">
          <div class="card-header">
            <h2 class="card-title">Verified answer</h2>
          </div>
          <div id="telemetryStrip" class="metrics-strip"></div>
          <div id="answerBanner" class="answer-banner"></div>

          <div id="deltasWrapper" style="display:none; margin-bottom: 20px;">
            <div class="section-label">Cross-column changes</div>
            <div id="deltasGrid" class="delta-grid"></div>
          </div>

          <div class="grid-2">
            <div>
              <h3 class="section-label">Matched table coordinates</h3>
              <table class="data-table">
                <thead>
                  <tr>
                    <th>Row Metric Key</th>
                    <th>Column Header</th>
                    <th>Verified Cell Value</th>
                    <th>Filing &amp; Page</th>
                  </tr>
                </thead>
                <tbody id="coordTableBody"></tbody>
              </table>
            </div>

            <div>
              <h3 class="section-label">Source evidence</h3>
              <div id="citationsContainer"></div>
            </div>
          </div>

          <details class="raw-response">
            <summary>View raw response</summary>
            <pre id="rawJsonBox" class="code-block"></pre>
          </details>
        </div>
      </div>
    </section>

    <!-- TAB 2: KNOWLEDGE BASE INSPECTOR, PDF UPLOAD & TEMPLATES -->
    <section id="tab-kb" class="tab-view">
      <div class="kb-layout">
        <div>
          <div class="card">
            <div class="card-header">
              <div>
                <h2 class="card-title">Indexed Knowledge Base</h2>
                <div class="card-subtitle">Inspect extracted pages and tables from the indexed corpus</div>
              </div>
            </div>
            <div id="kbDocList"></div>
          </div>

          <div class="card">
            <div class="card-header">
              <div>
                <h2 class="card-title">Upload a PDF</h2>
                <div class="card-subtitle">Extract narrative text and tables with PyMuPDF</div>
              </div>
            </div>
            <div class="form-group">
              <input type="file" id="pdfFileInput" accept=".pdf" class="input-field" />
            </div>
            <button id="uploadPdfBtn" class="btn btn-primary full-width">Upload and index PDF</button>
            <div id="pdfUploadStatus" class="status-text"></div>
          </div>

          <div class="card">
            <div class="card-header">
              <div>
                <h2 class="card-title">Save a document</h2>
                <div class="card-subtitle">Load a sample or enter text and table data</div>
              </div>
            </div>
            <div class="template-actions">
              <button class="btn btn-secondary template-btn" data-template="sla">Load cloud SLA sample</button>
              <button class="btn btn-secondary template-btn" data-template="lab">Load clinical lab sample</button>
            </div>
            <div class="form-group">
              <label class="form-label">Document Identifier</label>
              <input id="docNameInput" class="input-field" type="text" placeholder="e.g. CLOUD_SLA_2026" />
            </div>
            <div class="form-group">
              <label class="form-label">Page content and table rows</label>
              <textarea id="docTextInput" class="input-field mono" rows="5" placeholder="Paste text or table rows"></textarea>
            </div>
            <div class="action-row">
              <button id="saveDocBtn" class="btn btn-primary flex-button">Save document</button>
            </div>
            <div id="crudStatus" class="status-text"></div>
          </div>
        </div>

        <div>
          <div class="card">
            <div class="card-header">
              <div>
                <h2 class="card-title" id="inspectorTitle">Document Inspector</h2>
                <div class="card-subtitle" id="inspectorSubtitle">Select a document to view its indexed pages and rendered tables</div>
              </div>
              <div id="inspectorActions"></div>
            </div>
            <div id="inspectorContent"></div>
          </div>
        </div>
      </div>
    </section>

    <!-- TAB 3: BENCHMARKS & SYSTEM TELEMETRY -->
    <section id="tab-bench" class="tab-view">
      <div class="grid-2">
        <div class="card">
          <div class="card-header">
            <div>
              <h2 class="card-title">Live Retrieval Ablation Study</h2>
              <div class="card-subtitle">Scores are loaded from the current indexed corpus and real BGE model run:</div>
            </div>
              <button id="runBenchmarkBtn" class="btn btn-primary btn-compact">Run live benchmark</button>
          </div>
          <div id="benchmarkStatus" class="metric-pill">Loading latest evaluation</div>
          <div id="benchmarkBars" class="benchmark-bars"></div>
          <div id="benchmarkSummary" class="metrics-strip benchmark-summary"></div>
          <div class="benchmark-table-wrap">
            <h3 class="section-label">Per-query rank comparison</h3>
            <div id="perQueryTable"></div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">
            <div>
              <h2 class="card-title">Live Engine Telemetry &amp; Configuration</h2>
              <div class="card-subtitle">Real-time state from <code>GET /health</code> and <code>GET /metrics</code>:</div>
            </div>
              <a href="/docs" target="_blank" class="btn btn-secondary api-link">Open API docs &nearr;</a>
          </div>
      <pre id="metricsJsonBox" class="code-block">Loading telemetry</pre>
        </div>
      </div>
    </section>
  </main>
  <footer class="workbench-footer">Table coordinates, citations, and verification · UI refinement guided by unslop by Damir Sokolovsky</footer>

  <script>
    let scenarios = [];

    const TEMPLATES = {
      sla: {
        doc_name: "CLOUD_SLA_2026",
        text: "ENTERPRISE CLOUD INFRASTRUCTURE SLA SCHEDULE (2026)\n| Service Tier | Guaranteed Uptime SLO | Max Failover Latency | Monthly Service Credit |\n| --- | --- | --- | --- |\n| Mission Critical Tier | 99.995% | 15 ms | 35.0% |\n| Production Standard Tier | 99.95% | 85 ms | 15.0% |",
        sample_question: "In the Enterprise Cloud Infrastructure SLA Schedule table, what are the Guaranteed Uptime SLO and Monthly Service Credit for the Mission Critical Tier?"
      },
      lab: {
        doc_name: "CARDIOLOGY_LAB_2026",
        text: "CLINICAL BIOMARKER REFERENCE TABLE\n| Biomarker Panel | Treatment Group | Placebo Group | Reduction % |\n| --- | --- | --- | --- |\n| LDL Cholesterol (mg/dL) | 68.4 mg/dL | 134.2 mg/dL | 49.0% |\n| hs-CRP Inflammation (mg/L) | 1.1 mg/L | 3.4 mg/L | 67.6% |",
        sample_question: "In the Clinical Biomarker Reference Table, what was the LDL Cholesterol (mg/dL) for the Treatment Group vs Placebo Group?"
      }
    };

    function escapeHtml(str) {
      return String(str || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
    }

    function formatChunkContent(rawText) {
      const lines = String(rawText || "").split("\n");
      let html = "";
      let tableRows = [];

      function flushTable() {
        if (tableRows.length < 2) {
          tableRows.forEach(r => { html += "<div>" + escapeHtml(r.join(" | ")) + "</div>"; });
          tableRows = [];
          return;
        }
        html += '<table class="data-table" style="margin:8px 0;"><thead><tr>';
        tableRows[0].forEach(h => { html += "<th>" + escapeHtml(h) + "</th>"; });
        html += "</tr></thead><tbody>";
        for (let i = 1; i < tableRows.length; i++) {
          html += "<tr>";
          tableRows[i].forEach(c => { html += '<td class="mono">' + escapeHtml(c) + "</td>"; });
          html += "</tr>";
        }
        html += "</tbody></table>";
        tableRows = [];
      }

      lines.forEach(line => {
        const s = line.trim();
        if (s.startsWith("|") && s.endsWith("|")) {
          if (/^\|[\s\-:|]+\|$/.test(s)) return;
          const cells = s.slice(1, -1).split("|").map(c => c.trim());
          tableRows.push(cells);
        } else {
          if (tableRows.length) flushTable();
          if (s) html += '<div style="margin-bottom:4px;">' + escapeHtml(s) + "</div>";
        }
      });
      if (tableRows.length) flushTable();
      return html;
    }

    // Tab switching
    document.querySelectorAll(".tab-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
        document.querySelectorAll(".tab-view").forEach(v => v.classList.remove("active"));
        btn.classList.add("active");
        const target = document.getElementById(btn.getAttribute("data-tab"));
        if (target) target.classList.add("active");
        if (btn.getAttribute("data-tab") === "tab-kb") loadDocumentsList();
        if (btn.getAttribute("data-tab") === "tab-bench") {
          loadTelemetry();
          loadBenchmark();
        }
      });
    });

    async function loadScenarios() {
      const res = await fetch("/scenarios");
      if (!res.ok) return;
      const data = await res.json();
      scenarios = data.scenarios || [];
      const scenarioGrid = document.getElementById("scenarioGrid");
      scenarioGrid.innerHTML = scenarios.length
        ? ""
        : '<div class="card-subtitle">No indexed scenarios are available yet</div>';

      scenarios.forEach(sc => {
        const btn = document.createElement("button");
        btn.className = "scenario-card";
        const sourceLabel = {
          benchmark: "Benchmark question",
          live_table_schema: "Live table schema",
          indexed_document: "Indexed document"
        }[sc.source] || "Indexed question";
        btn.innerHTML =
          '<span class="scenario-tag">' + escapeHtml(sc.doc_name) + " · p." + sc.page_number + "</span>" +
          '<div class="scenario-name">' + escapeHtml(sourceLabel) + "</div>" +
          '<div class="scenario-q">' + escapeHtml(sc.question) + "</div>";
        btn.addEventListener("click", () => {
          document.getElementById("qInput").value = sc.question;
          executeQuery();
        });
        scenarioGrid.appendChild(btn);
      });

      if (!document.getElementById("qInput").value && scenarios.length) {
        document.getElementById("qInput").value = scenarios[0].question;
      }
    }

    document.getElementById("runQueryBtn").addEventListener("click", executeQuery);
    document.getElementById("qInput").addEventListener("keydown", e => {
      if (e.key === "Enter") executeQuery();
    });

    async function executeQuery() {
      const q = document.getElementById("qInput").value.trim();
      if (!q) return;

      const btn = document.getElementById("runQueryBtn");
      btn.textContent = "Running";
      try {
        const res = await fetch("/query", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({question: q, top_k: 3, use_cache: true})
        });
        const data = await res.json();
        document.getElementById("resultsContainer").style.display = "block";
        document.getElementById("rawJsonBox").textContent = JSON.stringify(data, null, 2);

        const strip = document.getElementById("telemetryStrip");
        const banner = document.getElementById("answerBanner");

        if (!res.ok) {
          strip.innerHTML = '<span class="metric-pill blocked">HTTP ' + res.status + " · Input guardrail</span>";
          banner.className = "answer-banner blocked";
          const rules = (data.detail && data.detail.triggered_rules) ? data.detail.triggered_rules.join(", ") : "policy rule";
          banner.textContent = "Query blocked · Triggered rule: " + rules;
          document.getElementById("deltasWrapper").style.display = "none";
          document.getElementById("coordTableBody").innerHTML = '<tr><td colspan="4">Blocked before retrieval</td></tr>';
          document.getElementById("citationsContainer").innerHTML = "";
          return;
        }

        const v = data.verification || {};
        const pct = ((v.numeric_grounding_score || 1) * 100).toFixed(0);
        const figs = (v.verified_figures || []).join(", ") || "none listed";

        strip.innerHTML =
          '<span class="metric-pill verified">Numeric grounding: ' + pct + "%</span>" +
          '<span class="metric-pill">Verified figures: ' + escapeHtml(figs) + "</span>" +
          '<span class="metric-pill">Latency: <b>' + data.latency_ms + " ms</b></span>" +
          '<span class="metric-pill">Semantic cache: <b>' + (data.cache_hit ? "hit (" + data.cache_similarity + ")" : "miss") + "</b></span>";

        banner.className = "answer-banner";
        banner.textContent = data.answer;

        const deltas = data.computed_deltas || [];
        const deltasWrapper = document.getElementById("deltasWrapper");
        const deltasGrid = document.getElementById("deltasGrid");
        if (deltas.length > 0) {
          deltasWrapper.style.display = "block";
          deltasGrid.innerHTML = deltas.map(d =>
            '<div class="delta-card">' +
              '<div class="delta-title">' + escapeHtml(d.row_key) + " (" + escapeHtml(d.comparison) + ")</div>" +
              '<div class="delta-val">' + escapeHtml(d.absolute_change) + " (" + escapeHtml(d.pct_change) + ")</div>" +
            "</div>"
          ).join("");
        } else {
          deltasWrapper.style.display = "none";
        }

        const cells = data.matched_cells || [];
        document.getElementById("coordTableBody").innerHTML = cells.length
          ? cells.map(c =>
              "<tr>" +
                "<td><b>" + escapeHtml(c.row_key) + "</b></td>" +
                '<td class="mono">' + escapeHtml(c.col_header) + "</td>" +
                '<td class="mono"><b>' + escapeHtml(c.value) + "</b></td>" +
                '<td class="mono">' + escapeHtml(c.doc_name) + " · p." + c.page_number + "</td>" +
              "</tr>"
            ).join("")
          : '<tr><td colspan="4">No table coordinates in the retrieved evidence</td></tr>';

        const citations = data.citations || [];
        document.getElementById("citationsContainer").innerHTML = citations.map(c =>
          '<div class="source-card">' +
            '<div class="source-header">' +
              "<b>[" + escapeHtml(c.doc_name) + " · Page " + c.page_number + "]</b>" +
              "<span>SCAR score: " + c.rerank_score + "</span>" +
            "</div>" +
            "<div>" + formatChunkContent(c.excerpt) + "</div>" +
          "</div>"
        ).join("");
      } finally {
        btn.textContent = "Run query";
      }
    }

    const benchmarkModes = [
      {key: "dense_only", label: "Dense-only RAG", color: "var(--text-muted)"},
      {key: "hybrid_rrf", label: "Hybrid RRF", color: "var(--accent)"},
      {key: "hybrid_plus_reranker", label: "Adaptive SCAR", color: "var(--success-text)"}
    ];

    function rankLabel(rank) {
      return rank === null || rank === undefined ? "—" : "#" + rank;
    }

    function renderBenchmark(report) {
      const bars = document.getElementById("benchmarkBars");
      const summary = document.getElementById("benchmarkSummary");
      const status = document.getElementById("benchmarkStatus");
      const breakdown = report.per_query_breakdown || [];

      bars.innerHTML = benchmarkModes.map(mode => {
        const metric = report[mode.key] || {};
        const precision = Number(metric.precision_at_1 || 0);
        const width = Math.max(0, Math.min(100, precision * 100));
        return '<div class="bar-row">' +
          '<div class="bar-label"><span>' + escapeHtml(mode.label) +
          '</span><span class="mono">' + (width).toFixed(1) + "% (" + precision.toFixed(2) + ")</span></div>" +
          '<div class="bar-track"><div class="bar-fill" style="width:' + width + "%; background:" + mode.color + ';"></div></div>' +
          '<div style="font-size:11px; color:var(--text-secondary); margin-top:3px;">Mean latency: ' +
          Number(metric.avg_latency_ms || 0).toFixed(2) + " ms · MRR: " + Number(metric.mrr || 0).toFixed(3) + "</div>" +
          "</div>";
      }).join("");

      const queryCount = report.dense_only ? report.dense_only.queries_evaluated : breakdown.length;
      status.textContent = "Live evaluation · " + queryCount + " queries · " + new Date().toLocaleTimeString();
      summary.innerHTML = benchmarkModes.map(mode => {
        const metric = report[mode.key] || {};
        return '<span class="metric-pill"><b>' + escapeHtml(mode.label) + "</b> · " +
          Number(metric.avg_latency_ms || 0).toFixed(2) + " ms · MRR " + Number(metric.mrr || 0).toFixed(3) + "</span>";
      }).join("");

      document.getElementById("perQueryTable").innerHTML = breakdown.length
        ? '<table class="data-table"><thead><tr><th>Document / Page</th><th>Question</th>' +
          benchmarkModes.map(mode => "<th>" + escapeHtml(mode.label) + " Rank</th>").join("") +
          "</tr></thead><tbody>" + breakdown.map(item =>
            "<tr><td class=\"mono\"><b>" + escapeHtml(item.doc_name) + "</b><br>p." + item.expected_page + "</td>" +
            "<td>" + escapeHtml(item.question) + "</td>" +
            benchmarkModes.map(mode => "<td class=\"mono\">" + rankLabel(item.ranks && item.ranks[mode.key]) + "</td>").join("") +
            "</tr>"
          ).join("") + "</tbody></table>"
        : '<div class="card-subtitle">No per-query evaluation rows are available</div>';
    }

    async function loadBenchmark() {
      const res = await fetch("/evaluate/latest");
      if (!res.ok) {
        document.getElementById("benchmarkStatus").textContent = "Latest evaluation unavailable";
        return;
      }
      renderBenchmark(await res.json());
    }

    document.getElementById("runBenchmarkBtn").addEventListener("click", async () => {
      const btn = document.getElementById("runBenchmarkBtn");
      btn.disabled = true;
      btn.textContent = "Running benchmark";
      document.getElementById("benchmarkStatus").textContent = "Evaluating the current indexed corpus";
      try {
        const res = await fetch("/evaluate", {method: "POST"});
        if (res.ok) renderBenchmark(await res.json());
        else document.getElementById("benchmarkStatus").textContent = "Live evaluation failed";
      } finally {
        btn.disabled = false;
        btn.textContent = "Run live benchmark";
      }
    });

    // Knowledge Base Management
    async function loadDocumentsList(selectedDoc) {
      const res = await fetch("/documents");
      const data = await res.json();
      const docs = data.documents || [];
      document.getElementById("headerStatusText").textContent =
        "Engine online · " + docs.length + " documents indexed";

      const listEl = document.getElementById("kbDocList");
      listEl.innerHTML = "";
      docs.forEach((d, idx) => {
        const row = document.createElement("div");
        const isTarget = selectedDoc ? d.doc_name === selectedDoc : idx === 0;
        row.className = "doc-row" + (isTarget ? " active" : "");
        row.innerHTML =
          "<div>" +
            '<div class="mono doc-name">' + escapeHtml(d.doc_name) + "</div>" +
            '<div class="doc-meta">' +
              d.total_chunks + " chunks · " + d.table_chunks + " tables · Pages: " + d.pages.join(", ") +
            "</div>" +
          "</div>" +
          '<button class="btn btn-secondary template-btn">Inspect &rarr;</button>';
        row.addEventListener("click", () => inspectDocument(d.doc_name));
        listEl.appendChild(row);
        if (isTarget) inspectDocument(d.doc_name);
      });
    }

    async function inspectDocument(docName) {
      const res = await fetch("/documents/" + encodeURIComponent(docName));
      if (!res.ok) return;
      const data = await res.json();
      document.getElementById("inspectorTitle").textContent = "Document: " + data.doc_name;
      document.getElementById("inspectorSubtitle").textContent =
        data.total_chunks + " indexed chunks (" + data.table_chunks + " table-bearing chunks)";

      const actionsEl = document.getElementById("inspectorActions");
      actionsEl.innerHTML = "";
      const delBtn = document.createElement("button");
      delBtn.className = "btn btn-danger btn-compact";
      delBtn.textContent = "Delete document";
      delBtn.addEventListener("click", async () => {
        await fetch("/documents/" + encodeURIComponent(docName), {method: "DELETE"});
        await loadScenarios();
        loadDocumentsList();
      });
      actionsEl.appendChild(delBtn);

      const contentEl = document.getElementById("inspectorContent");
      contentEl.innerHTML = (data.chunks || []).map(c =>
        '<div class="source-card">' +
          '<div class="source-header">' +
            "<b>Page " + c.page_number + " (Chunk " + c.chunk_id + ")</b>" +
            "<span>Contains Table: " + (c.contains_table ? "Yes" : "No") + "</span>" +
          "</div>" +
          "<div>" + formatChunkContent(c.text) + "</div>" +
        "</div>"
      ).join("");
    }

    // Templates
    document.querySelectorAll("[data-template]").forEach(btn => {
      btn.addEventListener("click", () => {
        const key = btn.getAttribute("data-template");
        const t = TEMPLATES[key];
        if (!t) return;
        document.getElementById("docNameInput").value = t.doc_name;
        document.getElementById("docTextInput").value = t.text;
        document.getElementById("qInput").value = t.sample_question;
      });
    });

    document.getElementById("saveDocBtn").addEventListener("click", async () => {
      const name = document.getElementById("docNameInput").value.trim() || "CUSTOM_DOC_2026";
      const text = document.getElementById("docTextInput").value.trim();
      if (!text) return;
      const res = await fetch("/documents/" + encodeURIComponent(name), {
        method: "PUT",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({doc_name: name, pages: [{page_number: 1, text: text}]})
      });
      if (res.ok) {
        document.getElementById("crudStatus").textContent = "Saved and indexed " + name;
        await loadScenarios();
        loadDocumentsList(name);
      }
    });

    // PDF Upload via Base64
    document.getElementById("uploadPdfBtn").addEventListener("click", () => {
      const fileInput = document.getElementById("pdfFileInput");
      const statusEl = document.getElementById("pdfUploadStatus");
      if (!fileInput.files || !fileInput.files[0]) {
        statusEl.textContent = "Select a PDF file first";
        return;
      }
      const file = fileInput.files[0];
      statusEl.textContent = "Parsing PDF tables and indexing";
      const reader = new FileReader();
      reader.onload = async () => {
        const base64Str = String(reader.result).split(",")[1] || "";
        const res = await fetch("/ingest/pdf", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({doc_name: file.name, pdf_base64: base64Str})
        });
        const out = await res.json();
        if (res.ok) {
          statusEl.textContent = "Indexed " + out.doc_name + " · " + out.chunks_created + " chunks · " + out.table_chunks + " tables";
          await loadScenarios();
          loadDocumentsList(out.doc_name);
        } else {
          statusEl.textContent = "Error · " + (out.detail || "Upload failed");
        }
      };
      reader.readAsDataURL(file);
    });

    async function loadTelemetry() {
      const [h, m] = await Promise.all([fetch("/health"), fetch("/metrics")]);
      const info = {health: await h.json(), metrics: await m.json()};
      document.getElementById("metricsJsonBox").textContent = JSON.stringify(info, null, 2);
    }

    // Initial startup load
    loadScenarios().then(() => {
      loadDocumentsList();
      if (scenarios.length) executeQuery();
    });
  </script>
</body>
</html>
"""
