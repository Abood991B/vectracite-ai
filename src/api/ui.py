WORKBENCH_HTML = r"""<!-- UI craft pass informed by unslop by @heydamir, CC BY 4.0 -->
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="theme-color" content="#0d1b2a" />
  <title>VectraCite-AI - PDF Verification Studio</title>
  <link rel="icon" type="image/svg+xml" href="/assets/logo.svg" />
  <style>
    :root {
      --ink: #122033;
      --ink-soft: #42546a;
      --muted: #748296;
      --paper: #f4f5f7;
      --paper-warm: #fbfaf7;
      --surface: #ffffff;
      --line: #d9e0e8;
      --line-dark: #9eacbc;
      --sidebar: #0d1b2a;
      --sidebar-2: #14283d;
      --sidebar-text: #e9f1f7;
      --sidebar-muted: #9aafc1;
      --blue: #2563eb;
      --blue-dark: #1d4ed8;
      --blue-pale: #eaf1ff;
      --green: #0f9f6e;
      --green-pale: #e9f8f1;
      --red: #b64242;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      --sans: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif;
      --shadow: 0 8px 24px rgba(18, 32, 51, 0.07);
      --ease: cubic-bezier(0.22, 1, 0.36, 1);
    }
    * { box-sizing: border-box; }
    html { background: var(--paper); }
    body {
      margin: 0;
      color: var(--ink);
      background: var(--paper);
      font-family: var(--sans);
      font-size: 14px;
      line-height: 1.5;
    }
    button, input, textarea { font: inherit; }
    button, a { -webkit-tap-highlight-color: transparent; }
    button { cursor: pointer; }
    .app-shell { min-height: 100vh; display: grid; grid-template-columns: 320px minmax(0, 1fr); }
    .sidebar {
      min-height: 100vh;
      color: var(--sidebar-text);
      background: var(--sidebar);
      padding: 24px 18px 18px;
      display: flex;
      flex-direction: column;
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: hidden;
    }
    .brand { display: flex; align-items: center; gap: 12px; padding: 0 8px 22px; }
    .brand img { width: 42px; height: 42px; border-radius: 12px; box-shadow: 0 4px 14px rgba(0, 0, 0, 0.22); }
    .brand-name { font-weight: 650; letter-spacing: -0.02em; line-height: 20px; }
    .brand-sub { color: var(--sidebar-muted); font-size: 11px; line-height: 16px; }
    .sidebar-kicker {
      color: var(--sidebar-muted);
      font-family: var(--mono);
      font-size: 10px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      padding: 0 8px 8px;
    }
    .upload-btn {
      display: flex;
      justify-content: center;
      align-items: center;
      width: 100%;
      min-height: 42px;
      border: 1px solid #3d7dff;
      border-radius: 9px;
      color: #fff;
      background: var(--blue);
      font-weight: 600;
      transition: background-color 160ms var(--ease), transform 160ms var(--ease);
    }
    .upload-btn:hover { background: var(--blue-dark); }
    .upload-btn:active, .action-btn:active, .mode-btn:active, .example-chip:active { transform: scale(0.98); }
    .sidebar-scroll { min-height: 0; overflow-y: auto; padding: 20px 0 12px; }
    .scope-item, .doc-card {
      width: 100%;
      border: 1px solid transparent;
      border-radius: 9px;
      color: var(--sidebar-text);
      background: transparent;
      text-align: left;
    }
    .scope-item { padding: 11px 12px; display: flex; align-items: center; gap: 10px; }
    .scope-item:hover, .scope-item.active { background: var(--sidebar-2); border-color: #2d4965; }
    .scope-symbol { width: 20px; color: #81a8ff; font-family: var(--mono); text-align: center; }
    .scope-label { font-weight: 600; }
    .scope-note { display: block; color: var(--sidebar-muted); font-size: 11px; margin-top: 1px; }
    .doc-list { display: grid; gap: 8px; margin-top: 8px; }
    .doc-card {
      padding: 11px 12px;
      border-color: #243b52;
      background: rgba(20, 40, 61, 0.42);
    }
    .doc-card.active { border-color: #5b8cff; background: #173452; }
    .doc-card:hover { border-color: #53708f; }
    .doc-card-head { display: flex; align-items: flex-start; gap: 8px; }
    .doc-filter {
      flex: 1;
      min-width: 0;
      padding: 0;
      border: 0;
      color: var(--sidebar-text);
      background: transparent;
      text-align: left;
    }
    .doc-name { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-family: var(--mono); font-size: 12px; font-weight: 600; }
    .doc-meta { display: block; color: var(--sidebar-muted); font-size: 11px; margin-top: 3px; }
    .doc-delete {
      width: 24px;
      height: 24px;
      padding: 0;
      border: 0;
      border-radius: 5px;
      color: #d2dfeb;
      background: transparent;
      font-size: 18px;
      line-height: 20px;
    }
    .doc-delete:hover { color: #fff; background: rgba(182, 66, 66, 0.7); }
    .doc-actions { display: flex; gap: 6px; margin-top: 9px; }
    .doc-actions a, .doc-actions button {
      border: 0;
      color: #b9d2ff;
      background: transparent;
      font-size: 11px;
      padding: 0;
      text-decoration: none;
    }
    .doc-actions a:hover, .doc-actions button:hover { color: #fff; text-decoration: underline; }
    .sidebar-footer { margin-top: auto; border-top: 1px solid #263d54; padding-top: 14px; }
    .sidebar-actions { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .sidebar-actions button {
      min-height: 34px;
      border: 1px solid #38516b;
      border-radius: 7px;
      color: var(--sidebar-text);
      background: transparent;
      font-size: 11px;
    }
    .sidebar-actions button:hover { background: var(--sidebar-2); border-color: #6d8ca8; }
    .sidebar-status { min-height: 20px; color: var(--sidebar-muted); font-size: 11px; margin-top: 8px; }
    .main-shell { min-width: 0; }
    .studio-header {
      min-height: 74px;
      padding: 16px 34px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.92);
      position: sticky;
      top: 0;
      z-index: 10;
      backdrop-filter: blur(12px);
    }
    .header-title { font-size: 18px; line-height: 24px; font-weight: 650; letter-spacing: -0.025em; }
    .header-note { color: var(--muted); font-size: 12px; margin-top: 2px; }
    .mode-switch { display: flex; gap: 4px; padding: 4px; border: 1px solid var(--line); border-radius: 10px; background: var(--paper); }
    .mode-btn {
      min-height: 34px;
      border: 0;
      border-radius: 7px;
      padding: 0 13px;
      color: var(--ink-soft);
      background: transparent;
      font-size: 12px;
      font-weight: 600;
      transition: color 160ms var(--ease), background-color 160ms var(--ease), transform 160ms var(--ease);
    }
    .mode-btn.active { color: var(--blue-dark); background: var(--surface); box-shadow: 0 1px 4px rgba(18, 32, 51, 0.1); }
    .view { display: none; }
    .view.active { display: block; }
    .workspace { max-width: 1440px; margin: 0 auto; padding: 34px; }
    .question-card, .panel, .benchmark-card {
      border: 1px solid var(--line);
      border-radius: 14px;
      background: var(--surface);
      box-shadow: var(--shadow);
    }
    .question-card { padding: 22px; }
    .eyebrow { color: var(--blue-dark); font-family: var(--mono); font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase; }
    .question-heading { margin: 4px 0 0; font-size: 24px; line-height: 32px; letter-spacing: -0.04em; }
    .question-sub { color: var(--ink-soft); margin-top: 4px; }
    .question-row { display: flex; gap: 10px; margin-top: 18px; }
    .question-input {
      flex: 1;
      min-width: 0;
      min-height: 50px;
      padding: 13px 15px;
      border: 1px solid var(--line-dark);
      border-radius: 9px;
      color: var(--ink);
      background: var(--paper-warm);
      outline: none;
    }
    .question-input:focus, .manual-input:focus { border-color: var(--blue); box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15); }
    .ask-btn, .action-btn {
      min-height: 42px;
      border: 1px solid var(--blue);
      border-radius: 8px;
      padding: 0 17px;
      color: #fff;
      background: var(--blue);
      font-weight: 650;
      transition: background-color 160ms var(--ease), transform 160ms var(--ease);
    }
    .ask-btn { min-height: 50px; }
    .ask-btn:hover, .action-btn:hover { background: var(--blue-dark); }
    .scope-bar { display: flex; align-items: center; gap: 8px; margin-top: 14px; flex-wrap: wrap; }
    .scope-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 9px;
      border: 1px solid #bfd1fb;
      border-radius: 6px;
      color: var(--blue-dark);
      background: var(--blue-pale);
      font-family: var(--mono);
      font-size: 11px;
    }
    .scope-description { color: var(--muted); font-size: 12px; }
    .examples { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-top: 16px; }
    .examples-label { color: var(--muted); font-size: 12px; }
    .example-chip {
      max-width: 300px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      padding: 6px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--ink-soft);
      background: var(--paper);
      font-size: 11px;
      transition: border-color 160ms var(--ease), color 160ms var(--ease), transform 160ms var(--ease);
    }
    .example-chip:hover { color: var(--blue-dark); border-color: #9cb7f5; }
    .inline-status { min-height: 22px; color: var(--muted); font-size: 12px; margin-top: 9px; }
    .inline-status.error { color: var(--red); }
    .inline-status.success { color: var(--green); }
    .verification-grid { display: grid; grid-template-columns: minmax(0, 0.92fr) minmax(0, 1.08fr); gap: 18px; margin-top: 20px; }
    .panel { min-width: 0; padding: 22px; }
    .panel-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 14px; margin-bottom: 16px; }
    .panel-title { margin: 0; font-size: 16px; line-height: 22px; letter-spacing: -0.02em; }
    .panel-sub { color: var(--muted); font-size: 12px; margin-top: 3px; }
    .verified-mark { width: 34px; height: 34px; display: grid; place-items: center; border-radius: 50%; color: var(--green); background: var(--green-pale); font-size: 18px; }
    .answer-box { min-height: 108px; border-left: 4px solid var(--green); border-radius: 8px; padding: 16px 17px; color: var(--ink); background: var(--green-pale); font-size: 16px; line-height: 25px; }
    .answer-box.empty { color: var(--muted); background: var(--paper); border-left-color: var(--line-dark); font-size: 14px; }
    .metric-row { display: flex; flex-wrap: wrap; gap: 0; margin: 17px 0 20px; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }
    .metric { min-width: 120px; padding: 11px 14px 11px 0; margin-right: 14px; border-right: 1px solid var(--line); }
    .metric:last-child { border-right: 0; }
    .metric-label { color: var(--muted); font-size: 10px; letter-spacing: 0.04em; text-transform: uppercase; }
    .metric-value { margin-top: 2px; font-family: var(--mono); font-size: 12px; font-weight: 600; }
    .section-heading { margin: 20px 0 9px; font-size: 12px; letter-spacing: 0.02em; text-transform: uppercase; }
    .delta-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 8px; }
    .delta-card { padding: 11px 12px; border: 1px solid #bfe1d0; border-radius: 8px; background: var(--green-pale); }
    .delta-name { color: var(--ink-soft); font-size: 11px; }
    .delta-value { margin-top: 3px; font-family: var(--mono); font-size: 13px; font-weight: 650; }
    .table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { padding: 9px 7px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
    th { color: var(--muted); font-size: 10px; font-weight: 650; letter-spacing: 0.04em; text-transform: uppercase; }
    td.mono, .mono { font-family: var(--mono); font-variant-numeric: tabular-nums; }
    .source-panel { background: var(--paper-warm); }
    .source-headline { display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }
    .source-title { margin: 0; font-family: var(--mono); font-size: 12px; }
    .source-meta { color: var(--muted); font-size: 11px; margin-top: 3px; }
    .pdf-frame { width: 100%; height: 330px; margin-top: 14px; border: 1px solid var(--line); border-radius: 8px; background: #e9edf1; }
    .source-empty { height: 330px; display: grid; place-items: center; margin-top: 14px; border: 1px dashed var(--line-dark); border-radius: 8px; color: var(--muted); text-align: center; padding: 30px; }
    .source-excerpt { margin-top: 14px; padding: 13px; border: 1px solid var(--line); border-radius: 8px; color: var(--ink-soft); background: var(--surface); font-size: 12px; line-height: 19px; }
    mark { padding: 0 2px; border-radius: 2px; color: var(--ink); background: #ffe58a; }
    .open-pdf { display: inline-flex; align-items: center; min-height: 34px; padding: 0 11px; border: 1px solid #aac3f9; border-radius: 7px; color: var(--blue-dark); background: var(--blue-pale); font-size: 11px; font-weight: 650; text-decoration: none; white-space: nowrap; }
    .open-pdf:hover { border-color: var(--blue); background: #dce8ff; }
    .benchmark-view { max-width: 1440px; margin: 0 auto; padding: 34px; }
    .benchmark-hero { padding: 25px; border: 1px solid var(--line); border-radius: 14px; background: var(--ink); color: #fff; box-shadow: var(--shadow); }
    .benchmark-hero .eyebrow { color: #86aaff; }
    .benchmark-hero h1 { margin: 5px 0 5px; font-size: 25px; line-height: 32px; letter-spacing: -0.04em; }
    .benchmark-hero p { max-width: 700px; margin: 0; color: #bed0df; }
    .benchmark-layout { display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(300px, 0.9fr); gap: 18px; margin-top: 18px; }
    .benchmark-card { padding: 22px; }
    .benchmark-card h2 { margin: 0; font-size: 16px; }
    .metric-definitions { display: grid; gap: 9px; margin-top: 16px; }
    .metric-definition { padding: 11px 12px; border-left: 3px solid var(--blue); background: var(--paper); }
    .metric-definition strong { display: block; font-size: 12px; }
    .metric-definition span { display: block; color: var(--ink-soft); font-size: 12px; margin-top: 2px; }
    .bar-stack { display: grid; gap: 18px; margin-top: 18px; }
    .bar-label { display: flex; justify-content: space-between; gap: 10px; font-size: 12px; font-weight: 650; }
    .bar-track { height: 9px; margin-top: 7px; border-radius: 5px; overflow: hidden; background: #e6ebf0; }
    .bar-fill { height: 100%; border-radius: 5px; transition: width 220ms var(--ease); }
    .bar-note { color: var(--muted); font-size: 11px; margin-top: 5px; }
    .rank-table { overflow-x: auto; margin-top: 14px; }
    .rank-badge { display: inline-flex; padding: 4px 7px; border-radius: 5px; font-size: 10px; font-weight: 650; white-space: nowrap; }
    .rank-badge.found { color: #08704b; background: var(--green-pale); }
    .rank-badge.missed { color: #936019; background: #fff5d9; }
    .model-list { display: grid; gap: 8px; margin-top: 15px; }
    .model-row { padding: 11px 12px; border: 1px solid var(--line); border-radius: 8px; background: var(--paper); }
    .model-role { color: var(--muted); font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em; }
    .model-name { margin-top: 2px; font-family: var(--mono); font-size: 11px; overflow-wrap: anywhere; }
    pre.telemetry { min-height: 250px; margin: 15px 0 0; padding: 14px; overflow: auto; border-radius: 8px; color: #d9e8f2; background: #102437; font-family: var(--mono); font-size: 11px; line-height: 18px; }
    .modal-backdrop { position: fixed; inset: 0; z-index: 30; display: none; place-items: center; padding: 20px; background: rgba(7, 18, 30, 0.66); }
    .modal-backdrop.open { display: grid; }
    .modal { width: min(760px, 100%); max-height: min(760px, 90vh); overflow: auto; padding: 22px; border: 1px solid var(--line); border-radius: 14px; background: var(--surface); box-shadow: 0 20px 70px rgba(0, 0, 0, 0.28); }
    .modal-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 14px; }
    .modal-header h2 { margin: 0; font-size: 17px; }
    .close-btn { width: 32px; height: 32px; border: 1px solid var(--line); border-radius: 7px; color: var(--ink-soft); background: var(--paper); font-size: 18px; }
    .preview-pages { display: grid; gap: 10px; margin-top: 17px; }
    .preview-page { padding: 13px; border: 1px solid var(--line); border-radius: 8px; background: var(--paper-warm); }
    .preview-page-head { display: flex; justify-content: space-between; gap: 10px; color: var(--ink-soft); font-family: var(--mono); font-size: 11px; }
    .preview-page p { margin: 9px 0 0; color: var(--ink-soft); font-size: 12px; white-space: pre-wrap; }
    .sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0; }
    .manual-input { width: 100%; margin-top: 7px; padding: 8px; border: 1px solid #38516b; border-radius: 6px; color: var(--sidebar-text); background: #102437; font-size: 11px; }
    .manual-input::placeholder { color: #7991a7; }
    .sidebar-action-link { display: block; padding: 0; border: 0; color: #b9d2ff; background: transparent; font-size: 11px; text-align: left; }
    .sidebar-action-link:hover { color: #fff; text-decoration: underline; }
    .sidebar-save { width: 100%; min-height: 31px; margin-top: 8px; border: 1px solid #547ca4; border-radius: 6px; color: #fff; background: #193a5a; font-size: 11px; }
    .sidebar-save:hover { background: #245078; }
    :focus-visible { outline: 3px solid rgba(37, 99, 235, 0.45); outline-offset: 2px; }
    @media (max-width: 1050px) {
      .app-shell { grid-template-columns: 270px minmax(0, 1fr); }
      .studio-header, .workspace, .benchmark-view { padding-left: 22px; padding-right: 22px; }
      .verification-grid, .benchmark-layout { grid-template-columns: 1fr; }
    }
    @media (max-width: 760px) {
      .app-shell { display: block; }
      .sidebar { min-height: auto; height: auto; position: relative; padding: 16px; }
      .sidebar-scroll { max-height: 350px; }
      .sidebar-footer { margin-top: 12px; }
      .studio-header { align-items: flex-start; flex-direction: column; padding: 16px; }
      .mode-switch { width: 100%; }
      .mode-btn { flex: 1; }
      .workspace, .benchmark-view { padding: 16px; }
      .question-card, .panel, .benchmark-card, .benchmark-hero { padding: 18px; }
      .question-row { flex-direction: column; }
      .ask-btn { width: 100%; }
      .answer-box { font-size: 14px; line-height: 22px; }
      .pdf-frame, .source-empty { height: 280px; }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { transition-duration: 0.01ms !important; animation-duration: 0.01ms !important; scroll-behavior: auto !important; }
    }
  </style>
</head>
<body>
  <div class="app-shell">
    <aside class="sidebar" aria-label="PDF knowledge base">
      <div class="brand">
        <img src="/assets/logo.svg" alt="VectraCite-AI logo" />
        <div>
          <div class="brand-name">VectraCite-AI</div>
          <div class="brand-sub">PDF verification studio</div>
        </div>
      </div>
      <label class="upload-btn" for="pdfUploadInput">+ Upload Any PDF (.pdf)</label>
      <input id="pdfUploadInput" class="sr-only" type="file" accept=".pdf,application/pdf" />
      <div id="sidebarStatus" class="sidebar-status" role="status"></div>
      <div class="sidebar-scroll">
        <div class="sidebar-kicker">Search scope</div>
        <button id="scopeAll" class="scope-item active" type="button">
          <span class="scope-symbol">∑</span>
          <span><span class="scope-label">All Indexed PDFs</span><span class="scope-note">Search Everything</span></span>
        </button>
        <div class="sidebar-kicker" style="margin-top:18px;">Documents</div>
        <div id="documentList" class="doc-list"></div>
        <details class="manual-drawer" style="margin-top:18px;">
          <summary class="sidebar-kicker" style="cursor:pointer;">Or paste text/table manually</summary>
          <div style="padding:10px 8px 0;">
            <button id="templateSla" class="sidebar-action-link" type="button">Load cloud SLA sample</button>
            <button id="templateLab" class="sidebar-action-link" type="button">Load clinical lab sample</button>
            <input id="manualName" class="manual-input" type="text" placeholder="Document name" />
            <textarea id="manualText" class="manual-input" rows="5" placeholder="Paste page text or a Markdown table"></textarea>
            <button id="manualSave" class="sidebar-save" type="button">Save document</button>
          </div>
        </details>
      </div>
      <div class="sidebar-footer">
        <div class="sidebar-actions">
          <button id="clearDefaults" type="button">Clear Demo PDFs</button>
          <button id="restoreDefaults" type="button">Restore Demo PDFs</button>
        </div>
        <div id="sidebarFooterStatus" class="sidebar-status" role="status"></div>
      </div>
    </aside>

    <div class="main-shell">
      <header class="studio-header">
        <div>
          <div class="header-title">Ask, inspect, verify</div>
          <div class="header-note">Every answer stays connected to a page, a table, and a document scope</div>
        </div>
        <nav class="mode-switch" aria-label="Studio views">
          <button id="modeWorkspace" class="mode-btn active" type="button">Workspace</button>
          <button id="modeBenchmark" class="mode-btn" type="button">System accuracy</button>
        </nav>
      </header>

      <main>
        <section id="workspaceView" class="view active" aria-label="Workspace">
          <div class="workspace">
            <section class="question-card">
              <div class="eyebrow">Question bar</div>
              <h1 class="question-heading">What do you need to verify?</h1>
              <div class="question-sub">Ask about a figure, a paragraph, or a technical detail in the active PDF scope</div>
              <div class="question-row">
                <label class="sr-only" for="questionInput">Question</label>
                <input id="questionInput" class="question-input" type="text" placeholder="Ask a question about your PDFs" />
                <button id="askButton" class="ask-btn" type="button">Ask question</button>
              </div>
              <div class="scope-bar">
                <span id="scopeBadge" class="scope-badge">Searching: All PDFs</span>
                <span class="scope-description">Choose a PDF in the left shelf to narrow the answer</span>
              </div>
              <div id="exampleChips" class="examples"></div>
              <div id="queryStatus" class="inline-status" role="status"></div>
            </section>

            <section class="verification-grid" aria-label="Answer and source verification">
              <article class="panel">
                <div class="panel-header">
                  <div>
                    <h2 class="panel-title">Answer &amp; extracted data</h2>
                    <div class="panel-sub">Deterministic figures, deltas, and coordinates from the response</div>
                  </div>
                  <div class="verified-mark" aria-label="Verification shield">✓</div>
                </div>
                <div id="answerBox" class="answer-box empty">Ask a question to bring the verified answer here.</div>
                <div id="metricRow" class="metric-row" hidden>
                  <div class="metric"><div class="metric-label">Figures verified</div><div id="metricGrounding" class="metric-value">-</div></div>
                  <div class="metric"><div class="metric-label">Latency</div><div id="metricLatency" class="metric-value">-</div></div>
                  <div class="metric"><div class="metric-label">Cache</div><div id="metricCache" class="metric-value">-</div></div>
                </div>
                <div id="deltaSection" hidden>
                  <h3 class="section-heading">Cross-column changes</h3>
                  <div id="deltaGrid" class="delta-grid"></div>
                </div>
                <div id="cellSection" hidden>
                  <h3 class="section-heading">Matched table coordinates</h3>
                  <div class="table-wrap"><table><thead><tr><th>Row</th><th>Column</th><th>Value</th><th>Source</th></tr></thead><tbody id="cellTable"></tbody></table></div>
                </div>
              </article>

              <article class="panel source-panel">
                <div class="panel-header">
                  <div>
                    <h2 class="panel-title">Highlighted Source PDF Page &amp; Table Preview</h2>
                    <div id="sourceMeta" class="panel-sub">The cited page preview appears here after a query</div>
                  </div>
                </div>
                <div id="sourceEmpty" class="source-empty">The exact cited PDF page and table preview will appear here.</div>
                <div id="sourceContent" hidden>
                  <div class="source-headline">
                    <div><h3 id="sourceTitle" class="source-title"></h3><div id="sourcePage" class="source-meta"></div></div>
                    <a id="openPdfLink" class="open-pdf" target="_blank" rel="noopener">Open original PDF at page ↗</a>
                  </div>
                  <iframe id="pdfFrame" class="pdf-frame" title="Cited PDF page preview"></iframe>
                  <div id="sourceExcerpt" class="source-excerpt"></div>
                  <div id="sourceTableWrap" class="table-wrap" style="margin-top:12px;" hidden><table><thead><tr><th>Matched row</th><th>Column</th><th>Value</th></tr></thead><tbody id="sourceTable"></tbody></table></div>
                </div>
              </article>
            </section>
          </div>
        </section>

        <section id="benchmarkView" class="view" aria-label="System accuracy benchmark">
          <div class="benchmark-view">
            <section class="benchmark-hero">
              <div class="eyebrow">System accuracy</div>
              <h1>How well does the studio find the right evidence?</h1>
              <p>Three retrieval stages are measured on the active corpus. The rank table explains each query in plain English, not just as a score.</p>
            </section>
            <div class="benchmark-layout">
              <article class="benchmark-card">
                <h2>Metric guide</h2>
                <div class="metric-definitions">
                  <div class="metric-definition"><strong>Precision@1</strong><span>How often the first result is the exact page we expected.</span></div>
                  <div class="metric-definition"><strong>Recall@3</strong><span>How often the expected page appears somewhere in the first three results.</span></div>
                  <div class="metric-definition"><strong>MRR</strong><span>Rewards finding the right page near the top of the list.</span></div>
                  <div class="metric-definition"><strong>Mean latency</strong><span>Average CPU time for one query after model warmup.</span></div>
                </div>
                <div id="benchmarkStatus" class="inline-status" role="status">Loading benchmark</div>
                <div id="benchmarkBars" class="bar-stack"></div>
              </article>
              <article class="benchmark-card">
                <h2>Active AI models</h2>
                <div id="modelList" class="model-list"></div>
                <pre id="telemetryBox" class="telemetry">Loading system telemetry</pre>
                <button id="refreshBenchmark" class="action-btn" type="button" style="margin-top:12px;">Refresh benchmark</button>
              </article>
            </div>
            <article class="benchmark-card" style="margin-top:18px;">
              <h2>Rank comparison by question</h2>
              <div id="rankTable" class="rank-table"></div>
            </article>
          </div>
        </section>
      </main>
    </div>
  </div>

  <div id="previewBackdrop" class="modal-backdrop" role="presentation">
    <section class="modal" role="dialog" aria-modal="true" aria-labelledby="previewTitle">
      <div class="modal-header">
        <div><h2 id="previewTitle">PDF pages</h2><div id="previewSubtitle" class="panel-sub"></div></div>
        <button id="closePreview" class="close-btn" type="button" aria-label="Close preview">×</button>
      </div>
      <div id="previewPages" class="preview-pages"></div>
    </section>
  </div>

  <script>
    const TEMPLATES = {
      sla: {
        name: "CLOUD_SLA_2026",
        text: "ENTERPRISE CLOUD INFRASTRUCTURE SLA SCHEDULE (2026)\n| Service Tier | Guaranteed Uptime SLO | Max Failover Latency | Monthly Service Credit |\n| --- | --- | --- | --- |\n| Mission Critical Tier | 99.995% | 15 ms | 35.0% |\n| Production Standard Tier | 99.95% | 85 ms | 15.0% |",
        question: "In the Enterprise Cloud Infrastructure SLA Schedule table, what are the Guaranteed Uptime SLO and Monthly Service Credit for the Mission Critical Tier?"
      },
      lab: {
        name: "CARDIOLOGY_LAB_2026",
        text: "CLINICAL BIOMARKER REFERENCE TABLE\n| Biomarker Panel | Treatment Group | Placebo Group | Reduction % |\n| --- | --- | --- | --- |\n| LDL Cholesterol (mg/dL) | 68.4 mg/dL | 134.2 mg/dL | 49.0% |\n| hs-CRP Inflammation (mg/L) | 1.1 mg/L | 3.4 mg/L | 67.6% |",
        question: "In the Clinical Biomarker Reference Table, what was the LDL Cholesterol (mg/dL) for the Treatment Group vs Placebo Group?"
      }
    };
    let documents = [];
    let scenarios = [];
    let selectedDoc = null;
    let lastQuestion = "";

    const byId = id => document.getElementById(id);
    const escapeHtml = value => String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");

    function setStatus(id, message, type = "") {
      const element = byId(id);
      if (!element) return;
      element.textContent = message || "";
      element.className = element.className.replace(/\s(error|success)\b/g, "");
      if (type) element.classList.add(type);
    }

    async function safeFetch(url, options = {}) {
      try {
        const response = await fetch(url, options);
        const raw = await response.text();
        let data = {};
        try {
          data = raw ? JSON.parse(raw) : {};
        } catch (_) {
          data = {detail: raw || "The server returned an unreadable response"};
        }
        if (!response.ok) {
          return {ok: false, status: response.status, data, error: data.detail || "Request failed"};
        }
        return {ok: true, status: response.status, data, error: null};
      } catch (_) {
        return {ok: false, status: 0, data: {}, error: "The server is offline or restarting"};
      }
    }

    async function runUiTask(task, statusId = "queryStatus") {
      try {
        await task();
      } catch (_) {
        setStatus(statusId, "The studio could not complete that action. Try again when the server is ready.", "error");
      }
    }

    function updateScope() {
      const label = selectedDoc ? "Searching only: " + selectedDoc + ".pdf" : "Searching: All PDFs";
      byId("scopeBadge").textContent = label;
      byId("scopeAll").classList.toggle("active", !selectedDoc);
      document.querySelectorAll(".doc-card").forEach(card => {
        card.classList.toggle("active", card.dataset.doc === selectedDoc);
      });
    }

    function renderDocuments() {
      const list = byId("documentList");
      if (!documents.length) {
        list.innerHTML = '<div style="padding:10px 8px;color:var(--sidebar-muted);font-size:12px;">No PDFs are indexed. Restore the demo shelf or upload a PDF.</div>';
        updateScope();
        return;
      }
      list.innerHTML = documents.map(doc => {
        const encoded = encodeURIComponent(doc.doc_name);
        return '<div class="doc-card" data-doc="' + escapeHtml(doc.doc_name) + '">' +
          '<div class="doc-card-head">' +
          '<button class="doc-filter" type="button" data-filter="' + escapeHtml(doc.doc_name) + '">' +
          '<span class="doc-name">' + escapeHtml(doc.doc_name) + '</span>' +
          '<span class="doc-meta">' + doc.pages.length + ' pages · ' + doc.total_chunks + ' chunks · ' + doc.table_chunks + ' tables</span>' +
          '</button>' +
          '<button class="doc-delete" type="button" data-delete="' + escapeHtml(doc.doc_name) + '" aria-label="Delete ' + escapeHtml(doc.doc_name) + '">×</button>' +
          '</div>' +
          '<div class="doc-actions">' +
          '<a href="/documents/' + encoded + '/pdf" target="_blank" rel="noopener">Open PDF ↗</a>' +
          '<button type="button" data-preview="' + escapeHtml(doc.doc_name) + '">Preview Pages</button>' +
          '</div></div>';
      }).join("");
      list.querySelectorAll("[data-filter]").forEach(button => {
        button.addEventListener("click", () => runUiTask(async () => {
          selectedDoc = button.dataset.filter;
          updateScope();
          setStatus("sidebarStatus", "Scope set to " + selectedDoc, "success");
        }, "sidebarStatus"));
      });
      list.querySelectorAll("[data-delete]").forEach(button => {
        button.addEventListener("click", () => runUiTask(async () => {
          const name = button.dataset.delete;
          const result = await safeFetch("/documents/" + encodeURIComponent(name), {method: "DELETE"});
          if (!result.ok) {
            setStatus("sidebarStatus", result.error, "error");
            return;
          }
          if (selectedDoc === name) selectedDoc = null;
          await loadDocuments();
          await loadScenarios();
          setStatus("sidebarStatus", "Removed " + name + " from the active shelf", "success");
        }, "sidebarStatus"));
      });
      list.querySelectorAll("[data-preview]").forEach(button => {
        button.addEventListener("click", () => runUiTask(() => openPreview(button.dataset.preview), "sidebarStatus"));
      });
      updateScope();
    }

    async function loadDocuments() {
      const result = await safeFetch("/documents");
      if (!result.ok) {
        setStatus("sidebarStatus", result.error, "error");
        return;
      }
      documents = result.data.documents || [];
      renderDocuments();
    }

    function renderExamples() {
      const container = byId("exampleChips");
      if (!scenarios.length) {
        container.innerHTML = '<span class="examples-label">No indexed examples yet</span>';
        return;
      }
      container.innerHTML = '<span class="examples-label">Try an example question:</span>' +
        scenarios.slice(0, 4).map((scenario, index) =>
          '<button class="example-chip" type="button" data-example="' + index + '" title="' + escapeHtml(scenario.question) + '">' +
          escapeHtml(scenario.doc_name + " · " + scenario.question) + '</button>'
        ).join("");
      container.querySelectorAll("[data-example]").forEach(button => {
        button.addEventListener("click", () => runUiTask(async () => {
          const scenario = scenarios[Number(button.dataset.example)];
          selectedDoc = scenario.doc_name;
          updateScope();
          byId("questionInput").value = scenario.question;
          await askQuestion(scenario.question);
        }));
      });
    }

    async function loadScenarios() {
      const result = await safeFetch("/scenarios");
      if (!result.ok) {
        setStatus("queryStatus", result.error, "error");
        return;
      }
      scenarios = result.data.scenarios || [];
      renderExamples();
      if (!byId("questionInput").value && scenarios.length) {
        byId("questionInput").value = scenarios[0].question;
      }
    }

    function highlightExcerpt(text, question) {
      let html = escapeHtml(text);
      const terms = [...new Set((String(question).toLowerCase().match(/[a-z0-9]{4,}/g) || []))].slice(0, 8);
      terms.forEach(term => {
        const safeTerm = term.replace(/[.*+?^()|[\]\\]/g, "\\$&");
        html = html.replace(new RegExp("(" + safeTerm + ")", "gi"), "<mark>$1</mark>");
      });
      return html;
    }

    function renderCells(cells) {
      const section = byId("cellSection");
      if (!cells.length) {
        section.hidden = true;
        return;
      }
      section.hidden = false;
      byId("cellTable").innerHTML = cells.map(cell =>
        "<tr><td>" + escapeHtml(cell.row_key) + "</td><td class='mono'>" +
        escapeHtml(cell.col_header) + "</td><td class='mono'>" + escapeHtml(cell.value) +
        "</td><td class='mono'>" + escapeHtml(cell.doc_name) + " · p." + cell.page_number + "</td></tr>"
      ).join("");
    }

    function renderSource(citation, cells) {
      if (!citation) {
        byId("sourceEmpty").hidden = false;
        byId("sourceContent").hidden = true;
        return;
      }
      const pdfUrl = "/documents/" + encodeURIComponent(citation.doc_name) + "/pdf";
      byId("sourceEmpty").hidden = true;
      byId("sourceContent").hidden = false;
      byId("sourceTitle").textContent = citation.doc_name;
      byId("sourcePage").textContent = "Page " + citation.page_number + " · SCAR score " + citation.rerank_score;
      byId("pdfFrame").src = pdfUrl + "#page=" + citation.page_number;
      byId("openPdfLink").href = pdfUrl + "#page=" + citation.page_number;
      byId("openPdfLink").textContent = "Open original PDF at page " + citation.page_number + " ↗";
      byId("sourceExcerpt").innerHTML = highlightExcerpt(citation.excerpt, lastQuestion);
      const matched = cells.filter(cell => cell.doc_name === citation.doc_name && cell.page_number === citation.page_number);
      byId("sourceTableWrap").hidden = !matched.length;
      byId("sourceTable").innerHTML = matched.map(cell =>
        "<tr><td>" + escapeHtml(cell.row_key) + "</td><td class='mono'>" +
        escapeHtml(cell.col_header) + "</td><td class='mono'>" + escapeHtml(cell.value) + "</td></tr>"
      ).join("");
    }

    function renderAnswer(data) {
      byId("answerBox").classList.remove("empty");
      byId("answerBox").textContent = data.answer || "No answer was returned";
      byId("metricRow").hidden = false;
      const verification = data.verification || {};
      byId("metricGrounding").textContent = Math.round(Number(verification.numeric_grounding_score || 0) * 100) + "%";
      byId("metricLatency").textContent = String(data.latency_ms ?? "-") + " ms";
      byId("metricCache").textContent = data.cache_hit ? "hit" : "miss";
      const deltas = data.computed_deltas || [];
      byId("deltaSection").hidden = !deltas.length;
      byId("deltaGrid").innerHTML = deltas.map(delta =>
        '<div class="delta-card"><div class="delta-name">' + escapeHtml(delta.row_key) +
        " · " + escapeHtml(delta.comparison) + '</div><div class="delta-value">' +
        escapeHtml(delta.absolute_change) + " (" + escapeHtml(delta.pct_change) + ")</div></div>"
      ).join("");
      renderCells(data.matched_cells || []);
      renderSource((data.citations || [])[0], data.matched_cells || []);
    }

    async function askQuestion(value = null) {
      const question = String(value ?? byId("questionInput").value).trim();
      if (!question) {
        setStatus("queryStatus", "Write a question first", "error");
        return;
      }
      lastQuestion = question;
      const button = byId("askButton");
      button.disabled = true;
      button.textContent = "Checking";
      setStatus("queryStatus", "Retrieving evidence from the active scope");
      const result = await safeFetch("/query", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({question, top_k: 4, use_cache: true, doc_name: selectedDoc})
      });
      button.disabled = false;
      button.textContent = "Ask question";
      if (!result.ok) {
        setStatus("queryStatus", result.error, "error");
        byId("answerBox").className = "answer-box empty";
        byId("answerBox").textContent = result.error;
        return;
      }
      renderAnswer(result.data);
      setStatus("queryStatus", "Answer verified against the selected evidence", "success");
    }

    function readFileAsDataUrl(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || ""));
        reader.onerror = () => reject(new Error("The selected PDF could not be read"));
        reader.readAsDataURL(file);
      });
    }

    async function uploadPdf() {
      const input = byId("pdfUploadInput");
      const file = input.files && input.files[0];
      if (!file) {
        setStatus("sidebarStatus", "Choose a PDF file first", "error");
        return;
      }
      setStatus("sidebarStatus", "Reading and indexing " + file.name);
      const dataUrl = await readFileAsDataUrl(file);
      const result = await safeFetch("/ingest/pdf", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({doc_name: file.name, pdf_base64: dataUrl.split(",")[1] || ""})
      });
      if (!result.ok) {
        setStatus("sidebarStatus", result.error, "error");
        return;
      }
      selectedDoc = result.data.doc_name;
      input.value = "";
      await loadDocuments();
      await loadScenarios();
      updateScope();
      setStatus("sidebarStatus", "Indexed " + result.data.doc_name + " and set its scope", "success");
    }

    function loadTemplate(key) {
      const template = TEMPLATES[key];
      if (!template) return;
      byId("manualName").value = template.name;
      byId("manualText").value = template.text;
      byId("questionInput").value = template.question;
      setStatus("sidebarStatus", "Sample loaded - save it to index the document", "success");
    }

    async function saveManualDocument() {
      const name = byId("manualName").value.trim() || "CUSTOM_DOC_2026";
      const text = byId("manualText").value.trim();
      if (!text) {
        setStatus("sidebarStatus", "Paste page text or choose a sample first", "error");
        return;
      }
      const result = await safeFetch("/documents/" + encodeURIComponent(name), {
        method: "PUT",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({doc_name: name, pages: [{page_number: 1, text}]})
      });
      if (!result.ok) {
        setStatus("sidebarStatus", result.error, "error");
        return;
      }
      selectedDoc = name;
      await loadDocuments();
      await loadScenarios();
      updateScope();
      setStatus("sidebarStatus", "Saved and indexed " + name, "success");
    }

    async function clearDefaults() {
      const result = await safeFetch("/documents/clear-all", {method: "POST"});
      if (!result.ok) {
        setStatus("sidebarFooterStatus", result.error, "error");
        return;
      }
      selectedDoc = null;
      await loadDocuments();
      scenarios = [];
      renderExamples();
      updateScope();
      setStatus("sidebarFooterStatus", "Demo shelf cleared. Upload your own PDFs to begin.", "success");
    }

    async function restoreDefaults() {
      setStatus("sidebarFooterStatus", "Restoring the default PDF shelf");
      const result = await safeFetch("/documents/restore-defaults", {method: "POST"});
      if (!result.ok) {
        setStatus("sidebarFooterStatus", result.error, "error");
        return;
      }
      selectedDoc = null;
      await loadDocuments();
      await loadScenarios();
      updateScope();
      setStatus("sidebarFooterStatus", "Default PDFs restored", "success");
    }

    async function openPreview(docName) {
      const result = await safeFetch("/documents/" + encodeURIComponent(docName));
      if (!result.ok) {
        setStatus("sidebarStatus", result.error, "error");
        return;
      }
      byId("previewTitle").textContent = docName;
      byId("previewSubtitle").textContent = result.data.total_chunks + " chunks · " + result.data.table_chunks + " table-bearing chunks";
      byId("previewPages").innerHTML = (result.data.chunks || []).map(chunk =>
        '<div class="preview-page"><div class="preview-page-head"><span>Page ' + chunk.page_number +
        " · Chunk " + escapeHtml(chunk.chunk_id) + '</span><a class="open-pdf" target="_blank" rel="noopener" href="/documents/' +
        encodeURIComponent(docName) + '/pdf#page=' + chunk.page_number + '">Open page ↗</a></div><p>' +
        escapeHtml(chunk.text.slice(0, 900)) + "</p></div>"
      ).join("");
      byId("previewBackdrop").classList.add("open");
    }

    function closePreview() {
      byId("previewBackdrop").classList.remove("open");
    }

    function renderBenchmark(report) {
      const modes = [
        {key: "dense_only", label: "Dense-only RAG", color: "#8796a8"},
        {key: "hybrid_rrf", label: "Hybrid RRF", color: "#2563eb"},
        {key: "hybrid_plus_reranker", label: "Adaptive SCAR", color: "#0f9f6e"}
      ];
      byId("benchmarkBars").innerHTML = modes.map(mode => {
        const metric = report[mode.key] || {};
        const precision = Number(metric.precision_at_1 || 0);
        return '<div><div class="bar-label"><span>' + mode.label + '</span><span class="mono">' +
          (precision * 100).toFixed(1) + "% · MRR " + Number(metric.mrr || 0).toFixed(3) +
          '</span></div><div class="bar-track"><div class="bar-fill" style="width:' + (precision * 100) +
          "%;background:" + mode.color + ';"></div></div><div class="bar-note">Mean latency ' +
          Number(metric.avg_latency_ms || 0).toFixed(2) + " ms · " + Number(metric.queries_evaluated || 0) + " queries</div></div>";
      }).join("");
      const rows = report.per_query_breakdown || [];
      byId("rankTable").innerHTML = rows.length ? '<table><thead><tr><th>Document</th><th>Question</th>' +
        modes.map(mode => "<th>" + mode.label + "</th>").join("") + "</tr></thead><tbody>" +
        rows.map(row => "<tr><td class='mono'>" + escapeHtml(row.doc_name) + "<br>p." + row.expected_page +
          "</td><td>" + escapeHtml(row.question) + "</td>" + modes.map(mode => {
            const rank = row.ranks && row.ranks[mode.key];
            return '<td><span class="rank-badge ' + (rank === 1 ? "found" : "missed") + '">' +
              (rank === 1 ? "✅ Found at Rank #1" : "⚠️ Missed") + "</span></td>";
          }).join("") + "</tr>").join("") + "</tbody></table>" :
        '<div class="inline-status">No benchmark rows are available yet</div>';
      setStatus("benchmarkStatus", "Live evaluation loaded at " + new Date().toLocaleTimeString(), "success");
    }

    async function loadBenchmark() {
      const [report, metrics, health] = await Promise.all([
        safeFetch("/evaluate/latest"),
        safeFetch("/metrics"),
        safeFetch("/health")
      ]);
      if (report.ok) renderBenchmark(report.data);
      else setStatus("benchmarkStatus", report.error, "error");
      const info = {health: health.data, metrics: metrics.data};
      byId("telemetryBox").textContent = JSON.stringify(info, null, 2);
      const metricData = metrics.data || {};
      byId("modelList").innerHTML = [
        ["Embedder", metricData.embedding_model || "Unavailable"],
        ["Reranker", metricData.reranker_model || "Unavailable"],
        ["Answer model", metricData.llm_model || "Deterministic fallback"]
      ].map(item => '<div class="model-row"><div class="model-role">' + item[0] + '</div><div class="model-name">' +
        escapeHtml(item[1]) + "</div></div>").join("");
    }

    function setView(view) {
      const workspace = view === "workspace";
      byId("workspaceView").classList.toggle("active", workspace);
      byId("benchmarkView").classList.toggle("active", !workspace);
      byId("modeWorkspace").classList.toggle("active", workspace);
      byId("modeBenchmark").classList.toggle("active", !workspace);
      if (!workspace) runUiTask(loadBenchmark, "benchmarkStatus");
    }

    byId("scopeAll").addEventListener("click", () => runUiTask(async () => {
      selectedDoc = null;
      updateScope();
      setStatus("sidebarStatus", "Scope set to all indexed PDFs", "success");
    }, "sidebarStatus"));
    byId("pdfUploadInput").addEventListener("change", () => runUiTask(uploadPdf, "sidebarStatus"));
    byId("templateSla").addEventListener("click", () => loadTemplate("sla"));
    byId("templateLab").addEventListener("click", () => loadTemplate("lab"));
    byId("manualSave").addEventListener("click", () => runUiTask(saveManualDocument, "sidebarStatus"));
    byId("clearDefaults").addEventListener("click", () => runUiTask(clearDefaults, "sidebarFooterStatus"));
    byId("restoreDefaults").addEventListener("click", () => runUiTask(restoreDefaults, "sidebarFooterStatus"));
    byId("askButton").addEventListener("click", () => runUiTask(askQuestion));
    byId("questionInput").addEventListener("keydown", event => {
      if (event.key === "Enter") runUiTask(askQuestion);
    });
    byId("modeWorkspace").addEventListener("click", () => setView("workspace"));
    byId("modeBenchmark").addEventListener("click", () => setView("benchmark"));
    byId("refreshBenchmark").addEventListener("click", () => runUiTask(loadBenchmark, "benchmarkStatus"));
    byId("closePreview").addEventListener("click", closePreview);
    byId("previewBackdrop").addEventListener("click", event => {
      if (event.target === byId("previewBackdrop")) closePreview();
    });

    async function startStudio() {
      await Promise.all([loadDocuments(), loadScenarios()]);
    }
    startStudio().catch(() => setStatus("sidebarStatus", "The studio is waiting for the API server", "error"));
  </script>
</body>
</html>
"""
