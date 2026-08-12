(function () {
  "use strict";

  var isChinese = document.documentElement.lang.toLowerCase().startsWith("zh");
  var config = window.CORPCHECK_CONFIG || {};
  var isLocal = ["localhost", "127.0.0.1"].includes(location.hostname);
  var apiBase = String(config.API_BASE_URL || (isLocal ? "http://127.0.0.1:8000" : "")).replace(/\/$/, "");
  var form = document.getElementById("evidence-form");
  if (!form) return;

  var query = document.getElementById("query");
  var panel = document.getElementById("result-panel");
  var empty = document.getElementById("result-empty");
  var loading = document.getElementById("result-loading");
  var content = document.getElementById("result-content");
  var mode = document.getElementById("demo-mode");
  var submit = form.querySelector("button[type=submit]");

  var copy = isChinese ? {
    pass: "证据通过门槛",
    fail: "证据不足，拒绝回答",
    llm: "LLM 调用",
    retrieved: "返回证据",
    top1: "Top-1 cosine",
    mean3: "Mean top-3",
    floor: "门槛",
    evidence: "检索到的证据",
    source: "查看 SEC 原文 ↗",
    noEvidence: "没有可展示的证据块。",
    live: "<strong>Live API mode</strong> · 当前结果来自本地 /answerability。",
    recorded: "<strong>Recorded evidence mode</strong> · 公共 API 尚未开放；这里展示 2026-08-12 从当前语料快照验证并记录的结果。自定义问题需要本地 API。",
    customUnavailable: "公共 API 尚未开放",
    customHelp: "当前公开页面只能回放两个已验证场景。运行本地 API 后，可以检查任意问题。",
    retry: "无法连接 evidence API",
    retryHelp: "确认 FastAPI 已在 127.0.0.1:8000 启动，或在 config.js 中配置公开 API origin。",
    unknown: "未知"
  } : {
    pass: "Evidence cleared the gate",
    fail: "Evidence too weak — refused",
    llm: "LLM consulted",
    retrieved: "Evidence returned",
    top1: "Top-1 cosine",
    mean3: "Mean top-3",
    floor: "floor",
    evidence: "Retrieved evidence",
    source: "Inspect SEC source ↗",
    noEvidence: "No evidence blocks to display.",
    live: "<strong>Live API mode</strong> · This result came from the local /answerability endpoint.",
    recorded: "<strong>Recorded evidence mode</strong> · The public API is not open yet. These results were verified against the current corpus snapshot on 2026-08-12. Custom questions require the local API.",
    customUnavailable: "Public API is not open yet",
    customHelp: "This public page can replay two verified scenarios. Run the local API to inspect a custom question.",
    retry: "Could not reach the evidence API",
    retryHelp: "Start FastAPI on 127.0.0.1:8000, or configure a public API origin in config.js.",
    unknown: "Unknown"
  };

  var recorded = {
    "What was Apple's total net sales in fiscal 2022?": {
      query: "What was Apple's total net sales in fiscal 2022?",
      answerable: true,
      gate_status: "pass",
      reason: "Evidence passed both confidence floors.",
      llm_consulted: false,
      similarity: { top1_cos_sim: 0.70896, mean_top3_cos_sim: 0.6743513333333334, top1_min: 0.42, mean_top3_min: 0.4 },
      coverage: { retrieved: 5, companies: ["AAPL"], filing_types: ["10-K", "10-Q"], fiscal_years: [2022] },
      chunks: [{ company: "AAPL", filing_type: "10-K", fiscal_year: 2022, filed_date: "2022-10-28", accession_number: "0000320193-22-000108", display_title: "Selected Financial Data", text: "Fiscal 2022 Highlights — Total net sales increased 8% or $28.5 billion during 2022 compared to 2021, driven primarily by higher net sales of iPhone, Services and Mac.", source_url: "https://www.sec.gov/Archives/edgar/data/320193/000032019322000108/0000320193-22-000108-index.htm", cos_sim: 0.70896 }]
    },
    "What is the best recipe for sourdough bread?": {
      query: "What is the best recipe for sourdough bread?",
      answerable: false,
      gate_status: "below_top1_floor",
      reason: "The filings searched do not contain passages relevant enough to answer this question.",
      llm_consulted: false,
      similarity: { top1_cos_sim: 0.295165, mean_top3_cos_sim: 0.29333866666666664, top1_min: 0.42, mean_top3_min: 0.4 },
      coverage: { retrieved: 5, companies: ["MPC"], filing_types: ["10-K"], fiscal_years: [2020, 2023, 2024, 2025] },
      chunks: []
    }
  };

  function escapeHtml(value) {
    return String(value == null ? "" : value).replace(/[&<>'"]/g, function (char) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char];
    });
  }

  function safeSourceUrl(value) {
    try {
      var url = new URL(value);
      return url.protocol === "https:" && url.hostname.endsWith("sec.gov") ? url.href : "";
    } catch (_error) { return ""; }
  }

  function number(value) { return typeof value === "number" ? value.toFixed(3) : "—"; }

  function setState(state) {
    empty.hidden = state !== "empty";
    loading.hidden = state !== "loading";
    content.hidden = state !== "content";
    panel.setAttribute("aria-busy", state === "loading" ? "true" : "false");
    submit.disabled = state === "loading";
  }

  function metric(label, value, floor) {
    return '<div class="metric"><span>' + escapeHtml(label) + '</span><b>' + escapeHtml(value) + '</b>' + (floor == null ? "" : '<small>' + copy.floor + " " + escapeHtml(floor.toFixed(2)) + "</small>") + "</div>";
  }

  function evidenceCard(chunk) {
    var url = safeSourceUrl(chunk.source_url);
    var tags = [chunk.company, chunk.filing_type, chunk.fiscal_year ? "FY" + chunk.fiscal_year : null, chunk.filed_date].filter(Boolean);
    return '<article class="evidence-card"><div class="evidence-meta">' + tags.map(function (tag) { return '<span class="tag">' + escapeHtml(tag) + "</span>"; }).join("") + '</div><h4>' + escapeHtml(chunk.display_title || chunk.article_title || chunk.accession_number || copy.evidence) + '</h4><p>' + escapeHtml(chunk.text || "") + "</p>" + (chunk.accession_number ? '<div class="source-link">' + escapeHtml(chunk.accession_number) + "</div>" : "") + (url ? '<br><a class="source-link" href="' + escapeHtml(url) + '" target="_blank" rel="noopener">' + copy.source + "</a>" : "") + "</article>";
  }

  function render(data, sourceMode) {
    var sim = data.similarity || {};
    var coverage = data.coverage || {};
    var chunks = Array.isArray(data.chunks) ? data.chunks.slice(0, 3) : [];
    var statusClass = data.answerable ? "pass" : "fail";
    content.innerHTML = [
      '<div class="result-head"><div><div class="verdict ', statusClass, '">',
      data.answerable ? copy.pass : copy.fail,
      "</div><h3>", escapeHtml(data.query), '</h3></div><span class="gate-code">',
      escapeHtml(data.gate_status || copy.unknown),
      '</span></div><p class="result-reason">', escapeHtml(data.reason),
      '</p><div class="metrics">',
      metric(copy.top1, number(sim.top1_cos_sim), sim.top1_min),
      metric(copy.mean3, number(sim.mean_top3_cos_sim), sim.mean_top3_min),
      metric(copy.retrieved, String(coverage.retrieved == null ? 0 : coverage.retrieved), null),
      metric(copy.llm, data.llm_consulted ? "Yes" : "No", null),
      '</div><p class="evidence-label">', copy.evidence,
      '</p><div class="evidence-list">',
      chunks.length ? chunks.map(evidenceCard).join("") : '<p class="loading-note">' + copy.noEvidence + "</p>",
      "</div>"
    ].join("");
    mode.innerHTML = sourceMode === "live" ? copy.live : copy.recorded;
    setState("content");
  }

  function renderError(title, detail) {
    content.innerHTML = '<div class="error-box"><h3>' + escapeHtml(title) + "</h3><p>" + escapeHtml(detail) + "</p></div>";
    setState("content");
  }

  document.querySelectorAll(".preset").forEach(function (button) {
    button.addEventListener("click", function () {
      document.querySelectorAll(".preset").forEach(function (item) { item.classList.remove("active"); });
      button.classList.add("active");
      query.value = button.dataset.query;
      query.focus();
    });
  });

  query.addEventListener("input", function () {
    document.querySelectorAll(".preset").forEach(function (button) {
      button.classList.toggle("active", button.dataset.query === query.value.trim());
    });
  });

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    var question = query.value.trim();
    if (!question) return;
    setState("loading");
    mode.textContent = "";

    if (!apiBase) {
      window.setTimeout(function () {
        if (recorded[question]) render(recorded[question], "recorded");
        else renderError(copy.customUnavailable, copy.customHelp);
      }, 650);
      return;
    }

    fetch(apiBase + "/answerability", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "application/json" },
      body: JSON.stringify({ query: question, k: 5 })
    }).then(function (response) {
      if (!response.ok) throw new Error("HTTP " + response.status);
      return response.json();
    }).then(function (data) { render(data, "live"); })
      .catch(function () { renderError(copy.retry, copy.retryHelp); });
  });
})();
