const API = "http://127.0.0.1:5000";
const $ = (id) => document.getElementById(id);

// Runs INSIDE the web page to pull out the job text. Selectors are best-effort; falls back to the whole page.
function extractJob() {
  const selectors = [
    ".jobs-description", ".show-more-less-html", "#jobDescriptionText",
    "[class*='JDC__dang-inner-html']", "[class*='job-desc']", "[class*='jobDescription']",
    "[class*='job-description']", "article", "main",
  ];
  let text = null;
  for (const s of selectors) {
    const el = document.querySelector(s);
    if (el && el.innerText.trim().length > 200) { text = el.innerText; break; }
  }
  const h1 = document.querySelector("h1");
  return { title: h1 ? h1.innerText : document.title, text: (text || document.body.innerText).slice(0, 15000) };
}

async function checkServer() {
  const el = $("server");
  try {
    const r = await fetch(`${API}/health`);
    if (!r.ok) throw new Error();
    el.textContent = "server online"; el.className = "server ok";
  } catch {
    el.textContent = "server offline"; el.className = "server down";
  }
}

function showError(msg) {
  $("result").hidden = true;
  $("error").textContent = msg;
  $("error").hidden = false;
}

function render(d) {
  $("error").hidden = true;
  $("result").hidden = false;
  const label = { low: "Low risk", medium: "Medium risk", high: "High risk" }[d.level];
  $("levelText").textContent = label;
  $("levelText").className = `level ${d.level}`;
  $("scoreText").textContent = `${Math.round(d.score * 100)} / 100`;
  $("marker").style.left = `${Math.min(Math.max(d.score, 0), 1) * 100}%`;

  const ul = $("flags");
  ul.replaceChildren();
  if (!d.flags.length) {
    const li = document.createElement("li");
    li.className = "none";
    li.textContent = "No known scam phrases found.";
    ul.appendChild(li);
  }
  for (const f of d.flags) {
    const li = document.createElement("li");
    const strong = document.createElement("strong");
    strong.textContent = `${f.flag}: “${f.match}”`;
    const span = document.createElement("span");
    span.textContent = f.why;
    li.append(strong, span);
    ul.appendChild(li);
  }
  $("terms").textContent = d.suspicious_terms.length ? d.suspicious_terms.join(", ") : "None stood out.";
}

$("scan").addEventListener("click", async () => {
  const btn = $("scan");
  btn.disabled = true; btn.textContent = "Checking…";
  try {
    let payload;
    const manual = $("manual").value.trim();
    if (manual) {
      payload = { title: "", text: manual };
    } else {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      const [res] = await chrome.scripting.executeScript({ target: { tabId: tab.id }, func: extractJob });
      payload = res.result;
    }
    const r = await fetch(`${API}/analyze`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || "The server could not analyse this page.");
    render(data);
  } catch (e) {
    showError(e.message.includes("fetch")
      ? "Cannot reach the server. Start it with: python backend/app.py"
      : (e.message || "Something went wrong. Open a job posting tab, or paste the text instead."));
  } finally {
    btn.disabled = false; btn.textContent = "Check this job posting";
  }
});

checkServer();
