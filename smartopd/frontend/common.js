// Point this at your deployed backend URL (e.g. https://smartopd-api.onrender.com).
// Kept in localStorage so you can change it from the browser without redeploying.
function getApiBase() {
  return localStorage.getItem("smartopd_api_base") || "http://localhost:8000";
}
function setApiBase(url) {
  localStorage.setItem("smartopd_api_base", url.replace(/\/$/, ""));
}

async function apiFetch(path, options = {}) {
  const token = localStorage.getItem("smartopd_token");
  const headers = options.headers || {};
  if (token) headers["Authorization"] = "Bearer " + token;
  if (options.body && !(options.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }
  const res = await fetch(getApiBase() + path, { ...options, headers });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (e) {}
    throw new Error(detail);
  }
  const ct = res.headers.get("content-type") || "";
  return ct.includes("application/json") ? res.json() : res;
}

function fmtMinutes(m) {
  if (m === null || m === undefined) return "—";
  return `~${Math.round(m)} min`;
}
function fmtTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso + (iso.endsWith("Z") ? "" : "Z"));
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}
