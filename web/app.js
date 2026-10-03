/* NFL Picks: reads data/picks.json (see research/SITE_SCHEMA.md) and renders the weekly picks sheet + model report. */
"use strict";

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const isNum = (x) => typeof x === "number" && Number.isFinite(x);
const pct = (p, d = 0) => (isNum(p) ? (p * 100).toFixed(d) + "%" : "—");
const MINUS = "−";
const signed = (x, d = 1) => (!isNum(x) ? "—" : (x > 0 ? "+" : x < 0 ? MINUS : "") + Math.abs(x).toFixed(d));
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const pts = (x) => { const n = Math.abs(x); return Number.isInteger(n) ? String(n) : n.toFixed(1); };
const fmtLine = (x) => (!isNum(x) ? "" : x === 0 ? "PK" : (x > 0 ? "+" : MINUS) + pts(x));
const BREAK_EVEN = 0.5238;
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

let D = null; // the data file
let MODEL = {}; // id -> model meta
let PRIMARY = "ensemble";
let GAME = {}; // game_id -> game
const state = { view: "picks", week: null, sort: "time", layout: "cards", metric: "su" };
const charts = new Map(); // element -> render fn

const store = {
  get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } },
};

/* ------------------------------------------------------------------ data */

function normalize(d) {
  d = d && typeof d === "object" ? d : {};
  d.teams = d.teams && typeof d.teams === "object" ? d.teams : {};
  d.models = Array.isArray(d.models) ? d.models.filter((m) => m && m.id) : [];
  d.games = Array.isArray(d.games) ? d.games.filter((g) => g && g.home && g.away) : [];
  if (!isNum(d.season)) d.season = d.games.find((g) => isNum(g.season))?.season ?? null;
  d.games = d.games.filter((g) => d.season == null || g.season == null || g.season === d.season);
  d.games.forEach((g, i) => { g.game_id ||= `g${i}`; g.models = g.models && typeof g.models === "object" ? g.models : {}; });
  const gw = [...new Set(d.games.map((g) => g.week).filter(isNum))];
  d.weeks = (Array.isArray(d.weeks) && d.weeks.length ? d.weeks.filter(isNum) : gw).sort((a, b) => a - b);
  for (const w of gw) if (!d.weeks.includes(w)) d.weeks.push(w);
  d.weeks.sort((a, b) => a - b);
  if (!isNum(d.week) || !d.weeks.includes(d.week)) d.week = d.weeks[d.weeks.length - 1] ?? null;
  d.backtest = d.backtest && typeof d.backtest === "object" ? d.backtest : {};

  MODEL = Object.fromEntries(d.models.map((m) => [m.id, m]));
  PRIMARY = d.models.find((m) => m.primary)?.id || (MODEL.ensemble ? "ensemble" : d.models[0]?.id || "ensemble");
  for (const g of d.games) for (const id of Object.keys(g.models)) {
    if (!MODEL[id]) { MODEL[id] = { id, name: id }; d.models.push(MODEL[id]); }
  }
  GAME = Object.fromEntries(d.games.map((g) => [g.game_id, g]));
  return d;
}

const mname = (id) => MODEL[id]?.name || id;
const mshort = (id) => mname(id).replace(/\s*\(.*?\)\s*/g, "").trim() || id;
const team = (t) => D.teams[t] || {};
const tname = (t) => team(t).name || t;
const tnick = (t) => team(t).nick || (team(t).name ? team(t).name.split(" ").pop() : t);

const played = (g) => isNum(g.home_score) && isNum(g.away_score);
const winner = (g) => (!played(g) ? null : g.home_score > g.away_score ? g.home : g.away_score > g.home_score ? g.away : null);

// "win" | "loss" | "push" | null (unplayed / no pick)
function suResult(g) {
  if (!played(g) || !g.su?.pick) return null;
  if (g.su.correct === true) return "win";
  if (g.su.correct === false) return "loss";
  const w = winner(g);
  if (!w) return "push";
  return g.su.correct === undefined ? (w === g.su.pick ? "win" : "loss") : "push";
}
function atsResult(g) {
  if (!played(g) || !g.ats?.pick) return null;
  if (g.ats.correct === true) return "win";
  if (g.ats.correct === false) return "loss";
  if (g.ats.correct === undefined && isNum(g.ats.line)) {
    const m = (g.ats.pick === g.home ? 1 : -1) * (g.home_score - g.away_score) + g.ats.line;
    return m > 0 ? "win" : m < 0 ? "loss" : "push";
  }
  return "push";
}

// how many of the individual models lean the same way as the ensemble's winner pick
function agreement(g) {
  if (!g.su?.pick) return null;
  const pickHome = g.su.pick === g.home;
  const ids = Object.keys(g.models).filter((id) => id !== PRIMARY && isNum(g.models[id]?.p_home));
  if (!ids.length) return isNum(g.agreement) ? { share: g.agreement } : null;
  const k = ids.filter((id) => (g.models[id].p_home >= 0.5) === pickHome).length;
  return { k, n: ids.length, share: k / ids.length };
}

function tally(games) {
  const r = { su: { w: 0, l: 0, t: 0 }, ats: { w: 0, l: 0, p: 0 }, played: 0, total: games.length };
  for (const g of games) {
    if (played(g)) r.played++;
    const s = suResult(g), a = atsResult(g);
    if (s === "win") r.su.w++; else if (s === "loss") r.su.l++; else if (s === "push") r.su.t++;
    if (a === "win") r.ats.w++; else if (a === "loss") r.ats.l++; else if (a === "push") r.ats.p++;
  }
  return r;
}
const rec = (o, third) => `${o.w}-${o.l}${o[third] ? `-${o[third]}` : ""}`;
const rate = (o) => (o.w + o.l ? o.w / (o.w + o.l) : null);

function seasonRecord() {
  const t = tally(D.games);
  const r = D.record;
  if (r && r.su && isNum(r.su.w) && isNum(r.su.l)) t.su = { w: r.su.w, l: r.su.l, t: r.su.t || 0 };
  if (r && r.ats && isNum(r.ats.w) && isNum(r.ats.l)) t.ats = { w: r.ats.w, l: r.ats.l, p: r.ats.p || 0 };
  return t;
}

/* -------------------------------------------------------------- formatting */

function dateObj(g) {
  const [y, m, d] = String(g.date || "").split("-").map(Number);
  return y && m && d ? new Date(Date.UTC(y, m - 1, d, 12)) : null;
}
const fmtDay = (g, opts = { weekday: "short", month: "short", day: "numeric" }) => {
  const d = dateObj(g);
  return d ? d.toLocaleDateString("en-US", { ...opts, timeZone: "UTC" }) : "TBD";
};
function fmtTime(t) {
  const m = /^(\d{1,2}):(\d{2})/.exec(t || "");
  if (!m) return "";
  const h = +m[1];
  return `${((h + 11) % 12) + 1}:${m[2]} ${h < 12 ? "AM" : "PM"}`;
}
const kickoff = (g) => [fmtDay(g), fmtTime(g.time) && fmtTime(g.time) + " ET"].filter(Boolean).join(" · ");
const kickKey = (g) => `${g.date || "9999-99-99"} ${g.time || "99:99"} ${g.game_id}`;

function spreadText(g) {
  const s = g.spread_line;
  if (!isNum(s)) return null;
  if (s === 0) return "Pick'em";
  return s > 0 ? `${g.home} ${MINUS}${pts(s)}` : `${g.away} ${MINUS}${pts(s)}`;
}
function marginText(g, m) {
  if (!isNum(m)) return "—";
  if (Math.abs(m) < 0.05) return "Even";
  return `${m > 0 ? g.home : g.away} by ${Math.abs(m).toFixed(1)}`;
}
function fmtVal(v, fmt) {
  if (v === null || v === undefined || v === "") return "—";
  if (typeof v !== "number") return esc(v);
  if (!Number.isFinite(v)) return "—";
  fmt = typeof fmt === "string" && fmt ? fmt : "0.0";
  const plus = fmt.startsWith("+"), asPct = fmt.endsWith("%"), comma = fmt.includes(",");
  const core = fmt.replace(/[+%,]/g, "");
  const dec = core.includes(".") ? core.split(".")[1].length : 0;
  const x = asPct ? v * 100 : v;
  const s = Math.abs(x).toLocaleString("en-US", { minimumFractionDigits: dec, maximumFractionDigits: dec, useGrouping: comma });
  const zero = Number(Math.abs(x).toFixed(dec)) === 0;
  const sign = zero ? "" : x < 0 ? MINUS : plus ? "+" : "";
  return sign + s + (asPct ? "%" : "");
}
function weatherText(g) {
  const roof = String(g.roof || "").toLowerCase();
  if (roof === "dome" || roof === "closed") return `Indoors (${roof === "dome" ? "dome" : "roof closed"})`;
  const bits = [];
  if (isNum(g.temp)) bits.push(`${Math.round(g.temp)}°F`);
  if (isNum(g.wind)) bits.push(`wind ${Math.round(g.wind)} mph`);
  if (bits.length) return bits.join(" · ") + (roof === "open" ? " (roof open)" : "");
  return roof ? (roof === "open" ? "Roof open" : "Outdoors · forecast pending") : "—";
}
function weatherChip(g) {
  const roof = String(g.roof || "").toLowerCase();
  if (roof === "dome" || roof === "closed") return `<span class="chip">Indoors</span>`;
  if (isNum(g.temp)) return `<span class="chip">${Math.round(g.temp)}°${isNum(g.wind) && g.wind >= 12 ? ` · ${Math.round(g.wind)} mph` : ""}</span>`;
  return "";
}

/* ------------------------------------------------------------------ logos */

const NO_LOCAL = new Set(["LA"]); // no PNG shipped for the Rams
const ESPN = (t) => `https://a.espncdn.com/i/teamlogos/nfl/500/${({ LA: "lar", WAS: "wsh" })[t] || String(t).toLowerCase()}.png`;
const logoSrc = (t) => (NO_LOCAL.has(t) ? team(t).logo || ESPN(t) : `logos/${t}.png`);
function logo(t, cls = "", size = 38) {
  return `<img class="logo ${cls}" src="${esc(logoSrc(t))}" alt="" width="${size}" height="${size}" decoding="async" data-team="${esc(t)}">`;
}
document.addEventListener("error", (e) => {
  const img = e.target;
  if (!(img instanceof HTMLImageElement) || !img.classList.contains("logo")) return;
  const t = img.dataset.team;
  const alts = [team(t).logo, ESPN(t)].filter((u) => u && !img.src.endsWith(u.replace(/^https?:/, "")));
  const n = +(img.dataset.try || 0);
  if (n < alts.length) { img.dataset.try = n + 1; img.src = alts[n]; return; }
  const span = document.createElement("span");
  span.className = img.className + " logo-missing";
  span.style.width = span.style.height = img.getAttribute("width") + "px";
  span.textContent = t;
  img.replaceWith(span);
}, true);

/* --------------------------------------------------------------- snippets */

const ICON_CHECK = `<svg width="11" height="11" viewBox="0 0 12 12" aria-hidden="true"><path d="M2.5 6.4 5 8.8l4.6-5.6" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
const ICON_X = `<svg width="11" height="11" viewBox="0 0 12 12" aria-hidden="true"><path d="m3.2 3.2 5.6 5.6m0-5.6L3.2 8.8" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>`;
const ICON_ARROW = `<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M2 6h7.5M6.5 2.8 9.7 6l-3.2 3.2" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
const ICON_CLOSE = `<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><path d="m3 3 8 8m0-8-8 8" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>`;

function badge(res, what) {
  if (res === "win") return `<span class="res win" title="${what} pick correct">${ICON_CHECK}Hit</span>`;
  if (res === "loss") return `<span class="res loss" title="${what} pick wrong">${ICON_X}Miss</span>`;
  if (res === "push") return `<span class="res push" title="${what}: push / tie">Push</span>`;
  return "";
}
function probBar(p, mkt) {
  if (!isNum(p)) return "";
  return `<div class="pbar" aria-hidden="true"><i class="bar-fill" style="width:${clamp(p, 0, 1) * 100}%"></i><i class="half"></i>${
    isNum(mkt) ? `<i class="mk" style="left:${clamp(mkt, 0, 1) * 100}%" title="Market ${pct(mkt)}"></i>` : ""}</div>`;
}
function agreeHTML(g) {
  const a = agreement(g);
  if (!a) return `<span class="agree muted">—</span>`;
  if (!isNum(a.n)) return `<span class="agree">${pct(a.share)} of models agree</span>`;
  const dots = Array.from({ length: a.n }, (_, i) => `<i class="${i < a.k ? "on" : ""}"></i>`).join("");
  return `<span class="agree" title="${a.k} of ${a.n} individual models pick ${esc(g.su.pick)} to win"><span class="dots" aria-hidden="true">${dots}</span>${a.k}/${a.n} models</span>`;
}

function suBlock(g, big = false) {
  const su = g.su;
  if (!su?.pick) return `<div class="pick none"><div class="pick-lbl"><span>Winner</span></div><div class="pick-main"><b>No pick yet</b></div></div>`;
  const sub = big
    ? `Market ${pct(su.market_prob)} · projected ${esc(marginText(g, su.margin))}`
    : isNum(su.market_prob) ? `Market <b>${pct(su.market_prob)}</b> · ${esc(marginText(g, su.margin))}` : esc(marginText(g, su.margin));
  return `<div class="pick su">
    <div class="pick-lbl"><span>Winner</span>${badge(suResult(g), "Winner")}</div>
    <div class="pick-main">${logo(su.pick, "sm", 22)}<b>${esc(su.pick)}</b><span class="big" aria-label="win probability">${pct(su.prob)}</span></div>
    ${probBar(su.prob, su.market_prob)}
    <div class="pick-sub">${sub}</div>
  </div>`;
}
function atsBlock(g, big = false) {
  const a = g.ats;
  if (!a?.pick) return `<div class="pick none"><div class="pick-lbl"><span>Spread</span></div><div class="pick-main"><b>${isNum(g.spread_line) ? "No pick" : "No line yet"}</b></div><div class="pick-sub">${isNum(g.spread_line) ? "" : "Pick posts when the line does"}</div></div>`;
  const sub = big
    ? `Covers ${pct(a.prob)} · line ${esc(spreadText(g) || "—")}`
    : `Covers <b>${pct(a.prob)}</b> · ${isNum(a.edge) ? `edge ${Math.abs(a.edge).toFixed(1)}` : ""}`;
  const bb = a.best_bet ? `<span class="bb" title="Edge of ${esc(String(D.backtest?.best_bet_edge ?? 2))}+ points: the ensemble's strongest spread leans">Best bet</span>` : "";
  return `<div class="pick ats${a.best_bet ? " best" : ""}">
    <div class="pick-lbl"><span>Spread${bb}</span>${badge(atsResult(g), "Spread")}</div>
    <div class="pick-main">${logo(a.pick, "sm", 22)}<b>${esc(a.pick)} ${fmtLine(a.line)}</b><span class="big" title="Edge: model margin vs the line">${isNum(a.edge) ? signed(Math.abs(a.edge)) : ""}<small>${isNum(a.edge) ? "pts" : ""}</small></span></div>
    ${probBar(a.prob)}
    <div class="pick-sub">${sub}</div>
  </div>`;
}

/* -------------------------------------------------------------- picks view */

function weekGames(w) {
  return D.games.filter((g) => g.week === w);
}

function renderWeeks() {
  const seg = $("#weeks");
  seg.innerHTML = D.weeks.map((w) => {
    const t = tally(weekGames(w));
    const sub = t.played ? rec(t.su, "t") : "next";
    return `<button data-week="${w}" aria-pressed="${w === state.week}" aria-label="Week ${w}${t.played ? `, ${sub} straight up` : ""}">Wk ${w}<span class="wk-res">${sub}</span></button>`;
  }).join("");
  segInit(seg);
  const on = seg.querySelector('[aria-pressed="true"]');
  const sc = $("#weeks-scroll");
  if (on && sc.scrollWidth > sc.clientWidth) sc.scrollLeft = on.offsetLeft - sc.clientWidth / 2 + on.offsetWidth / 2;
}

function tile(label, val, small, foot, cls = "", bar = "") {
  return `<div class="tile ${cls}"><p class="tile-label">${label}</p><div class="tile-val">${val}${small ? `<small>${small}</small>` : ""}</div>${bar}${foot ? `<div class="tile-foot">${foot}</div>` : ""}</div>`;
}
function wlBar(w, l, p = 0) {
  const n = w + l + p;
  if (!n) return "";
  return `<div class="mini-bar" aria-hidden="true"><i style="width:${(w / n) * 100}%;background:var(--good)"></i>${p ? `<i style="width:${(p / n) * 100}%;background:var(--ink-3)"></i>` : ""}<i style="width:${(l / n) * 100}%;background:var(--bad)"></i></div>`;
}
function btHeadline() {
  const lb = D.backtest.leaderboard || [];
  const e = lb.find((r) => r.id === PRIMARY), m = lb.find((r) => r.id === "market");
  return { e, m, window: D.backtest.window || "" };
}

function renderHero() {
  const w = state.week;
  const gs = weekGames(w);
  const t = tally(gs);
  $("#season-label").textContent = `${D.season ?? ""} season`.trim();
  $("#week-title").textContent = w != null ? `Week ${w} picks` : "Picks";
  const dates = gs.map(dateObj).filter(Boolean).sort((a, b) => a - b);
  const span = dates.length ? (dates.length > 1 && +dates[0] !== +dates[dates.length - 1]
    ? `${fmtDay({ date: gs.find((g) => dateObj(g) && +dateObj(g) === +dates[0]).date }, { month: "short", day: "numeric" })} – ${fmtDay({ date: gs.find((g) => dateObj(g) && +dateObj(g) === +dates[dates.length - 1]).date }, { month: "short", day: "numeric" })}`
    : fmtDay({ date: gs.find((g) => dateObj(g)).date })) : "";
  const parts = [`${gs.length} game${gs.length === 1 ? "" : "s"}`, span];
  if (t.played) parts.push(`${t.played === gs.length ? "all final" : `${t.played} final, ${gs.length - t.played} to play`}`);
  $("#week-sub").innerHTML = parts.filter(Boolean).map((p) => `<span>${esc(p)}</span>`).join('<span class="muted">·</span>');

  const s = seasonRecord();
  const suR = rate(s.su), atsR = rate(s.ats);
  const { e, m, window: win } = btHeadline();
  const wkTile = t.played
    ? tile(`Week ${w}`, rec(t.su, "t"), "SU", `Spread ${rec(t.ats, "p")}${t.played < gs.length ? ` · ${gs.length - t.played} to play` : ""}`, "", wlBar(t.su.w, t.su.l))
    : (() => {
      const probs = gs.map((g) => g.su?.prob).filter(isNum);
      const avg = probs.length ? probs.reduce((a, b) => a + b, 0) / probs.length : null;
      const edges = gs.filter((g) => isNum(g.ats?.edge) && Math.abs(g.ats.edge) >= 3).length;
      return tile(`Week ${w}`, String(gs.length), "games", `Avg winner prob ${pct(avg)}${edges ? ` · ${edges} spread edge${edges > 1 ? "s" : ""} 3+ pts` : ""}`);
    })();
  $("#tiles").innerHTML = [
    tile(`${D.season ?? ""} straight-up`, rec(s.su, "t"), pct(suR, 1), "Ensemble winner picks", "", wlBar(s.su.w, s.su.l)),
    tile(`${D.season ?? ""} vs the spread`, rec(s.ats, "p"), pct(atsR, 1), atsR != null ? (atsR > BREAK_EVEN ? "Above the 52.4% break-even" : "Break-even is 52.4%") : "", "", wlBar(s.ats.w, s.ats.l, s.ats.p)),
    wkTile,
    e ? tile(`Backtest ${esc(win)}`, pct(e.su_acc, 1), "SU", `Favorite ${pct(m?.su_acc, 1)} · spread ${pct(e.ats_acc, 1)}`, "accent") : "",
  ].join("");

  $("#bt-line").innerHTML = e
    ? `Backtest ${esc(win)}: <b>${pct(e.su_acc, 1)}</b> straight-up${m ? ` vs <b>${pct(m.su_acc, 1)}</b> for the betting favorite` : ""}; against the spread <b>${pct(e.ats_acc, 1)}</b> (break-even 52.4%). <a href="#models" data-go="models">How the models work ${ICON_ARROW}</a>`
    : "";
  $("#tb-ctx").textContent = w != null ? `Week ${w}` : "";
}

function sortGames(gs) {
  const a = [...gs];
  if (state.sort === "conf") a.sort((x, y) => (y.su?.prob ?? -1) - (x.su?.prob ?? -1) || kickKey(x).localeCompare(kickKey(y)));
  else a.sort((x, y) => kickKey(x).localeCompare(kickKey(y)));
  return a;
}

function teamRow(g, side) {
  const t = g[side], other = side === "home" ? g.away : g.home;
  const f = g.team_form?.[side];
  const qb = g[`${side}_qb`];
  const sub = [f?.record, qb].filter(Boolean).join(" · ");
  let right = "";
  if (played(g)) right = `<span class="t-score">${g[`${side}_score`]}</span>`;
  else if (isNum(g.spread_line)) {
    const fav = g.spread_line > 0 ? g.home : g.spread_line < 0 ? g.away : null;
    right = `<span class="t-line">${fav === t ? `${MINUS}${pts(g.spread_line)}` : fav === null && side === "away" ? "PK" : isNum(g.total_line) && fav === other ? `O/U ${pts(g.total_line)}` : ""}</span>`;
  }
  const w = winner(g);
  const cls = played(g) && w ? (w === t ? "won" : "lost") : "";
  return `<div class="t-row ${cls}">${logo(t)}<div class="t-name"><b>${esc(t)}</b>${side === "home" ? '<span class="at">HOME</span>' : ""}<span class="nick">${esc(tnick(t))}</span><span class="rec">${esc(sub) || "&nbsp;"}</span></div>${right}</div>`;
}

function cardHTML(g, i) {
  const meta = [
    `<span class="when">${esc(kickoff(g))}</span>`,
    weatherChip(g),
    played(g) ? `<span class="chip final status">Final${isNum(g.spread_line) ? ` · ${esc(spreadText(g))}` : ""}</span>`
      : !isNum(g.spread_line) ? `<span class="chip warn status">No line yet</span>` : "",
  ].join("");
  return `<article class="game" data-id="${esc(g.game_id)}" style="--n:${i}">
    <div class="g-meta">${meta}</div>
    <div class="g-teams">${teamRow(g, "away")}${teamRow(g, "home")}</div>
    <div class="g-picks">${suBlock(g)}${atsBlock(g)}</div>
    <div class="g-foot">${agreeHTML(g)}<button class="more-btn" type="button" data-open="${esc(g.game_id)}" aria-haspopup="dialog">Data behind the pick ${ICON_ARROW}</button></div>
  </article>`;
}

function tableHTML(gs) {
  const res = (g) => {
    const s = suResult(g), a = atsResult(g);
    if (!s && !a) return '<span class="muted">—</span>';
    return `<span class="res-pair">${s ? badge(s, "Winner").replace(/Hit|Miss|Push/, "SU") : ""}${a ? badge(a, "Spread").replace(/Hit|Miss|Push/, "ATS") : ""}</span>`;
  };
  const rows = gs.map((g) => {
    const w = winner(g);
    const sc = (side) => (played(g) ? `<span class="sc ${w === g[side] ? "w" : ""}">${g[side + "_score"]}</span>` : "");
    const ag = agreement(g);
    return `<tr data-id="${esc(g.game_id)}" tabindex="0" aria-label="${esc(g.away)} at ${esc(g.home)}: open details">
      <td class="when hide-sm">${esc(kickoff(g))}</td>
      <td><span class="mu"><span class="tm">${logo(g.away, "xs", 18)}${esc(g.away)}${sc("away")}</span><span class="vs">@</span><span class="tm">${logo(g.home, "xs", 18)}${esc(g.home)}${sc("home")}</span></span></td>
      <td class="hide-sm">${esc(spreadText(g) || "—")}${isNum(g.total_line) ? ` <span class="muted">· ${pts(g.total_line)}</span>` : ""}</td>
      <td>${g.su?.pick ? `<span class="pk">${logo(g.su.pick, "xs", 18)}${esc(g.su.pick)} <span class="muted" style="font-weight:600">${pct(g.su.prob)}</span></span>` : '<span class="muted">—</span>'}</td>
      <td>${g.ats?.pick ? `<span class="pk">${esc(g.ats.pick)} ${fmtLine(g.ats.line)}</span>` : '<span class="muted">—</span>'}</td>
      <td class="num hide-sm">${isNum(g.ats?.edge) ? Math.abs(g.ats.edge).toFixed(1) : "—"}</td>
      <td class="num hide-sm agree-n">${ag && isNum(ag.n) ? `${ag.k}/${ag.n}` : ag ? pct(ag.share) : "—"}</td>
      <td>${res(g)}</td>
    </tr>`;
  }).join("");
  return `<div class="sheet-table-wrap"><div class="table-scroll"><table class="stats st">
    <thead><tr><th class="hide-sm">Kickoff</th><th>Matchup</th><th class="hide-sm">Line</th><th>Winner</th><th>Spread</th><th class="num hide-sm">Edge</th><th class="num hide-sm">Models</th><th>Result</th></tr></thead>
    <tbody>${rows}</tbody></table></div></div>`;
}

function renderSheet() {
  const gs = sortGames(weekGames(state.week));
  const el = $("#sheet");
  if (!gs.length) { el.innerHTML = `<div class="empty">No games for this week yet.</div>`; return; }
  if (state.layout === "table") {
    el.innerHTML = tableHTML(gs);
  } else {
    let html = "", last = null, i = 0;
    for (const g of gs) {
      if (state.sort === "time" && g.date !== last) {
        last = g.date;
        const n = gs.filter((x) => x.date === g.date).length;
        html += `<div class="day-head"><h3>${esc(fmtDay(g, { weekday: "long", month: "short", day: "numeric" }))}</h3><span>${n} game${n > 1 ? "s" : ""}</span></div>`;
      }
      html += cardHTML(g, i++);
    }
    el.innerHTML = `<div class="cards">${html}</div>`;
  }
  if (!reduceMotion) { const c = el.firstElementChild; c.classList.add("enter"); setTimeout(() => c.classList.remove("enter"), 1200); }
}

function renderPicks() {
  renderWeeks();
  renderHero();
  renderSheet();
}

/* ------------------------------------------------------------ detail sheet */

function modelStrip(g) {
  const ids = D.models.map((m) => m.id);
  const ens = g.models[PRIMARY]?.p_home ?? (g.su?.pick ? (g.su.pick === g.home ? g.su.prob : 1 - g.su.prob) : null);
  const pickHome = g.su?.pick ? g.su.pick === g.home : null;
  const rows = ids.map((id) => {
    const m = g.models[id] || (id === PRIMARY && isNum(ens) ? { p_home: ens, margin: g.su?.margin } : null);
    const cls = [id === PRIMARY ? "ens" : "", id === "market" ? "mkt" : ""];
    if (!m || !isNum(m.p_home)) {
      return `<div class="mrow missing ${cls.join(" ")}"><span class="mn" title="${esc(mname(id))}">${esc(mshort(id))}</span><div class="track"><i class="mid"></i></div><span class="mv">n/a</span></div>`;
    }
    const p = clamp(m.p_home, 0, 1);
    if (pickHome !== null && id !== PRIMARY && (p >= 0.5) !== pickHome) cls.push("dis");
    const fav = p >= 0.5 ? g.home : g.away;
    const tipTxt = `${mname(id)}: ${g.home} ${pct(p)} / ${g.away} ${pct(1 - p)}${isNum(m.margin) ? ` · ${marginText(g, m.margin)}` : ""}`;
    const ref = isNum(ens) && id !== PRIMARY ? `<i class="ens-ref" style="left:${ens * 100}%"></i>` : "";
    const line = id === PRIMARY ? `<i class="seg-line" style="left:${Math.min(p, 0.5) * 100}%;width:${Math.abs(p - 0.5) * 100}%"></i>` : "";
    return `<div class="mrow ${cls.join(" ")}" title="${esc(tipTxt)}"><span class="mn">${esc(mshort(id))}</span><div class="track"><i class="mid"></i>${ref}${line}<i class="dot" style="left:${p * 100}%"></i></div><span class="mv">${esc(fav)} ${pct(Math.max(p, 1 - p))}</span></div>`;
  }).join("");
  const a = agreement(g);
  return `<div class="d-sec"><p class="d-sec-title">What each model says <span>${a && isNum(a.n) ? `${a.k} of ${a.n} agree with ${esc(g.su.pick)}` : ""}</span></p>
    <div class="mstrip">
      <div class="mstrip-axis"><span></span><div class="ends"><span>${esc(g.away)} wins</span><span>50%</span><span>${esc(g.home)} wins</span></div><span></span></div>
      ${rows}
    </div>
    <p class="note">Each dot is one model's chance that ${esc(g.home)} wins (right) or ${esc(g.away)} wins (left). Hollow dots disagree with the pick; the dashed line marks the ensemble.</p>
  </div>`;
}

function marketCompare(g) {
  const su = g.su;
  if (!su?.pick || !isNum(su.prob)) return "";
  const diff = isNum(su.market_prob) ? su.prob - su.market_prob : null;
  return `<div class="d-sec"><p class="d-sec-title">Ensemble vs the market <span>${diff != null ? `${signed(diff * 100, 1)} pts on ${esc(su.pick)}` : "no market price"}</span></p>
    <div class="compare-bars">
      <div class="cb"><span>Ensemble</span><div class="pbar"><i class="bar-fill" style="width:${su.prob * 100}%"></i><i class="half"></i></div><span class="v">${pct(su.prob)}</span></div>
      <div class="cb"><span>Market${isNum(g.home_ml) ? ` <span class="muted">(${g.su.pick === g.home ? fmtML(g.home_ml) : fmtML(g.away_ml)})</span>` : ""}</span><div class="pbar">${isNum(su.market_prob) ? `<i class="bar-fill mkt" style="width:${su.market_prob * 100}%"></i>` : ""}<i class="half"></i></div><span class="v">${pct(su.market_prob)}</span></div>
    </div>
    <p class="note">${esc(su.pick)}'s chance to win. Market = moneyline odds with the bookmaker's margin removed.</p>
  </div>`;
}
const fmtML = (x) => (isNum(x) ? (x > 0 ? "+" : MINUS) + Math.abs(x) : "—");

function factorsTable(g) {
  const fs = Array.isArray(g.factors) ? g.factors.filter((f) => f && (f.label || f.key)) : [];
  if (!fs.length) return `<div class="d-sec"><p class="d-sec-title">Data behind the pick</p><div class="empty">No factor data for this game yet.</div></div>`;
  let hw = 0, aw = 0, lastGroup = null;
  const cell = (v, txt, f) => {
    const num = isNum(v) ? `<span>${fmtVal(v, f.fmt)}</span>` : "";
    return txt ? `${num}<small class="ftxt">${esc(txt)}</small>` : num || `<span class="muted">—</span>`;
  };
  const rows = fs.map((f) => {
    const h = f.home, a = f.away;
    let better = null;
    if (isNum(h) && isNum(a) && h !== a && (f.better === "high" || f.better === "low")) {
      better = (f.better === "high") === (h > a) ? "home" : "away";
      better === "home" ? hw++ : aw++;
    }
    const lab = esc(f.label || f.key) + (f.better === "low" ? ' <span class="muted" title="lower is better">↓</span>' : "");
    let head = "";
    if (f.group && f.group !== lastGroup) { lastGroup = f.group; head = `<tr class="fgroup"><th colspan="3">${esc(f.group)}</th></tr>`; }
    return `${head}<tr><td class="v ${better === "away" ? "better" : ""}">${cell(a, f.away_text, f)}</td><td class="lab">${lab}</td><td class="v ${better === "home" ? "better" : ""}">${cell(h, f.home_text, f)}</td></tr>`;
  }).join("");
  const n = hw + aw;
  const sum = n ? (hw === aw ? `Even split, ${hw}–${aw}` : `${hw > aw ? g.home : g.away} better in ${Math.max(hw, aw)} of ${n}`) : "";
  return `<div class="d-sec"><p class="d-sec-title">Data behind the pick <span>${esc(sum)}</span></p>
    <table class="ftable"><thead><tr><th>${logo(g.away, "sm", 22)}${esc(g.away)}</th><th>Factor</th><th>${logo(g.home, "sm", 22)}${esc(g.home)}</th></tr></thead><tbody>${rows}</tbody></table>
    <p class="note">Highlighted = the better side for that factor. ↓ means lower is better. Every value is what was known before kickoff.</p>
  </div>`;
}

function infoGrid(g) {
  const tf = g.team_form || {};
  const formLine = (side) => {
    const f = tf[side];
    if (!f) return `<div><span>${esc(g[side])}</span><span class="muted">—</span></div>`;
    return `<div><span>${esc(g[side])}</span><span>${esc(f.record ?? "—")} · ATS ${esc(f.ats ?? "—")} · ${isNum(f.pd) ? signed(f.pd, 0) : "—"}</span></div>`;
  };
  const qb = (side) => `<div><span>${esc(g[side])}</span><span>${esc(g[side + "_qb"] || "TBD")}</span></div>`;
  const lines = [
    `<div><span>Spread</span><span>${esc(spreadText(g) || "No line")}</span></div>`,
    `<div><span>Total</span><span>${isNum(g.total_line) ? pts(g.total_line) : "—"}</span></div>`,
    `<div><span>Moneyline</span><span>${esc(g.away)} ${fmtML(g.away_ml)} · ${esc(g.home)} ${fmtML(g.home_ml)}</span></div>`,
  ].join("");
  const venue = [g.stadium, g.roof ? String(g.roof).replace(/^./, (c) => c.toUpperCase()) : ""].filter(Boolean).join(" · ");
  return `<div class="d-sec"><p class="d-sec-title">Game info</p><div class="info-grid">
    <div class="info"><div class="k">Quarterbacks</div><div class="v">${qb("away")}${qb("home")}</div></div>
    <div class="info"><div class="k">Form coming in (record · ATS · point diff)</div><div class="v">${formLine("away")}${formLine("home")}</div></div>
    <div class="info"><div class="k">Betting line</div><div class="v">${lines}</div></div>
    <div class="info"><div class="k">Venue &amp; weather</div><div class="v"><div><span>${esc(venue || "—")}</span></div><div><span>${esc(weatherText(g))}</span></div></div></div>
  </div></div>`;
}

function detailHTML(g) {
  const w = winner(g);
  const side = (s) => `<div class="d-team ${s}">${logo(g[s], "", 56)}<div style="min-width:0"><div class="nm">${esc(g[s])}</div><div class="full">${esc(tname(g[s]))}</div><div class="sm">${esc(g.team_form?.[s]?.record || "")}</div></div></div>`;
  const mid = played(g)
    ? `<div class="d-score"><span class="${w === g.away ? "" : "l"}">${g.away_score}</span><span class="l"> – </span><span class="${w === g.home ? "" : "l"}">${g.home_score}</span></div><div class="d-line">Final</div>`
    : `<div class="d-at">@</div><div class="d-line">${esc(spreadText(g) || "No line")}${isNum(g.total_line) ? ` · O/U ${pts(g.total_line)}` : ""}</div>`;
  const where = [g.week != null ? `Week ${g.week}` : "", kickoff(g), g.stadium].filter(Boolean).join(" · ");
  const su = g.su, a = g.ats;
  let why = "";
  if (su?.pick) {
    why = `The ensemble makes <b>${esc(tname(su.pick))}</b> a ${pct(su.prob)} favorite${isNum(su.market_prob) ? ` (market ${pct(su.market_prob)})` : ""} and projects <b>${esc(marginText(g, su.margin))}</b>.`;
    if (a?.pick && isNum(g.spread_line)) why += ` Against a line of ${esc(spreadText(g))}, that's a <b>${isNum(a.edge) ? Math.abs(a.edge).toFixed(1) : "?"}-point edge</b> on ${esc(a.pick)} ${fmtLine(a.line)}.`;
  }
  return `<header class="d-head">
      <div class="d-top"><p class="eyebrow">${esc(where)}</p><button class="icon-btn" type="button" data-close aria-label="Close">${ICON_CLOSE}</button></div>
      <div class="d-match">${side("away")}<div class="d-mid">${mid}</div>${side("home")}</div>
      <h2 id="detail-title" class="sr" style="position:absolute;left:-9999px">${esc(tname(g.away))} at ${esc(tname(g.home))}</h2>
    </header>
    <div class="d-body">
      <div class="d-picks">${suBlock(g, true)}${atsBlock(g, true)}</div>
      ${why ? `<p class="lede" style="margin:0;font-size:13.5px">${why}</p>` : ""}
      ${marketCompare(g)}
      ${Object.keys(g.models).length || g.su ? modelStrip(g) : ""}
      ${factorsTable(g)}
      ${infoGrid(g)}
    </div>`;
}

function openDetail(id) {
  const g = GAME[id];
  if (!g) return;
  const dlg = $("#detail");
  $("#detail-inner").innerHTML = detailHTML(g);
  $("#detail-inner").scrollTop = 0;
  if (!dlg.open) { if (dlg.showModal) dlg.showModal(); else dlg.setAttribute("open", ""); }
}
function closeDetail() { const dlg = $("#detail"); if (dlg.open) dlg.close(); }

/* -------------------------------------------------------------- models view */

function liveStats(id) {
  let sw = 0, sl = 0, aw = 0, al = 0;
  for (const g of D.games) {
    if (!played(g)) continue;
    if (id === PRIMARY) {
      const s = suResult(g), a = atsResult(g);
      if (s === "win") sw++; else if (s === "loss") sl++;
      if (a === "win") aw++; else if (a === "loss") al++;
      continue;
    }
    const m = g.models[id];
    if (!m || !isNum(m.p_home)) continue;
    const w = winner(g);
    if (w) ((m.p_home >= 0.5 ? g.home : g.away) === w ? sw++ : sl++);
    if (isNum(g.spread_line) && isNum(m.margin) && Math.abs(m.margin - g.spread_line) > 0.05) {
      const c = g.home_score - g.away_score - g.spread_line;
      if (c !== 0) ((c > 0) === (m.margin > g.spread_line) ? aw++ : al++);
    }
  }
  return { su: { w: sw, l: sl }, ats: { w: aw, l: al } };
}

function renderModels() {
  const bt = D.backtest;
  const lb = (bt.leaderboard || []).filter((r) => r && r.id);
  const { e, m, window: win } = btHeadline();
  const base = D.models.filter((x) => x.id !== PRIMARY);
  $("#stack-lede").innerHTML = `Every pick comes from a <b>stacked ensemble</b> of ${base.length} models${e ? `. In a walk-forward backtest over ${esc(win)} it picked <b>${pct(e.su_acc, 1)}</b> of winners${m ? ` (the betting favorite: ${pct(m.su_acc, 1)})` : ""} and went <b>${pct(e.ats_acc, 1)}</b> against the spread` : ""}.`;
  if (D.ensemble?.design?.su?.design === "guard") {
    $("#stack-note").innerHTML = `Each model estimates the chance the home team wins and the expected margin. The ensemble learns, only from seasons it never trained on, how much to trust the betting market and each family of models: the models "talking to each other". <b>What it learned:</b> when a model picks against the betting favorite, the favorite usually wins (table below). So the winner pick stays with the favorite. The models adjust the win probability and drive the spread pick, which is where they disagree with the line usefully.`;
  }
  $("#stack-diagram").innerHTML = `
    <div class="stack-models">${base.map((x) => `<span class="chip" title="${esc(x.description || "")}">${esc(mshort(x.id))}</span>`).join("")}</div>
    <div class="stack-arrow" aria-hidden="true"><svg width="22" height="12" viewBox="0 0 22 12"><path d="M1 6h18M15 1.5 19.5 6 15 10.5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg></div>
    <div class="stack-box"><b>${esc(mshort(PRIMARY))}</b><span>${esc(MODEL[PRIMARY]?.description || "Learns how much to trust each model")}</span></div>
    <div class="stack-arrow" aria-hidden="true"><svg width="22" height="12" viewBox="0 0 22 12"><path d="M1 6h18M15 1.5 19.5 6 15 10.5" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg></div>
    <div class="stack-box out"><b>The pick</b><span>Winner + win probability; spread side + edge in points</span></div>`;

  const best = (k, lowGood) => {
    const v = lb.map((r) => r[k]).filter(isNum);
    return v.length ? (lowGood ? Math.min(...v) : Math.max(...v)) : null;
  };
  $("#model-tiles").innerHTML = e ? [
    tile("Straight-up", pct(e.su_acc, 1), "", m ? `Favorite ${pct(m.su_acc, 1)} · ${signed((e.su_acc - m.su_acc) * 100, 1)} pts` : "", "accent"),
    tile("Against the spread", pct(e.ats_acc, 1), "", "Break-even 52.4% at −110"),
    tile("Log loss", isNum(e.logloss) ? e.logloss.toFixed(3) : "—", "", m && isNum(m.logloss) ? `Market ${m.logloss.toFixed(3)} · lower is better` : "Lower is better"),
    tile("Games tested", isNum(e.n) ? e.n.toLocaleString() : "—", "", esc(win)),
  ].join("") : "";
  $("#lb-window").textContent = win ? `· ${win}` : "";

  // leaderboard
  const sorted = [...lb].sort((a, b) => (b.su_acc ?? 0) - (a.su_acc ?? 0));
  const bSU = best("su_acc"), bATS = best("ats_acc"), bLL = best("logloss", true), bBR = best("brier", true);
  const cell = (v, b, d, f = (x) => pct(x, 1)) => `<td class="num ${isNum(v) && v === b ? "best" : ""}">${isNum(v) ? f(v, d) : "—"}</td>`;
  const lo = Math.min(...sorted.map((r) => r.su_acc).filter(isNum), 0.6) - 0.01, hi = Math.max(...sorted.map((r) => r.su_acc).filter(isNum), 0.7);
  $("#leaderboard").innerHTML = `<thead><tr><th>Model</th><th>Straight-up</th><th class="num">Spread</th><th class="num">Log loss</th><th class="num hide-sm">Brier</th><th class="num">${D.season ?? ""} SU</th><th class="num hide-sm">${D.season ?? ""} ATS</th></tr></thead><tbody>${
    sorted.map((r) => {
      const live = liveStats(r.id);
      const cls = r.id === PRIMARY ? "primary" : r.id === "market" ? "baseline" : "";
      const tag = r.id === PRIMARY ? '<span class="tag ens">Picks</span>' : r.id === "market" ? '<span class="tag mkt">Baseline</span>' : "";
      const w = isNum(r.su_acc) ? clamp((r.su_acc - lo) / (hi - lo), 0.02, 1) * 100 : 0;
      const barColor = r.id === PRIMARY ? "var(--series-1)" : r.id === "market" ? "var(--series-2)" : "var(--series-muted)";
      return `<tr class="${cls}"><td title="${esc(MODEL[r.id]?.description || "")}">${esc(mshort(r.id))}${tag}</td>
        <td><span class="accbar"><span class="${r.su_acc === bSU ? "best" : ""}">${pct(r.su_acc, 1)}</span><span class="pbar"><i class="bar-fill" style="width:${w}%;background:${barColor}"></i></span></span></td>
        ${cell(r.ats_acc, bATS, 1)}${cell(r.logloss, bLL, 3, (x) => x.toFixed(3))}${cell(r.brier, bBR, 3, (x) => x.toFixed(3)).replace('class="num', 'class="num hide-sm')}
        <td class="num">${live.su.w + live.su.l ? rec(live.su) : "—"}</td><td class="num hide-sm">${r.id === "market" ? '<span class="muted" title="The market is the line">n/a</span>' : live.ats.w + live.ats.l ? rec(live.ats) : "—"}</td></tr>`;
    }).join("")}</tbody>`;

  // model list
  $("#model-list").innerHTML = D.models.map((x) => `<div class="model-item ${x.id === PRIMARY ? "ens" : x.id === "market" ? "mkt" : ""}"><i aria-hidden="true"></i><div><b>${esc(x.name || x.id)}</b>${x.family ? ` <span class="muted" style="font-size:11.5px">${esc(x.family)}</span>` : ""}<p>${esc(x.description || "")}</p></div></div>`).join("");

  renderEnsembleInternals();
  registerChart($("#season-chart"), drawSeasonChart);
  registerChart($("#conf-chart"), (el) => barChart(el, bt.by_confidence || [], {
    label: (r) => `${Math.round(r.min_prob * 100)}%+`, val: (r) => r.acc, tip: (r) => `<div class="tt-t">Win prob ≥ ${pct(r.min_prob)}</div><div class="tt-r"><span>Correct</span><b>${pct(r.acc, 1)}</b></div><div class="tt-r"><span>Games</span><b>${isNum(r.n) ? r.n.toLocaleString() : "—"}</b></div>`,
    base: 0.5, color: "var(--series-1)",
  }));
  registerChart($("#edge-chart"), (el) => barChart(el, bt.ats_by_edge || [], {
    label: (r) => `${r.min_edge}+`, val: (r) => r.acc, tip: (r) => `<div class="tt-t">Edge ≥ ${r.min_edge} pts</div><div class="tt-r"><span>Covered</span><b>${pct(r.acc, 1)}</b></div><div class="tt-r"><span>Games</span><b>${isNum(r.n) ? r.n.toLocaleString() : "—"}</b></div>`,
    base: 0.45, ref: BREAK_EVEN, refLabel: "52.4%", color: "var(--series-1)", xTitle: "edge, points",
  }));
  registerChart($("#cal-chart"), (el) => calChart(el, bt.calibration || []));
}

const FAMILY_LABEL = {
  efficiency: "Efficiency models (no line)", efficiency_mkt: "Efficiency + line", personnel: "Lineup model (no line)",
  personnel_mkt: "QB & injuries + line", ratings: "Rating models (no line)", ratings_mkt: "Ratings + line",
  situational: "Situational, neural net, similar games", market: "Market", roster: "Player lineup", closegames: "Close games",
};
const ATS_LABEL = {
  home_dog: "Home underdog", spread: "Size of the spread", playoff_dog_home: "Playoff home underdog",
  inj_gap: "Starters-out gap (injuries)", c_efficiency: "Efficiency models' cover lean",
  c_efficiency_mkt: "Efficiency + line cover lean", c_personnel: "Lineup model's cover lean",
  c_personnel_mkt: "QB & injuries cover lean", c_ratings: "Rating models' cover lean",
  c_ratings_mkt: "Ratings + line cover lean", c_situational: "Situational models' cover lean",
};

function renderEnsembleInternals() {
  const E = D.ensemble || {};
  const su = (E.su_weights || []).filter((w) => isNum(w.influence));
  const ats = (E.ats_weights || []).filter((w) => isNum(w.influence));
  const all = [...su, ...ats].map((w) => Math.abs(w.influence));
  const max = Math.max(0.01, ...all);
  const bars = (rows, lab) => rows.sort((a, b) => Math.abs(b.influence) - Math.abs(a.influence)).map((w) => {
    const v = w.influence, wid = (Math.abs(v) / max) * 50;
    const title = w.models ? `Models: ${w.models.join(", ")}` : "";
    return `<div class="lrow" title="${esc(title)}"><span class="ln">${esc(lab(w))}</span><div class="ltrack"><i class="lmid"></i><i class="lbar ${v < 0 ? "neg" : ""}" style="${v < 0 ? `right:50%` : `left:50%`};width:${wid}%"></i></div><span class="lv">${signed(v, 3)}</span></div>`;
  }).join("");
  $("#listen").innerHTML = su.length || ats.length ? `
    <p class="lsub">Who wins <span class="muted">· weight on each family's disagreement with the market</span></p>${bars(su, (w) => FAMILY_LABEL[w.family] || w.family)}
    <p class="lsub">Against the spread <span class="muted">· positive = leans to the home side</span></p>${bars(ats, (w) => ATS_LABEL[w.input] || w.input)}
    ${E.design?.su?.summary ? `<p class="note">${esc(E.design.su.summary)}</p>` : ""}` : `<div class="empty">No ensemble weights yet.</div>`;

  const ctx = E.context || [];
  const fams = ["ratings", "efficiency", "personnel"];
  const fl = { ratings: "Ratings", efficiency: "Efficiency", personnel: "Lineup" };
  const dcell = (acc, n) => isNum(acc) && n ? `<td class="num ${acc < 0.5 ? "lose" : "win"}">${pct(acc)}<small> of ${n}</small></td>` : `<td class="num muted">—</td>`;
  $("#dissent").innerHTML = ctx.length ? `<thead><tr><th>Spread</th><th class="num">Games</th><th class="num">Favorite won</th>${fams.map((f) => `<th class="num">${fl[f]} dissent</th>`).join("")}</tr></thead><tbody>${
    ctx.map((c) => `<tr><td>${esc(c.spread_bucket)}</td><td class="num">${c.n}</td><td class="num">${pct(c.market_acc)}</td>${fams.map((f) => dcell(c[f + "_dissent_acc"], c[f + "_dissent_n"])).join("")}</tr>`).join("")}</tbody>` : "";

  const bt = D.backtest || {}, bbs = bt.best_bets || [];
  $("#bb-sub").innerHTML = `A spread pick is a <b>best bet</b> when the ensemble's edge over the line is at least <b>${bt.best_bet_edge ?? 2} points</b>. That's rare: a handful of games a season. Its record by era is below. 2018–2025 is a true holdout: the ensemble's design was frozen before those seasons were scored. Break-even at −110 is 52.4%. Samples are small, so treat this as a lean, not a lock.`;
  $("#bestbets").innerHTML = bbs.length ? `<thead><tr><th>Seasons</th><th class="num">Record</th><th class="num">Cover rate</th><th class="num">Games</th></tr></thead><tbody>${
    bbs.map((b) => `<tr class="${b.holdout ? "primary" : ""}"><td>${esc(b.window)}${b.holdout ? '<span class="tag ens">Holdout</span>' : ""}</td><td class="num">${b.w}-${b.l}</td><td class="num ${b.acc >= BREAK_EVEN ? "win" : "lose"}">${pct(b.acc, 1)}</td><td class="num">${b.n}</td></tr>`).join("")}</tbody>` : "";
}

/* ------------------------------------------------------------------ charts */

const NS = "http://www.w3.org/2000/svg";
const tipEl = () => $("#tip");
function showTip(html, x, y) {
  const t = tipEl();
  t.innerHTML = html;
  t.hidden = false;
  const r = t.getBoundingClientRect();
  let left = x + 14, top = y + 14;
  if (left + r.width > innerWidth - 8) left = x - r.width - 14;
  if (top + r.height > innerHeight - 8) top = y - r.height - 14;
  t.style.left = clamp(left, 8, innerWidth - r.width - 8) + "px";
  t.style.top = clamp(top, 8, innerHeight - r.height - 8) + "px";
}
const hideTip = () => { tipEl().hidden = true; };

const ro = "ResizeObserver" in window ? new ResizeObserver((entries) => {
  for (const en of entries) {
    const el = en.target, w = Math.round(en.contentRect.width);
    if (!w || el._w === w) continue;
    el._w = w;
    charts.get(el)?.(el);
  }
}) : null;
function registerChart(el, fn) {
  if (!el) return;
  charts.set(el, fn);
  el._w = el.clientWidth;
  fn(el);
  ro?.observe(el);
}
function niceStep(range, target) {
  const raw = range / target, mag = 10 ** Math.floor(Math.log10(raw)), n = raw / mag;
  return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * mag;
}
function emptyChart(el, msg = "No backtest data yet.") { el.innerHTML = `<div class="empty">${msg}</div>`; }

function drawSeasonChart(el) {
  const metric = state.metric;
  const rows = ((metric === "ats" ? D.backtest.by_season_ats : D.backtest.by_season) || []).filter((r) => r && isNum(r.season)).sort((a, b) => a.season - b.season);
  $("#season-chart-title").textContent = metric === "ats" ? "Spread accuracy by season" : "Straight-up accuracy by season";
  $("#season-chart-sub").textContent = metric === "ats"
    ? "Share of games each model picked correctly against the closing spread (pushes excluded). Dashed line: 52.4% break-even."
    : "Share of games each model picked the winner. The orange line is simply taking the betting favorite.";
  if (!rows.length) { $("#season-legend").innerHTML = ""; return emptyChart(el); }
  const ids = [...new Set(rows.flatMap((r) => Object.keys(r).filter((k) => k !== "season" && isNum(r[k]))))];
  if (metric === "ats" && ids.includes("market") && rows.every((r) => !isNum(r.market) || Math.abs(r.market - 0.5) < 0.04)) { /* keep */ }
  const hl = [PRIMARY, "market"].filter((id) => ids.includes(id));
  const others = ids.filter((id) => !hl.includes(id));
  $("#season-legend").innerHTML = [
    ids.includes(PRIMARY) ? `<span><i style="background:var(--series-1);height:3px"></i>${esc(mshort(PRIMARY))}</span>` : "",
    ids.includes("market") ? `<span><i style="background:var(--series-2)"></i>Market (favorite)</span>` : "",
    others.length ? `<span><i></i>Other models (${others.length})</span>` : "",
  ].join("");

  const W = el.clientWidth || 640, narrow = W < 520, H = narrow ? 240 : 290;
  const M = { t: 14, r: narrow ? 66 : 92, b: 26, l: 40 };
  const vals = rows.flatMap((r) => ids.map((id) => r[id]).filter(isNum));
  if (metric === "ats") vals.push(BREAK_EVEN);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const step = niceStep(hi - lo || 0.05, 5);
  lo = Math.floor((lo - step * 0.2) / step) * step; hi = Math.ceil((hi + step * 0.2) / step) * step;
  const x = (i) => M.l + (rows.length === 1 ? (W - M.l - M.r) / 2 : (i / (rows.length - 1)) * (W - M.l - M.r));
  const y = (v) => M.t + (1 - (v - lo) / (hi - lo)) * (H - M.t - M.b);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" aria-hidden="true"><g class="grid">`;
  for (let v = lo; v <= hi + 1e-9; v += step) s += `<line x1="${M.l}" x2="${W - M.r + 8}" y1="${y(v)}" y2="${y(v)}"/><text x="${M.l - 8}" y="${y(v) + 4}" text-anchor="end">${(v * 100).toFixed(step < 0.01 ? 1 : 0)}%</text>`;
  s += `</g>`;
  if (metric === "ats") s += `<line x1="${M.l}" x2="${W - M.r + 8}" y1="${y(BREAK_EVEN)}" y2="${y(BREAK_EVEN)}" style="stroke:var(--ink-3);stroke-dasharray:4 4"/>`;
  const every = Math.ceil(rows.length / (narrow ? 5 : 14));
  rows.forEach((r, i) => { if ((rows.length - 1 - i) % every === 0) s += `<text x="${x(i)}" y="${H - 6}" text-anchor="middle">${narrow ? "’" + String(r.season).slice(2) : r.season}</text>`; });
  const path = (id) => {
    let d = "", pen = false;
    rows.forEach((r, i) => { if (isNum(r[id])) { d += `${pen ? "L" : "M"}${x(i).toFixed(1)},${y(r[id]).toFixed(1)}`; pen = true; } else pen = false; });
    return d;
  };
  for (const id of others) s += `<path d="${path(id)}" fill="none" style="stroke:var(--series-muted);stroke-width:1.5" stroke-linejoin="round" opacity=".8" data-id="${esc(id)}"/>`;
  const color = (id) => (id === PRIMARY ? "var(--series-1)" : "var(--series-2)");
  for (const id of [...hl].reverse()) s += `<path d="${path(id)}" fill="none" style="stroke:${color(id)};stroke-width:${id === PRIMARY ? 2.5 : 2}" stroke-linejoin="round" stroke-linecap="round"/>`;
  // direct end labels for the highlighted series (nudged apart)
  const ends = hl.map((id) => { const i = rows.map((r) => isNum(r[id])).lastIndexOf(true); return i < 0 ? null : { id, i, v: rows[i][id], ly: y(rows[i][id]) }; }).filter(Boolean).sort((a, b) => a.ly - b.ly);
  for (let k = 1; k < ends.length; k++) if (ends[k].ly - ends[k - 1].ly < 26) ends[k].ly = ends[k - 1].ly + 26;
  for (const en of ends) {
    s += `<circle cx="${x(en.i)}" cy="${y(en.v)}" r="4" style="fill:${color(en.id)};stroke:var(--card);stroke-width:2"/>`;
    s += `<text class="lbl" x="${x(en.i) + 10}" y="${en.ly - 1}">${en.id === PRIMARY ? esc(mshort(PRIMARY)) : "Market"}</text><text class="lbl-2" x="${x(en.i) + 10}" y="${en.ly + 12}">${pct(en.v, 1)}</text>`;
  }
  s += `<g class="hov" style="display:none"><line class="xhair" y1="${M.t}" y2="${H - M.b}"/>${hl.map((id) => `<circle r="4.5" data-h="${esc(id)}" style="fill:${color(id)};stroke:var(--card);stroke-width:2"/>`).join("")}</g>`;
  s += `<rect class="hit" x="${M.l - 10}" y="0" width="${W - M.l - M.r + 20}" height="${H}"/></svg>`;
  el.innerHTML = s;
  el.setAttribute("aria-label", `${metric === "ats" ? "Spread" : "Straight-up"} accuracy by season, ${rows[0].season}–${rows[rows.length - 1].season}`);

  const svg = el.querySelector("svg"), hov = svg.querySelector(".hov");
  const move = (ev) => {
    const b = svg.getBoundingClientRect();
    const px = ((ev.clientX - b.left) / b.width) * W;
    const i = clamp(Math.round(((px - M.l) / (W - M.l - M.r)) * (rows.length - 1)), 0, rows.length - 1);
    const r = rows[i];
    hov.style.display = "";
    hov.querySelector("line").setAttribute("x1", x(i)); hov.querySelector("line").setAttribute("x2", x(i));
    for (const c of hov.querySelectorAll("circle")) {
      const v = r[c.dataset.h];
      if (isNum(v)) { c.setAttribute("cx", x(i)); c.setAttribute("cy", y(v)); c.style.display = ""; } else c.style.display = "none";
    }
    const list = ids.filter((id) => isNum(r[id])).sort((a, b) => r[b] - r[a]);
    showTip(`<div class="tt-t">${r.season}</div>${list.map((id) => `<div class="tt-r ${hl.includes(id) ? "hl" : ""}"><span>${hl.includes(id) ? `<i class="sw" style="background:${color(id)}"></i>` : ""}${esc(mshort(id))}</span><b>${pct(r[id], 1)}</b></div>`).join("")}`, ev.clientX, ev.clientY);
  };
  const hit = svg.querySelector(".hit");
  hit.addEventListener("pointermove", move);
  hit.addEventListener("pointerdown", move);
  hit.addEventListener("pointerleave", () => { hov.style.display = "none"; hideTip(); });
}

function barPath(x, y, w, h, r) {
  r = Math.max(0, Math.min(r, w / 2, h));
  return `M${x},${y + h}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h}Z`;
}
function barChart(el, rows, o) {
  rows = rows.filter((r) => r && isNum(o.val(r)));
  if (!rows.length) return emptyChart(el);
  const W = el.clientWidth || 400, H = W < 420 ? 220 : 240;
  const M = { t: 22, r: 8, b: 40, l: 40 };
  const vals = rows.map(o.val);
  const lo = o.base;
  const step = niceStep(Math.max(...vals, o.ref || 0) - lo, 4);
  const hi = Math.ceil((Math.max(...vals, o.ref || 0) + step * 0.3) / step) * step;
  const y = (v) => M.t + (1 - (v - lo) / (hi - lo)) * (H - M.t - M.b);
  const bw = (W - M.l - M.r) / rows.length, gap = Math.max(4, bw * 0.28);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" aria-hidden="true"><g class="grid">`;
  for (let v = lo; v <= hi + 1e-9; v += step) s += `<line x1="${M.l}" x2="${W - M.r}" y1="${y(v)}" y2="${y(v)}"/><text x="${M.l - 8}" y="${y(v) + 4}" text-anchor="end">${Math.round(v * 100)}%</text>`;
  s += `</g>`;
  rows.forEach((r, i) => {
    const v = o.val(r), x0 = M.l + i * bw + gap / 2, w = bw - gap, top = y(Math.max(v, lo)), h = y(lo) - top;
    s += `<g class="bar" data-i="${i}"><rect class="hit" x="${M.l + i * bw}" y="${M.t - 18}" width="${bw}" height="${H - M.t - M.b + 18}"/>`;
    s += `<path d="${barPath(x0, top, w, h, 4)}" style="fill:${o.color}" opacity="${o.ref && v < o.ref ? 0.45 : 1}"/>`;
    s += `<text class="lbl" x="${x0 + w / 2}" y="${top - 6}" text-anchor="middle">${(v * 100).toFixed(1)}</text>`;
    s += `<text x="${x0 + w / 2}" y="${H - M.b + 16}" text-anchor="middle" class="lbl-2">${esc(o.label(r))}</text>`;
    if (isNum(r.n)) s += `<text x="${x0 + w / 2}" y="${H - M.b + 30}" text-anchor="middle" style="font-size:10px">${r.n >= 1000 ? (r.n / 1000).toFixed(1) + "k" : r.n}</text>`;
    s += `</g>`;
  });
  if (isNum(o.ref)) s += `<line x1="${M.l}" x2="${W - M.r}" y1="${y(o.ref)}" y2="${y(o.ref)}" style="stroke:var(--ink-2);stroke-dasharray:4 4;stroke-width:1.2"/>`;
  s += `<line x1="${M.l}" x2="${W - M.r}" y1="${y(lo)}" y2="${y(lo)}" style="stroke:var(--line-2)"/>`;
  if (o.baseLabel) s += `<text x="${W - M.r}" y="${y(lo) - 5}" text-anchor="end" style="font-size:10px">${esc(o.baseLabel)}</text>`;
  s += `</svg>`;
  el.innerHTML = s;
  el.setAttribute("aria-label", rows.map((r) => `${o.label(r)}: ${pct(o.val(r), 1)}`).join(", "));
  el.querySelectorAll(".bar").forEach((g) => {
    const r = rows[+g.dataset.i];
    const f = (ev) => showTip(o.tip(r), ev.clientX, ev.clientY);
    g.addEventListener("pointermove", f); g.addEventListener("pointerdown", f);
    g.addEventListener("pointerleave", hideTip);
  });
}

function calChart(el, rows) {
  rows = rows.filter((r) => r && isNum(r.pred) && isNum(r.actual));
  if (!rows.length) return emptyChart(el);
  const W = el.clientWidth || 400, H = Math.min(320, Math.max(220, W * 0.72));
  const M = { t: 10, r: 12, b: 34, l: 54 };
  const all = rows.flatMap((r) => [r.pred, r.actual]);
  const lo = Math.floor(Math.min(0.5, ...all) * 10) / 10, hi = Math.min(1, Math.ceil(Math.max(...all) * 10) / 10);
  const x = (v) => M.l + ((v - lo) / (hi - lo)) * (W - M.l - M.r);
  const y = (v) => M.t + (1 - (v - lo) / (hi - lo)) * (H - M.t - M.b);
  const maxN = Math.max(...rows.map((r) => r.n || 1));
  let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" aria-hidden="true"><g class="grid">`;
  for (let v = lo; v <= hi + 1e-9; v += 0.1) {
    s += `<line x1="${M.l}" x2="${W - M.r}" y1="${y(v)}" y2="${y(v)}"/><text x="${M.l - 8}" y="${y(v) + 4}" text-anchor="end">${Math.round(v * 100)}%</text>`;
    s += `<text x="${x(v)}" y="${H - M.b + 16}" text-anchor="middle">${Math.round(v * 100)}%</text>`;
  }
  s += `</g><line x1="${x(lo)}" y1="${y(lo)}" x2="${x(hi)}" y2="${y(hi)}" style="stroke:var(--ink-3);stroke-dasharray:4 4"/>`;
  s += `<text x="${x(hi) - 4}" y="${y(hi) + 14}" text-anchor="end" style="font-size:10px">perfect</text>`;
  s += `<path d="${rows.map((r, i) => `${i ? "L" : "M"}${x(r.pred).toFixed(1)},${y(r.actual).toFixed(1)}`).join("")}" fill="none" style="stroke:var(--series-1);stroke-width:2" opacity=".5"/>`;
  rows.forEach((r, i) => {
    const rad = 4 + 5 * Math.sqrt((r.n || 1) / maxN);
    s += `<g class="pt" data-i="${i}"><circle cx="${x(r.pred)}" cy="${y(r.actual)}" r="${rad + 8}" class="hit"/><circle cx="${x(r.pred)}" cy="${y(r.actual)}" r="${rad}" style="fill:var(--series-1);stroke:var(--card);stroke-width:2"/></g>`;
  });
  s += `<text x="${(M.l + W - M.r) / 2}" y="${H - 4}" text-anchor="middle">Predicted win probability</text>`;
  s += `<text transform="translate(11 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle">Actual win rate</text></svg>`;
  el.innerHTML = s;
  el.setAttribute("aria-label", "Calibration: " + rows.map((r) => `predicted ${pct(r.pred)} actual ${pct(r.actual)}`).join("; "));
  el.querySelectorAll(".pt").forEach((g) => {
    const r = rows[+g.dataset.i];
    const f = (ev) => showTip(`<div class="tt-t">Predicted ${pct(r.pred, 1)}</div><div class="tt-r"><span>Actual</span><b>${pct(r.actual, 1)}</b></div><div class="tt-r"><span>Games</span><b>${isNum(r.n) ? r.n.toLocaleString() : "—"}</b></div>`, ev.clientX, ev.clientY);
    g.addEventListener("pointermove", f); g.addEventListener("pointerdown", f);
    g.addEventListener("pointerleave", hideTip);
  });
}

/* --------------------------------------------------------------- controls */

function segInit(seg) {
  if (!seg) return;
  if (!seg.querySelector(".seg-ind")) seg.insertAdjacentHTML("afterbegin", '<span class="seg-ind" aria-hidden="true"></span>');
  segMove(seg);
}
function segMove(seg) {
  const ind = seg.querySelector(".seg-ind");
  const on = seg.querySelector('button[aria-pressed="true"], button[aria-selected="true"]');
  if (!ind) return;
  if (!on || !on.offsetWidth) { ind.style.opacity = on ? "" : "0"; return; }
  ind.style.opacity = "";
  ind.style.width = on.offsetWidth + "px";
  ind.style.transform = `translateX(${on.offsetLeft}px)`;
  if (!ind.classList.contains("ready")) requestAnimationFrame(() => requestAnimationFrame(() => ind.classList.add("ready")));
}
function press(seg, btn) {
  const attr = btn.hasAttribute("aria-selected") ? "aria-selected" : "aria-pressed";
  seg.querySelectorAll("button").forEach((b) => b.setAttribute(attr, String(b === btn)));
  segMove(seg);
}
const allSegs = () => $$(".seg");

function setView(view, push = true) {
  state.view = view === "models" ? "models" : "picks";
  $$("#tabs button").forEach((b) => { b.setAttribute("aria-selected", String(b.dataset.view === state.view)); b.tabIndex = b.dataset.view === state.view ? 0 : -1; });
  segMove($("#tabs"));
  for (const v of ["picks", "models"]) {
    const el = $(`#view-${v}`);
    el.hidden = v !== state.view;
    el.classList.toggle("on", v === state.view);
  }
  if (state.view === "models") {
    if (!renderModels.done) { renderModels(); renderModels.done = true; segInit($("#bt-metric")); }
    else { for (const [el, fn] of charts) { if (el._w !== el.clientWidth) { el._w = el.clientWidth; fn(el); } } segMove($("#bt-metric")); }
  } else {
    segMove($("#weeks")); segMove($("#sort-seg")); segMove($("#layout-seg"));
  }
  if (push) writeHash();
}
function setWeek(w, push = true) {
  if (!D.weeks.includes(w)) return;
  state.week = w;
  renderPicks();
  if (push) writeHash();
}
function writeHash() {
  const h = state.view === "models" ? "#models" : state.week != null && state.week !== D.week ? `#week-${state.week}` : "#picks";
  if (location.hash !== h) history.replaceState(null, "", h === "#picks" ? location.pathname + location.search : h);
}
function readHash() {
  const h = location.hash;
  const m = /week-(\d+)/.exec(h);
  if (m && D.weeks.includes(+m[1])) state.week = +m[1];
  state.view = /^#models/.test(h) ? "models" : "picks";
}

function bind() {
  $("#tabs").addEventListener("click", (e) => { const b = e.target.closest("button[data-view]"); if (b) { setView(b.dataset.view); scrollTo({ top: 0, behavior: reduceMotion ? "auto" : "smooth" }); } });
  $("#tabs").addEventListener("keydown", (e) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    const next = state.view === "picks" ? "models" : "picks";
    setView(next); $(`#tab-${next}`).focus();
  });
  $("#weeks").addEventListener("click", (e) => { const b = e.target.closest("button[data-week]"); if (b) setWeek(+b.dataset.week); });
  $("#sort-seg").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-sort]"); if (!b) return;
    press($("#sort-seg"), b); state.sort = b.dataset.sort; store.set("sort", state.sort); renderSheet();
  });
  $("#layout-seg").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-layout]"); if (!b) return;
    press($("#layout-seg"), b); state.layout = b.dataset.layout; store.set("layout", state.layout); renderSheet();
  });
  $("#bt-metric").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-metric]"); if (!b) return;
    press($("#bt-metric"), b); state.metric = b.dataset.metric; drawSeasonChart($("#season-chart"));
  });
  document.addEventListener("click", (e) => {
    const go = e.target.closest("[data-go]");
    if (go) { e.preventDefault(); setView(go.dataset.go); scrollTo({ top: 0, behavior: reduceMotion ? "auto" : "smooth" }); return; }
    const card = e.target.closest("[data-open], .game[data-id], .st tbody tr[data-id]");
    if (card) { openDetail(card.dataset.open || card.dataset.id); return; }
    if (e.target.closest("[data-close]")) closeDetail();
  });
  $("#sheet").addEventListener("keydown", (e) => {
    const tr = e.target.closest("tr[data-id]");
    if (tr && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); openDetail(tr.dataset.id); }
  });
  const dlg = $("#detail");
  dlg.addEventListener("click", (e) => { if (e.target === dlg) closeDetail(); });
  dlg.addEventListener("close", hideTip);

  $("#theme-btn").addEventListener("click", () => {
    const cur = document.documentElement.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    store.set("theme", next);
  });

  // header shadow + pinned toolbar
  const hdr = $("#site-header");
  const measure = () => document.documentElement.style.setProperty("--hdr", hdr.offsetHeight + "px");
  measure();
  let ticking = false;
  const onScroll = () => {
    if (ticking) return; ticking = true;
    requestAnimationFrame(() => {
      ticking = false;
      hdr.classList.toggle("scrolled", scrollY > 4);
      const sen = $("#tb-sentinel");
      if (state.view === "picks" && sen) $("#tb-wrap").classList.toggle("pinned", sen.getBoundingClientRect().top < hdr.offsetHeight);
    });
  };
  addEventListener("scroll", onScroll, { passive: true });
  addEventListener("resize", () => { measure(); allSegs().forEach(segMove); onScroll(); });
  addEventListener("hashchange", () => { const w = state.week; readHash(); if (state.week !== w) renderPicks(); setView(state.view, false); });
  document.fonts?.ready.then(() => { allSegs().forEach(segMove); measure(); });
  addEventListener("scroll", hideTip, { passive: true });
}

/* ------------------------------------------------------------------- boot */

function fmtUpdated(s) {
  const d = new Date(s);
  if (!s || isNaN(d)) return "";
  return "Updated " + d.toLocaleString("en-US", { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
}

async function boot() {
  try {
    const res = await fetch("data/picks.json", { cache: "no-cache" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    D = normalize(await res.json());
  } catch (err) {
    console.warn("picks.json failed to load", err);
    $("#updated").textContent = "Data unavailable";
    $("#sheet").innerHTML = `<div class="error">Couldn't load this week's picks. Try again in a minute.</div>`;
    return;
  }
  state.week = D.week;
  state.sort = store.get("sort") === "conf" ? "conf" : "time";
  state.layout = store.get("layout") === "table" ? "table" : "cards";
  readHash();
  $("#updated").textContent = fmtUpdated(D.updated) || (D.season ? `${D.season} season` : "");
  press($("#sort-seg"), $(`#sort-seg [data-sort="${state.sort}"]`));
  press($("#layout-seg"), $(`#layout-seg [data-layout="${state.layout}"]`));
  bind();
  ["#tabs", "#sort-seg", "#layout-seg"].forEach((s) => segInit($(s)));
  renderPicks();
  setView(state.view, false);
}

boot();
