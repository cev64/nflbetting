"use strict";

// ---------- state ----------

const state = {
  seasons: [],
  season: null,
  type: "REG",
  span: 0,
  venue: "all",
  mode: "pg",
  view: "under",
  a: null,
  b: null,
  sort: { key: "diff", dir: "desc" },
  atsSample: "season", // "season" | "all"
  atsSort: { key: "ats_pct", dir: "desc" },
  kickSort: {}, // per side ("off" / "def"): { key, dir }
  kickAll: false, // weekly kicker chart: all kickers instead of the top 10
  underSeason: null, // 1H Under lookback: season and week ("next" = upcoming)
  underWeek: null,
};
const cache = {};
let data = null; // current season payload

const $ = (sel) => document.querySelector(sel);

// ---------- motion kit (see the Fluid Glass guide) ----------

const EASE = "cubic-bezier(.22,1,.36,1)";
const REDUCE = matchMedia("(prefers-reduced-motion: reduce)");
const TINT = { up: "21,128,61", down: "220,38,38", accent: "16,89,252" };

// Restart a CSS animation class.
function replay(el, cls) {
  el.classList.remove(cls);
  void el.offsetWidth;
  el.classList.add(cls);
}

// Fading colour wash.
function tint(el, rgb, { delay = 0, a = 0.16, duration = 1400 } = {}) {
  if (REDUCE.matches) return;
  el.animate([{ backgroundColor: `rgba(${rgb},${a})` }, { backgroundColor: `rgba(${rgb},0)` }], { duration, easing: "ease-out", delay });
}

// One white pill that glides to the selected button of a segmented control.
function glideIndicator(box) {
  let ind = box.querySelector(":scope > .seg-ind");
  if (!ind) {
    ind = document.createElement("span");
    ind.className = "seg-ind";
    box.prepend(ind);
  }
  const on = box.querySelector('[aria-selected="true"],[aria-pressed="true"]');
  if (!on || !on.offsetWidth) return;
  ind.style.width = `${on.offsetWidth}px`;
  ind.style.transform = `translateX(${on.offsetLeft}px)`;
  // Transition only after the first placement, so nothing slides in on load.
  if (!ind.classList.contains("ready")) requestAnimationFrame(() => requestAnimationFrame(() => ind.classList.add("ready")));
}
const glideAll = () => document.querySelectorAll(".seg").forEach(glideIndicator);

// FLIP: rows keyed by data-k slide from where they were, washing green when
// they moved up and red when they moved down; new rows fade in.
function snapshot(root) {
  if (!root || !root.offsetParent || REDUCE.matches) return null;
  const m = new Map();
  root.querySelectorAll("[data-k]").forEach((el) => m.set(el.dataset.k, el.getBoundingClientRect().top));
  return m.size ? m : null;
}
function flip(root, before) {
  if (!before || !root) return;
  root.querySelectorAll("[data-k]").forEach((el) => {
    const was = before.get(el.dataset.k);
    if (was == null) {
      el.animate([{ opacity: 0, transform: "translateY(6px)" }, { opacity: 1, transform: "none" }], { duration: 450, easing: EASE });
      return;
    }
    const dy = was - el.getBoundingClientRect().top;
    if (Math.abs(dy) > 1) {
      el.animate([{ transform: `translateY(${dy}px)` }, { transform: "none" }], { duration: 650, easing: EASE, fill: "backwards" });
      tint(el, dy > 0 ? TINT.up : TINT.down, { a: 0.1 });
    }
  });
}

// Tooltip: a glass popover that follows the pointer.
const tip = () => $("#tooltip");
function showTip(html, ev) {
  const t = tip();
  t.innerHTML = html;
  t.classList.add("on");
  const x = Math.min(ev.clientX + 14, window.innerWidth - t.offsetWidth - 8);
  const y = ev.clientY + 16 + t.offsetHeight > window.innerHeight - 8 ? ev.clientY - t.offsetHeight - 12 : ev.clientY + 16;
  t.style.left = `${Math.max(8, x)}px`;
  t.style.top = `${y}px`;
}
const hideTip = () => tip().classList.remove("on");

// ---------- data ----------

async function loadJSON(path) {
  const res = await fetch(path, { cache: "no-cache" });
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json();
}

async function loadSeason(season) {
  if (!cache[season]) cache[season] = await loadJSON(`data/${season}.json`);
  return cache[season];
}

function teamGames(team, d = data) {
  let g = d.games.filter((x) => x.team === team);
  if (state.type === "REG") g = g.filter((x) => x.type === "REG");
  else if (state.type === "POST") g = g.filter((x) => x.type !== "REG");
  if (state.venue === "home") g = g.filter((x) => x.home);
  else if (state.venue === "away") g = g.filter((x) => !x.home);
  g.sort((a, b) => (a.date < b.date ? -1 : 1));
  if (state.span > 0) g = g.slice(-state.span);
  return g;
}

// Against the spread, from the team's side. `line` is the team's spread
// (negative = favored), so it covers when final margin + line > 0.
function ats(g) {
  if (g.line == null) return null;
  const margin = g.pf - g.pa + g.line;
  return { margin, res: margin > 0 ? "W" : margin < 0 ? "L" : "P" };
}
const toDiff = (g) => g.int_made + g.fum_rec - g.int_thrown - g.fum_lost;

function atsTally(games) {
  const t = { w: 0, l: 0, p: 0, n: 0, margin: 0 };
  for (const g of games) {
    const a = ats(g);
    if (!a) continue;
    t[a.res.toLowerCase()]++;
    t.n++;
    t.margin += a.margin;
  }
  t.pct = t.w + t.l ? t.w / (t.w + t.l) : null;
  t.avg = t.n ? t.margin / t.n : null;
  t.rec = t.n ? `${t.w}-${t.l}${t.p ? `-${t.p}` : ""}` : "–";
  return t;
}

// Drive / red-zone / kicking counts stored per team-game (and opp_ for the defense).
const DRIVE_KEYS = ["drives", "t40", "rz", "rz_td", "td", "fga", "fgm", "fg50", "xpa", "xpm"];

function aggregate(team, d = data) {
  const games = teamGames(team, d);
  const s = { team, games, gp: games.length, w: 0, l: 0, t: 0 };
  const keys = ["int_made", "fum_rec", "int_thrown", "fum_lost", "fumbles", "opp_fumbles", "pf", "pa", ...DRIVE_KEYS, ...DRIVE_KEYS.map((k) => `opp_${k}`)];
  for (const k of keys) s[k] = 0;
  for (const g of games) {
    for (const k of keys) s[k] += g[k] ?? 0;
    if (g.pf > g.pa) s.w++;
    else if (g.pf < g.pa) s.l++;
    else s.t++;
  }
  s.take = s.int_made + s.fum_rec;
  s.give = s.int_thrown + s.fum_lost;
  s.diff = s.take - s.give;
  s.kept_pct = s.fumbles ? (s.fumbles - s.fum_lost) / s.fumbles : null;
  s.opp_rec_pct = s.opp_fumbles ? s.fum_rec / s.opp_fumbles : null;
  // Drives & red zone: offense, and (def_) what the defense allowed.
  s.has_drives = games.some((g) => g.t40 != null);
  s.off_t40 = s.t40;
  s.def_t40 = s.opp_t40;
  s.off_rz_pct = s.rz ? s.rz_td / s.rz : null;
  s.def_rz_pct = s.opp_rz ? s.opp_rz_td / s.opp_rz : null;
  s.off_fga_trip = s.t40 ? s.fga / s.t40 : null;
  s.def_fga_trip = s.opp_t40 ? s.opp_fga / s.opp_t40 : null;
  s.ats = atsTally(games);
  s.ats_pct = s.ats.pct;
  s.ats_margin = s.ats.avg;
  return s;
}

// Metric definitions. `better`: +1 higher is better, -1 lower is better.
const METRICS = {
  take: { label: "Takeaways", short: "Total", better: 1, count: true },
  int_made: { label: "INT made", short: "INT", better: 1, count: true },
  fum_rec: { label: "Fumbles recovered", short: "FR", better: 1, count: true },
  give: { label: "Giveaways", short: "Total", better: -1, count: true },
  int_thrown: { label: "INT thrown", short: "INT", better: -1, count: true },
  fum_lost: { label: "Fumbles lost", short: "FL", better: -1, count: true },
  diff: { label: "TO differential", short: "Diff", better: 1, count: true, signed: true },
  kept_pct: { label: "Own fumbles kept", short: "Fum kept %", better: 1, pct: true },
  opp_rec_pct: { label: "Opp fumbles recovered", short: "Opp fum rec %", better: 1, pct: true },
  ats_pct: { label: "ATS cover %", short: "Cover %", better: 1, pct: true },
  ats_margin: { label: "Avg cover margin", short: "Cover margin", better: 1, signed: true, avg: true },
  // better: 0 = no "good" direction (ranked high to low, never highlighted)
  off_t40: { label: "Trips inside the 40", short: "Trips 40", better: 1, count: true },
  off_rz_pct: { label: "Red-zone TD %", short: "RZ TD %", better: 1, pct: true },
  fgm: { label: "Field goals made", short: "FGM", better: 0, count: true },
  def_t40: { label: "Trips inside the 40 allowed", short: "Trips 40", better: -1, count: true },
  def_rz_pct: { label: "Red-zone TD % allowed", short: "RZ TD %", better: -1, pct: true },
  opp_fgm: { label: "Field goals allowed", short: "FGM", better: 0, count: true },
};

function value(s, key) {
  const m = METRICS[key];
  if (!m) return s[key];
  const v = s[key];
  if (v == null) return null;
  if (m.count && state.mode === "pg") return s.gp ? v / s.gp : null;
  return v;
}

function fmt(v, key) {
  const m = METRICS[key] || {};
  if (v == null || Number.isNaN(v)) return "–";
  if (m.pct) return `${Math.round(v * 100)}%`;
  let out = m.count && state.mode === "pg" ? v.toFixed(2) : m.avg ? v.toFixed(1) : String(Math.round(v * 100) / 100);
  if (m.signed && v > 0) out = `+${out}`;
  return out;
}

function leagueStats() {
  const teams = Object.keys(data.teams).sort();
  const stats = teams.map((t) => aggregate(t)).filter((s) => s.gp > 0);
  // Ranks per metric: 1 = best.
  const ranks = {};
  for (const key of Object.keys(METRICS)) {
    const m = METRICS[key];
    const sorted = stats
      .filter((s) => value(s, key) != null)
      .sort((x, y) => (value(y, key) - value(x, key)) * (m.better || 1));
    ranks[key] = {};
    sorted.forEach((s, i) => {
      // ties share the better rank
      const prev = sorted[i - 1];
      ranks[key][s.team] = prev && value(prev, key) === value(s, key) ? ranks[key][prev.team] : i + 1;
    });
  }
  const avg = {};
  for (const key of Object.keys(METRICS)) {
    const vals = stats.map((s) => value(s, key)).filter((v) => v != null);
    avg[key] = vals.reduce((a, b) => a + b, 0) / (vals.length || 1);
  }
  return { stats, ranks, avg, byTeam: Object.fromEntries(stats.map((s) => [s.team, s])) };
}

// Local logos live in web/logos/<abbr>.png; fall back to nflverse's (ESPN) logo.
const logo = (t) => {
  const remote = data.teams[t]?.logo || "";
  return `<img class="logo" src="logos/${t}.png" alt="" loading="lazy" onerror="this.onerror=null;this.src='${remote}'">`;
};
const signed = (v, dp = 1) => (v == null ? "–" : `${v > 0 ? "+" : ""}${v.toFixed(dp)}`);
const posneg = (v) => (v > 0 ? "pos" : v < 0 ? "neg" : "");
function weekLabel(w) {
  return { 19: "Wild Card round", 20: "Divisional round", 21: "Conference championships", 22: "Super Bowl" }[w] || `week ${w}`;
}
const ordinal = (n) => {
  const s = ["th", "st", "nd", "rd"], v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
};

// Diverging shading around the league average: blue = good, red = bad.
function shade(v, key, L) {
  if (v == null) return "";
  const m = METRICS[key];
  const vals = L.stats.map((s) => value(s, key)).filter((x) => x != null);
  const spread = Math.max(...vals.map((x) => Math.abs(x - L.avg[key]))) || 1;
  const z = ((v - L.avg[key]) / spread) * m.better; // -1..1
  const alpha = Math.min(Math.abs(z), 1) * 0.32;
  if (alpha < 0.03) return "";
  const rgb = z > 0 ? "var(--good-rgb)" : "var(--bad-rgb)";
  return `background: rgba(${rgb}, ${alpha.toFixed(3)})`;
}

// ---------- league table ----------

const LEAGUE_COLS = [
  { key: "gp", label: "GP", group: "" },
  { key: "take", group: "Takeaways", sep: true, key_col: true },
  { key: "int_made", group: "Takeaways" },
  { key: "fum_rec", group: "Takeaways" },
  { key: "give", group: "Giveaways", sep: true, key_col: true },
  { key: "int_thrown", group: "Giveaways" },
  { key: "fum_lost", group: "Giveaways" },
  { key: "diff", group: "", sep: true, key_col: true },
  { key: "kept_pct", group: "Fumble luck", sep: true },
  { key: "opp_rec_pct", group: "Fumble luck" },
  { key: "ats_pct", group: "Against the spread", sep: true, label: "ATS", title: "Record against the spread (W-L-Push)", text: (s) => s.ats.rec },
  { key: "ats_pct", group: "Against the spread" },
  { key: "ats_margin", group: "Against the spread" },
];

function renderLeague(L) {
  renderLeagueCharts(L);
  const table = $("#league");
  const groups = [];
  for (const c of LEAGUE_COLS) {
    const last = groups[groups.length - 1];
    if (last && last.name === c.group && c.group) last.span++;
    else groups.push({ name: c.group, span: 1, sep: c.sep });
  }
  const { key: sk, dir } = state.sort;
  table.tHead.innerHTML =
    `<tr class="group"><th></th>${groups.map((g) => `<th colspan="${g.span}" class="${g.sep ? "sep" : ""}">${g.name}</th>`).join("")}</tr>` +
    `<tr><th class="sortable" data-key="team" ${sk === "team" ? `aria-sort="${dir}ending"` : ""}>Team</th>${LEAGUE_COLS.map(
      (c) =>
        `<th class="sortable ${c.sep ? "sep" : ""}" data-key="${c.key}" title="${c.title || METRICS[c.key]?.label || "Games played"}" ${
          sk === c.key && !c.text ? `aria-sort="${dir}ending"` : ""
        }>${c.label || METRICS[c.key]?.short}</th>`
    ).join("")}</tr>`;

  const rows = [...L.stats].sort((x, y) => {
    let a, b;
    if (sk === "team") [a, b] = [data.teams[x.team].name, data.teams[y.team].name];
    else [a, b] = [value(x, sk), value(y, sk)];
    if (a == null) return 1;
    if (b == null) return -1;
    const c = a < b ? -1 : a > b ? 1 : 0;
    return dir === "asc" ? c : -c;
  });

  table.tBodies[0].innerHTML = rows
    .map((s, i) => {
      const cells = LEAGUE_COLS.map((c) => {
        const v = value(s, c.key);
        const cls = [c.sep ? "sep" : "", c.key_col ? "key" : "", c.key === "diff" ? (v > 0 ? "pos" : v < 0 ? "neg" : "") : ""].join(" ");
        const style = c.key_col ? shade(v, c.key, L) : "";
        const txt = c.text ? c.text(s) : c.key === "gp" ? s.gp : fmt(v, c.key);
        return `<td class="${cls}" style="${style}">${txt}</td>`;
      }).join("");
      return `<tr data-k="${s.team}"><td class="team"><span class="rank">${i + 1}</span>${logo(s.team)}<a href="#" data-team="${s.team}" title="${data.teams[s.team].name}">${s.team}</a></td>${cells}</tr>`;
    })
    .join("");
}

// ---------- upcoming ----------

function lineText(g) {
  if (g.spread_line == null) return "No line yet";
  const fav = g.spread_line > 0 ? g.home : g.away;
  const spread = Math.abs(g.spread_line);
  const s = spread === 0 ? "Pick'em" : `${fav} −${spread}`;
  return `${s}${g.total_line != null ? ` · O/U ${g.total_line}` : ""}`;
}

function gameWhen(g) {
  const d = new Date(`${g.date}T12:00:00`);
  return d.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" });
}

const isPair = (g, a, b) => (g.home === a && g.away === b) || (g.home === b && g.away === a);

function renderUpcoming(L) {
  const wrap = $("#upcoming-wrap");
  const up = data.upcoming || [];
  const isLatest = state.season === state.seasons[0];
  wrap.hidden = !(isLatest && up.length);
  if (wrap.hidden) return;
  $("#upcoming-title").textContent = `${weekLabel(up[0].week).replace(/^w/, "W")} games · turnover diff per game`;
  const was = new Set([...document.querySelectorAll("#upcoming .game.on")].map((el) => el.dataset.k));
  const pg = (t) => {
    const s = L.byTeam[t];
    if (!s || !s.gp) return "–";
    const v = s.diff / s.gp;
    return `<span class="${v > 0 ? "pos" : v < 0 ? "neg" : ""}">${v > 0 ? "+" : ""}${v.toFixed(2)}</span>`;
  };
  const html = up
    .map(
      (g, i) => `<button class="game" style="--n:${i}" data-k="${g.game_id}" data-a="${g.away}" data-b="${g.home}">
        <div class="when"><span>${gameWhen(g)}</span><span>${
          g.home_score != null ? `Final ${g.away_score}-${g.home_score}` : g.time ? `${g.time} ET` : ""
        }</span></div>
        <div class="row">${logo(g.away)}<span class="abbr">${g.away}</span><span class="val">${pg(g.away)}</span></div>
        <div class="row">${logo(g.home)}<span class="abbr">@${g.home}</span><span class="val">${pg(g.home)}</span></div>
        <div class="line">${lineText(g)}</div>
      </button>`
    )
    .join("");
  // Only rebuild when the tiles' content changed, so the strip keeps its scroll position.
  const box = $("#upcoming");
  if (box.dataset.html !== html) {
    box.dataset.html = html;
    box.innerHTML = html;
    replay(box, "enter");
  }
  box.querySelectorAll(".game").forEach((el) => {
    const g = { home: el.dataset.b, away: el.dataset.a };
    const sel = state.view === "matchup" && isPair(g, state.a, state.b);
    el.classList.toggle("on", sel);
    el.setAttribute("aria-pressed", sel);
    if (sel && !was.has(el.dataset.k)) replay(el, "pop");
  });
}

// ---------- matchup ----------

const CMP_ROWS = [
  "take", "give", "diff", "ats_pct", "ats_margin",
  { sec: "Red zone & kicking" }, "off_t40", "off_rz_pct", "fgm", "def_t40", "def_rz_pct", "opp_fgm",
];

function renderMatchup(L) {
  const teams = Object.keys(data.teams).sort((x, y) => data.teams[x].name.localeCompare(data.teams[y].name));
  if (!state.a || !data.teams[state.a]) state.a = teams[0];
  if (!state.b || !data.teams[state.b] || state.b === state.a) state.b = teams.find((t) => t !== state.a);
  for (const [id, cur] of [["#m-a", state.a], ["#m-b", state.b]]) {
    $(id).innerHTML = teams.map((t) => `<option value="${t}" ${t === cur ? "selected" : ""}>${data.teams[t].name}</option>`).join("");
  }

  const A = L.byTeam[state.a], B = L.byTeam[state.b];
  const up = (data.upcoming || []).find(
    (g) => (g.home === state.a && g.away === state.b) || (g.home === state.b && g.away === state.a)
  );
  $("#m-line").innerHTML =
    up && state.season === state.seasons[0]
      ? `<b>Week ${up.week}</b> · ${gameWhen(up)} ${up.time ? `${up.time} ET` : ""} · ${up.away} @ ${up.home} · ${lineText(up)}`
      : "";

  if (!A || !B) {
    $("#m-compare").innerHTML = `<div class="cmp-row"><span class="muted">No games for one of these teams with the current filters.</span></div>`;
    $("#m-edges").innerHTML = $("#m-chart").innerHTML = $("#m-legend").innerHTML = "";
    $("#m-log-a").innerHTML = $("#m-log-b").innerHTML = "";
    return;
  }

  // Head-to-head: mirrored bars, longer = better league rank (or simply "more" where neither is better).
  const rec = (s) => `${s.w}-${s.l}${s.t ? `-${s.t}` : ""}`;
  const n = L.stats.length;
  const len = (key, t) => (L.ranks[key][t] ? ((n - L.ranks[key][t] + 1) / n) * 100 : 0);
  const side = (s, cls) => `<div class="t ${cls}">${cls === "b" ? "" : logo(s.team)}<div><div class="nm">${data.teams[s.team].nick || s.team}</div>
    <div class="rec">${rec(s)} · ATS ${s.ats.rec}</div></div>${cls === "b" ? logo(s.team) : ""}</div>`;
  const rows = (A.has_drives && B.has_drives ? CMP_ROWS : CMP_ROWS.slice(0, CMP_ROWS.findIndex((k) => k.sec))).map((key) => {
    if (key.sec) return `<div class="bf-sec micro">${key.sec}</div>`;
    const m = METRICS[key];
    const va = value(A, key), vb = value(B, key);
    const rank = (t) => (L.ranks[key][t] ? ordinal(L.ranks[key][t]) : "");
    return `<div class="bf-row"><div class="val">${fmt(va, key)}<small>${rank(A.team)}</small></div>
      <div class="bar a"><i style="width:${len(key, A.team)}%"></i></div><div class="lbl">${m.label}</div>
      <div class="bar b"><i style="width:${len(key, B.team)}%"></i></div><div class="val b">${fmt(vb, key)}<small>${rank(B.team)}</small></div></div>`;
  }).join("");
  $("#m-compare").innerHTML = `<div class="bf-head">${side(A, "a")}<span class="micro">vs</span>${side(B, "b")}</div>
    <div class="bf-key">${state.mode === "pg" ? "Per game" : "Totals"} · longer bar = better league rank (for field goals: more)</div>${rows}`;

  // Offense-vs-defense edges, always per game so the two sides are comparable.
  const per = (s, k) => (s.gp ? s[k] / s.gp : 0);
  const lgMax = Math.max(...L.stats.flatMap((s) => [per(s, "give"), per(s, "take")]), 0.1);
  const lgAvg = L.stats.reduce((a, s) => a + per(s, "give"), 0) / L.stats.length;
  const edge = (off, def, offColor, defColor) => {
    const g = per(off, "give"), t = per(def, "take");
    const proj = (g + t) / 2;
    const bar = (label, color, v) =>
      `<div class="bar-row"><span class="lbl"><i class="swatch" style="background:${color}"></i>${label}</span><div class="track"><div class="fill" style="width:${(v / lgMax) * 100}%;background:${color}"></div></div><span class="num">${v.toFixed(2)}</span></div>`;
    return {
      proj,
      html: `<div class="edge">
        <h4>${off.team} offense vs ${def.team} defense</h4>
        <p>Per game. League average is ${lgAvg.toFixed(2)} turnovers per team-game.</p>
        ${bar(`${off.team} giveaways (${ordinal(L.ranks.give[off.team])})`, offColor, g)}
        ${bar(`${def.team} takeaways (${ordinal(L.ranks.take[def.team])})`, defColor, t)}
        <div class="verdict">Blended expectation: <b>${proj.toFixed(2)}</b> ${off.team} turnovers
          (${proj > lgAvg + 0.15 ? "above" : proj < lgAvg - 0.15 ? "below" : "near"} league average).</div>
      </div>`,
    };
  };
  const e1 = edge(A, B, "var(--series-1)", "var(--series-2)");
  const e2 = edge(B, A, "var(--series-2)", "var(--series-1)");
  const margin = e2.proj - e1.proj;
  const leader = margin >= 0 ? A.team : B.team;
  $("#m-edges").innerHTML =
    e1.html +
    e2.html +
    `<div class="edge"><h4>Projected turnover margin</h4><p>Naive average of each offense's giveaway rate and the opposing defense's takeaway rate.</p>
      <div class="big">${Math.abs(margin) < 0.05 ? "Even" : `${leader} +${Math.abs(margin).toFixed(2)}`}</div>
      <div class="verdict">${A.team} ${e2.proj.toFixed(2)} takeaways vs ${B.team} ${e1.proj.toFixed(2)} expected.
      Small samples swing hard; check the span filter.</div></div>`;

  renderChart(A, B);
  renderLog("#m-log-a", A);
  renderLog("#m-log-b", B);
  $("#m-log-a-title").innerHTML = `${data.teams[A.team].name} game log`;
  $("#m-log-b-title").innerHTML = `${data.teams[B.team].name} game log`;
}

function renderLog(sel, s) {
  const rows = [...s.games].reverse().map((g) => {
    const res = g.pf > g.pa ? "W" : g.pf < g.pa ? "L" : "T";
    const ta = g.int_made + g.fum_rec, ga = g.int_thrown + g.fum_lost, d = ta - ga;
    const wk = g.type === "REG" ? g.week : g.type;
    const a = ats(g);
    const line = g.line == null ? "–" : g.line === 0 ? "PK" : signed(g.line);
    const atsCell = a ? `<span class="${a.res === "W" ? "pos" : a.res === "L" ? "neg" : ""}">${a.res} ${signed(a.margin)}</span>` : "–";
    return `<tr><td>${wk}</td><td>${g.home ? "vs" : "@"} ${g.opp}</td><td>${res} ${g.pf}-${g.pa}</td>
      <td>${line}</td><td>${atsCell}</td>
      <td class="sep key">${ta}</td><td>${g.int_made}</td><td>${g.fum_rec}</td>
      <td class="sep key">${ga}</td><td>${g.int_thrown}</td><td>${g.fum_lost}</td>
      <td class="sep key ${d > 0 ? "pos" : d < 0 ? "neg" : ""}">${d > 0 ? "+" : ""}${d}</td></tr>`;
  });
  $(sel).innerHTML = `<thead><tr><th>Wk</th><th>Opp</th><th>Result</th><th title="Team's spread (nflverse)">Line</th><th title="Against the spread: result and cover margin">ATS</th><th class="sep" title="Takeaways">TA</th><th title="Interceptions made">INT</th><th title="Fumbles recovered">FR</th><th class="sep" title="Giveaways">GA</th><th title="Interceptions thrown">INT</th><th title="Fumbles lost">FL</th><th class="sep">Diff</th></tr></thead><tbody>${rows.join("")}</tbody>`;
}

// ---------- chart: cumulative TO diff by game ----------

function renderChart(A, B) {
  const box = $("#m-chart");
  const series = [
    { s: A, color: "var(--series-1)" },
    { s: B, color: "var(--series-2)" },
  ].map(({ s, color }) => {
    let cum = 0;
    const pts = s.games.map((g, i) => {
      const d = g.int_made + g.fum_rec - g.int_thrown - g.fum_lost;
      cum += d;
      return { i: i + 1, d, cum, g };
    });
    return { team: s.team, color, pts };
  });
  $("#m-legend").innerHTML = series
    .map((x) => `<span><i style="background:${x.color}"></i>${data.teams[x.team].name}</span>`)
    .join("");

  const W = Math.max(box.clientWidth, 280), H = 240;
  const m = { t: 12, r: 60, b: 28, l: 34 };
  const n = Math.max(...series.map((x) => x.pts.length), 1);
  const vals = series.flatMap((x) => x.pts.map((p) => p.cum)).concat([0]);
  let lo = Math.min(...vals), hi = Math.max(...vals);
  if (hi - lo < 4) { hi += 2; lo -= 2; }
  const step = Math.max(1, Math.ceil((hi - lo) / 6));
  lo = Math.floor(lo / step) * step;
  hi = Math.ceil(hi / step) * step;
  const x = (i) => m.l + (n === 1 ? (W - m.l - m.r) / 2 : ((i - 1) / (n - 1)) * (W - m.l - m.r));
  const y = (v) => m.t + ((hi - v) / (hi - lo)) * (H - m.t - m.b);

  let svg = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Cumulative turnover differential by game">`;
  svg += `<g class="grid">`;
  for (let v = lo; v <= hi; v += step) svg += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}"/>`;
  svg += `</g><line class="zero" x1="${m.l}" x2="${W - m.r}" y1="${y(0)}" y2="${y(0)}"/>`;
  svg += `<g class="axis">`;
  for (let v = lo; v <= hi; v += step) svg += `<text x="${m.l - 8}" y="${y(v) + 4}" text-anchor="end">${v > 0 ? "+" : ""}${v}</text>`;
  const every = Math.ceil(n / Math.floor((W - m.l - m.r) / 36));
  for (let i = 1; i <= n; i++) if ((i - 1) % every === 0 ? n - i >= every || i === n : i === n) svg += `<text x="${x(i)}" y="${H - 8}" text-anchor="middle">G${i}</text>`;
  svg += `</g><line class="hover-line" id="hover-line" y1="${m.t}" y2="${H - m.b}" x1="-10" x2="-10"/>`;
  for (const s of series) {
    if (!s.pts.length) continue;
    const d = s.pts.map((p, k) => `${k ? "L" : "M"}${x(p.i)},${y(p.cum)}`).join("");
    svg += `<g class="series"><path d="${d}" style="stroke:${s.color}"/>`;
    for (const p of s.pts) svg += `<circle cx="${x(p.i)}" cy="${y(p.cum)}" r="4" style="fill:${s.color}"/>`;
    const last = s.pts[s.pts.length - 1];
    svg += `</g><text class="endlabel" x="${x(last.i) + 8}" y="${y(last.cum) + 4}">${s.team} ${last.cum > 0 ? "+" : ""}${last.cum}</text>`;
  }
  svg += `<rect id="hover-zone" x="${m.l}" y="0" width="${W - m.l - m.r}" height="${H}" fill="transparent"/></svg>`;
  box.innerHTML = svg;

  // Crosshair tooltip: snap to the nearest game index.
  const zone = $("#hover-zone"), line = $("#hover-line");
  const svgEl = box.querySelector("svg");
  zone.addEventListener("mousemove", (ev) => {
    const r = svgEl.getBoundingClientRect();
    const px = ((ev.clientX - r.left) / r.width) * W;
    const i = n === 1 ? 1 : Math.min(n, Math.max(1, Math.round(((px - m.l) / (W - m.l - m.r)) * (n - 1)) + 1));
    line.setAttribute("x1", x(i));
    line.setAttribute("x2", x(i));
    const rows = series
      .map((s) => {
        const p = s.pts[i - 1];
        if (!p) return "";
        const wk = p.g.type === "REG" ? `Wk ${p.g.week}` : p.g.type;
        return `<div class="tt-row"><i style="background:${s.color}"></i><b>${s.team}</b>&nbsp;${wk} ${p.g.home ? "vs" : "@"} ${p.g.opp}: ${p.d > 0 ? "+" : ""}${p.d} (total ${p.cum > 0 ? "+" : ""}${p.cum})</div>`;
      })
      .join("");
    showTip(`<div class="tt-title">Game ${i}</div>${rows}`, ev);
  });
  zone.addEventListener("mouseleave", () => {
    hideTip();
    line.setAttribute("x1", -10);
    line.setAttribute("x2", -10);
  });
}

// ---------- ATS view: turnover margin vs covering the spread ----------

const IN_GAME_BUCKETS = [
  { label: "−3 or worse", test: (v) => v <= -3 },
  { label: "−2", test: (v) => v === -2 },
  { label: "−1", test: (v) => v === -1 },
  { label: "0", test: (v) => v === 0 },
  { label: "+1", test: (v) => v === 1 },
  { label: "+2", test: (v) => v === 2 },
  { label: "+3 or better", test: (v) => v >= 3 },
];
const EDGE_BUCKETS = [
  { label: "< −1.0", test: (v) => v < -1 },
  { label: "−1.0 to −0.5", test: (v) => v >= -1 && v < -0.5 },
  { label: "−0.5 to 0", test: (v) => v >= -0.5 && v < 0 },
  { label: "0 to +0.5", test: (v) => v >= 0 && v < 0.5 },
  { label: "+0.5 to +1.0", test: (v) => v >= 0.5 && v < 1 },
  { label: "≥ +1.0", test: (v) => v >= 1 },
];
const MIN_PRIOR = 3; // games of history needed before a team's "entering" TO rate counts

// One row per team-game with a line: the in-game TO margin and the gap in
// season-to-date TO diff/game between the team and its opponent entering it.
function atsSamples(datasets) {
  const out = [];
  for (const d of datasets) {
    const byTeam = {};
    for (const g of d.games) (byTeam[g.team] ||= []).push(g);
    for (const list of Object.values(byTeam)) list.sort((a, b) => (a.date < b.date ? -1 : 1));
    const entering = (team, date) => {
      const prior = (byTeam[team] || []).filter((x) => x.date < date);
      return prior.length >= MIN_PRIOR ? prior.reduce((a, x) => a + toDiff(x), 0) / prior.length : null;
    };
    for (const team of Object.keys(d.teams)) {
      for (const g of teamGames(team, d)) {
        const a = ats(g);
        if (!a) continue;
        const mine = entering(team, g.date), theirs = entering(g.opp, g.date);
        out.push({ season: d.season, team, g, a, to: toDiff(g), edge: mine != null && theirs != null ? mine - theirs : null });
      }
    }
  }
  return out;
}

function bucketize(samples, buckets, key) {
  return buckets.map((b) => {
    const t = { label: b.label, w: 0, l: 0, p: 0 };
    for (const x of samples) if (x[key] != null && b.test(x[key])) t[x.a.res.toLowerCase()]++;
    t.n = t.w + t.l + t.p;
    t.pct = t.w + t.l ? t.w / (t.w + t.l) : null;
    return t;
  });
}

function renderBars(box, bars, ariaLabel) {
  const W = Math.max(box.clientWidth, 280), H = 250;
  const m = { t: 22, r: 8, b: 44, l: 40 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const band = iw / bars.length;
  const bw = Math.min(32, band * 0.6);
  const y = (v) => m.t + (1 - v) * ih;
  let svg = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${ariaLabel}"><g class="grid">`;
  for (const v of [0, 0.25, 0.75, 1]) svg += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}"/>`;
  svg += `</g><g class="axis">`;
  for (const v of [0, 0.25, 0.5, 0.75, 1]) svg += `<text x="${m.l - 8}" y="${y(v) + 4}" text-anchor="end">${v * 100}%</text>`;
  svg += `</g>`;
  bars.forEach((b, i) => {
    const cx = m.l + band * (i + 0.5);
    svg += `<g class="bar" data-i="${i}">`;
    svg += `<rect class="hit" x="${cx - band / 2}" y="${m.t}" width="${band}" height="${ih + m.b}" fill="transparent"/>`;
    if (b.pct != null) {
      const top = y(b.pct), h = y(0) - top;
      const r = Math.min(4, h);
      // rounded data-end, square at the baseline
      svg += `<path d="M${cx - bw / 2},${y(0)} V${top + r} Q${cx - bw / 2},${top} ${cx - bw / 2 + r},${top} H${cx + bw / 2 - r} Q${cx + bw / 2},${top} ${cx + bw / 2},${top + r} V${y(0)} Z" style="fill:var(--series-1)"/>`;
      svg += `<text class="cap" x="${cx}" y="${top - 6}" text-anchor="middle">${Math.round(b.pct * 100)}%</text>`;
    }
    svg += `<text class="xl" x="${cx}" y="${H - m.b + 16}" text-anchor="middle">${b.label}</text>`;
    svg += `<text class="xn" x="${cx}" y="${H - m.b + 30}" text-anchor="middle">n=${b.n}</text></g>`;
  });
  svg += `<line class="ref" x1="${m.l}" x2="${W - m.r}" y1="${y(0.5)}" y2="${y(0.5)}"/></svg>`;
  box.innerHTML = svg;
  box.querySelectorAll(".bar").forEach((g) => {
    const b = bars[Number(g.dataset.i)];
    g.addEventListener("mousemove", (ev) => {
      showTip(`<div class="tt-title">${b.label}</div>ATS ${b.w}-${b.l}${b.p ? `-${b.p}` : ""} · ${
        b.pct == null ? "no decisions" : `covered ${(b.pct * 100).toFixed(1)}%`
      }`, ev);
    });
    g.addEventListener("mouseleave", hideTip);
  });
}

function renderATS() {
  document.querySelectorAll("#ats-sample button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.sample === state.atsSample));
  const datasets = state.atsSample === "all" ? state.seasons.map((x) => cache[x]).filter(Boolean) : [data];
  const seasons = datasets.map((d) => d.season).sort();
  $("#ats-sample-all").textContent = `All seasons (${Math.min(...state.seasons)}–${Math.max(...state.seasons)})`;
  const samples = atsSamples(datasets);

  // Headline tiles: won / even / lost the turnover battle.
  const tally = (f) => atsTally(samples.filter(f).map((x) => x.g));
  const tiles = [
    { label: "Won the turnover battle", t: tally((x) => x.to > 0) },
    { label: "Even turnovers", t: tally((x) => x.to === 0) },
    { label: "Lost the turnover battle", t: tally((x) => x.to < 0) },
  ];
  $("#ats-tiles").innerHTML = tiles
    .map(
      ({ label, t }) => `<div class="tile"><div class="tile-label">${label}</div>
        <div class="tile-value">${t.pct == null ? "–" : `${Math.round(t.pct * 100)}%`}</div>
        <div class="tile-sub">covered · ATS ${t.rec} · avg cover margin ${signed(t.avg)}</div></div>`
    )
    .join("");

  const scope = `${seasons.length > 1 ? `${seasons[0]}–${seasons[seasons.length - 1]}` : seasons[0]} · ${samples.length} team-games with a line`;
  $("#ats-scope").textContent = scope;
  renderBars($("#ats-chart-ingame"), bucketize(samples, IN_GAME_BUCKETS, "to"), "Cover rate by turnover margin in the game");
  renderBars($("#ats-chart-edge"), bucketize(samples, EDGE_BUCKETS, "edge"), "Cover rate by turnover edge entering the game");

  // Team table
  const teams = {};
  for (const x of samples) (teams[x.team] ||= []).push(x);
  const rows = Object.entries(teams).map(([team, xs]) => {
    const all = atsTally(xs.map((x) => x.g));
    const won = atsTally(xs.filter((x) => x.to > 0).map((x) => x.g));
    const even = atsTally(xs.filter((x) => x.to === 0).map((x) => x.g));
    const lost = atsTally(xs.filter((x) => x.to < 0).map((x) => x.g));
    return { team, all, won, even, lost, n: xs.length, todiff: xs.reduce((a, x) => a + x.to, 0) / xs.length };
  });
  const cols = [
    { key: "n", label: "Games", get: (r) => r.n, show: (r) => r.n },
    { key: "ats_pct", label: "ATS", get: (r) => r.all.pct, show: (r) => r.all.rec, sep: true },
    { key: "ats_pct2", label: "Cover %", get: (r) => r.all.pct, show: (r) => pct(r.all.pct) },
    { key: "ats_margin", label: "Cover margin", get: (r) => r.all.avg, show: (r) => `<span class="${posneg(r.all.avg)}">${signed(r.all.avg)}</span>` },
    { key: "todiff", label: "TO diff/g", get: (r) => r.todiff, show: (r) => `<span class="${posneg(r.todiff)}">${signed(r.todiff, 2)}</span>`, sep: true },
    { key: "won", label: "Won TO battle", get: (r) => r.won.pct, show: (r) => cell(r.won), sep: true },
    { key: "even", label: "Even", get: (r) => r.even.pct, show: (r) => cell(r.even) },
    { key: "lost", label: "Lost TO battle", get: (r) => r.lost.pct, show: (r) => cell(r.lost) },
  ];
  function pct(v) { return v == null ? "–" : `${Math.round(v * 100)}%`; }
  function cell(t) { return t.n ? `${t.rec} <span class="muted">(${pct(t.pct)})</span>` : "–"; }
  const { key: sk, dir } = state.atsSort;
  const col = cols.find((c) => c.key === sk);
  rows.sort((x, y) => {
    let a, b;
    if (sk === "team") [a, b] = [data.teams[x.team]?.name || x.team, data.teams[y.team]?.name || y.team];
    else [a, b] = [col.get(x), col.get(y)];
    if (a == null) return 1;
    if (b == null) return -1;
    const c = a < b ? -1 : a > b ? 1 : 0;
    return dir === "asc" ? c : -c;
  });
  renderScatter($("#ats-team-chart"), rows.filter((r) => r.all.pct != null).map((r) => ({
    team: r.team,
    x: r.todiff,
    y: r.all.pct,
    tip: ttTitle(r.team) + ttGrid([
      ["ATS", `${r.all.rec} (${pct(r.all.pct)})`],
      ["Avg cover margin", signed(r.all.avg)],
      ["TO diff / g", signed(r.todiff, 2)],
      ["Won TO battle", cell(r.won).replace(/<[^>]+>/g, "")],
      ["Lost TO battle", cell(r.lost).replace(/<[^>]+>/g, "")],
    ]),
  })), {
    quad: false, xAvg: 0, yAvg: 0.5, xTitle: "Turnover diff per game →", yTitle: "Cover %", aria: "Turnover margin vs cover rate",
    xSpan: 1, ySpan: 0.4, xFmt: (v) => signed(v, 1).replace("+0.0", "0").replace("-0.0", "0"),
  });
  const table = $("#ats-table");
  table.tHead.innerHTML = `<tr><th class="sortable" data-key="team" ${sk === "team" ? `aria-sort="${dir}ending"` : ""}>Team</th>${cols
    .map((c) => `<th class="sortable ${c.sep ? "sep" : ""}" data-key="${c.key}" ${sk === c.key ? `aria-sort="${dir}ending"` : ""}>${c.label}</th>`)
    .join("")}</tr>`;
  table.tBodies[0].innerHTML = rows
    .map(
      (r, i) =>
        `<tr data-k="${r.team}"><td class="team"><span class="rank">${i + 1}</span>${logo(r.team)}<a href="#" data-team="${r.team}">${r.team}</a></td>${cols
          .map((c) => `<td class="${c.sep ? "sep" : ""}">${c.show(r)}</td>`)
          .join("")}</tr>`
    )
    .join("");
}

// ---------- Spots: spread signals for this week's games ----------

let model = null; // data/model.json, built by backend/model.py

// "KC −3.5" style, from a team's own line (negative = favored).
function teamLine(team, line) {
  if (line == null) return `${team} –`;
  if (line === 0) return `${team} PK`;
  return `${team} ${line > 0 ? "+" : "−"}${Math.abs(line)}`;
}
// Favorite-side text for a home-view number (positive = home favored).
function favLine(home, away, homeFav) {
  if (homeFav == null) return "–";
  if (Math.abs(homeFav) < 0.05) return "Pick'em";
  return homeFav > 0 ? teamLine(home, -round1(homeFav)) : teamLine(away, -round1(-homeFav));
}
const round1 = (v) => Math.round(v * 10) / 10;
const rec = (r) => (r && r.w + r.l + r.p ? `${r.w}-${r.l}${r.p ? `-${r.p}` : ""}` : "–");
const pctTxt = (r) => (r && r.pct != null ? `${(r.pct * 100).toFixed(1)}%` : "–");

const SIGNAL_NAMES = { power: "Power edge", to_fade: "Turnover fade", strong: "Both agree" };

function renderSpots() {
  const box = $("#spots");
  if (!model) {
    box.innerHTML = `<p class="muted">No model data yet. Run backend/fetch_data.py.</p>`;
    return;
  }
  const be = model.break_even;
  const [b0, b1] = model.backtest_seasons || ["?", "?"];

  // Backtest + live record per signal, as tiles
  $("#spots-tiles").innerHTML = Object.entries(model.signals)
    .map(([key, sg]) => {
      const beat = sg.backtest.pct != null && sg.backtest.pct >= be;
      return `<div class="tile ${beat ? "hi" : ""}"><div class="tile-label">${SIGNAL_NAMES[key]}</div>
        <div class="tile-value">${pct0(sg.backtest.pct)}<small>${rec(sg.backtest)}</small></div>
        <div class="tile-sub">covered, ${b0}–${b1}</div>
        <div class="tile-sub muted">${model.season}: ${rec(sg.live)}${sg.live.pct != null ? ` (${pct0(sg.live.pct)})` : ""}</div></div>`;
    })
    .join("");
  const best = Math.max(...Object.values(model.signals).map((sg) => sg.backtest.pct || 0));
  $("#spots-verdict").innerHTML =
    best >= be
      ? `At least one signal has cleared the <b>${(be * 100).toFixed(1)}%</b> break-even at −110 in the backtest. Samples are small, so size bets accordingly.`
      : `<b>None of these beat the ${(be * 100).toFixed(1)}% break-even at −110</b> over ${b0}–${b1}. The spread already prices in
         turnover luck, so use the flags to find games worth a closer look, not as automatic bets.`;

  const spots = model.spots || [];
  $("#spots-title").textContent = spots.length ? `Week ${spots[0].week}: market spread vs model fair line` : "No upcoming games";
  box.innerHTML = spots.map(spotCard).join("") || `<p class="muted">No upcoming games in the schedule.</p>`;
  if (spots.length) renderDumbbells($("#spots-chart"), spots);
  else $("#spots-chart").innerHTML = `<p class="muted">No upcoming games in the schedule.</p>`;

  // 2026 flagged games already graded
  const hist = [...(model.live_history || [])].reverse();
  $("#spots-history").innerHTML = hist.length
    ? `<thead><tr><th>Wk</th><th class="l">Game</th><th>Line</th><th class="l">Signals</th><th>Result</th></tr></thead><tbody>${hist
        .map((h) => {
          const cells = Object.entries(h.signals)
            .map(([k, side]) => {
              const won = h.home_ats === "P" ? null : (h.home_ats === "W") === (side === h.home);
              return `<span class="chip ${won == null ? "" : won ? "chip-win" : "chip-loss"}">${SIGNAL_NAMES[k]}: ${side} ${
                won == null ? "push" : won ? "✓" : "✗"
              }</span>`;
            })
            .join(" ");
          return `<tr><td>${h.week}</td><td class="l">${h.away} @ ${h.home}</td><td>${favLine(h.home, h.away, h.spread_line)}</td><td class="l"><span class="chips">${cells}</span></td><td>${
            h.home_ats === "P" ? "Push" : `${h.home_ats === "W" ? h.home : h.away} covered`
          }</td></tr>`;
        })
        .join("")}</tbody>`
    : `<tbody><tr><td class="muted">No ${model.season} games have been flagged and graded yet.</td></tr></tbody>`;
}

function spotCard(s, i) {
  const final = s.home_score != null;
  const edgeSide = s.edge == null ? null : s.edge > 0 ? s.home : s.away;
  const sideLine = (t) => (t === s.home ? -s.spread_line : s.spread_line);
  const chips = [];
  if (s.signals.power) chips.push(`<span class="chip chip-on">Power edge → ${teamLine(s.signals.power, sideLine(s.signals.power))}</span>`);
  else chips.push(`<span class="chip">Power edge: ${s.edge == null ? "no line" : `${Math.abs(s.edge).toFixed(1)} pts (needs ${model.params.power_edge})`}</span>`);
  if (s.signals.to_fade) chips.push(`<span class="chip chip-on">Turnover fade → ${teamLine(s.signals.to_fade, sideLine(s.signals.to_fade))}</span>`);
  else if (Math.min(s.home_games, s.away_games) < model.params.min_games)
    chips.push(`<span class="chip">Turnover fade: needs ${model.params.min_games} games (${s.away_games}/${s.home_games})</span>`);
  else
    chips.push(`<span class="chip">Turnover fade: gap ${Math.abs(s.home_to_pg - s.away_to_pg).toFixed(2)}/g (needs ${model.params.to_gap})</span>`);

  let verdict;
  if (s.strength === 2) verdict = `<span class="tag tag-strong">Both signals</span> ${teamLine(s.lean, sideLine(s.lean))}`;
  else if (s.lean) verdict = `<span class="tag tag-lean">Lean</span> ${teamLine(s.lean, sideLine(s.lean))}`;
  else if (Object.keys(s.signals).length) verdict = `<span class="tag">Signals disagree</span>`;
  else verdict = `<span class="tag">No spot</span>`;

  let result = "";
  if (final && s.lean && s.spread_line != null) {
    const m = s.home_score - s.away_score - s.spread_line; // home ATS margin
    const won = m === 0 ? null : (m > 0) === (s.lean === s.home);
    result = `<div class="spot-result ${won == null ? "" : won ? "pos" : "neg"}">Final ${s.away_score}-${s.home_score} · ${
      won == null ? "push" : won ? "lean covered" : "lean lost"
    } (graded when the week is published)</div>`;
  } else if (final) result = `<div class="spot-result muted">Final ${s.away_score}-${s.home_score}</div>`;

  return `<div class="spot ${s.strength === 2 ? "spot-strong" : s.lean ? "spot-lean" : ""}" data-k="${s.game_id}" style="--n:${i}">
    <div class="spot-head">
      <div class="spot-teams">${logo(s.away)}<b>${s.away}</b><span class="muted">@</span>${logo(s.home)}<b>${s.home}</b></div>
      <div class="muted">${gameWhen(s)}${!final && s.time ? ` · ${s.time} ET` : ""}</div>
    </div>
    <div class="spot-lines">
      <div><span class="muted">Market</span><b>${favLine(s.home, s.away, s.spread_line)}</b></div>
      <div><span class="muted">Model fair line</span><b>${favLine(s.home, s.away, s.fair_spread_line)}</b></div>
      <div><span class="muted">Edge</span><b>${s.edge == null ? "–" : `${Math.abs(s.edge).toFixed(1)} pts → ${edgeSide}`}</b></div>
    </div>
    <div class="chips">${chips.join("")}</div>
    <div class="spot-verdict">${verdict}</div>
    ${result}
  </div>`;
}

// ---------- Kicks: field-goal props from drive and red-zone profiles ----------

const pct0 = (v) => (v == null ? "–" : `${Math.round(v * 100)}%`);
const pct1 = (v) => (v == null ? "–" : `${(v * 100).toFixed(1)}%`);
const odds = (o) => (o == null ? "–" : o > 0 ? `+${o}` : `−${Math.abs(o)}`);
const breakEven = (o) => (o < 0 ? -o / (-o + 100) : 100 / (o + 100));

// Per-game offense / defense profiles for every team under the current filters,
// plus league averages and "stall" / "bend" scores (z of trips minus z of RZ TD %).
function kickProfiles(L) {
  const teams = L.stats.filter((s) => s.has_drives && s.gp);
  if (!teams.length) return null;
  const sum = (k) => teams.reduce((a, s) => a + s[k], 0);
  const gp = sum("gp");
  const lg = {
    t40: sum("t40") / gp,
    rz_pct: sum("rz_td") / (sum("rz") || 1),
    fgm: sum("fgm") / gp,
    fga_trip: sum("fga") / (sum("t40") || 1),
  };
  const rows = teams.map((s) => ({
    team: s.team,
    gp: s.gp,
    off: { t40: s.t40 / s.gp, rz: s.rz / s.gp, rz_pct: s.off_rz_pct, fga: s.fga / s.gp, fgm: s.fgm / s.gp, fg50: s.fg50 / s.gp, fga_trip: s.off_fga_trip, kpts: (3 * s.fgm + s.xpm) / s.gp },
    def: { t40: s.opp_t40 / s.gp, rz: s.opp_rz / s.gp, rz_pct: s.def_rz_pct, fga: s.opp_fga / s.gp, fgm: s.opp_fgm / s.gp, fg50: s.opp_fg50 / s.gp, fga_trip: s.def_fga_trip, kpts: (3 * s.opp_fgm + s.opp_xpm) / s.gp },
  }));
  const sd = (vals) => {
    const m = vals.reduce((a, b) => a + b, 0) / vals.length;
    return Math.sqrt(vals.reduce((a, v) => a + (v - m) ** 2, 0) / vals.length) || 1;
  };
  for (const side of ["off", "def"]) {
    const sdT = sd(rows.map((r) => r[side].t40));
    const sdR = sd(rows.filter((r) => r[side].rz_pct != null).map((r) => r[side].rz_pct));
    for (const r of rows) {
      const p = r[side];
      p.score = p.rz_pct == null ? null : (p.t40 - lg.t40) / sdT - (p.rz_pct - lg.rz_pct) / sdR;
      p.target = p.rz_pct != null && p.t40 >= lg.t40 && p.rz_pct < lg.rz_pct;
    }
  }
  return { lg, rows };
}

// Scatter of team logos: x = trips inside the 40 per game, y = red-zone TD %.
// The bottom-right quadrant (above-average trips, below-average TD rate) is shaded.
// quad: shade the bottom-right quadrant (and dim teams outside it); xFmt / yFmt: tick labels;
// xSpan / ySpan: smallest axis range, so a tight cluster isn't blown up.
function renderScatter(box, pts, { xAvg, yAvg, xTitle, yTitle, quadLabel, aria, quad = true, xSpan = 1, ySpan = 0.1,
  xFmt = (v) => +v.toFixed(2), yFmt = (v) => `${Math.round(v * 100)}%` }) {
  if (!pts.length) {
    box.innerHTML = `<p class="muted">Not enough games for this chart with the current filters.</p>`;
    return;
  }
  const W = Math.max(box.clientWidth, 280), H = Math.round(Math.min(380, Math.max(280, W * 0.62)));
  const m = { t: 14, r: 16, b: 40, l: 56 };
  const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
  const pad = (lo, hi, f) => [lo - (hi - lo) * f, hi + (hi - lo) * f];
  let [x0, x1] = pad(Math.min(...xs, xAvg), Math.max(...xs, xAvg), 0.08);
  let [y0, y1] = pad(Math.min(...ys, yAvg), Math.max(...ys, yAvg), 0.08);
  if (x1 - x0 < xSpan) [x0, x1] = [(x0 + x1) / 2 - xSpan / 2, (x0 + x1) / 2 + xSpan / 2];
  if (y1 - y0 < ySpan) [y0, y1] = [(y0 + y1) / 2 - ySpan / 2, (y0 + y1) / 2 + ySpan / 2];
  const x = (v) => m.l + ((v - x0) / (x1 - x0)) * (W - m.l - m.r);
  const y = (v) => m.t + ((y1 - v) / (y1 - y0)) * (H - m.t - m.b);
  const ticks = (lo, hi, n) => {
    const raw = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map((k) => k * mag).find((s) => s >= raw);
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Math.round(v / step) * step);
    return out;
  };
  const S = 22;
  let svg = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${aria}">`;
  if (quad) {
    svg += `<rect class="quad" x="${x(xAvg)}" y="${y(yAvg)}" width="${W - m.r - x(xAvg)}" height="${H - m.b - y(yAvg)}" rx="6"/>`;
    svg += `<text class="quad-label" x="${W - m.r - 8}" y="${H - m.b - 10}" text-anchor="end">${quadLabel}</text>`;
  }
  svg += `<g class="grid">`;
  for (const v of ticks(x0, x1, 5)) svg += `<line x1="${x(v)}" x2="${x(v)}" y1="${m.t}" y2="${H - m.b}"/>`;
  for (const v of ticks(y0, y1, 5)) svg += `<line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}"/>`;
  svg += `</g><g class="axis">`;
  for (const v of ticks(x0, x1, 5)) svg += `<text x="${x(v)}" y="${H - m.b + 16}" text-anchor="middle">${xFmt(v)}</text>`;
  for (const v of ticks(y0, y1, 5)) svg += `<text x="${m.l - 8}" y="${y(v) + 4}" text-anchor="end">${yFmt(v)}</text>`;
  svg += `<text class="axis-title" x="${(m.l + W - m.r) / 2}" y="${H - 6}" text-anchor="middle">${xTitle}</text>`;
  svg += `<text class="axis-title" transform="translate(12 ${(m.t + H - m.b) / 2}) rotate(-90)" text-anchor="middle">${yTitle}</text>`;
  svg += `</g><line class="ref dash" x1="${x(xAvg)}" x2="${x(xAvg)}" y1="${m.t}" y2="${H - m.b}"/>`;
  svg += `<line class="ref dash" x1="${m.l}" x2="${W - m.r}" y1="${y(yAvg)}" y2="${y(yAvg)}"/>`;
  pts.forEach((p, i) => {
    const remote = data.teams[p.team]?.logo || "";
    svg += `<g class="dot ${quad && !p.target ? "dim" : ""}" data-i="${i}"><circle cx="${x(p.x)}" cy="${y(p.y)}" r="${S / 2 + 3}" fill="transparent"/>`;
    svg += `<image href="logos/${p.team}.png" x="${x(p.x) - S / 2}" y="${y(p.y) - S / 2}" width="${S}" height="${S}" onerror="this.onerror=null;this.setAttribute('href','${remote}')"/></g>`;
  });
  svg += `</svg>`;
  box.innerHTML = svg;
  box.querySelectorAll(".dot").forEach((g) => {
    const p = pts[Number(g.dataset.i)];
    g.addEventListener("mousemove", (ev) => showTip(p.tip, ev));
    g.addEventListener("mouseleave", hideTip);
    g.addEventListener("click", () => openMatchup(p.team, null));
  });
}

function kickTip(team, side, p, lg) {
  const what = side === "off" ? "offense" : "defense (allowed)";
  return `<div class="tt-title">${logo(team)}${data.teams[team].name} ${what}</div><div class="tt-grid">
    <span>Trips inside 40 / g</span><span>${p.t40.toFixed(2)} <small class="muted">lg ${lg.t40.toFixed(2)}</small></span>
    <span>Red-zone TD %</span><span>${pct0(p.rz_pct)} <small class="muted">lg ${pct0(lg.rz_pct)}</small></span>
    <span>FG tries per trip</span><span>${p.fga_trip == null ? "–" : p.fga_trip.toFixed(2)}</span>
    <span>FG made / g</span><span>${p.fgm.toFixed(2)}</span></div>`;
}

function renderKickTable(sel, K, side) {
  const table = $(sel);
  const sortState = (state.kickSort[side] ||= { key: "score", dir: "desc" });
  const cols = [
    { key: "gp", label: "GP", get: (r) => r.gp, show: (r) => r.gp },
    { key: "t40", label: "Trips 40/g", title: "Trips inside the 40 per game", get: (r) => r[side].t40, show: (r) => r[side].t40.toFixed(2), sep: true },
    { key: "rz", label: "RZ trips/g", get: (r) => r[side].rz, show: (r) => r[side].rz.toFixed(2) },
    { key: "rz_pct", label: "RZ TD %", get: (r) => r[side].rz_pct, show: (r) => pct0(r[side].rz_pct) },
    { key: "fga_trip", label: "FGA/trip", title: "Field-goal tries per trip inside the 40", get: (r) => r[side].fga_trip, show: (r) => (r[side].fga_trip == null ? "–" : r[side].fga_trip.toFixed(2)), sep: true },
    { key: "fgm", label: "FGM/g", get: (r) => r[side].fgm, show: (r) => r[side].fgm.toFixed(2) },
    { key: "kpts", label: "K pts/g", title: "Kicker points per game (3 x FG made + XP made)", get: (r) => r[side].kpts, show: (r) => r[side].kpts.toFixed(1) },
    { key: "score", label: side === "off" ? "Stall" : "Bend", title: "Trips above average minus red-zone TD % above average, in standard deviations", get: (r) => r[side].score, sep: true,
      show: (r) => `${r[side].target ? `<span class="pill accent">${side === "off" ? "Stall" : "Bend"}</span> ` : ""}<b>${signed(r[side].score)}</b>` },
  ];
  const col = cols.find((c) => c.key === sortState.key) || cols[cols.length - 1];
  const rows = [...K.rows].sort((x, y) => {
    const a = col.get(x), b = col.get(y);
    if (a == null) return 1;
    if (b == null) return -1;
    return (a - b) * (sortState.dir === "asc" ? 1 : -1);
  });
  table.tHead.innerHTML = `<tr><th>Team</th>${cols
    .map((c) => `<th class="sortable ${c.sep ? "sep" : ""}" data-key="${c.key}" title="${c.title || ""}" ${sortState.key === c.key ? `aria-sort="${sortState.dir}ending"` : ""}>${c.label}</th>`)
    .join("")}</tr>`;
  table.tBodies[0].innerHTML = rows
    .map(
      (r, i) => `<tr data-k="${side}:${r.team}"><td class="team"><span class="rank">${i + 1}</span>${logo(r.team)}<a href="#" data-team="${r.team}">${r.team}</a></td>${cols
        .map((c) => `<td class="${c.sep ? "sep" : ""}">${c.show(r)}</td>`)
        .join("")}</tr>`
    )
    .join("");
}

const KICK_TOP = 10; // kickers shown in the weekly chart before "Show all"

function renderKicks(L) {
  const K = kickProfiles(L);
  const kx = model?.kicks;

  // Backtest verdict
  if (kx) {
    const sig = kx.signals;
    const [b0, b1] = kx.backtest_seasons || ["?", "?"];
    const base = sig.all.backtest, sb = sig.stall_bend.backtest, p2 = sig.proj2.backtest;
    const tile = (label, r, live, hi) => `<div class="tile ${hi ? "hi" : ""}"><div class="tile-label">${label}</div>
      <div class="tile-value">${pct0(r.rate)}<small>n=${r.n}</small></div>
      <div class="tile-sub">made 2+ FGs · ${r.fgm_pg ?? "–"} FGM/g${r.total_only != null ? ` · total-only price ${pct0(r.total_only)}` : ""}</div>
      ${live ? `<div class="tile-sub muted">${kx.season}: ${live.n ? `${live.hit}/${live.n} (${pct0(live.rate)})` : "no games yet"}</div>` : ""}</div>`;
    $("#kicks-tiles").innerHTML =
      tile(`Every kicker, ${b0}–${b1}`, base, sig.all.live) +
      tile("Stall offense × bend defense", sb, sig.stall_bend.live) +
      tile(`Projects ${kx.params.flag_fgm}+ FGs made`, p2, sig.proj2.live, true);
    $("#kicks-verdict").innerHTML = `<b>The short version:</b> "drives but stalls" on its own didn't produce extra field goals.
      Projecting trips × field-goal tries per trip did: kickers it put at ${kx.params.flag_fgm}+ made 2+ field goals
      <b>${pct0(p2.rate)}</b> of the time. Over 1.5 needs ${pct1(breakEven(-120))} at −120 and ${pct1(breakEven(-140))} at −140,
      so it only pays when your book's price is short.`;
    const live = Object.fromEntries((kx.calibration.live || []).map((r) => [r.lo, r]));
    $("#kicks-cal").innerHTML = `<thead><tr><th>Raw projection</th><th class="sep">${b0}–${b1} games</th><th>Model P(2+)</th><th>Total-only P(2+)</th><th>Actual 2+</th><th>FGM/g</th>
      <th class="sep">${kx.season} games</th><th>Actual 2+</th></tr></thead><tbody>${kx.calibration.backtest
        .map((r) => {
          const lv = live[r.lo] || {};
          const name = r.hi == null ? `${r.lo.toFixed(1)}+ FGM` : r.lo === 0 ? `Under ${r.hi.toFixed(1)} FGM` : `${r.lo.toFixed(1)}–${r.hi.toFixed(1)} FGM`;
          return `<tr><td>${name}</td><td class="sep">${r.n}</td><td>${pct1(r.model)}</td><td>${pct1(r.total_only)}</td><td><b>${pct1(r.rate)}</b></td><td>${r.fgm_pg ?? "–"}</td>
            <td class="sep">${lv.n ?? 0}</td><td>${lv.n ? pct0(lv.rate) : "–"}</td></tr>`;
        })
        .join("")}</tbody>`;
  } else {
    $("#kicks-verdict").textContent = "No kicking model yet. Run backend/fetch_data.py.";
    $("#kicks-tiles").innerHTML = $("#kicks-cal").innerHTML = "";
  }

  // This week's board (current season only)
  const board = kx && state.season === kx.season ? kx.board || [] : [];
  $("#kicks-board-wrap").hidden = !board.length;
  if (board.length) {
    $("#kicks-title").textContent = `Week ${board[0].week} kickers`;
    const flagAt = kx.params.flag_fgm;
    const shown = state.kickAll ? board : board.slice(0, KICK_TOP);
    renderRankDots($("#kicks-chart"), shown.map((r) => ({
      team: r.team,
      opp: r.opp,
      sub: `${r.home ? "vs" : "@"} ${r.opp}`,
      value: r.fgm_raw,
      flagged: r.signals.length > 0,
      valText: `${r.fgm_raw.toFixed(1)} · ${pct0(r.p2)}`,
      tip: ttTitle(r.team, ` ${r.home ? "vs" : "@"} ${r.opp}`) + ttGrid([
        ["Projected FGs made", r.fgm_raw.toFixed(2)],
        ["Chance of 2+", pct0(r.p2)],
        ["Fair odds, over 1.5", odds(r.p2_odds)],
        ["Implied team points", r.implied ?? "–"],
        ["Kicker points", `${r.kpts.toFixed(1)} (${pct0(r.p_kpts)} for ${kx.params.kpts_line + 0.5}+)`],
        ["Profile", [r.stall && "stall offense", r.bend && `${r.opp} bends`].filter(Boolean).join(", ") || "–"],
      ]),
    })), { ref: { value: flagAt, label: `flag at ${flagAt.toFixed(1)}` }, min: 1, max: 2.2, step: 0.25, fmtTick: (v) => +v.toFixed(2),
      aria: "Projected field goals made per kicker" });
    const btn = $("#kicks-all");
    btn.hidden = board.length <= KICK_TOP;
    btn.textContent = state.kickAll ? `Show top ${KICK_TOP}` : `Show all ${board.length} kickers`;
    const table = $("#kicks-board");
    table.tHead.innerHTML = `<tr><th>Kicker</th><th title="Team's implied points from the total and spread">Implied</th>
      <th class="sep" title="Projected trips inside the 40">Trips 40</th><th title="Projected field-goal tries per trip">FGA/trip</th>
      <th title="Raw projection of field goals made (used for the flag)">Proj FGM</th><th class="sep" title="Calibrated chance of 2+ FGs made">P(2+ FGM)</th>
      <th title="No-vig odds for over 1.5 FGs made">Fair o1.5</th><th class="sep" title="Projected kicker points">K pts</th>
      <th title="Chance of ${kx.params.kpts_line + 0.5}+ kicker points">P(${kx.params.kpts_line + 0.5}+)</th><th class="sep">Profile</th></tr>`;
    table.tBodies[0].innerHTML = board
      .map((r) => {
        const pills = [
          r.stall ? `<span class="pill accent" title="Offense: average-or-better trips, below-average RZ TD %">Stall O</span>` : "",
          r.bend ? `<span class="pill accent" title="${r.opp} defense: average-or-more trips allowed, below-average RZ TD % allowed">Bend D</span>` : "",
          r.signals.includes("proj2") ? `<span class="pill solid">Projects ${kx.params.flag_fgm}+</span>` : "",
        ].join(" ");
        const final = (data.upcoming || []).find((u) => u.game_id === r.game_id && u.home_score != null);
        return `<tr data-k="b:${r.team}" class="${r.signals.length ? "flag" : ""}">
          <td class="team">${logo(r.team)}<a href="#" data-team="${r.team}" data-opp="${r.opp}">${r.team}</a><span class="vs">${r.home ? "vs" : "@"}</span>${r.opp}${
            final ? ` <span class="pill">Final</span>` : ""
          }</td>
          <td>${r.implied ?? "–"}</td>
          <td class="sep">${r.trips.toFixed(1)}</td><td>${r.fga_per_trip.toFixed(2)}</td><td><b>${r.fgm_raw.toFixed(2)}</b></td>
          <td class="sep"><span class="meter"><span class="track"><span class="fill" style="width:${Math.round(r.p2 * 100)}%"></span></span>${pct0(r.p2)}</span></td>
          <td>${odds(r.p2_odds)}</td><td class="sep">${r.kpts.toFixed(1)}</td><td>${pct0(r.p_kpts)}</td>
          <td class="sep"><span class="chips" style="justify-content:flex-end">${pills || `<span class="muted">–</span>`}</span></td></tr>`;
      })
      .join("");
  }

  // Scatter charts + team tables (follow the filters)
  if (K) {
    const mk = (side) =>
      K.rows.filter((r) => r[side].rz_pct != null).map((r) => ({ team: r.team, x: r[side].t40, y: r[side].rz_pct, target: r[side].target, tip: kickTip(r.team, side, r[side], K.lg) }));
    renderScatter($("#kicks-off-chart"), mk("off"), {
      xAvg: K.lg.t40, yAvg: K.lg.rz_pct, xTitle: "Trips inside the 40 per game →", yTitle: "Red-zone TD %",
      quadLabel: "Moves it, then kicks", aria: "Offenses: trips inside the 40 vs red-zone touchdown rate",
    });
    renderScatter($("#kicks-def-chart"), mk("def"), {
      xAvg: K.lg.t40, yAvg: K.lg.rz_pct, xTitle: "Trips inside the 40 allowed per game →", yTitle: "Red-zone TD % allowed",
      quadLabel: "Bends, doesn't break", aria: "Defenses: trips inside the 40 allowed vs red-zone touchdown rate allowed",
    });
    renderKickTable("#kicks-off", K, "off");
    renderKickTable("#kicks-def", K, "def");
  } else {
    for (const id of ["#kicks-off-chart", "#kicks-def-chart"]) $(id).innerHTML = `<p class="muted">No drive data for this season yet.</p>`;
  }

  // Flagged kickers this season, graded
  const hist = kx?.live_history || [];
  $("#kicks-history").innerHTML = hist.length
    ? `<thead><tr><th>Wk</th><th class="l">Kicker</th><th class="l">Flags</th><th>Proj FGM</th><th>P(2+)</th><th>FG made</th><th>K pts</th><th>Over 1.5</th></tr></thead><tbody>${hist
        .map((h) => {
          const hit = h.fgm_actual >= 2;
          return `<tr><td>${h.week}</td><td class="l">${h.team} ${h.home ? "vs" : "@"} ${h.opp}</td><td class="l"><span class="chips">${h.signals
            .map((k) => `<span class="chip chip-on">${k === "proj2" ? `Projects ${kx.params.flag_fgm}+` : "Stall × bend"}</span>`)
            .join("")}</span></td><td>${h.fgm_raw.toFixed(2)}</td><td>${pct0(h.p2)}</td><td>${h.fgm_actual}/${h.fga_actual}</td><td>${h.kpts_actual}</td>
            <td><span class="chip ${hit ? "chip-win" : "chip-loss"}">${hit ? "✓ over" : "✗ under"}</span></td></tr>`;
        })
        .join("")}</tbody>`
    : `<tbody><tr><td class="muted">No ${kx?.season ?? ""} kickers have been flagged and graded yet.</td></tr></tbody>`;
}

// ---------- simple charts: league scatters, ranked bars, dumbbells ----------

const ttGrid = (pairs) => `<div class="tt-grid">${pairs.map(([k, v]) => `<span>${k}</span><span>${v}</span>`).join("")}</div>`;
const ttTitle = (team, extra = "") => `<div class="tt-title">${logo(team)}${data.teams[team]?.name || team}${extra}</div>`;

function renderLeagueCharts(L) {
  const teams = L.stats.filter((s) => s.gp);
  const gp = teams.reduce((a, s) => a + s.gp, 0) || 1;
  const avgTake = teams.reduce((a, s) => a + s.take, 0) / gp;
  renderScatter($("#league-to-chart"), teams.map((s) => ({
    team: s.team,
    x: s.take / s.gp,
    y: s.give / s.gp,
    target: s.take / s.gp >= avgTake && s.give / s.gp < avgTake,
    tip: ttTitle(s.team) + ttGrid([
      ["Takeaways / g", (s.take / s.gp).toFixed(2)],
      ["Giveaways / g", (s.give / s.gp).toFixed(2)],
      ["TO diff / g", signed(s.diff / s.gp, 2)],
      ["ATS", s.ats.rec],
    ]),
  })), {
    xAvg: avgTake, yAvg: avgTake, xTitle: "Takeaways per game →", yTitle: "Giveaways per game",
    quadLabel: "Takes it, protects it", aria: "Takeaways vs giveaways per game", ySpan: 1,
    xFmt: (v) => v.toFixed(1), yFmt: (v) => v.toFixed(1),
  });

  // Fumble luck: share of opponents' fumbles recovered vs share of own fumbles lost.
  const MIN_FUM = 2;
  const fum = teams.filter((s) => s.fumbles >= MIN_FUM && s.opp_fumbles >= MIN_FUM);
  renderScatter($("#league-fum-chart"), fum.map((s) => ({
    team: s.team,
    x: s.opp_rec_pct,
    y: 1 - s.kept_pct,
    target: s.opp_rec_pct >= 0.5 && 1 - s.kept_pct < 0.5,
    tip: ttTitle(s.team) + ttGrid([
      ["Opp fumbles recovered", `${s.fum_rec} of ${s.opp_fumbles} (${pct0(s.opp_rec_pct)})`],
      ["Own fumbles lost", `${s.fum_lost} of ${s.fumbles} (${pct0(1 - s.kept_pct)})`],
    ]),
  })), {
    xAvg: 0.5, yAvg: 0.5, xTitle: "Opponent fumbles recovered →", yTitle: "Own fumbles lost",
    quadLabel: "Lucky so far", aria: "Fumble recovery luck", xSpan: 0.4, ySpan: 0.4,
    xFmt: (v) => `${Math.round(v * 100)}%`,
  });
}

// One row per team (logo + label), sorted as given, as a dot plot on a [min, max]
// axis: a thin stem from the axis start to a dot at the value. Dots don't need a
// zero baseline, so close values stay readable.
// rows: { team, opp, sub, value, flagged, valText, tip }; ref: { value, label } draws a dashed line.
function renderRankDots(box, rows, { ref, min, max, step, fmtTick = (v) => v.toFixed(1), aria }) {
  const W = Math.max(box.clientWidth, 280), RH = 28;
  const m = { t: 22, r: 92, b: ref ? 22 : 6, l: 112 };
  const H = m.t + rows.length * RH + m.b;
  const hi = Math.max(max, ...rows.map((r) => r.value)), lo = Math.min(min, ...rows.map((r) => r.value));
  const x = (v) => m.l + ((v - lo) / (hi - lo)) * (W - m.l - m.r);
  let svg = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${aria}"><g class="grid">`;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) ticks.push(+v.toFixed(4));
  for (const v of ticks) svg += `<line x1="${x(v)}" x2="${x(v)}" y1="${m.t - 4}" y2="${H - m.b}"/>`;
  svg += `</g><g class="axis">`;
  for (const v of ticks) svg += `<text x="${x(v)}" y="${m.t - 9}" text-anchor="middle">${fmtTick(v)}</text>`;
  svg += `</g>`;
  if (ref) {
    svg += `<line class="ref dash" x1="${x(ref.value)}" x2="${x(ref.value)}" y1="${m.t - 4}" y2="${H - m.b + 4}"/>`;
    svg += `<text class="rowsub" x="${x(ref.value)}" y="${H - 4}" text-anchor="middle">${ref.label}</text>`;
  }
  rows.forEach((r, i) => {
    const y = m.t + i * RH, cy = y + RH / 2;
    const remote = data.teams[r.team]?.logo || "";
    svg += `<g class="rrow" data-i="${i}"><rect class="row-hit" x="0" y="${y}" width="${W}" height="${RH}" rx="6"/>`;
    svg += `<image href="logos/${r.team}.png" x="4" y="${cy - 9}" width="18" height="18" onerror="this.onerror=null;this.setAttribute('href','${remote}')"/>`;
    svg += `<text class="rowlabel" x="28" y="${cy + 4}">${r.team}</text><text class="rowsub" x="62" y="${cy + 4}">${r.sub || ""}</text>`;
    svg += `<line class="db-line ${r.flagged ? "flagged" : ""}" x1="${x(lo)}" x2="${x(r.value)}" y1="${cy}" y2="${cy}"/>`;
    svg += `<circle class="db-model ${r.flagged ? "flagged" : ""}" cx="${x(r.value)}" cy="${cy}" r="6"/>`;
    svg += `<text class="val ${r.flagged ? "flagged" : ""}" x="${W - m.r + 12}" y="${cy + 4}">${r.valText}</text></g>`;
  });
  box.innerHTML = svg + `</svg>`;
  box.querySelectorAll(".rrow").forEach((g) => {
    const r = rows[Number(g.dataset.i)];
    g.addEventListener("mousemove", (ev) => showTip(r.tip, ev));
    g.addEventListener("mouseleave", hideTip);
    g.addEventListener("click", () => openMatchup(r.team, r.opp));
  });
}

// One row per game: the market spread (hollow ring) and the model's fair line (dot),
// on a home-margin axis (right = home favored).
function renderDumbbells(box, spots) {
  const W = Math.max(box.clientWidth, 280), RH = 34, narrow = W < 520;
  const m = { t: 26, r: narrow ? 8 : 128, b: 8, l: 92 };
  const H = m.t + spots.length * RH + m.b;
  const vals = spots.flatMap((s) => [s.spread_line, s.fair_spread_line]).filter((v) => v != null);
  const span = Math.max(7, Math.ceil(Math.max(...vals.map(Math.abs), 0) + 1));
  const x = (v) => m.l + ((v + span) / (2 * span)) * (W - m.l - m.r);
  let svg = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Market spread vs model fair line">`;
  svg += `<g class="axis"><text x="${m.l}" y="12">← away${narrow ? "" : " favored"}</text><text x="${W - m.r}" y="12" text-anchor="end">home${narrow ? "" : " favored"} →</text>`;
  svg += `<text x="${x(0)}" y="12" text-anchor="middle">even</text></g>`;
  svg += `<g class="grid">`;
  for (let v = -Math.floor(span / 7) * 7; v <= span; v += 7) svg += `<line x1="${x(v)}" x2="${x(v)}" y1="${m.t - 6}" y2="${H - m.b}"/>`;
  svg += `</g><line class="ref" x1="${x(0)}" x2="${x(0)}" y1="${m.t - 6}" y2="${H - m.b}"/>`;
  spots.forEach((s, i) => {
    const cy = m.t + i * RH + RH / 2;
    const flagged = !!s.lean;
    svg += `<g class="rrow" data-i="${i}"><rect class="row-hit" x="0" y="${cy - RH / 2}" width="${W}" height="${RH}" rx="6"/>`;
    svg += `<text class="rowlabel" x="4" y="${cy + 4}">${s.away}</text><text class="rowsub" x="40" y="${cy + 4}">@</text><text class="rowlabel" x="54" y="${cy + 4}">${s.home}</text>`;
    if (s.spread_line != null) {
      const a = x(s.spread_line), b = x(s.fair_spread_line);
      svg += `<line class="db-line ${flagged ? "flagged" : ""}" x1="${a}" x2="${b}" y1="${cy}" y2="${cy}"/>`;
      svg += `<circle class="db-model ${flagged ? "flagged" : ""}" cx="${b}" cy="${cy}" r="5.5"/>`;
      svg += `<circle class="db-market" cx="${a}" cy="${cy}" r="5.5"/>`;
    } else svg += `<text class="rowsub" x="${x(0)}" y="${cy + 4}" text-anchor="middle">no line yet</text>`;
    if (!narrow && flagged) {
      const side = s.lean === s.home ? -s.spread_line : s.spread_line;
      svg += `<text class="lean" x="${W - m.r + 12}" y="${cy + 4}">${s.strength === 2 ? "★ " : ""}${teamLine(s.lean, side)}</text>`;
    }
    svg += `</g>`;
  });
  box.innerHTML = svg + `</svg>`;
  box.querySelectorAll(".rrow").forEach((g) => {
    const s = spots[Number(g.dataset.i)];
    const sideLine = (t) => (t === s.home ? -s.spread_line : s.spread_line);
    const sig = Object.entries(s.signals).filter(([k]) => k !== "strong").map(([k, t]) => [SIGNAL_NAMES[k], teamLine(t, sideLine(t))]);
    const tipHtml = `<div class="tt-title">${s.away} @ ${s.home} · ${gameWhen(s)}</div>` + ttGrid([
      ["Market", favLine(s.home, s.away, s.spread_line)],
      ["Model fair line", favLine(s.home, s.away, s.fair_spread_line)],
      ["Edge", s.edge == null ? "–" : `${Math.abs(s.edge).toFixed(1)} pts`],
      ...sig,
      ["Verdict", s.strength === 2 ? `Both signals: ${s.lean}` : s.lean ? `Lean ${s.lean}` : sig.length ? "Signals disagree" : "No spot"],
    ]);
    g.addEventListener("mousemove", (ev) => showTip(tipHtml, ev));
    g.addEventListener("mouseleave", hideTip);
    g.addEventListener("click", () => openMatchup(s.away, s.home));
  });
}

// ---------- 1H Under: chance each game is 24 or fewer at halftime ----------

const UNDER_PRICES = [-150, -200, -250]; // break-even reference lines
const LINE_1H = 24.5;
const TOP_N = 3;

function renderUnder() {
  const fh = model?.first_half;
  const box = $("#under-chart");
  if (!fh) {
    $("#under-lede").textContent = "No first-half model yet. Run backend/fetch_data.py.";
    box.innerHTML = "";
    return;
  }
  const board = fh.board || [];
  const bt = fh.backtest, lv = fh.live;
  const seasons = fh.backtest_seasons;
  const span = seasons.length ? `${seasons[0]}–${seasons[seasons.length - 1]}` : "";
  renderUnderWeek(fh, board, box, bt, span);

  // How this was tested
  const tile = (label, r, sub, hi) => `<div class="tile ${hi ? "hi" : ""}"><div class="tile-label">${label}</div>
    <div class="tile-value">${pct0(r.rate)}<small>${r.under} of ${r.n}</small></div><div class="tile-sub">${sub}</div></div>`;
  $("#under-tiles").innerHTML =
    tile(`Every game, ${span}`, bt.all, `24 or fewer at half · model said ${pct0(bt.all.avg_p)}`) +
    tile("Top 3 each week", bt.top3, `model said ${pct0(bt.top3.avg_p)} · ${fh.season}: ${lv.top3.n ? `${lv.top3.under} of ${lv.top3.n}` : "none yet"}`, true) +
    tile("Model 70%+", bt.p70, `model said ${pct0(bt.p70.avg_p)} · ${fh.season}: ${lv.p70.n ? `${lv.p70.under} of ${lv.p70.n}` : "none yet"}`);
  const be = (o) => pct1(breakEven(o));
  $("#under-notes").innerHTML = `<b>How it works:</b> the projection is ${fh.coef ? `${fh.coef.intercept} + ${fh.coef.per_total_point} × the full-game total` : "fitted on past seasons"},
    and the chance of 24 or fewer comes from how far real halftime scores landed from that projection in past seasons.
    Each team's first-half scoring (points scored and allowed) was tested as a second input but didn't improve the forecast
    (Brier ${bt.all.brier_profile} with it vs ${bt.all.brier} without; lower is better, ${bt.all.brier_base} for a flat base rate):
    the market's total already reflects it. <b>What can't be tested:</b> nflverse has no first-half odds, so this measures how
    accurate the probabilities are, not whether betting them made money. Past weeks in the week strip are rebuilt with the same
    model and the total nflverse has on file for each game, which is usually close to the closing line rather than the 8am number. Break-even is ${be(-150)} at −150, ${be(-200)} at −200, ${be(-250)} at −250.`;
  const live = Object.fromEntries((fh.calibration.live || []).map((r) => [r.lo, r]));
  $("#under-cal").innerHTML = `<thead><tr><th>Model said</th><th class="sep">${span} games</th><th>Avg model</th><th>Actually under</th>
    <th class="sep">${fh.season} games</th><th>Actually under</th></tr></thead><tbody>${fh.calibration.backtest
      .map((r) => {
        const l = live[r.lo] || {};
        const name = r.lo === 0 ? `Under ${pct0(r.hi)}` : r.hi >= 1 ? `${pct0(r.lo)}+` : `${pct0(r.lo)}–${pct0(r.hi)}`;
        return `<tr><td>${name}</td><td class="sep">${r.n}</td><td>${pct1(r.avg_p)}</td><td><b>${r.n ? `${pct1(r.rate)}` : "–"}</b> <span class="muted">${r.under}/${r.n}</span></td>
          <td class="sep">${l.n || 0}</td><td>${l.n ? `${pct0(l.rate)} <span class="muted">${l.under}/${l.n}</span>` : "–"}</td></tr>`;
      })
      .join("")}</tbody>`;
}

let lastUnderKey = null;
const shortWeek = (w) => ({ 19: "WC", 20: "DIV", 21: "CONF", 22: "SB" })[w] || `Wk ${w}`;
const UNDER_NEXT = "next";

// Week strip (each past week's top-3 results as pips) + that week's chart.
// The upcoming week shows fair prices; past weeks show what happened at halftime.
function renderUnderWeek(fh, board, box, bt, span) {
  const hist = fh.history || {};
  const seasons = Object.keys(hist).map(Number);
  if (!seasons.includes(fh.season)) seasons.push(fh.season);
  seasons.sort((a, b) => b - a);
  if (!seasons.includes(state.underSeason)) state.underSeason = fh.season;
  const sel = $("#under-season");
  sel.innerHTML = seasons.map((y) => `<option value="${y}" ${y === state.underSeason ? "selected" : ""}>${y}</option>`).join("");

  const rows = hist[state.underSeason] || [];
  const weeks = [...new Set(rows.map((r) => r.week))].sort((a, b) => a - b);
  const hasNext = state.underSeason === fh.season && board.length > 0;
  const valid = [...weeks, ...(hasNext ? [UNDER_NEXT] : [])];
  if (!valid.includes(state.underWeek)) state.underWeek = hasNext ? UNDER_NEXT : weeks[weeks.length - 1];

  // Season record
  const line = fh.line;
  const top = rows.filter((r) => r.top3), topW = top.filter((r) => r.h1 <= line).length, allW = rows.filter((r) => r.h1 <= line).length;
  $("#under-record").innerHTML = rows.length
    ? `<b>${state.underSeason}</b> · top 3 each week: <b>${topW} of ${top.length}</b> under (${pct0(topW / top.length)}) · every game: ${allW} of ${rows.length} (${pct0(allW / rows.length)})`
    : `${state.underSeason} · no completed weeks yet`;

  // Strip
  const strip = $("#under-weeks");
  const prevSel = strip.querySelector(".wk.on")?.dataset.w;
  strip.innerHTML = weeks
    .map((w) => {
      const t = rows.filter((r) => r.week === w && r.top3);
      const won = t.filter((r) => r.h1 <= line).length;
      return `<button class="wk" data-w="${w}" title="${weekLabel(w)}: top 3 went ${won} of ${t.length} under">${shortWeek(w)}
        <span class="pips">${t.map((r) => `<i class="pip ${r.h1 <= line ? "w" : "l"}"></i>`).join("")}</span><span class="rec">${won}/${t.length}</span></button>`;
    })
    .join("") + (hasNext ? `<button class="wk" data-w="${UNDER_NEXT}">${shortWeek(board[0].week)}<span class="pips"><i class="pip"></i><i class="pip"></i><i class="pip"></i></span><span class="rec">next</span></button>` : "");
  strip.querySelectorAll(".wk").forEach((b) => {
    const on = String(state.underWeek) === b.dataset.w;
    b.classList.toggle("on", on);
    b.setAttribute("aria-pressed", on);
    if (on && prevSel != null && prevSel !== b.dataset.w) replay(b, "pop");
    if (on) strip.scrollLeft = Math.max(0, b.offsetLeft - strip.clientWidth / 2 + b.offsetWidth / 2);
  });

  // Chart (slides in when the selected week changes)
  const key = `${state.underSeason}:${state.underWeek}`;
  if (lastUnderKey && lastUnderKey !== key) requestAnimationFrame(() => replay(box, "swap"));
  lastUnderKey = key;
  if (state.underWeek === UNDER_NEXT) {
    $("#under-title").textContent = `Week ${board[0].week}: chance of 24 or fewer at half`;
    $("#under-lede").innerHTML = `Ranked best to worst. Bet a game only when your book's under 24.5 price is better than the fair price on the right.
      The week's top 3 went under <b>${pct0(bt.top3.rate)}</b> of the time in ${span} (${bt.top3.under} of ${bt.top3.n}); every game, ${pct0(bt.all.rate)}.`;
    renderUnderChart(box, board);
    return;
  }
  const wk = rows.filter((r) => r.week === state.underWeek).map((r) => ({ ...r, total_line: r.total, fair_odds: fairOdds(r.p) }));
  const t = wk.filter((r) => r.top3), tw = t.filter((r) => r.h1 <= line).length, aw = wk.filter((r) => r.h1 <= line).length;
  $("#under-title").textContent = `${weekLabel(state.underWeek).replace(/^w/, "W")} ${state.underSeason}: how it went`;
  $("#under-lede").innerHTML = `Top 3: <b>${tw} of ${t.length}</b> under · every game: ${aw} of ${wk.length}. Right side: the model's chance,
    then the actual halftime points (green = 24 or fewer).`;
  if (!wk.length) box.innerHTML = `<p class="muted">No games this week.</p>`;
  else renderUnderChart(box, wk);
}

// No-vig American odds for probability p.
const fairOdds = (p) => (p <= 0 || p >= 1 ? null : p >= 0.5 ? Math.round((-100 * p) / (1 - p)) : Math.round((100 * (1 - p)) / p));

// One row per game: a dot at the chance of 24 or fewer at half, with dashed
// break-even lines for common under 24.5 prices. Right label: chance and fair price.
function renderUnderChart(box, board) {
  const W = Math.max(box.clientWidth, 280), RH = 40, narrow = W < 560;
  const m = { t: narrow ? 38 : 26, r: narrow ? 66 : 132, b: 8, l: narrow ? 128 : 168 };
  const H = m.t + board.length * RH + m.b;
  const lo = narrow ? 0.4 : 0.3, hi = narrow ? 0.8 : 0.85;
  const x = (v) => m.l + ((Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo)) * (W - m.l - m.r);
  const top = new Set(
    board.some((g) => g.top3 != null) ? board.filter((g) => g.top3).map((g) => g.game_id) : board.filter((g) => g.p != null).slice(0, TOP_N).map((g) => g.game_id)
  );
  let svg = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Chance each game is 24 or fewer at halftime"><g class="grid">`;
  for (let v = lo; v <= hi + 1e-9; v += 0.1) svg += `<line x1="${x(v)}" x2="${x(v)}" y1="${m.t - 4}" y2="${H - m.b}"/>`;
  svg += `</g>`;
  UNDER_PRICES.forEach((o, k) => {
    const v = breakEven(o);
    // On narrow screens the lines sit close together, so stagger short labels.
    const ty = narrow ? m.t - 10 - (k % 2) * 13 : m.t - 12;
    svg += `<line class="ref dash" x1="${x(v)}" x2="${x(v)}" y1="${ty + 3}" y2="${H - m.b}"/>`;
    svg += `<text class="rowsub" x="${x(v)}" y="${ty}" text-anchor="middle">−${-o}${narrow ? "" : ` · ${Math.round(v * 100)}%`}</text>`;
  });
  board.forEach((g, i) => {
    const cy = m.t + i * RH + RH / 2;
    const isTop = top.has(g.game_id);
    const logoAt = (t, lx) => `<image href="logos/${t}.png" x="${lx}" y="${cy - 17}" width="16" height="16" onerror="this.onerror=null;this.setAttribute('href','${data.teams[t]?.logo || ""}')"/>`;
    svg += `<g class="rrow" data-i="${i}"><rect class="row-hit" x="0" y="${cy - RH / 2}" width="${W}" height="${RH}" rx="6"/>`;
    svg += logoAt(g.away, 4) + `<text class="rowlabel" x="24" y="${cy - 4}">${g.away}</text><text class="rowsub" x="57" y="${cy - 4}">@</text>`;
    svg += logoAt(g.home, 72) + `<text class="rowlabel" x="92" y="${cy - 4}">${g.home}</text>`;
    const final = g.home_score != null;
    const sub = final ? `Final ${g.away_score}-${g.home_score}` : `${gameWhen(g).split(",")[0]}${g.total_line != null ? ` · total ${g.total_line}` : ""}`;
    const graded = g.h1 != null;
    svg += `<text class="rowsub" x="4" y="${cy + 13}">${sub}</text>`;
    if (g.p == null) {
      svg += `<text class="rowsub" x="${m.l + 6}" y="${cy + 4}">no line yet</text></g>`;
      return;
    }
    svg += `<line class="db-line ${isTop ? "flagged" : ""}" x1="${x(lo)}" x2="${x(g.p)}" y1="${cy}" y2="${cy}"/>`;
    svg += `<circle class="db-model ${isTop ? "flagged" : ""}" cx="${x(g.p)}" cy="${cy}" r="7"/>`;
    svg += `<text class="val ${isTop ? "flagged" : ""}" x="${W - m.r + 12}" y="${cy - 1}" style="font-size:15px">${pct0(g.p)}</text>`;
    svg += graded
      ? `<text class="rowsub ${g.h1 <= LINE_1H ? "res-w" : "res-l"}" x="${W - m.r + 12}" y="${cy + 13}">${g.h1}${narrow ? "" : " at half"} ${g.h1 <= LINE_1H ? "✓" : "✗"}</text></g>`
      : `<text class="rowsub" x="${W - m.r + 12}" y="${cy + 13}">fair ${odds(g.fair_odds)}</text></g>`;
  });
  box.innerHTML = svg + `</svg>`;
  box.querySelectorAll(".rrow").forEach((el) => {
    const g = board[Number(el.dataset.i)];
    const team = (t) => {
      const s = g.teams[t];
      if (!s || !s.n) return [`${t} 1H pts for / against`, "no games yet"];
      return [`${t} 1H pts for / against`, `${s.scored.toFixed(1)} / ${s.allowed.toFixed(1)}, ${s.under} of ${s.n} under`];
    };
    const html = `<div class="tt-title">${g.away} @ ${g.home} · ${gameWhen(g)}${g.time ? ` ${g.time} ET` : ""}</div>` + ttGrid([
      ["Full-game total", g.total_line ?? "no line yet"],
      ...(g.spread_line !== undefined ? [["Spread", lineText(g).split(" · ")[0]]] : []),
      ...(g.h1 != null ? [["Halftime", `${g.away} ${g.h1_away}, ${g.home} ${g.h1_home} (${g.h1}) ${g.h1 <= LINE_1H ? "✓ under" : "✗ over"}`]] : []),
      ["Projected 1H points", g.proj == null ? "–" : `${g.proj} (80%: ${Math.round(g.lo)}–${Math.round(g.hi)})`],
      ["Chance of 24 or fewer", g.p == null ? "–" : pct0(g.p)],
      ["Fair price, under 24.5", odds(g.fair_odds)],
      ...(g.teams ? [team(g.away), team(g.home)] : []),
    ]);
    el.addEventListener("mousemove", (ev) => showTip(html, ev));
    el.addEventListener("mouseleave", hideTip);
    el.addEventListener("click", () => openMatchup(g.away, g.home));
  });
}

// ---------- routing & wiring ----------

const VIEWS = ["under", "league", "matchup", "ats", "spots", "kicks"];
const VIEW_NAMES = { under: "1H Under", league: "League", matchup: "Matchup", ats: "vs Spread", spots: "Spots", kicks: "Kicks" };

function writeHash() {
  const p = new URLSearchParams();
  p.set("season", state.season);
  if (state.view === "matchup") p.set("a", state.a), p.set("b", state.b);
  history.replaceState(null, "", `#${state.view}?${p}`);
}

function readHash() {
  const [view, q] = location.hash.slice(1).split("?");
  if (VIEWS.includes(view)) state.view = view;
  const p = new URLSearchParams(q || "");
  if (p.get("season")) state.season = Number(p.get("season"));
  if (p.get("a")) state.a = p.get("a");
  if (p.get("b")) state.b = p.get("b");
}

let lastView = null;

function render() {
  const sec = $(`#view-${state.view}`);
  const switched = lastView !== state.view;
  // Same view re-rendering (sort, filter, season): rows travel to their new spots.
  const before = switched ? null : snapshot(sec);
  const L = leagueStats();
  document.querySelectorAll("#tabs button").forEach((b) => b.setAttribute("aria-selected", b.dataset.view === state.view));
  for (const v of VIEWS) $(`#view-${v}`).hidden = state.view !== v;
  document.body.dataset.view = state.view;
  if (switched) chromeSetup?.();
  document.querySelectorAll("#mode-seg button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.mode === state.mode));
  const upd = new Date(data.updated);
  $("#updated").textContent = `${data.season} season · through ${weekLabel(data.last_week)} · updated ${upd.toLocaleString(undefined, {
    month: "short", day: "numeric", hour: "numeric", minute: "2-digit",
  })}`;
  $("#tb-ctx").textContent = `${VIEW_NAMES[state.view]} · ${state.season}`;
  renderUpcoming(L);
  if (state.view === "league") renderLeague(L);
  else if (state.view === "matchup") renderMatchup(L);
  else if (state.view === "ats") renderATS();
  else if (state.view === "spots") renderSpots();
  else if (state.view === "under") renderUnder();
  else renderKicks(L);
  if (switched) {
    replay(sec, "on");
    sec.querySelectorAll(".list").forEach((l) => replay(l, "enter"));
  } else flip(sec, before);
  glideAll();
  writeHash();
  lastView = state.view;
}

function openMatchup(a, b) {
  state.view = "matchup";
  state.a = a;
  if (b) state.b = b;
  else {
    const up = (data.upcoming || []).find((g) => g.home === a || g.away === a);
    if (up) state.b = up.home === a ? up.away : up.home;
    else if (state.b === a) state.b = null;
  }
  render();
  window.scrollTo({ top: 0, behavior: REDUCE.matches ? "auto" : "smooth" });
}

// Sticky glass header that gains a shadow once scrolled, and a filter toolbar
// that pins into a floating glass card. Pinning is detected with an
// IntersectionObserver on a 0-height sentinel, and the toolbar's footprint is
// frozen at its open height so pinning never moves the page.
let chromeSetup = null; // re-measures the sticky toolbar (it's hidden on some views)

function setupChrome() {
  const header = $("#site-header"), wrap = $("#tb-wrap"), bar = $("#toolbar");
  const wide = matchMedia("(min-width: 641px)");
  new IntersectionObserver(([e]) => header.classList.toggle("scrolled", !e.isIntersecting)).observe($("#top-sentinel"));
  let io;
  const setup = () => {
    const h = header.offsetHeight;
    document.documentElement.style.setProperty("--hdr", `${h}px`);
    const pinned = wrap.classList.contains("pinned");
    wrap.classList.add("measuring");
    wrap.classList.remove("pinned");
    wrap.style.height = "";
    wrap.style.height = wide.matches ? `${bar.offsetHeight}px` : "";
    if (pinned) wrap.classList.add("pinned");
    void wrap.offsetWidth;
    wrap.classList.remove("measuring");
    io?.disconnect();
    io = new IntersectionObserver(
      ([e]) => wrap.classList.toggle("pinned", wide.matches && !e.isIntersecting && e.boundingClientRect.top < h + 1),
      { rootMargin: `-${h + 1}px 0px 0px 0px` }
    );
    io.observe($("#tb-sentinel"));
  };
  chromeSetup = setup;
  setup();
  let t;
  window.addEventListener("resize", () => {
    clearTimeout(t);
    t = setTimeout(() => {
      setup();
      glideAll();
      render();
    }, 150);
  });
  document.fonts?.ready.then(() => {
    setup();
    glideAll();
  });
}

function sortClick(sortObj, key, defaultDir = "desc") {
  if (sortObj.key === key) sortObj.dir = sortObj.dir === "desc" ? "asc" : "desc";
  else Object.assign(sortObj, { key, dir: defaultDir });
}

function wire() {
  $("#f-season").addEventListener("change", async (e) => {
    state.season = Number(e.target.value);
    data = await loadSeason(state.season);
    render();
  });
  $("#f-type").addEventListener("change", (e) => { state.type = e.target.value; render(); });
  $("#f-span").addEventListener("change", (e) => { state.span = Number(e.target.value); render(); });
  $("#f-venue").addEventListener("change", (e) => { state.venue = e.target.value; render(); });
  document.querySelectorAll("#mode-seg button").forEach((b) => b.addEventListener("click", () => { state.mode = b.dataset.mode; render(); }));
  document.querySelectorAll("#ats-sample button").forEach((b) =>
    b.addEventListener("click", async () => {
      state.atsSample = b.dataset.sample;
      if (state.atsSample === "all") await Promise.all(state.seasons.map(loadSeason));
      render();
    })
  );
  $("#ats-table").tHead.addEventListener("click", (e) => {
    const th = e.target.closest("th[data-key]");
    if (!th) return;
    sortClick(state.atsSort, th.dataset.key, th.dataset.key === "team" ? "asc" : "desc");
    render();
  });
  for (const side of ["off", "def"]) {
    $(`#kicks-${side}`).tHead.addEventListener("click", (e) => {
      const th = e.target.closest("th[data-key]");
      if (!th) return;
      sortClick((state.kickSort[side] ||= { key: "score", dir: "desc" }), th.dataset.key);
      render();
    });
  }
  document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => { state.view = b.dataset.view; render(); }));
  // Arrow keys move between tabs.
  $("#tabs").addEventListener("keydown", (e) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    const i = VIEWS.indexOf(state.view) + (e.key === "ArrowRight" ? 1 : -1);
    state.view = VIEWS[(i + VIEWS.length) % VIEWS.length];
    render();
    $(`#tabs [data-view="${state.view}"]`).focus();
  });

  $("#league").tHead.addEventListener("click", (e) => {
    const th = e.target.closest("th[data-key]");
    if (!th) return;
    const key = th.dataset.key;
    const m = METRICS[key];
    sortClick(state.sort, key, key === "team" ? "asc" : m && m.better < 0 ? "asc" : "desc");
    render();
  });
  // Team links anywhere in the tables open that team's matchup (vs its next opponent).
  document.querySelector("main").addEventListener("click", (e) => {
    const a = e.target.closest("tbody a[data-team]");
    if (!a) return;
    e.preventDefault();
    openMatchup(a.dataset.team, a.dataset.opp || null);
  });
  // "Show …" disclosures expand in place.
  document.querySelector("main").addEventListener("click", (e) => {
    const btn = e.target.closest(".more-btn");
    if (!btn) return;
    const box = btn.parentElement;
    const open = !box.classList.contains("open");
    box.classList.toggle("open", open);
    btn.setAttribute("aria-expanded", open);
    btn.firstChild.textContent = open ? btn.firstChild.textContent.replace(/^Show/, "Hide") : btn.firstChild.textContent.replace(/^Hide/, "Show");
  });
  $("#kicks-all").addEventListener("click", () => { state.kickAll = !state.kickAll; render(); });
  $("#under-season").addEventListener("change", (e) => { state.underSeason = Number(e.target.value); state.underWeek = null; render(); });
  $("#under-weeks").addEventListener("click", (e) => {
    const b = e.target.closest(".wk");
    if (!b) return;
    state.underWeek = b.dataset.w === UNDER_NEXT ? UNDER_NEXT : Number(b.dataset.w);
    render();
  });
  $("#upcoming").addEventListener("click", (e) => {
    const g = e.target.closest(".game");
    if (g) openMatchup(g.dataset.a, g.dataset.b);
  });
  $("#m-a").addEventListener("change", (e) => { state.a = e.target.value; render(); });
  $("#m-b").addEventListener("change", (e) => { state.b = e.target.value; render(); });
  $("#m-swap").addEventListener("click", () => { [state.a, state.b] = [state.b, state.a]; render(); });
  setupChrome();
}

async function init() {
  try {
    const idx = await loadJSON("data/seasons.json");
    state.seasons = idx.seasons;
    readHash();
    if (!state.seasons.includes(state.season)) state.season = state.seasons[0];
    $("#f-season").innerHTML = state.seasons.map((s) => `<option value="${s}" ${s === state.season ? "selected" : ""}>${s}</option>`).join("");
    data = await loadSeason(state.season);
    model = await loadJSON("data/model.json").catch(() => null);
    wire();
    render();
  } catch (err) {
    $("#updated").textContent = `Could not load data (${err.message}). Run backend/fetch_data.py and serve the web/ folder.`;
  }
}

init();
