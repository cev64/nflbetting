"use strict";

// ---------- state ----------

const state = {
  seasons: [],
  season: null,
  type: "REG",
  span: 0,
  venue: "all",
  mode: "pg",
  view: "league",
  a: null,
  b: null,
  sort: { key: "diff", dir: "desc" },
};
const cache = {};
let data = null; // current season payload

const $ = (sel) => document.querySelector(sel);

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

function teamGames(team) {
  let g = data.games.filter((x) => x.team === team);
  if (state.type === "REG") g = g.filter((x) => x.type === "REG");
  else if (state.type === "POST") g = g.filter((x) => x.type !== "REG");
  if (state.venue === "home") g = g.filter((x) => x.home);
  else if (state.venue === "away") g = g.filter((x) => !x.home);
  g.sort((a, b) => (a.date < b.date ? -1 : 1));
  if (state.span > 0) g = g.slice(-state.span);
  return g;
}

function aggregate(team) {
  const games = teamGames(team);
  const s = { team, games, gp: games.length, w: 0, l: 0, t: 0 };
  const keys = ["int_made", "fum_rec", "int_thrown", "fum_lost", "fumbles", "opp_fumbles", "pf", "pa"];
  for (const k of keys) s[k] = 0;
  for (const g of games) {
    for (const k of keys) s[k] += g[k];
    if (g.pf > g.pa) s.w++;
    else if (g.pf < g.pa) s.l++;
    else s.t++;
  }
  s.take = s.int_made + s.fum_rec;
  s.give = s.int_thrown + s.fum_lost;
  s.diff = s.take - s.give;
  s.kept_pct = s.fumbles ? (s.fumbles - s.fum_lost) / s.fumbles : null;
  s.opp_rec_pct = s.opp_fumbles ? s.fum_rec / s.opp_fumbles : null;
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
  let out = m.count && state.mode === "pg" ? v.toFixed(2) : String(Math.round(v * 100) / 100);
  if (m.signed && v > 0) out = `+${out}`;
  return out;
}

function leagueStats() {
  const teams = Object.keys(data.teams).sort();
  const stats = teams.map(aggregate).filter((s) => s.gp > 0);
  // Ranks per metric: 1 = best.
  const ranks = {};
  for (const key of Object.keys(METRICS)) {
    const m = METRICS[key];
    const sorted = stats
      .filter((s) => value(s, key) != null)
      .sort((x, y) => (value(y, key) - value(x, key)) * m.better);
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

const logo = (t) => `<img class="logo" src="${data.teams[t]?.logo || ""}" alt="" loading="lazy">`;
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
];

function renderLeague(L) {
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
        `<th class="sortable ${c.sep ? "sep" : ""}" data-key="${c.key}" title="${METRICS[c.key]?.label || "Games played"}" ${
          sk === c.key ? `aria-sort="${dir}ending"` : ""
        }>${METRICS[c.key]?.short || c.label}</th>`
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
        return `<td class="${cls}" style="${style}">${c.key === "gp" ? s.gp : fmt(v, c.key)}</td>`;
      }).join("");
      return `<tr><td class="team"><span class="rank">${i + 1}</span>${logo(s.team)}<a href="#" data-team="${s.team}" title="${data.teams[s.team].name}">${s.team}</a></td>${cells}</tr>`;
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

function renderUpcoming(L) {
  const wrap = $("#upcoming-wrap");
  const up = data.upcoming || [];
  const isLatest = state.season === state.seasons[0];
  wrap.hidden = !(isLatest && up.length);
  if (wrap.hidden) return;
  $("#upcoming-title").textContent = `Week ${up[0].week} games · turnover diff per game`;
  const pg = (t) => {
    const s = L.byTeam[t];
    if (!s || !s.gp) return "–";
    const v = s.diff / s.gp;
    return `<span class="${v > 0 ? "pos" : v < 0 ? "neg" : ""}">${v > 0 ? "+" : ""}${v.toFixed(2)}</span>`;
  };
  $("#upcoming").innerHTML = up
    .map(
      (g) => `<button class="game" data-a="${g.away}" data-b="${g.home}">
        <div class="when"><span>${gameWhen(g)}</span><span>${g.time ? `${g.time} ET` : ""}</span></div>
        <div class="row">${logo(g.away)}<span class="abbr">${g.away}</span><span class="val">${pg(g.away)}</span></div>
        <div class="row">${logo(g.home)}<span class="abbr">@${g.home}</span><span class="val">${pg(g.home)}</span></div>
        <div class="line">${lineText(g)}</div>
      </button>`
    )
    .join("");
}

// ---------- matchup ----------

const CMP_ROWS = ["take", "give", "diff", "int_made", "fum_rec", "int_thrown", "fum_lost", "kept_pct", "opp_rec_pct"];

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

  // Side-by-side comparison
  const rk = (key, t) => (L.ranks[key][t] ? `<small>${ordinal(L.ranks[key][t])}</small>` : "");
  const rec = (s) => `${s.w}-${s.l}${s.t ? `-${s.t}` : ""}`;
  $("#m-compare").innerHTML =
    `<div class="cmp-head"><div class="t">${logo(A.team)}${data.teams[A.team].name}</div><div class="label muted">${
      state.mode === "pg" ? "per game · league rank" : "totals · league rank"
    }</div><div class="t b">${data.teams[B.team].name}${logo(B.team)}</div></div>` +
    `<div class="cmp-row"><div class="v">${rec(A)}</div><div class="label">Record (${A.gp} / ${B.gp} games)</div><div class="v b">${rec(B)}</div></div>` +
    CMP_ROWS.map((key) => {
      const va = value(A, key), vb = value(B, key);
      const m = METRICS[key];
      const aWin = va != null && vb != null && (va - vb) * m.better > 0;
      const bWin = va != null && vb != null && (vb - va) * m.better > 0;
      return `<div class="cmp-row"><div class="v ${aWin ? "win" : ""}">${fmt(va, key)}${rk(key, A.team)}</div><div class="label">${m.label}</div><div class="v b ${bWin ? "win" : ""}">${rk(key, B.team)}${fmt(vb, key)}</div></div>`;
    }).join("");

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
      <div style="font-size:28px;font-weight:600">${Math.abs(margin) < 0.05 ? "Even" : `${leader} +${Math.abs(margin).toFixed(2)}`}</div>
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
    return `<tr><td>${wk}</td><td>${g.home ? "vs" : "@"} ${g.opp}</td><td>${res} ${g.pf}-${g.pa}</td>
      <td class="sep key">${ta}</td><td>${g.int_made}</td><td>${g.fum_rec}</td>
      <td class="sep key">${ga}</td><td>${g.int_thrown}</td><td>${g.fum_lost}</td>
      <td class="sep key ${d > 0 ? "pos" : d < 0 ? "neg" : ""}">${d > 0 ? "+" : ""}${d}</td></tr>`;
  });
  $(sel).innerHTML = `<thead><tr><th>Wk</th><th>Opp</th><th>Result</th><th class="sep" title="Takeaways">TA</th><th title="Interceptions made">INT</th><th title="Fumbles recovered">FR</th><th class="sep" title="Giveaways">GA</th><th title="Interceptions thrown">INT</th><th title="Fumbles lost">FL</th><th class="sep">Diff</th></tr></thead><tbody>${rows.join("")}</tbody>`;
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
  const zone = $("#hover-zone"), line = $("#hover-line"), tip = $("#tooltip");
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
    tip.innerHTML = `<div class="tt-title">Game ${i}</div>${rows}`;
    tip.hidden = false;
    const tw = tip.offsetWidth;
    tip.style.left = `${Math.min(ev.clientX + 14, window.innerWidth - tw - 8)}px`;
    tip.style.top = `${ev.clientY + 14}px`;
  });
  zone.addEventListener("mouseleave", () => {
    tip.hidden = true;
    line.setAttribute("x1", -10);
    line.setAttribute("x2", -10);
  });
}

// ---------- routing & wiring ----------

function writeHash() {
  const p = new URLSearchParams();
  p.set("season", state.season);
  if (state.view === "matchup") p.set("a", state.a), p.set("b", state.b);
  history.replaceState(null, "", `#${state.view}?${p}`);
}

function readHash() {
  const [view, q] = location.hash.slice(1).split("?");
  if (view === "matchup" || view === "league") state.view = view;
  const p = new URLSearchParams(q || "");
  if (p.get("season")) state.season = Number(p.get("season"));
  if (p.get("a")) state.a = p.get("a");
  if (p.get("b")) state.b = p.get("b");
}

function render() {
  const L = leagueStats();
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", b.dataset.view === state.view));
  $("#view-league").hidden = state.view !== "league";
  $("#view-matchup").hidden = state.view !== "matchup";
  document.querySelectorAll(".seg button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.mode === state.mode));
  const upd = new Date(data.updated);
  $("#updated").textContent = `${data.season} season · through week ${data.last_week} · updated ${upd.toLocaleString(undefined, {
    month: "short", day: "numeric", hour: "numeric", minute: "2-digit",
  })}`;
  renderUpcoming(L);
  if (state.view === "league") renderLeague(L);
  else renderMatchup(L);
  writeHash();
}

function openMatchup(a, b) {
  state.view = "matchup";
  state.a = a;
  if (b) state.b = b;
  render();
  window.scrollTo({ top: 0, behavior: "smooth" });
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
  document.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => { state.mode = b.dataset.mode; render(); }));
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => { state.view = b.dataset.view; render(); }));

  $("#league").tHead.addEventListener("click", (e) => {
    const th = e.target.closest("th[data-key]");
    if (!th) return;
    const key = th.dataset.key;
    if (state.sort.key === key) state.sort.dir = state.sort.dir === "desc" ? "asc" : "desc";
    else {
      const m = METRICS[key];
      state.sort = { key, dir: key === "team" ? "asc" : m && m.better < 0 ? "asc" : "desc" };
    }
    render();
  });
  $("#league").tBodies[0].addEventListener("click", (e) => {
    const a = e.target.closest("a[data-team]");
    if (!a) return;
    e.preventDefault();
    const t = a.dataset.team;
    const up = (data.upcoming || []).find((g) => g.home === t || g.away === t);
    openMatchup(t, up ? (up.home === t ? up.away : up.home) : state.b === t ? null : state.b);
  });
  $("#upcoming").addEventListener("click", (e) => {
    const g = e.target.closest(".game");
    if (g) openMatchup(g.dataset.a, g.dataset.b);
  });
  $("#m-a").addEventListener("change", (e) => { state.a = e.target.value; render(); });
  $("#m-b").addEventListener("change", (e) => { state.b = e.target.value; render(); });
  $("#m-swap").addEventListener("click", () => { [state.a, state.b] = [state.b, state.a]; render(); });

  let t;
  window.addEventListener("resize", () => {
    clearTimeout(t);
    t = setTimeout(() => state.view === "matchup" && render(), 150);
  });
}

async function init() {
  try {
    const idx = await loadJSON("data/seasons.json");
    state.seasons = idx.seasons;
    readHash();
    if (!state.seasons.includes(state.season)) state.season = state.seasons[0];
    $("#f-season").innerHTML = state.seasons.map((s) => `<option value="${s}" ${s === state.season ? "selected" : ""}>${s}</option>`).join("");
    data = await loadSeason(state.season);
    wire();
    render();
  } catch (err) {
    $("#updated").textContent = `Could not load data (${err.message}). Run backend/fetch_data.py and serve the web/ folder.`;
  }
}

init();
