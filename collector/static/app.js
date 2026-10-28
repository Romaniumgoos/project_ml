"use strict";
// ================================================================ состояние
const CHEAT = new URLSearchParams(location.search).get("cheat") === "1";
// режим отладки: /?debug=<токен из KT_DEBUG_TOKEN> — без дневного лимита сессий
const DEBUG = new URLSearchParams(location.search).get("debug") || "";
const S = { uid: null, me: null, plan: [], idx: 0, total: 0, offset: 0, sessionNo: 0,
            att: null, results: [] };
const $ = (id) => document.getElementById(id);
const store = {
  get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) {} },
};

async function api(path, body) {
  const opt = body ? { method: "POST", headers: { "Content-Type": "application/json" },
                       body: JSON.stringify(body) } : {};
  const r = await fetch(path, opt);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw Object.assign(new Error(data.error || r.statusText), { data, status: r.status });
  return data;
}

function show(id) {
  for (const s of ["scrWelcome", "scrHome", "scrType", "scrDone"]) $(s).classList.toggle("hidden", s !== id);
}

function clientInfo() {
  // оценка разрешения таймера браузера (влияет на точность таймингов)
  let minStep = Infinity, prev = performance.now();
  for (let i = 0; i < 20000; i++) {
    const t = performance.now();
    if (t !== prev) { minStep = Math.min(minStep, t - prev); prev = t; }
  }
  return { ua: navigator.userAgent, platform: navigator.platform, lang: navigator.language,
           screen: `${screen.width}x${screen.height}`, dpr: devicePixelRatio,
           timer_res_ms: +minStep.toFixed(4), touch: "ontouchstart" in window };
}
const CLIENT = clientInfo();

// ================================================================ регистрация / вход
$("regForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = new FormData(e.target);
  const meta = Object.fromEntries(["lang", "keyboard", "typing", "hand", "hours"].map(k => [k, f.get(k)]));
  meta.client = CLIENT;
  try {
    const r = await api("/api/register", { nickname: f.get("nickname"), consent: !!f.get("consent"), meta });
    S.uid = r.user_id; store.set("kt_uid", S.uid);
    await openHome();
  } catch (err) { alert(err.message); }
});

$("loginForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  S.uid = new FormData(e.target).get("code").trim().toUpperCase();
  try { await openHome(); store.set("kt_uid", S.uid); } catch (err) { alert(err.message); }
});

// ================================================================ кабинет
function fmtWait(sec) {
  const h = Math.floor(sec / 3600), m = Math.ceil((sec % 3600) / 60);
  return h ? `${h} ч ${m} мин` : `${m} мин`;
}

async function openHome() {
  S.me = await api(`/api/me?uid=${encodeURIComponent(S.uid)}&debug=${encodeURIComponent(DEBUG)}`);
  $("debugBanner").classList.toggle("hidden", !S.me.debug);
  show("scrHome");
  $("whoami").textContent = `${S.me.nickname} · ${S.uid}`;
  $("helloName").textContent = `Привет, ${S.me.nickname}!`;
  $("myCode").textContent = S.uid;
  const done = S.me.sessions.filter(s => s.attempts >= 16).length;
  $("sessionDots").innerHTML = Array.from({ length: Math.max(3, done + 1) }, (_, k) => k + 1).map(i => `<div class="dot ${i <= done ? "done" : ""}">${i}</div>`).join("");
  $("cheatTypeRow").style.display = CHEAT ? "" : "none";
  const btn = $("btnStart");
  btn.disabled = false;
  if (CHEAT) {
    $("startHint").textContent = "Каждая попытка в этом режиме попадёт в датасет как пример читера.";
  } else if (S.me.wait_seconds > 0) {
    btn.disabled = true;
    $("startHint").textContent = `На сегодня все ${S.me.max_per_day} сессии пройдены, спасибо! Новые откроются `
      + `через ${fmtWait(S.me.wait_seconds)} (после полуночи) — данные из разных дней особенно ценны.`;
  } else {
    const left = S.me.max_per_day > 0 ? S.me.max_per_day - S.me.sessions_today : null;
    $("startHint").textContent = `Сессия ${done + 1}: калибровочная фраза ×10 и несколько текстов (~6 минут). `
      + (S.me.debug ? "Режим отладки: лимит сессий отключён."
         : left !== null ? `Сегодня можно пройти ещё ${left} из ${S.me.max_per_day}.` : "");
  }
  loadProfile($("profileBox"));
  loadLeaders();
}

async function loadLeaders() {
  try {
    const r = await api("/api/leaderboard");
    $("leaderBox").innerHTML = `<div class="muted">Участников: ${r.users}, попыток: ${r.attempts}</div>`
      + (r.top.length ? `<table><tr><th>#</th><th>ник</th><th>лучший WPM</th><th>точность</th></tr>`
      + r.top.map((x, i) => `<tr><td>${i + 1}</td><td>${esc(x.nickname)}</td><td>${x.best}</td>`
      + `<td>${Math.round(x.acc * 100)}%</td></tr>`).join("") + "</table>" : "<p>Пока пусто.</p>");
  } catch (e) { $("leaderBox").textContent = "—"; }
}

async function loadProfile(box) {
  try {
    const r = await api(`/api/profile?uid=${encodeURIComponent(S.uid)}`);
    const p = r.profile;
    if (!p.bigrams || !p.bigrams.length) return;
    const maxW = Math.max(...p.bigrams.map(b => b.weakness), 0.01);
    let html = `<p>Общий темп: <b>${(1 / p.speed_factor).toFixed(2)}×</b> от среднего участника.</p>
      <table><tr><th>переход</th><th>у вас</th><th>ожидалось</th><th></th></tr>`;
    for (const b of p.bigrams.slice(0, 7)) {
      html += `<tr><td><span class="pill">${esc(b.label)}</span></td><td>${Math.round(b.post_ms)} мс</td>
        <td>${Math.round(b.expected_ms)} мс</td>
        <td><span class="bar" style="width:${Math.max(4, 80 * b.weakness / maxW)}px"></span></td></tr>`;
    }
    html += "</table>";
    if (p.keys && p.keys.length) {
      html += `<p>Чаще всего ошибаетесь: ${p.keys.slice(0, 4).map(k => `<span class="pill">${esc(k.label)}</span>`).join(" ")}</p>`;
    }
    if (r.history.length) {
      html += `<p class="muted">Скорость по сессиям: ${r.history.map(h => `${h.session_no}: ${h.wpm}`).join(" → ")}</p>`;
    }
    box.innerHTML = html + `<p class="muted">Оценки сглажены эмпирическим Байесом: редкие переходы
      не объявляются «слабыми» по одной случайной задержке.</p>`;
  } catch (e) { /* профиль пока недоступен */ }
}

$("btnStart").addEventListener("click", async () => {
  try {
    const r = await api("/api/session/start", { uid: S.uid, mode: CHEAT ? "cheat" : "normal",
                                                 cheat_type: CHEAT ? $("cheatType").value : null, debug: DEBUG });
    Object.assign(S, { plan: r.plan, idx: 0, total: r.total, offset: r.offset, sessionNo: r.session_no, results: [] });
    show("scrType");
    nextAttempt();
  } catch (err) {
    alert(err.message);
    openHome();
  }
});
$("btnHome").addEventListener("click", openHome);

// ================================================================ печать
const CYR = /[а-яё]/i, LAT = /[a-z]/i;

function nextAttempt() {
  if (S.idx >= S.plan.length) return finishSession();
  const item = S.plan[S.idx];
  S.att = { item, typed: "", events: [], t0: performance.now(), started: false, finished: false,
            flags: { blur: 0, paste: 0, long_pause: 0, ctrl_bs: 0 }, lastDown: null, wrongLayout: 0 };
  const n = S.offset + S.idx + 1;
  const kindName = { calib: "Калибровочная фраза", text: "Текст", adaptive: "Персональное упражнение" }[item.kind];
  const nCalib = S.offset + S.plan.filter(p => p.kind === "calib").length;
  $("stageLabel").textContent = item.kind === "calib"
    ? `${kindName} — повтор ${n} из ${nCalib}` : `${kindName} · задание ${n} из ${S.total}`;
  $("progBar").style.width = `${100 * (n - 1) / S.total}%`;
  $("liveWpm").textContent = "0";
  $("typeHint").classList.remove("hidden");
  render();
  $("prompt").focus();
}

function render() {
  const { item, typed } = S.att;
  const p = item.prompt;
  let html = "";
  for (let i = 0; i < p.length; i++) {
    const ch = p[i] === " " ? "&nbsp;" : esc(p[i]);
    let cls = "todo";
    if (i < typed.length) cls = typed[i] === p[i] ? "ok" : "bad";
    else if (i === typed.length) cls = "cur";
    html += `<span class="${cls}">${ch}</span>`;
  }
  $("prompt").innerHTML = html;
}

function rec(type, e) {
  const t = performance.now() - S.att.t0;
  S.att.events.push([type, e.code || ("Unidentified:" + e.key), (e.key || "").slice(0, 12),
                     Math.round(t * 10) / 10, e.isTrusted ? 1 : 0]);
}

window.addEventListener("keydown", (e) => {
  if ($("scrType").classList.contains("hidden") || !S.att || S.att.finished) return;
  if (e.repeat) { e.preventDefault(); return; }
  const a = S.att;
  if ((e.ctrlKey || e.metaKey) && !e.altKey) {
    if (e.key === "v" || e.key === "м") a.flags.paste++;
    if (e.key === "Backspace") a.flags.ctrl_bs++;
    if (e.key !== "Backspace") return;   // остальные сочетания не трогаем
  }
  if (e.key === "Tab" || e.key === "Escape" || e.key.startsWith("F")  && e.key.length > 1) return;
  e.preventDefault();
  const now = performance.now();
  if (a.lastDown !== null && now - a.lastDown > 20000) a.flags.long_pause++;
  a.lastDown = now;
  if (!a.started) { a.started = true; a.firstDown = now; $("typeHint").classList.add("hidden"); }
  rec("d", e);
  if (e.key === "Backspace") {
    a.typed = a.typed.slice(0, -1);
  } else if (e.key.length === 1) {
    const exp = a.item.prompt[a.typed.length] || "";
    if ((CYR.test(exp) && LAT.test(e.key)) || (LAT.test(exp) && CYR.test(e.key))) a.wrongLayout++;
    else a.wrongLayout = 0;
    $("layoutWarn").classList.toggle("hidden", a.wrongLayout < 3);
    a.typed += e.key;
  }
  render();
  const mins = (now - a.firstDown) / 60000;
  if (mins > 0.02) $("liveWpm").textContent = Math.round(a.typed.length / 5 / mins);
  if (a.typed.length >= a.item.prompt.length) {
    a.finished = true;
    // ещё немного ловим keyup последних клавиш
    setTimeout(submitAttempt, 300);
  }
});

window.addEventListener("keyup", (e) => {
  if ($("scrType").classList.contains("hidden") || !S.att || !S.att.started) return;
  rec("u", e);
});
window.addEventListener("blur", () => { if (S.att && S.att.started && !S.att.finished) S.att.flags.blur++; });
document.addEventListener("paste", (e) => { if (S.att) { S.att.flags.paste++; e.preventDefault(); } });

async function submitAttempt() {
  const a = S.att;
  const body = { uid: S.uid, session_no: S.sessionNo, kind: a.item.kind, lang: a.item.lang,
                 prompt: a.item.prompt, typed: a.typed, events: a.events, flags: a.flags,
                 mode: CHEAT ? "cheat" : "normal", cheat_type: CHEAT ? $("cheatType").value : null, debug: DEBUG,
                 client: { timer_res_ms: CLIENT.timer_res_ms } };
  let res = null;
  for (let tries = 0; tries < 3 && !res; tries++) {
    try { res = await api("/api/attempt", body); }
    catch (e) { await new Promise(r => setTimeout(r, 1000)); }
  }
  if (res) {
    S.results.push(res);
    let txt = `Прошлое задание: ${res.wpm} WPM, точность ${Math.round(res.acc * 100)}%`;
    if (res.p_cheat !== null && res.p_cheat !== undefined) {
      const pc = (x) => (x < 0.01 ? "<1" : x > 0.99 ? ">99" : Math.round(x * 100)) + "%";
      const [lo, hi] = res.p_cheat_ci90.map(pc);
      txt += ` · вероятность читерства ${pc(res.p_cheat)} (90% интервал ${lo === hi ? lo : lo + "–" + hi})`;
    }
    if (res.note) txt += ` · ${res.note}`;
    $("lastResult").textContent = txt;
  } else {
    $("lastResult").textContent = "Не удалось отправить попытку (нет связи?) — продолжаем.";
  }
  S.idx++;
  setTimeout(nextAttempt, 400);
}

async function finishSession() {
  show("scrDone");
  const rs = S.results.filter(r => r.wpm > 0);
  const avg = (k) => rs.length ? rs.reduce((s, r) => s + r[k], 0) / rs.length : 0;
  $("doneStats").innerHTML = `<div><b>${avg("wpm").toFixed(0)}</b>средний WPM</div>
    <div><b>${Math.round(avg("acc") * 100)}%</b>точность</div><div><b>${rs.length}</b>заданий</div>`;
  $("doneNext").textContent = CHEAT ? "" :
    "Можно начать ещё одну сессию из кабинета (до 3 в день). Для исследования особенно полезно вернуться и в другой день. "
    + `Ваш код: ${S.uid}`;
  $("doneProfile").innerHTML = "";
  if (!CHEAT) loadProfile($("doneProfile"));
}

function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ================================================================ старт
(async () => {
  if (CHEAT) $("cheatBanner").classList.remove("hidden");
  if (CLIENT.touch && !matchMedia("(pointer:fine)").matches) {
    alert("Тренажёр рассчитан на физическую клавиатуру — откройте его на компьютере.");
  }
  S.uid = store.get("kt_uid");
  if (S.uid) { try { await openHome(); return; } catch (e) { S.uid = null; } }
  show("scrWelcome");
})();
