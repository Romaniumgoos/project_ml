// Самый примитивный читер: события из JavaScript (вставить в консоль браузера, F12 → Console).
// Такие события помечены isTrusted=false — сервер ловит их правилом, без ML.
// Нужен только чтобы убедиться, что правило срабатывает.
(async () => {
  const text = document.getElementById("prompt").innerText.replace(/ /g, " ");
  for (const ch of text) {
    const code = ch === " " ? "Space" : "Key" + ch.toUpperCase();
    window.dispatchEvent(new KeyboardEvent("keydown", { key: ch, code }));
    await new Promise(r => setTimeout(r, 40));
    window.dispatchEvent(new KeyboardEvent("keyup", { key: ch, code }));
    await new Promise(r => setTimeout(r, 60));
  }
})();
