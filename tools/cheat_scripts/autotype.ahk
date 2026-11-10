; AutoHotkey v2 — «читер» для сбора реальных негативных примеров (тип в тренажёре: AutoHotkey / макрос).
; Скачать AutoHotkey v2: https://www.autohotkey.com/ (официальный сайт), затем двойной клик по этому файлу.
; 1) откройте тренажёр с ?cheat=1, выберите «AutoHotkey / макрос-скрипт», начните сессию;
; 2) нажмите на сайте «Скопировать текст задания»;
; 3) нажмите F8 — скрипт «напечатает» текст. Esc — выход.
; Раскладка Windows должна совпадать с языком задания.
; Клавиши отправляются по одной как обычные нажатия (не {Text}), поэтому браузер видит настоящий event.code.
#Requires AutoHotkey v2.0
SendMode "Event"

F8:: {
    delay := Random(40, 160)      ; задержка между нажатиями, мс (меняйте между прогонами)
    press := Random(10, 70)       ; длительность нажатия, мс
    SetKeyDelay delay, press
    text := StrReplace(A_Clipboard, Chr(160), " ")
    Loop Parse, text {
        ch := A_LoopField
        if (ch = " ")
            SendEvent "{Space}"
        else
            SendEvent "{" ch "}"
        if (Random(0, 100) < 30)  ; случайный джиттер
            Sleep Random(0, 60)
    }
}
Esc::ExitApp
