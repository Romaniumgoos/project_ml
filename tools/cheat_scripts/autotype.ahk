; AutoHotkey v2: «читер» для сбора негативных примеров.
; 1) откройте тренажёр с ?cheat=1, начните сессию
; 2) скопируйте текст задания в буфер обмена (выделите мышью, Ctrl+C)
; 3) кликните в поле задания и нажмите F8 — скрипт «напечатает» текст
; Меняйте задержки между прогонами, чтобы данные были разнообразнее.
#Requires AutoHotkey v2.0
SendMode "Event"

F8:: {
    delay := Random(40, 160)      ; задержка между нажатиями, мс
    press := Random(10, 70)       ; длительность нажатия, мс
    SetKeyDelay delay, press
    text := A_Clipboard
    Loop Parse, text {
        SendEvent "{Text}" A_LoopField
        if (Random(0, 100) < 30)  ; случайный джиттер
            Sleep Random(0, 60)
    }
}
Esc::ExitApp
