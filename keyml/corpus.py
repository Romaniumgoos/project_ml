"""Тексты для тренажёра. Все предложения написаны для проекта (без чужих текстов)."""

# Калибровочные фразы — полные панграммы (все буквы алфавита). Печатаются по CALIB_REPS раз
# за сессию: на фиксированном тексте учится биометрическая верификация
# (сравнимые векторы таймингов одинаковой длины, как в датасете CMU).
CALIB = {
    "ru": "Съешь же ещё этих мягких французских булок, да выпей чаю.",
    "en": "The quick brown fox jumps over the lazy dog.",
}

TEXTS = {
    "ru": [
        "утром в общежитии всегда пахнет кофе и немного паникой перед зачётом",
        "модель обучалась всю ночь, а утром выяснилось, что метки были перепутаны",
        "на остановке стоял троллейбус, но водитель явно не спешил никуда ехать",
        "если код работает с первого раза, значит, где-то спрятана очень хитрая ошибка",
        "в библиотеке было тихо, только шуршали страницы и гудел старый проектор",
        "лабораторная работа по физике заняла больше времени, чем вся сессия",
        "кот сидел на подоконнике и с презрением наблюдал за голубями",
        "градиентный спуск похож на поиск выхода из тумана на ощупь",
        "весной в парке появляются велосипедисты, бегуны и продавцы мороженого",
        "староста группы напомнил, что отчёт нужно сдать до пятницы",
        "чтобы печатать быстро, нужно сначала научиться печатать правильно",
        "в столовой опять закончились блины, и очередь медленно разошлась",
        "переобучение случается, когда модель запоминает ответы вместо закономерностей",
        "дождь начался внезапно, и все побежали прятаться под навес",
        "нормализация признаков часто помогает алгоритму сойтись быстрее",
        "на семинаре преподаватель рисовал графики прямо в воздухе руками",
        "поезд пришёл вовремя, что само по себе было маленьким чудом",
        "хорошая клавиатура не сделает вас программистом, но поможет",
        "в субботу мы собрались у друга и до ночи играли в настольные игры",
        "датасет оказался больше, чем помещался в оперативную память ноутбука",
        "зимой город становится тише, а фонари кажутся ярче",
        "каждое нажатие клавиши оставляет след, по которому можно узнать человека",
        "в конце семестра все внезапно вспоминают про долги и консультации",
        "свежий хлеб из пекарни на углу заканчивается уже к обеду",
        "дерево решений задаёт вопросы до тех пор, пока ответ не станет очевидным",
        "сосед по комнате учит японский и разговаривает сам с собой",
        "мы долго спорили, какой цвет выбрать для графиков в презентации",
        "в лесу пахло хвоей, грибами и чуть-чуть дымом от костра",
        "скорость печати растёт незаметно, если заниматься каждый день понемногу",
        "ночью сервер упал, и утром в чате было сорок непрочитанных сообщений",
    ],
    "en": [
        "the coffee machine on the third floor has been broken since monday",
        "a good model is simple enough to explain and accurate enough to trust",
        "we missed the bus and had to walk across the whole city in the rain",
        "my laptop fan sounds like a jet engine whenever i open the browser",
        "she wrote the first version of the parser in a single afternoon",
        "every keystroke tells a small story about the person who typed it",
        "the library closes at nine, so we moved our study group to the cafe",
        "random forests rarely overfit, but they can still surprise you",
        "the cat knocked a glass of water onto the keyboard during the call",
        "learning to type without looking at the keys takes patience",
        "our team spent two days arguing about the name of a single variable",
        "the train was late again, so i finished reading the paper on my phone",
        "clean data is worth more than a clever algorithm",
        "the old printer in the dean's office only works on tuesdays",
        "he practiced the guitar every evening until his fingers hurt",
        "a validation set should never be used to train the model",
        "the market near the station sells fresh bread and cheap flowers",
        "after the exam we celebrated with pizza and a long walk",
        "the server logs showed thousands of identical requests per second",
        "small daily practice beats rare marathon sessions",
    ],
}

# Частотные слова для адаптивных упражнений (подбираются под слабые биграммы).
WORDS = {
    "ru": """и в не на я быть он с что а по это она этот к но они мы как из у который то за
свой что весь год от так о для ты же все тот мочь вы человек такой его сказать только или
ещё бы себя один как уже до время если сам когда другой вот говорить наш мой знать стать
при чтобы дело жизнь кто первый очень два день её новый рука даже во со раз где там под
можно ну какой после их работа без самый потом надо хотеть ли слово идти большой должен
место иметь ничто то сейчас тут лицо каждый друг нет теперь ни глаз тоже тогда видеть
вопрос через да здесь дом сторона думать сделать страна жить чем мир об последний случай
голова более делать что-то смотреть ребёнок просто конечно сила российский конец перед
несколько вид система всегда работать между три нет понять пойти часть спросить город
давать также никто понимать получить отношение лишь второй именно хороший деньги нужно
земля главный решение задача модель данные обучение признак выборка ошибка точность
клавиша палец скорость текст пример студент учёба экзамен лекция проект группа неделя
вечер утро ночь окно дверь стол книга письмо рынок завод фабрика щука объём съезд
подъезд шёпот чашка щётка жёлтый счёт съёмка экскурсия цифра хищник чувство взгляд""".split(),
    "en": """the be to of and a in that have i it for not on with he as you do at this but his
by from they we say her she or an will my one all would there their what so up out if
about who get which go me when make can like time no just him know take people into year
your good some could them see other than then now look only come its over think also back
after use two how our work first well way even new want because any these give day most
us model data train test error feature value learn speed finger quick brown jump lazy
keyboard practice network vector matrix gradient random forest sample label class score
quiz jazz zebra fizz buzz exact extra squeeze puzzle wizard oxygen lynx sphinx crypt""".split(),
}


# Самые частотные слова (словоформы) — из них собираются случайные задания.
# Короткие служебные слова оставлены: они дают самые частые биграммы реальной печати.
FREQ_WORDS = {
    "ru": """и в не на я что он с это а как по но они к у мы из за вы так же от все она
то для ты бы только его если уже когда был нет да вот было или где там еще ее мне их до
был тоже чтобы при ну даже под меня потом может очень себя тут без тебя есть него можно
сейчас надо много чем время после были кто раз здесь нас всё через этого них ничего
теперь со год два день жизнь дело человек люди рука глаза голова слово место дом мир
лицо друг сторона вопрос работа страна город земля вода конец случай час минута путь
ночь утро вечер стол дверь окно книга машина школа деньги сила система история
хорошо больше просто сразу снова всегда никогда опять почти совсем вместе почему
сказал знать мочь стать говорить видеть идти думать хотеть делать сделать дать взять
жить понять помнить сидеть стоять смотреть любить ждать спросить ответить решить
начать прийти найти пойти работать писать читать играть сказать знаю могу хочу
новый большой первый последний хороший главный другой старый молодой маленький
белый черный долгий важный простой полный высокий русский каждый такой который
какой свой весь этот тот один три пять сто первый второй сегодня завтра вчера
быстро медленно тихо громко рядом далеко внизу наверх вперед назад справа слева
мама папа брат сестра сын дочь отец мать семья ребенок девушка парень учитель
студент урок экзамен задача ответ пример число буква текст клавиша палец""".split(),
    "en": """the be to of and a in that have i it for not on with he as you do at this but his
by from they we say her she or an will my one all would there their what so up out if
about who get which go me when make can like time no just him know take people into year
your good some could them see other than then now look only come its over think also back
after use two how our work first well way even new want because any these give day most
us is was are been has had did said made went came took got know thought told found
man woman child world life hand part place case week company system program question
government number night point home water room mother area money story fact month lot
right study book eye job word business issue side kind head house service friend father
power hour game line end member law car city community name team minute idea kid body
small large next early young important few public bad same able great little long""".split(),
}


def random_task(lang: str, rng, min_chars: int = 70, max_words: int = 16) -> str:
    """Случайное задание из частотных слов: ~70+ символов, без повторов слов подряд."""
    pool = FREQ_WORDS[lang]
    words: list[str] = []
    while len(" ".join(words)) < min_chars and len(words) < max_words:
        w = rng.choice(pool)
        if not words or w != words[-1]:
            words.append(w)
    return " ".join(words)
