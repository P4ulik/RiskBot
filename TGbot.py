import json
import os
import re
from datetime import datetime
from telebot import TeleBot
from telebot.types import ReplyKeyboardMarkup, KeyboardButton
from docx import Document
import fitz
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from apscheduler.schedulers.background import BackgroundScheduler
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd

# КОНФИГУРАЦИЯ И ЗАГРУЗКА МОДЕЛИ
TOKEN = 'YOUR_TOKEN'
YOUR_CHAT_ID = 'YOUR_CHAT_ID'
bot = TeleBot(TOKEN)
DB_FILE = 'documents.json'

user_states, user_texts, user_temp = {}, {}, {}

model_path = 'model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
print("ML-модель загружена")

# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
def classify_risk_ml(text):
    inputs = tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    pred_class = probs.argmax().item()
    return ['Красный', 'Желтый', 'Зеленый'][pred_class], probs[0][pred_class].item()


def load_docs():
    return json.load(open(DB_FILE, 'r', encoding='utf-8')) if os.path.exists(DB_FILE) else []


def save_docs(docs):
    json.dump(docs, open(DB_FILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)


def compare_dates(date1_str, date2_str):
    """Сравнивает две даты и возвращает более раннюю"""
    try:
        d1 = datetime.strptime(date1_str, '%Y-%m-%d')
        d2 = datetime.strptime(date2_str, '%Y-%m-%d')
        return d1.strftime('%Y-%m-%d') if d1 <= d2 else d2.strftime('%Y-%m-%d')
    except:
        return date1_str if date1_str else date2_str


def extract_deadline_from_text(text):
    """
    Ищет ДАТУ ОТВЕТА/ИСПОЛНЕНИЯ по ключевым фразам.
    """
    patterns = [
        # Срок - до 25 декабря 2023 г.
        r'(?:срок|дата|не позднее)\s*[-–—]?\s*(?:до|исполнения)?\s*до\s+(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+(\d{4})',

        # проинформировать до 18 июня 2026 года
        r'(?:проинформировать|направить|представить|обеспечить|просим|необходимо|предоставить|уведомить)\s+.*?(?:в срок|до)\s+(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+(\d{4})',

        # Срок до 25.12.2023 (с пробелом и без)
        r'(?:срок|ответ|направить|представить|просим|подтвердить|участие|доклад|представить)\s*(?:до|в срок)\s*[-–—]?\s*(\d{1,2})\.(\d{1,2})\.(\d{4})',

        # до 25.12.2023 (просто дата с предлогом)
        r'до\s+(\d{1,2})\.(\d{1,2})\.(\d{4})',

        # Дедлайн: 25.12.2023
        r'(?:дедлайн|deadline)\s*[-–—]?\s*(\d{1,2})\.(\d{1,2})\.(\d{4})',

        # Срок - до 25.12.2023
        r'срок\s*[-–—]?\s*до\s+(\d{1,2})\.(\d{1,2})\.(\d{4})',

        # --- НОВОЕ: дата с дефисом (20-12-2023) ---
        r'(?:срок|до|ответ)\s*[-–—]?\s*(\d{1,2})[-–—](\d{1,2})[-–—](\d{4})',
    ]

    months = {
        'января': 1, 'февраля': 2, 'марта': 3, 'апреля': 4,
        'мая': 5, 'июня': 6, 'июля': 7, 'августа': 8,
        'сентября': 9, 'октября': 10, 'ноября': 11, 'декабря': 12
    }

    found_dates = []

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            groups = match.groups()
            try:
                if len(groups) == 3:
                    # Проверяем текстовый месяц
                    if groups[1].lower() in months:
                        day = int(groups[0])
                        month = months[groups[1].lower()]
                        year = int(groups[2])
                        date_obj = datetime(year, month, day)
                        found_dates.append(date_obj)
                    # Формат ДД.ММ.ГГГГ или ДД-ММ-ГГГГ
                    elif groups[0].isdigit() and groups[1].isdigit() and groups[2].isdigit():
                        day = int(groups[0])
                        month = int(groups[1])
                        year = int(groups[2])
                        date_obj = datetime(year, month, day)
                        found_dates.append(date_obj)
            except:
                continue

    if not found_dates:
        return None

    # Берем САМУЮ РАННЮЮ дату (самый строгий дедлайн)
    earliest_date = min(found_dates)
    return earliest_date.strftime('%Y-%m-%d')


def get_summary(text, max_len=300):
    # Разбираем на строки, убираем пробелы по краям и отсекаем совсем пустые строки
    raw_lines = [l.strip() for l in text.split('\n') if l.strip()]

    skip_patterns = [
        'кому:', 'от:', 'исп.:', 'копия:', 'тема:', 'ооо', 'ао', 'пао', 'зао', 'оао', 'ип', 'гбу', 'гуп', 'фгуп', 'г.',
        'директору', 'руководителю', 'начальнику', 'заместителю', 'главному инженеру', 'председателю', 'декану',
        'профессору',
        'уважаемый', 'уважаемая', 'уважаемые'
    ]

    lines = []
    for l in raw_lines:
        l_low = l.lower()

        # 1. Если строка начинается со служебного слова (даже после очистки пробелов) — пропускаем
        if any(l_low.startswith(p) for p in skip_patterns):
            continue

        # 2. проверка коротких строк (проверяем длину УЖЕ ОЧИЩЕННОЙ строки)
        if len(l) < 30:
            # Если строка заканчивается на точку инициала (например, "Иванову И.И.")
            if len(l) >= 3 and l[-2] == '.' or (
                    l[-1] == '.' and l[-2].isalpha() and (len(l) == 2 or not l[-3].isalpha())):
                continue  # Это инициал, пропускаем строку целиком!

            # Если в короткой строке вообще нет знаков конца предложения (.!?) — это заголовок/ФИО/должность, пропускаем
            if not l[-1] in '.!?':
                continue

        lines.append(l)

    # Собираем очищенный текст в одну строку
    full_text = ' '.join(lines)

    # Список маркеров сути
    markers = [
        'просим', 'уведомляем', 'напоминаем', 'сообщаем', 'информируем', 'требуем',
        'приглашаем', 'направляем', 'извещаем', 'рекомендуем', 'предлагаем',
        'указываем', 'обращаем', 'доводим', 'представляем',
        'в связи с', 'в соответствии с', 'согласно', 'во исполнение',
        'в целях', 'в рамках', 'по факту', 'по вопросу',
        'просим вас', 'обращаемся к вам', 'настоятельно просим', 'категорически требуем',
        'докладываем', 'направляем ответ', 'информируем о',
    ]

    positions = [full_text.lower().find(m) for m in markers if full_text.lower().find(m) != -1]
    if positions:
        full_text = full_text[min(positions):]

    # Находим финишную точку для обрезки
    end_pos = -1
    for i, ch in enumerate(full_text):
        if ch in '.!?':
            # Защита от дат типа 20.06.2026
            if ch == '.' and i > 0 and i + 1 < len(full_text) and full_text[i - 1].isdigit() and full_text[
                i + 1].isdigit():
                continue
            end_pos = i + 1
            break

    # СОХРАНЯЕМ ОСНОВНОЙ РЕЗУЛЬТАТ
    if end_pos != -1 and end_pos <= max_len:
        main_result = full_text[:end_pos].strip()
    else:
        main_result = (full_text[:max_len].rsplit(' ', 1)[0] + '...').strip() if len(
            full_text) > max_len else full_text.strip()

    # НОВЫЙ БЛОК: ИЩЕМ ПРЕДЛОЖЕНИЕ С ДАТОЙ
    # Паттерны для поиска дат
    date_patterns = [
        r'\d{1,2}\.\d{1,2}\.\d{4}',  # 12.05.2026
        r'\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+\d{4}',
        # 12 мая 2026
    ]

    # Ищем предложения с датами в оригинальном тексте (до обрезки)
    original_full_text = ' '.join(lines)  # Берем текст до обрезки по маркеру
    date_sentences = []

    # Разбиваем оригинальный текст на предложения
    temp_text = re.sub(r'(\d{1,2})\.(\d{1,2})\.(\d{4})', r'\1<DOT>\2<DOT>\3', original_full_text)
    raw_sentences = []
    current = ""
    for ch in temp_text:
        current += ch
        if ch in '.!?':
            current = re.sub(r'<DOT>', '.', current)
            raw_sentences.append(current.strip())
            current = ""
    if current:
        current = re.sub(r'<DOT>', '.', current)
        raw_sentences.append(current.strip())

    # Проверяем каждое предложение на наличие даты
    for sent in raw_sentences:
        sent_lower = sent.lower()
        # Пропускаем предложения с обращениями
        if any(sent_lower.startswith(p) for p in ['уважаемый', 'уважаемая', 'уважаемые']):
            continue
        # Проверяем наличие даты
        for pattern in date_patterns:
            if re.search(pattern, sent):
                # Проверяем, не содержится ли это предложение уже в main_result
                if sent not in main_result:
                    date_sentences.append(sent)
                break
        if len(date_sentences) >= 1:  # Берем только первое предложение с датой
            break

    # ФОРМИРУЕМ ФИНАЛЬНЫЙ РЕЗУЛЬТАТ
    final_result = main_result

    if date_sentences:
        # Добавляем предложение с датой к основному результату
        final_result = main_result + ' ' + date_sentences[0]
        # Если получилось слишком длинно — обрезаем
        if len(final_result) > max_len * 1.5:  # Разрешаем чуть больше
            final_result = final_result[:max_len * 1.5].rsplit(' ', 1)[0] + '...'

    # Проверка на нумерацию: 1., 2., 1.1., 2.1., 1.1, 2.1 и т.д.
    has_numbering = False
    for line in text.split('\n'):
        l = line.strip()
        if not l:
            continue
        # Проверяем, начинается ли строка с нумерации
        if re.match(r'^\d+(\.\d+)?\.?\s+[А-Яа-я]', l):
            has_numbering = True
            break

    if has_numbering:
        if final_result and len(final_result) > 10:
            final_result = final_result + ' (в тексте есть перечень поручений! Рекомендуется ручное ознакомление)'
        else:
            final_result = 'В тексте есть перечень поручений (рекомендуется ручное ознакомление)'

    # ПРОВЕРКА ДЛИНЫ ИТОГОВОГО РЕЗУЛЬТАТА
    if not final_result or len(final_result) < 10:
        # ЗАПАСНОЙ ВАРИАНТ: берем первые 300 символов из ОРИГИНАЛЬНОГО текста
        original_text = ' '.join([l.strip() for l in text.split('\n') if l.strip()])
        if len(original_text) > 20:
            return original_text[:max_len].rsplit(' ', 1)[0] + '...'
        return "Не удалось извлечь содержание текста, рекомендуется ручной просмотр"

    return final_result.strip()


def get_doc_status(doc, today):
    try:
        days = (datetime.strptime(doc['deadline'], '%Y-%m-%d') - today).days
    except:
        days = 999
    risk = doc.get('risk', 'Зеленый')
    if days <= 1 or risk == 'Красный':
        return '🔴', 'ВЫСОКИЙ', days
    if days <= 3 or risk == 'Желтый':
        return '🟡', 'СРЕДНИЙ', days
    return '🟢', 'НИЗКИЙ', days


def send_morning_tasks():
    """Автоматическая функция для отправки задач в 10:00"""
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    docs = [d for d in load_docs() if d.get('status') != 'выполнено']

    if not docs:
        msg = "☀️ **Доброе утро!**\n\nАктивных задач на сегодня нет."
    else:
        try:
            docs.sort(
                key=lambda d: ({'Красный': 0, 'Желтый': 1, 'Зеленый': 2}.get(d.get('risk', 'Зеленый'), 2), d['deadline']))
            msg = "☀️ **Доброе утро! Напоминание об активных задачах:**\n\n" + render_docs_list(docs, today)
        except Exception as sort_err:
            print(f"❌ Ошибка сортировки в планировщике: {sort_err}")
            msg = "☀️ **Доброе утро!**\n\n Возникла ошибка при сортировке задач, но база сохранена. Проверьте список через меню."
    try:
        bot.send_message(YOUR_CHAT_ID, msg, parse_mode='Markdown')
        print(f"Утреннее уведомление успешно отправлено в {datetime.now()}")
    except Exception as e:
        print(f"Ошибка отправки утреннего уведомления: {e}")


def render_docs_list(docs_list, today, full_info=True):
    msg = ""
    for idx, doc in enumerate(docs_list, 1):
        try:
            emoji, label, days = get_doc_status(doc, today)
            summary = doc.get('summary', get_summary(doc.get('text', '')))
            file_name = doc.get('file_name', '📝 Текстовое сообщение')
            days_text = "Просрочка!" if days < 0 else f"{days} дн."

            if full_info:
                msg += f"{idx}. {emoji} **{label}** (ML уверенность: {doc.get('confidence', 0):.0%})\n"
                msg += f"   📁 *Файл:* `{file_name}`\n"
                msg += f"   📄 {summary}\n   ⏳ Дедлайн: {doc['deadline']} ({days_text})\n\n"
            else:
                msg += f"{idx}. {emoji} `{file_name}` — {summary}\n"

        except Exception as err:
            print(f"❌ Ошибка отрисовки документа под индексом {idx}: {err}")
            msg += f"{idx}. ⚠️ _Ошибка чтения данных этого документа (пропущен)_\n\n"
            continue
    return msg


@bot.message_handler(commands=['chart'])
def chart(message):
    chat_id = message.chat.id
    docs = load_docs()
    active = [d for d in docs if d.get('status') != 'выполнено']

    if not active:
        return bot.send_message(chat_id, "📊 Нет активных документов для графика.", reply_markup=get_main_keyboard())

    # Считаем количество по рискам
    risks = {'Красный': 0, 'Желтый': 0, 'Зеленый': 0}
    for d in active:
        risk = d.get('risk', 'Зеленый')
        risks[risk] += 1

    # Создаем график
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#ff4444', '#ffaa00', '#44bb44']
    bars = ax.bar(risks.keys(), risks.values(), color=colors)

    # Добавляем подписи на столбцах
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_title('Распределение активных документов по уровню риска', fontsize=14, fontweight='bold')
    ax.set_ylabel('Количество документов', fontsize=12)
    ax.set_xlabel('Уровень риска', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Сохраняем в буфер и отправляем
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close()

    # Отправляем график и статистику
    total = len(active)
    red = risks['Красный']
    yellow = risks['Желтый']
    green = risks['Зеленый']

    msg = f"📊 **Статистика по рискам:**\n\n"
    msg += f"📄 Всего активных: {total}\n"
    msg += f"🔴 Высокий риск: {red}\n"
    msg += f"🟡 Средний риск: {yellow}\n"
    msg += f"🟢 Низкий риск: {green}\n"

    bot.send_message(chat_id, msg, parse_mode='Markdown', reply_markup=get_main_keyboard())
    bot.send_photo(chat_id, buf)


@bot.message_handler(commands=['search'])
def search(message):
    chat_id = message.chat.id
    # Извлекаем запрос из команды
    query = message.text.replace('/search', '').strip()

    if not query:
        return bot.send_message(
            chat_id,
            "📝 Введите слово для поиска: `/search <слово>`\n\n"
            "Например: `/search штраф`",
            parse_mode='Markdown',
            reply_markup=get_main_keyboard()
        )

    docs = load_docs()
    found = []
    for d in docs:
        text = d.get('text', '')
        # Ищем в тексте и в выжимке
        if query.lower() in text.lower() and d.get('status') != 'выполнено':
            found.append(d)

    if not found:
        return bot.send_message(
            chat_id,
            f"🔍 Ничего не найдено по запросу **'{query}'**\n\n"
            "Попробуйте другое слово.",
            parse_mode='Markdown',
            reply_markup=get_main_keyboard()
        )

    # Формируем результат
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    msg = f"🔍 **Результаты поиска:** '{query}'\n\n"
    msg += f"📄 Найдено документов: {len(found)}\n\n"

    for i, d in enumerate(found[:10], 1):
        file_name = d.get('file_name', 'Без имени')
        summary = d.get('summary', get_summary(d.get('text', ''))[:50])
        risk = d.get('risk', 'Зеленый')
        deadline = d.get('deadline', 'не указан')

        # Определяем эмодзи для риска
        risk_emoji = '🔴' if risk == 'Красный' else '🟡' if risk == 'Желтый' else '🟢'

        msg += f"{i}. {risk_emoji} **{file_name}**\n"
        msg += f"   📝 {summary[:60]}...\n"
        msg += f"   ⏳ Дедлайн: {deadline}\n\n"

    if len(found) > 10:
        msg += f"\n_... и еще {len(found) - 10} документов_"

    bot.send_message(chat_id, msg, parse_mode='Markdown', reply_markup=get_main_keyboard())


@bot.message_handler(commands=['export'])
def export(message):
    chat_id = message.chat.id
    docs = load_docs()

    if not docs:
        return bot.send_message(chat_id, "📭 Нет документов для экспорта.", reply_markup=get_main_keyboard())

    # Создаем DataFrame
    data = []
    for d in docs:
        # Получаем статус с человеческим названием
        status = d.get('status', 'не известно')
        if status == 'не выполнено':
            status_display = 'Активен'
        elif status == 'выполнено':
            status_display = 'Выполнен'
        elif status == 'просрочено':
            status_display = 'Просрочен'
        else:
            status_display = status

        # Получаем выжимку (краткое содержание)
        summary = d.get('summary', get_summary(d.get('text', ''))[:100])

        data.append({
            'Файл': d.get('file_name', ''),
            'Риск': d.get('risk', ''),
            'Дедлайн': d.get('deadline', ''),
            'Статус': status_display,
            'Краткое содержание': summary,
            'Создан': d.get('created_at', ''),
        })

    # Сортируем по дате создания (сначала новые)
    df = pd.DataFrame(data)
    df = df.sort_values('Создан', ascending=False)

    # Сохраняем в Excel
    excel_file = 'export.xlsx'
    df.to_excel(excel_file, index=False, sheet_name='Документы')

    # Отправляем файл
    with open(excel_file, 'rb') as f:
        bot.send_document(
            chat_id,
            f,
            caption=f"📊 **Экспорт данных**\n\n📄 Всего документов: {len(docs)}\n✅ Выполнено: {len([d for d in docs if d.get('status') == 'выполнено'])}\n⏳ Активных: {len([d for d in docs if d.get('status') != 'выполнено'])}",
            parse_mode='Markdown'
        )

    # Удаляем временный файл
    os.remove(excel_file)

def parse_input_ids(text, max_limit):
    ids = [int(p) for p in re.sub(r'[,，\s]+', ' ', text).split() if p.isdigit()]
    return (ids, True) if ids and all(1 <= i <= max_limit for i in ids) else (ids, False)


def process_document_save(chat_id, date_obj, today):
    """Сохраняет документ с датой (используется после подтверждения или ручного ввода)"""
    doc_text = user_texts.get(chat_id, '')
    risk_class, confidence = classify_risk_ml(doc_text)
    days_left = (date_obj - today).days
    summary = get_summary(doc_text)

    saved_file_name = user_temp.get(chat_id, {}).pop('current_file_name', '📝 Текстовое сообщение')

    emoji, status_text, _ = get_doc_status({'deadline': date_obj.strftime('%Y-%m-%d'), 'risk': risk_class}, today)
    rec = 'Срочно принять меры!' if emoji == '🔴' else 'Запланировать ответ в ближайшие дни.' if emoji == '🟡' else 'Можно выполнить в штатном режиме.'

    reasons = [
        f"{'Высокий' if risk_class == 'Красный' else 'Средний' if risk_class == 'Желтый' else 'Низкий'} уровень риска документа по ML-модели"] #(ML, уверенность {confidence:.1%})"]
    if days_left < 0:
        reasons.append(f"Документ просрочен на {abs(days_left)} дней")
    elif days_left == 0:
        reasons.append("Дедлайн сегодня!")
    elif days_left == 1:
        reasons.append("Остался 1 день!")
    elif days_left <= 3:
        reasons.append(f"Осталось {days_left} дня(ей)")

    reply = f"📁 **Документ:** `{saved_file_name}`\n📄 **Краткое содержание:**\n> {summary}\n\n{emoji} **{status_text} УРОВЕНЬ ВАЖНОСТИ**\n"
    reply += f"📌 *Причины:* {'; '.join(reasons)}\n⏳ *До дедлайна:* {'Просрочка' if days_left < 0 else f'{days_left} дн.'}\n"
    reply += f"💡 *Рекомендация:* {rec}\n\n✅ Документ сохранён в память."

    load_d = load_docs()
    load_d.append(
        {'text': doc_text, 'summary': summary, 'deadline': date_obj.strftime('%Y-%m-%d'), 'risk': risk_class,
         'confidence': confidence, 'status': 'не выполнено', 'file_name': saved_file_name,
         'created_at': datetime.now().isoformat()})
    save_docs(load_d)

    user_states[chat_id] = None
    user_texts[chat_id] = ''
    user_temp[chat_id] = {}
    bot.send_message(chat_id, reply, parse_mode='Markdown', reply_markup=get_main_keyboard())


# --- КЛАВИАТУРЫ ---
def get_main_keyboard():
    markup = ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    buttons = [
        '📋 Все задачи', '🔥 Срочные задачи',
        '✅ Отметить выполненным', '📦 Архив',
        '🗑️ Очистить архив', '📝 Ввести текст',
        '📊 График рисков', '🔍 Поиск',
        '📤 Экспорт в Excel'
    ]
    markup.add(*[KeyboardButton(b) for b in buttons])
    return markup


def show_clear_archive_menu(chat_id):
    markup = ReplyKeyboardMarkup(row_width=2, resize_keyboard=True)
    bot.send_message(chat_id, "🗑️ **Очистка архива**\n\nВыберите действие:",
                     reply_markup=markup.add('🧹 Удалить все', '🔢 Выборочно удалить', '◀️ Назад'), parse_mode='Markdown')


# --- КОМАНДЫ /START И /CANCEL ---
@bot.message_handler(commands=['start', 'cancel'])
def handle_start_cancel(message):
    chat_id = message.chat.id
    user_states[chat_id] = 'waiting_text' if message.text == '/start' else None
    user_texts[chat_id] = ''
    if chat_id in user_temp:
        user_temp[chat_id].pop('current_file_name', None)

    msg = ("👋 Здравствуйте!\n\nЯ помогу вам отслеживать документы, а так же их содержание и контролировать сроки ответов.\n\n"
           "📎 Отправьте мне файл .docx или .pdf с текстом документа.\nИли просто скопируйте текст в чат.\n\n"
           "Не забывайте своевременно чистить архив бота\n\n" 
           "Затем я попрошу вас указать дату, до которой нужно дать ответ на документ.") if message.text == '/start' else "Действие отменено."
    bot.send_message(chat_id, msg, reply_markup=get_main_keyboard())


# --- ОБРАБОТЧИК ФАЙЛОВ ---
@bot.message_handler(content_types=['document'])
def handle_document(message):
    chat_id = message.chat.id
    file_name = message.document.file_name
    file_name_lower = file_name.lower()

    if not (file_name_lower.endswith('.docx') or file_name_lower.endswith('.pdf')):
        return bot.send_message(chat_id, "❌ Поддерживаются только файлы .docx и .pdf.",
                                reply_markup=get_main_keyboard())

    temp_path = f"temp_{chat_id}_{file_name_lower}"
    with open(temp_path, 'wb') as f:
        f.write(bot.download_file(bot.get_file(message.document.file_id).file_path))

    try:
        if file_name_lower.endswith('.docx'):
            text = '\n'.join([p.text for p in Document(temp_path).paragraphs if p.text.strip()])
        else:
            doc = fitz.open(temp_path)
            text = '\n'.join([page.get_text() for page in doc if page.get_text().strip()])
            doc.close()
        os.remove(temp_path)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return bot.send_message(chat_id, f"❌ Ошибка чтения файла: {e}", reply_markup=get_main_keyboard())

    if not text.strip():
        return bot.send_message(chat_id, "❌ В документе нет текста.", reply_markup=get_main_keyboard())

    # Сохраняем текст и имя файла
    user_texts[chat_id] = text
    if chat_id not in user_temp:
        user_temp[chat_id] = {}
    user_temp[chat_id]['current_file_name'] = file_name

    # --- АВТОМАТИЧЕСКИЙ ПОИСК ДАТЫ ПРИ ЗАГРУЗКЕ ФАЙЛА ---
    found_deadline = extract_deadline_from_text(text)

    if found_deadline:
        user_temp[chat_id]['found_deadline'] = found_deadline
        user_states[chat_id] = 'waiting_deadline_confirmation'
        bot.send_message(
            chat_id,
            f"📄 Текст документа `{file_name}` получен.\n\n"
            f"🔍 В тексте найдена дата дедлайна: **{found_deadline}**\n\n"
            "✅ Если дата верна, напишите **Да**.\n"
            "✏️ Если хотите указать другую дату, просто введите её в формате ДД.ММ.ГГГГ.",
            parse_mode='Markdown'
        )
    else:
        user_states[chat_id] = 'waiting_date_manual'
        bot.send_message(
            chat_id,
            f"📄 Текст документа `{file_name}` получен.\n\n"
            "📅 Не удалось автоматически найти дату в тексте.\n"
            "Введите дату дедлайна в формате ДД.ММ.ГГГГ (например, 25.06.2025).",
            parse_mode='Markdown'
        )


# --- ОБРАБОТЧИК ТЕКСТА И КНОПОК ---
@bot.message_handler(func=lambda m: True)
def handle_message(message):
    chat_id, text = message.chat.id, message.text.strip()
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    if text and not text.startswith('📋') and not text.startswith('🔥') and not text.startswith(
            '✅') and not text.startswith('📦') and not text.startswith('🗑️') and not text.startswith(
            '📝') and not text.startswith('◀️') and not text.startswith('🧹') and not text.startswith('🔢'):
        # Проверяем, есть ли активное состояние
        state = user_states.get(chat_id)
        if state is None:
            # Если нет состояния — предлагаем начать с /start или нажать кнопку
            bot.send_message(
                chat_id,
                "📝 Чтобы ввести текст документа, нажмите кнопку **📝 Ввести текст**\n"
                "или используйте команду /start",
                reply_markup=get_main_keyboard(),
                parse_mode='Markdown'
            )
            return
    # Статические команды меню
    if text == '📋 Все задачи':
        docs = [d for d in load_docs() if d.get('status') != 'выполнено']
        if not docs:
            return bot.send_message(chat_id, "📭 Нет активных документов.", reply_markup=get_main_keyboard())
        docs.sort(
            key=lambda d: ({'Красный': 0, 'Желтый': 1, 'Зеленый': 2}.get(d.get('risk', 'Зеленый'), 2), d['deadline']))
        return bot.send_message(chat_id, "📋 **Все активные задачи:**\n\n" + render_docs_list(docs, today),
                                parse_mode='Markdown')

    elif text == '🔥 Срочные задачи':
        active_docs = [doc for doc in load_docs() if doc.get('status') != 'выполнено']
        urgent = []
        for doc in active_docs:
            _, _, days = get_doc_status(doc, today)
            if days <= 1:
                urgent.append(doc)

        if not urgent:
            return bot.send_message(chat_id, "✅ Нет срочных задач на сегодня.", reply_markup=get_main_keyboard())
        urgent.sort(key=lambda d: d['deadline'])
        return bot.send_message(chat_id,
                                "🔥 **Срочные задачи (дедлайн сегодня/завтра):**\n\n" + render_docs_list(urgent, today),
                                parse_mode='Markdown')

    elif text in ['✅ Отметить выполненным', '/done']:
        active = [d for d in load_docs() if d.get('status') != 'выполнено']
        if not active:
            return bot.send_message(chat_id, "Нет активных документов для отметки.", reply_markup=get_main_keyboard())
        user_states[chat_id] = 'waiting_done_id'
        return bot.send_message(chat_id,
                                "📋 **Выберите номера документов для отметки выполненными:**\n\n" + render_docs_list(
                                    active, today, full_info=False) + "\nВведите номера через пробел или запятую.",
                                parse_mode='Markdown')

    elif text in ['📦 Архив', '/archive']:
        done = [d for d in load_docs() if d.get('status') == 'выполнено']
        if not done:
            return bot.send_message(chat_id, "📭 Архив пуст.", reply_markup=get_main_keyboard())

        msg = "📦 **Выполненные документы:**\n\n"
        for i, d in enumerate(done, 1):
            summary = d.get('summary', get_summary(d.get('text', '')))
            f_name = d.get('file_name', '📝 Текстовое сообщение')
            msg += f"{i}. 📁 `{f_name}`\n   📄 {summary}...\n   ⏳ Дедлайн был: {d['deadline']}\n\n"
        return bot.send_message(chat_id, msg, parse_mode='Markdown')

    elif text == '🗑️ Очистить архив':
        return show_clear_archive_menu(chat_id)

    elif text == '🧹 Удалить все':
        save_docs([d for d in load_docs() if d.get('status') != 'выполнено'])
        return bot.send_message(chat_id, "✅ Архив полностью очищен.", reply_markup=get_main_keyboard())

    elif text == '🔢 Выборочно удалить':
        done = [d for d in load_docs() if d.get('status') == 'выполнено']
        if not done:
            return bot.send_message(chat_id, "📭 Архив пуст. Нечего удалять.", reply_markup=get_main_keyboard())

        user_temp[chat_id] = user_temp.get(chat_id, {})
        user_temp[chat_id]['done_docs'] = done
        user_states[chat_id] = 'waiting_delete_ids'

        msg = "📋 **Выберите документы для удаления из архива:**\n\n"
        for i, d in enumerate(done, 1):
            summary = d.get('summary', get_summary(d.get('text', '')))
            f_name = d.get('file_name', '📝 Текстовое сообщение')
            msg += f"{i}. 📁 `{f_name}` — {summary}... (дедлайн: {d['deadline']})\n"
        return bot.send_message(chat_id, msg + "\nВведите номера документов через пробел или запятую.",
                                parse_mode='Markdown')

    elif text == '◀️ Назад':
        user_states[chat_id] = None
        return bot.send_message(chat_id, "Возврат в главное меню.", reply_markup=get_main_keyboard())

    elif text == '📝 Ввести текст':
        # Сбрасываем предыдущие состояния
        user_states[chat_id] = 'waiting_text'
        user_texts[chat_id] = ''
        if chat_id in user_temp:
            user_temp[chat_id].pop('current_file_name', None)
        bot.send_message(
            chat_id,
            "📝 Вставьте текст документа в чат.",
            reply_markup=get_main_keyboard()
        )
        return

    elif text == '📊 График рисков':
        chart(message)
        return

    elif text == '🔍 Поиск':
        # Отправляем инструкцию
        bot.send_message(
            chat_id,
            "📝 Введите слово для поиска: `/search <слово>`\n\n"
            "Например: `/search штраф`",
            parse_mode='Markdown',
            reply_markup=get_main_keyboard()
        )
        return

    elif text == '📤 Экспорт в Excel':
        export(message)
        return

    # --- ОБРАБОТКА ДИАЛОГОВ ---
    state = user_states.get(chat_id)
    if state is None:
        return bot.send_message(chat_id, "Чтобы начать, используйте /start", reply_markup=get_main_keyboard())

    if state == 'waiting_text':
        user_texts[chat_id] = text
        user_states[chat_id] = 'waiting_document_name'
        bot.send_message(
            chat_id,
            "📄 Текст документа получен.\n\n"
            "📝 Введите **название документа** (например: Письмо от 25.06.2025).\n"
            "Это поможет вам ориентироваться в списке задач.",
            parse_mode='Markdown'
        )
        return

    # --- ПОЛУЧАЕМ НАЗВАНИЕ ДОКУМЕНТА ---
    if state == 'waiting_document_name':
        doc_name = text.strip()
        if not doc_name:
            bot.send_message(chat_id, "❌ Название не может быть пустым. Введите название документа.")
            return

        if chat_id not in user_temp:
            user_temp[chat_id] = {}
        user_temp[chat_id]['current_file_name'] = doc_name

        doc_text = user_texts.get(chat_id, '')
        found_deadline = extract_deadline_from_text(doc_text)

        if found_deadline:
            user_temp[chat_id]['found_deadline'] = found_deadline
            user_states[chat_id] = 'waiting_deadline_confirmation'
            bot.send_message(
                chat_id,
                f"📄 **Документ:** `{doc_name}`\n\n"
                f"🔍 В тексте найдена дата дедлайна: **{found_deadline}**\n\n"
                "✅ Если дата верна, напишите **Да**.\n"
                "✏️ Если хотите указать другую дату, просто введите её в формате ДД.ММ.ГГГГ.",
                parse_mode='Markdown'
            )
        else:
            user_states[chat_id] = 'waiting_date_manual'
            bot.send_message(
                chat_id,
                f"📄 **Документ:** `{doc_name}`\n\n"
                "📅 Не удалось автоматически найти дату в тексте.\n"
                "Введите дату дедлайна в формате ДД.ММ.ГГГГ (например, 25.06.2025).",
                parse_mode='Markdown'
            )
        return

    # --- ПОДТВЕРЖДЕНИЕ НАЙДЕННОЙ ДАТЫ ---
    if state == 'waiting_deadline_confirmation':
        found_deadline = user_temp.get(chat_id, {}).get('found_deadline')

        if not found_deadline:
            bot.send_message(chat_id, "❌ Что-то пошло не так. Начните заново с /start.")
            user_states[chat_id] = None
            return

        # --- ПРОВЕРЯЕМ, ВВЕЛ ЛИ ПОЛЬЗОВАТЕЛЬ ДАТУ ---
        is_new_date = False
        try:
            if '.' in text:
                datetime.strptime(text, '%d.%m.%Y')
                is_new_date = True
            else:
                months = {'января': 1, 'февраля': 2, 'марта': 3, 'апреля': 4, 'мая': 5, 'июня': 6, 'июля': 7,
                          'августа': 8, 'сентября': 9, 'октября': 10, 'ноября': 11, 'декабря': 12}
                parts = text.split()
                if len(parts) == 3 and parts[1].lower() in months:
                    is_new_date = True
        except:
            pass

        if is_new_date:
            # Пользователь ввел новую дату — используем её
            try:
                if '.' in text:
                    date_obj = datetime.strptime(text, '%d.%m.%Y')
                else:
                    months = {'января': 1, 'февраля': 2, 'марта': 3, 'апреля': 4, 'мая': 5, 'июня': 6, 'июля': 7,
                              'августа': 8, 'сентября': 9, 'октября': 10, 'ноября': 11, 'декабря': 12}
                    parts = text.split()
                    date_obj = datetime(int(parts[2]), months[parts[1].lower()], int(parts[0]))
                user_temp[chat_id].pop('found_deadline', None)
                process_document_save(chat_id, date_obj, today)
                return
            except:
                bot.send_message(chat_id, "❌ Не удалось распознать дату. Введите в формате ДД.ММ.ГГГГ.")
                return

        # --- ПРОВЕРЯЕМ, НЕ НАЖАЛ ЛИ ПОЛЬЗОВАТЕЛЬ "НЕТ" ---
        if text.lower() in ['нет', 'no', 'н', 'n']:
            user_states[chat_id] = 'waiting_date_manual'
            bot.send_message(
                chat_id,
                "✏️ Введите дату дедлайна в формате ДД.ММ.ГГГГ (например, 25.06.2025)."
            )
            return

        # --- ВСЕ ОСТАЛЬНОЕ СЧИТАЕМ СОГЛАСИЕМ ---
        # (включая "Да", "дв", "lf", "да", просто нажатие Enter и т.д.)
        date_obj = datetime.strptime(found_deadline, '%Y-%m-%d')
        user_temp[chat_id].pop('found_deadline', None)
        process_document_save(chat_id, date_obj, today)
        return

    # --- РУЧНОЙ ВВОД ДАТЫ ---
    if state == 'waiting_date_manual':
        date_input = text.strip()
        if not date_input:
            bot.send_message(chat_id, "❌ Введите дату в формате ДД.ММ.ГГГГ.")
            return

        try:
            if '.' in date_input:
                date_obj = datetime.strptime(date_input, '%d.%m.%Y')
            else:
                months = {'января': 1, 'февраля': 2, 'марта': 3, 'апреля': 4, 'мая': 5, 'июня': 6, 'июля': 7,
                          'августа': 8, 'сентября': 9, 'октября': 10, 'ноября': 11, 'декабря': 12}
                parts = date_input.split()
                if len(parts) == 3 and parts[1].lower() in months:
                    date_obj = datetime(int(parts[2]), months[parts[1].lower()], int(parts[0]))
                else:
                    raise ValueError("Неверный формат")
        except:
            bot.send_message(chat_id, "❌ Не удалось распознать дату. Введите в формате ДД.ММ.ГГГГ.")
            return

        process_document_save(chat_id, date_obj, today)
        return

    # --- ОТМЕТКА ВЫПОЛНЕННЫМ И УДАЛЕНИЕ ---
    # --- ОТМЕТКА ВЫПОЛНЕННЫМ И УДАЛЕНИЕ ---
    if state in ['waiting_done_id', 'waiting_delete_ids']:
        all_docs = load_docs()
        target_list = [d for d in all_docs if
                       d.get('status') != 'выполнено'] if state == 'waiting_done_id' else user_temp.get(chat_id,
                                                                                                        {}).get(
            'done_docs', [])

        if not target_list:
            bot.send_message(chat_id, "❌ Нет документов для обработки.", reply_markup=get_main_keyboard())
            user_states[chat_id] = None
            return

        ids, valid = parse_input_ids(text, len(target_list))
        if not valid:
            bot.send_message(chat_id,
                             f"❌ Ошибка ввода или номера вне диапазона (1-{len(target_list)}). Введите числа через пробел или запятую.",
                             reply_markup=get_main_keyboard())
            return

        chosen_docs = [target_list[i - 1] for i in ids]
        new_docs = []
        marked_or_deleted = 0

        if state == 'waiting_done_id':
            for d in all_docs:
                if any(d['text'] == t['text'] and d['deadline'] == t['deadline'] for t in chosen_docs):
                    d['status'] = 'выполнено'
                    marked_or_deleted += 1
                new_docs.append(d)

            # Формируем сообщение с именами файлов
            filenames = [d.get('file_name', 'Без имени') for d in chosen_docs]
            if len(ids) == 1:
                msg = f"✅ Документ **{filenames[0]}** отмечен как выполненный."
            else:
                msg = f"✅ Отмечено выполненными {len(ids)} документов: {', '.join(filenames)}."

            save_docs(new_docs)
            user_states[chat_id] = None
            user_temp[chat_id] = {}
            bot.send_message(chat_id, msg, parse_mode='Markdown', reply_markup=get_main_keyboard())
            return

        else:  # waiting_delete_ids
            for d in all_docs:
                if d.get('status') == 'выполнено' and any(
                        d['text'] == t['text'] and d['deadline'] == t['deadline'] for t in chosen_docs):
                    marked_or_deleted += 1
                    continue
                new_docs.append(d)

            # Формируем сообщение с именами файлов
            filenames = [d.get('file_name', 'Без имени') for d in chosen_docs]
            msg = f"✅ Успешно удалено {len(ids)} документов из архива: {', '.join(filenames)}."

            save_docs(new_docs)
            user_states[chat_id] = None
            user_temp[chat_id] = {}
            bot.send_message(chat_id, msg, parse_mode='Markdown', reply_markup=get_main_keyboard())
            return


if __name__ == '__main__':
    # Настраиваем фоновый планировщик
    scheduler = BackgroundScheduler(timezone="Europe/Moscow")  # Укажи свой часовой пояс, если не МСК

    # Добавляем задачу: выполнять функцию send_morning_tasks каждый день в 10:30
    scheduler.add_job(send_morning_tasks, 'cron', hour=10, minute=30)

    # Стартуем планировщик
    scheduler.start()
    print("Фоновый планировщик запущен (уведомления настроены на 10:30)")

    print("Бот запущен")
    bot.infinity_polling()

