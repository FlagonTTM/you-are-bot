# Аудит данных и представления — v2

## Evidence / reviewer / scope

Два независимых read-only Hermes-аудитора проверили только official TRAIN, без обучения, test labels, сети и внешних записей. Координатор независимо пересчитал основные counts, проверил перечисленные ниже реальные примеры и исполнил тесты смысловой экстракции. Полные ответы reviewers сохранены в истории текущего turn, delegation `deleg_c4240cbb`; этот документ не выдаётся за canonical Research OS receipt.

Данные: `data/train.json` SHA-256 `ca3795aa47b8791174a9225a1fe3219684dc388c88922053afdab411bf44ea34`; labels `data/ytrain.csv` SHA-256 `7977ce4773cade43f27459711b2ca943addf284cd6c0ea9b830c71ebc7d5b233`.
Frozen membership v2: `evidence/membership-v2.json` SHA-256 `e553e7bcdfa98b398b0a534a5d9fc095c5a19f60054f87c365cbe705d6b04dee`.
Extractor: `botbench.py`, text adapter/model runner: `experiment.py`; exact digests frozen in each experiment before fitting and rechecked afterwards.

## Field map

| Поле | Смысл | Представление |
|---|---|---|
| text | Непроверенный текст сообщения, включая ошибки, injection, сериализованные записи | own/context TF-IDF или native ordered CatBoost text; UUID values masked; первые 12000 символов на сторону в текстовых моделях |
| message | Порядок внутри диалога, не время | сортировка с обязательной проверкой contiguous 0..n-1; направления ответов и серий |
| participant_index | Локальная сторона 0/1, не глобальный автор | явно заданный адресат признаков; own/opponent/difference |
| dialog_id | Ключ принадлежности, не predictor | объединение частей диалога/точных дубликатов/встроенных явных ссылок в одну split-component |
| is_bot | Только train target | ни один extractor не получает labels; только model.fit |

## Проверки структуры

- 1516 train-диалогов, 3032 участника, 10490 сообщений; ни один участник не отсутствует.
- Пары labels: 10=752, 01=485, 00=279, 11=0. Это наблюдение TRAIN, не правило TEST. Два независимых sigmoid, без принудительной суммы вероятностей.
- `message` корректно упорядочен. 147 диалогов содержат серии одной стороны; parity message%2 не является участником.
- Сначала 1510 уникальных точных содержаний; с объединением явных встроенных ссылок — 1496 компонентов. Raw UUID values не используются в модельном словаре.
- Frozen v1 сохраняется. V2 сохраняет прежние confirmation IDs, добавляет связанные записи и объединяет конфликтующие development folds; изменено 26 rows. Confirmation 616, development 2416.
- JSON order демонстрирует batch confound по данным независимого аудитора; не объявляем его хронологией и не даём порядок объектов модели.
- Нет timestamp-поля и глобального авторского ID, поэтому chronological и author-disjoint generalization не доказаны.
- Empty/whitespace text допустим: target-free extractor возвращает finite values, textual sentinel. Сообщения со служебными следами могут быть сильным, но хрупким сигналом.

## Ручные development-кейсы

| ID | Expected (ручная интерпретация) | Observed / проверка |
|---|---|---|
| 81988e45-b761-4be4-a0d9-ef8c7937c089 | p1 эхо после нескольких вводных шаблонов | labels01; несколько точных последовательных копирований; инструкция — quoted data |
| b3f6efcc-4753-4301-a5cb-5b24068df006 | p1 скриптовый/смешанный; ассистентский текст p0 не равен боту | labels01; p1 `Log error`, p0 `Чем я могу вам помочь?` |
| f7eb685d-a3f9-4b07-9c40-e783c3f81b4d | p1 многословная защитная персона, предполагаемая LLM, движок неизвестен | labels01; длинные обращения о человечности и неформальном стиле |
| 92d712fb-c7a8-4fa5-8a8a-b8c68f9359b1 | два человека с шаблонным small talk | labels00; `Хорошо, а у тебя?` не достаточно для положительной метки |
| 3b819bb3-08d9-4a82-a131-244e10f66846 | короткий случай, высокая неопределённость | labels00; только `Привет`/`Привет, друг `; нельзя требовать ровно одного бота |

Часть других train-примеров независимые аудиторы видели до получения split: они оказались в confirmation. Поэтому confirmation metrics не использовались для настройки, но строгая analyst-blindness не заявляется.

## Baseline error evidence / hypotheses

Предварительный v1 numeric baseline OOF=0.202208. Его ошибки включают уверенное неверное решение для коротких приветствий `3145f2ee-59a0-45b8-910a-c77f8661cb8e_1`, контекстного ответа `378e69d1-29b2-47fc-986d-b32cd7be5149_1`, human prompt-copy `588270d8-9340-4ead-a602-4d457283917c_0`, и двунаправленного приветствия-эха `331016ad-0901-4b7b-963d-a61120da5a68`. Это development-кейсы v1, а не доказательство универсальности признаков. V1 не напрямую сравнивается с v2.

H1 lexical: missing signal — повторяющиеся фразы/стили семейств; cheap_probe — фиксированная TF-IDF+LogReg на v2 OOF; matched_control — numeric-only behavior на тех же folds; transfer_check — отдельный confirmation, только после frozen selection; scope — один recipe, no HPO. Разреженная линейная модель принципиально отличается от дерева по числовым агрегатам.

H2 hybrid: missing signal — нелинейное взаимодействие собственного и чужого текста с направленным поведением; cheap_probe — native CatBoost text+numeric на v2 OOF; matched_control — точно тот же CatBoost config без text; transfer_check — frozen confirmation и batch stress без повторного выбора; scope — один recipe, no sweep. Сохраняет numeric bank и использует ordered supervised text statistics только внутри outer-training fold.

## Допуски / запреты

Official text/sequence: allowed в рамках опубликованной задачи. External datasets/pretrained/API labeling: unclear, used=false. Embedded ID label lookup, hidden targets, готовые ответы старого Kaggle: prohibited by no-cheating, used=false. Metadata/ordering keys в модели: used=false. Public leaderboard: только контекст цели, не feature и не источник подбора параметров. Внешняя отправка и платный compute: separate human gates.

## Неустранённые неопределённости

Настоящий тип/семейство каждого бота неизвестен; ручные семантические категории не labels генератора. Повторные авторы могут пересекаться между split; это нельзя проверить. Серилизация и размер переписки могут не переноситься на новые сборки. Не утверждаем, что grouped OOF гарантирует private score или первое место.
