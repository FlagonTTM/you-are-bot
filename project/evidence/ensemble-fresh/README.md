# Реальный fresh-запуск ансамбля

`ensemble-fresh-qa-v1`: исходные официальные данные → обработка → пять CV fits и один full CatBoost refit → TRAIN-only retrieval → **десять новых LLM inference-вызовов** для всех 379 TEST-диалогов → фиксированная смесь 50/50 → 758 строк CSV.

Независимым чтением SQL/JSON и файлов подтверждено:

- `FINISHED`; 28 групп параметров, 12 CatBoost-метрик, 163 индексированных артефакта.
- Все исходные 163 payload-артефакта проверены по SHA-256 и размерам в полном локальном каталоге.
- Во всех десяти batches совпали pre/post boundary locks; реальные event types содержали только разрешённые inference items, без tool execution.
- Входные данные и весь runtime/config/cache source closure остались неизменными и совпали с репозиторием.
- ID/order/schema соответствуют sample; все вероятности конечны и находятся в диапазоне. Формула смеси проверена отдельно по native и новым LLM CSV.

SHA-256 нового CSV:

`24818fbd65a92c78fda05ca646b55c3db4d09125249e765cf213a93a7413fdfe`

Он **отличается от исторического CSV**. Это ожидаемое свойство свежей внешней LLM, а не ошибка exact-replay режима. На соревнование он не отправлялся: **новый public score неизвестен**. OOF-метрика относится только к CatBoost, не к полному ансамблю.

Здесь curated subset результатов. Не включены сырые inputs/prompts/output rationales/events, native checkpoints и snapshot-копии. Некоторые пути в исходном artifact index поэтому отсутствуют в Git; полный каталог создаётся при запуске pipeline.

```bash
sqlite3 project/evidence/ensemble-fresh/tracking.sqlite3 \
  "SELECT run_id,status,started_utc,ended_utc FROM runs;"
```
