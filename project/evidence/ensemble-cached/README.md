# Полный cached-запуск из исходных данных

`ensemble-cached-qa-v1` выполнен из отдельной standalone-копии проекта, а не из исходного research workspace.

- Заново обработаны пять официальных файлов.
- Реально обучены пять CV-моделей и полный CatBoost refit; все параметры и defaults унаследованы в родительский SQL/JSON tracker.
- Новые native priors совпали с историческими точно.
- Заново построены десять blind LLM payloads; input/prompt SHA каждого совпал с исходным batch manifest до использования сохранённых ответов.
- Итоговый `submission.csv` побайтово совпал с исторической отправкой #10588: `4bde02a00f00d009ad69c3aa967c00e468a41c0bcb2b90f65b93416abb3d4639`.
- SQL и JSON прочитаны независимо: `FINISHED`, 28 групп параметров, 12 метрик, 114 артефактов. Все 114 исходных payload-артефактов проверены по SHA-256 и размеру в полном локальном run directory.

Здесь хранится **curated subset**, а не полный run directory. Не включены native checkpoints, snapshot-копии исходников и сырые LLM inputs/prompts/events. Поэтому пути в исходном artifact index могут отсутствовать в этой выборке. Полный набор создаёт повторный запуск `project/run_ensemble.sh`.

Этот режим не делал новых LLM-вызовов и contest uploads. Public score — историческая оценка #10588, а не локально пересчитанная метрика. Свежий режим имеет отдельный run ID и evidence.

```bash
sqlite3 project/evidence/ensemble-cached/tracking.sqlite3 \
  "SELECT run_id,status,started_utc,ended_utc FROM runs;"
sqlite3 project/evidence/ensemble-cached/tracking.sqlite3 \
  "SELECT name,value FROM metrics ORDER BY name;"
```
