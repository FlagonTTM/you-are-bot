# Проверенный запуск native-reproduction-01

Это **выборка результатов реального end-to-end запуска**, выполненного из отдельной standalone-копии `project/`, а не полный каталог запуска.

- Статус прочитан из SQLite и JSON: `FINISHED`.
- OOF LogLoss пересчитан независимо по CSV и официальным TRAIN labels: **0.16919850572003053** на **2416** development rows.
- Финальный CSV содержит **758** TEST rows с точными ID, порядком и схемой официального sample.
- Все шесть собственных native CatBoost checkpoints повторили вероятности точно, `numpy.array_equal`, max absolute difference **0.0**.
- У всех пяти CV-fits нулевое пересечение компонентов с validation; confirmation rows в CV-fit отсутствуют.
- 15 SQL-записей — **группы параметров**, включая полные effective parameter dictionaries шести моделей; это не число отдельных scalar hyperparameters. Также сохранены 12 метрик и исходный индекс 30 артефактов.
- Все исходные 30 артефактов были проверены по SHA-256 и размерам в полном локальном каталоге запуска. Consumed source/config/lock bytes совпали с репозиторием и snapshot, данные не изменились.

В Git намеренно **не включены** `models/*.cbm`, копии `snapshot/` и исходный `index.html` с ссылками на полный локальный run directory. Поэтому некоторые пути в сохранённой таблице `artifacts` здесь отсутствуют. Это исходный индекс проверенного запуска, а не утверждение о наличии всех payload-файлов в этой выборке. Повторный `project/run.sh` создаёт полный набор с моделями и working tracking index.

Машинное доказательство и SHA-256 всех экспортированных файлов: [`../reproduction-proof.json`](../reproduction-proof.json).

Просмотр сохранённого трекера:

```bash
sqlite3 project/evidence/reproduction/tracking.sqlite3 \
  "SELECT run_id,status,started_utc,ended_utc FROM runs;"
sqlite3 project/evidence/reproduction/tracking.sqlite3 \
  "SELECT name,value FROM metrics ORDER BY name;"
sqlite3 project/evidence/reproduction/tracking.sqlite3 \
  "SELECT value_json FROM params WHERE name='model/full';"
```

Этот запуск не обращался к LLM и не отправлял CSV на платформу. Официальный historical public LogLoss 0.1303426569 относится к другому ансамблю.
