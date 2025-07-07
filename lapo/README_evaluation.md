# Оценка агентов LAPO

Этот документ описывает, как оценивать обученных агентов в среде без нормализации наград.

## Изменения

Убраны все компоненты, связанные с нормализацией наград:
- `NormalizeReward` wrapper
- `TransformReward` wrapper  
- `normalize_return` функция
- `episodic_return_norm` в логировании

Теперь агенты оцениваются на чистых наградах среды.

## Использование

### Быстрая оценка (10 эпизодов)

```bash
python quick_eval.py <exp_name> <env_name>
```

Пример:
```bash
python quick_eval.py my_experiment coinrun
```

### Полная оценка (настраиваемое количество эпизодов)

```bash
python evaluate.py <exp_name> <env_name> [num_episodes] [device]
```

Примеры:
```bash
# 100 эпизодов на cuda:0
python evaluate.py my_experiment coinrun 100 cuda:0

# 50 эпизодов на cpu
python evaluate.py my_experiment coinrun 50 cpu
```

## Результаты

Скрипт выводит:
- Возврат для каждого эпизода
- Средний возврат ± стандартное отклонение
- Минимальный и максимальный возврат

## Поддерживаемые среды

Все среды ProcGen:
- bigfish, bossfight, caveflyer, chaser, climber
- coinrun, dodgeball, fruitbot, heist, jumper
- leaper, maze, miner, ninja, plunder, starpilot

## Структура файлов

- `evaluate.py` - основной скрипт оценки
- `quick_eval.py` - быстрая оценка для тестирования
- `env_utils.py` - обновленная функция создания среды без нормализации
- `ppo.py` - обновленное логирование без нормализованных наград 