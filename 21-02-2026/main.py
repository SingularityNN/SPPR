import itertools
import warnings
from typing import Callable, List, Union, Tuple, Optional

def grid_search(
    f: Callable[[List[float]], float],
    lb: List[float],
    ub: List[float],
    num_points: Union[int, List[int]],
    minimize: bool = True
) -> Tuple[Optional[List[float]], float]:
    """
    Перебор по равномерной сетке для многомерной оптимизации.

    Параметры:
        f : целевая функция, принимает список координат и возвращает число.
        lb : список нижних границ для каждой переменной.
        ub : список верхних границ для каждой переменной.
        num_points : число точек разбиения по каждому измерению.
                     Если задано целое число, оно используется для всех измерений.
        minimize : если True (по умолчанию), ищется минимум; иначе — максимум.

    Возвращает:
        (лучшая точка, лучшее значение). Если сетка пуста (например, размерность 0),
        возвращается (None, inf) или (None, -inf) в зависимости от режима.
    """
    N = len(lb)
    if N == 0:
        warnings.warn("Размерность задачи равна 0. Возвращаем None и бесконечность.")
        return None, (float('inf') if minimize else -float('inf'))

    # Приводим num_points к списку нужной длины
    if isinstance(num_points, int):
        num_points = [num_points] * N
    else:
        if len(num_points) != N:
            raise ValueError("Длина списка num_points должна совпадать с размерностью N.")

    # Проверка корректности границ и числа точек
    for i in range(N):
        if lb[i] > ub[i]:
            raise ValueError(f"Нижняя граница lb[{i}] = {lb[i]} больше верхней ub[{i}] = {ub[i]}.")
        if num_points[i] < 1:
            raise ValueError(f"Число точек разбиения num_points[{i}] должно быть >= 1.")

    # Предупреждение о вычислительной сложности
    total_points = 1
    for n in num_points:
        total_points *= n
    if total_points > 10_000_000:
        warnings.warn(f"Общее число точек сетки очень велико: {total_points}. "
                      f"Вычисление может занять много времени.")

    best_x = None
    best_val = float('inf') if minimize else -float('inf')

    # Генерация всех комбинаций индексов
    index_ranges = [range(n) for n in num_points]
    for indices in itertools.product(*index_ranges):
        # Вычисляем координаты точки
        x = []
        for i, idx in enumerate(indices):
            if num_points[i] == 1:
                coord = lb[i]  # единственная точка — нижняя граница
            else:
                step = (ub[i] - lb[i]) / (num_points[i] - 1)
                coord = lb[i] + idx * step
            x.append(coord)

        val = f(x)

        if minimize:
            if val < best_val:
                best_val = val
                best_x = x
        else:
            if val > best_val:
                best_val = val
                best_x = x

    return best_x, best_val


def main():
    """
    Демонстрация работы алгоритма на тестовой функции,
    сгенерированной библиотекой GKLS (синтаксис: from gkls import GKLS).
    """
    try:
        from gkls import GKLS
    except ImportError:
        print("Библиотека gkls не установлена. Демонстрация будет выполнена на функции Розенброка.")
        # Запасной вариант — функция Розенброка (двумерная)
        def test_func(x):
            return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
        dim = 2
        lb = [-2.0, -1.0]
        ub = [2.0, 3.0]
        true_min = (1.0, 1.0)   # известный минимум
        true_val = 0.0
        global_min_info = True
    else:
        # GKLS params
        dim = 2                     # размерность
        num_minima = 2               # число локальных минимумов (класс функции)
        domain = [-5.0, 5.0]         # единые границы для всех переменных
        global_min_value = -1.0       # желаемое значение глобального минимума
        # seed для воспроизводимости (gen=<int>)
        gkls = GKLS(
            dim=dim,
            num_minima=num_minima,
            domain=domain,
            global_min=global_min_value,
            gen=43                     # фиксированный seed для детерминированности
        )

        # Целевая функция — метод get_d_f (или get_nd_f — они эквивалентны)
        test_func = lambda x: gkls.get_d_f(x)

        # Границы едины для всех измерений
        lb = [domain[0]] * dim
        ub = [domain[1]] * dim

        # Предполагается, что у объекта есть атрибуты global_min_point и global_min_value
        if hasattr(gkls, 'global_min_point') and hasattr(gkls, 'global_min_value'):
            true_min = gkls.global_min_point
            true_val = gkls.global_min_value
            global_min_info = True
        else:
            # Если атрибуты отсутствуют, попробуем другие возможные имена
            if hasattr(gkls, 'global_min'):
                true_min = gkls.global_min
                true_val = global_min_value  # мы его задавали, но не факт, что совпадает
                global_min_info = True
            else:
                print("Не удалось получить информацию об истинном глобальном минимуме.")
                true_min = None
                true_val = None
                global_min_info = False

    # Настройки сетки
    points_per_dim = 8   # всего 8^dim точек (dim=3 -> 512 точек)
    print(f"Размерность: {dim}")
    print(f"Границы: lb={lb}, ub={ub}")
    print(f"Число точек на измерение: {points_per_dim} -> всего {points_per_dim**dim} точек")

    # Запуск перебора
    best_x, best_val = grid_search(test_func, lb, ub, points_per_dim, minimize=True)

    print("\nРезультаты перебора по равномерной сетке:")
    print(f"Лучшая найденная точка: {best_x}")
    print(f"Лучшее значение: {best_val}")

    if global_min_info and true_min is not None and true_val is not None:
        print(f"Истинный глобальный минимум: {true_min} со значением {true_val}")
        print(f"Абсолютная ошибка по значению: {abs(best_val - true_val):.2e}")
        # Оценим евклидово расстояние до истинного минимума
        distance = sum((bx - tx)**2 for bx, tx in zip(best_x, true_min))**0.5
        print(f"Расстояние до истинного минимума: {distance:.2e}")
    else:
        print("Истинный глобальный минимум неизвестен (библиотека GKLS не предоставила информацию).")


if __name__ == "__main__":
    main()