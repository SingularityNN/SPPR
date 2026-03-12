import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.utilities.lambdify import lambdify


def get_function_from_user():
    print("Введите функцию от одной переменной (используйте x) или от двух переменных (используйте x и y).")
    print("Примеры: sin(x), x**2 + 2*x, x*y + cos(x), x**2 + y**2")
    print("Если оставить строку пустой, будет использован пример: sin(x) для одномерной или x*y для двумерной.")
    func_str = input("f(x, y) = ").strip()

    # Default example
    if not func_str:
        dim = input("Выберите размерность для примера (1 - одномерная, 2 - двумерная): ").strip()
        if dim == '2':
            func_str = "x*y"
        else:
            func_str = "sin(x)"
        print(f"Используется пример: f = {func_str}")

    # Parse
    try:
        expr = sp.sympify(func_str)
    except Exception as e:
        raise ValueError(f"Ошибка при разборе выражения: {e}")

    variables = list(expr.free_symbols)
    var_names = [str(var) for var in variables]

    allowed = {'x', 'y'}
    for name in var_names:
        if name not in allowed:
            raise ValueError(f"Переменная '{name}' недопустима. Используйте только x и/или y.")

    if len(var_names) > 2:
        raise ValueError("Функция имеет больше двух переменных. Поддерживаются только одномерные и двумерные функции.")

    return expr, var_names


def estimate_lipschitz_1d(f_lambda, x_vals):
    f_vals = f_lambda(x_vals)
    df = np.gradient(f_vals, x_vals)
    L = np.max(np.abs(df))
    return L * 1.2


def estimate_lipschitz_2d(f_lambda, x_vals, y_vals):
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    f_vals = f_lambda(X, Y)
    # Градиент по x и y
    df_dx = np.gradient(f_vals, x_vals, axis=0)
    df_dy = np.gradient(f_vals, y_vals, axis=1)
    grad_norm = np.sqrt(df_dx ** 2 + df_dy ** 2)
    L = np.max(grad_norm)
    return L * 1.2


def build_minorant_1d(f_lambda, x_domain, L, num_support=5):
    x_min, x_max = x_domain
    support_points = np.linspace(x_min, x_max, num_support)
    support_values = f_lambda(support_points)

    x_plot = np.linspace(x_min, x_max, 200)
    minorant = np.full_like(x_plot, -np.inf)
    for i, (xs, fs) in enumerate(zip(support_points, support_values)):
        cone = fs - L * np.abs(x_plot - xs)
        minorant = np.maximum(minorant, cone)
    return x_plot, minorant, support_points, support_values


def build_minorant_2d(f_lambda, x_domain, y_domain, L, num_support_x=4, num_support_y=4):
    x_min, x_max = x_domain
    y_min, y_max = y_domain
    support_x = np.linspace(x_min, x_max, num_support_x)
    support_y = np.linspace(y_min, y_max, num_support_y)
    support_X, support_Y = np.meshgrid(support_x, support_y, indexing='ij')
    support_points = np.array([support_X.ravel(), support_Y.ravel()]).T
    support_values = f_lambda(support_X, support_Y).ravel()

    x_plot = np.linspace(x_min, x_max, 50)
    y_plot = np.linspace(y_min, y_max, 50)
    X, Y = np.meshgrid(x_plot, y_plot, indexing='ij')
    minorant = np.full_like(X, -np.inf, dtype=float)

    for (xs, ys), fs in zip(support_points, support_values):
        dist = np.sqrt((X - xs) ** 2 + (Y - ys) ** 2)
        cone = fs - L * dist
        minorant = np.maximum(minorant, cone)

    return X, Y, minorant, support_points, support_values


def plot_1d(f_lambda, x_domain):
    x_min, x_max = x_domain
    x_plot = np.linspace(x_min, x_max, 200)
    f_vals = f_lambda(x_plot)

    L = estimate_lipschitz_1d(f_lambda, x_plot)
    x_minor, minorant, sup_x, sup_f = build_minorant_1d(f_lambda, x_domain, L)

    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, f_vals, 'b-', label='Исходная функция f(x)')
    plt.plot(x_minor, minorant, 'r--', label='Миноранта (липшицева)')
    plt.scatter(sup_x, sup_f, color='red', marker='o', label='Опорные точки')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Одномерная функция и её миноранта (L = {L:.3f})')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_2d(f_lambda, x_domain, y_domain):
    x_min, x_max = x_domain
    y_min, y_max = y_domain
    x_plot = np.linspace(x_min, x_max, 50)
    y_plot = np.linspace(y_min, y_max, 50)
    X, Y = np.meshgrid(x_plot, y_plot, indexing='ij')
    f_vals = f_lambda(X, Y)

    L = estimate_lipschitz_2d(f_lambda, x_plot, y_plot)
    Xm, Ym, minorant, sup_points, sup_vals = build_minorant_2d(f_lambda, x_domain, y_domain, L)

    fig = plt.figure(figsize=(14, 6))

    # График исходной функции
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X, Y, f_vals, cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)
    ax1.scatter(sup_points[:, 0], sup_points[:, 1], sup_vals, color='red', s=50, label='Опорные точки')
    ax1.set_title('Исходная функция f(x,y)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

    # График миноранты
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(Xm, Ym, minorant, cmap='plasma', alpha=0.8, linewidth=0, antialiased=True)
    ax2.scatter(sup_points[:, 0], sup_points[:, 1], sup_vals, color='red', s=50, label='Опорные точки')
    ax2.set_title(f'Миноранта (L = {L:.3f})')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('m(x,y)')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

    plt.tight_layout()
    plt.show()


def main():
    try:
        expr, var_names = get_function_from_user()
        print(f"Распознано выражение: {expr}")
        print(f"Переменные: {var_names}")

        dim = len(var_names)
        if dim == 0:
            # Нет переменных — константа. Будем считать одномерной с переменной x
            var_names = ['x']
            dim = 1
            print("Обнаружена константа, интерпретируем как функцию от x.")

        if dim == 1:
            var = sp.Symbol(var_names[0])
            f_lambda = lambdify(var, expr, modules='numpy')
            # Область по умолчанию
            x_domain = (-10, 10)
            plot_1d(f_lambda, x_domain)
        elif dim == 2:
            vars_sym = [sp.Symbol(name) for name in var_names]
            f_lambda = lambdify(vars_sym, expr, modules='numpy')
            # Область по умолчанию
            x_domain = (-5, 5)
            y_domain = (-5, 5)
            plot_2d(f_lambda, x_domain, y_domain)
        else:
            print("Ошибка: функция имеет более двух переменных.")

    except Exception as e:
        print(f"Произошла ошибка: {e}")


if __name__ == "__main__":
    main()