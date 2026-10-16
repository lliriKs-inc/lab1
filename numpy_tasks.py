"""Заготовки задач на NumPy."""

import numpy as np

from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput,
    MatrixStatistics, MatrixVectorBatchInput, OneHotInput,
    RandomMatrixInput, RectangleInput, TimeSeriesInput,
    TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices = np.asarray(data.matrices)
    vectors = np.asarray(data.vectors)
    return np.matmul(matrices, vectors).sum(axis=0)


def binarize(data: BinarizeInput) -> np.ndarray:
    return (np.asarray(data.matrix) > data.threshold).astype(int)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [np.unique(row).tolist() for row in matrix]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    matrix = np.asarray(data.matrix)
    return [np.unique(column).tolist() for column in matrix.T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rng = np.random.default_rng(data.seed)
    matrix = rng.normal(
        loc=data.mean,
        scale=data.std,
        size=(data.rows, data.columns),
    )

    return MatrixStatistics(
        matrix=matrix,
        row_means=matrix.mean(axis=1),
        column_means=matrix.mean(axis=0),
        row_variances=matrix.var(axis=1),
        column_variances=matrix.var(axis=0),
    )


def plot_histograms(data: MatrixStatistics) -> None:
    import matplotlib.pyplot as plt

    matrix = np.asarray(data.matrix)

    plt.figure()
    plt.hist(matrix.T, bins=10)
    plt.title("Distributions by row")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.figure()
    plt.hist(matrix, bins=10)
    plt.title("Distributions by column")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.show()


def chess(data: ChessInput) -> np.ndarray:
    rows, columns = np.indices((data.rows, data.columns))
    return np.where(
        (rows + columns) % 2 == 0,
        data.first,
        data.second,
    )


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    if data.width < 0 or data.height < 0:
        raise ValueError("Размеры прямоугольника не могут быть отрицательными")

    image = np.full(
        (data.image_height, data.image_width, 3),
        data.background_color,
        dtype=np.uint8,
    )

    left = (data.image_width - data.width) // 2
    top = (data.image_height - data.height) // 2

    image[
        max(0, top):min(data.image_height, top + data.height),
        max(0, left):min(data.image_width, left + data.width),
    ] = data.shape_color

    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    if data.semi_axis_x <= 0 or data.semi_axis_y <= 0:
        raise ValueError("Полуоси эллипса должны быть положительными")

    image = np.full(
        (data.image_height, data.image_width, 3),
        data.background_color,
        dtype=np.uint8,
    )

    y, x = np.ogrid[:data.image_height, :data.image_width]
    center_x = data.image_width // 2
    center_y = data.image_height // 2
    a, b = int(data.semi_axis_x), int(data.semi_axis_y)

    dx = x - center_x
    dy = y - center_y

    # При больших числах используем целые без ограничения разрядности.
    bound = center_x ** 2 * b ** 2 + center_y ** 2 * a ** 2
    if max(bound, a ** 2 * b ** 2) > np.iinfo(np.int64).max:
        dx = dx.astype(object)
        dy = dy.astype(object)

    # Целочисленная формула сохраняет все точки на границе.
    mask = dx ** 2 * b ** 2 + dy ** 2 * a ** 2 <= a ** 2 * b ** 2
    image[mask] = data.shape_color

    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values = np.asarray(data.values)
    window = data.window

    if values.ndim != 1 or values.size == 0:
        raise ValueError("Временной ряд должен быть непустым и одномерным")
    if not 1 <= window <= values.size:
        raise ValueError("Размер окна должен быть от 1 до длины ряда")

    # Сравниваем исходные значения, сохраняя точность целых чисел.
    middle = values[1:-1]
    maxima = (middle > values[:-2]) & (middle > values[2:])
    minima = (middle < values[:-2]) & (middle < values[2:])

    integer_series = np.issubdtype(values.dtype, np.integer) or (
        values.dtype == object
        and all(isinstance(v, (int, np.integer)) for v in values)
    )

    if integer_series:
        # Python-целые в object-массиве не переполняются.
        exact = np.array([int(v) for v in values], dtype=object)
        count = values.size
        total = exact.sum()

        mean = float(total / count)
        variance = float(
            (count * (exact * exact).sum() - total * total)
            / (count * count)
        )

        prefix = np.concatenate((
            np.array([0], dtype=object),
            exact.cumsum(),
        ))
        moving_average = (
            (prefix[window:] - prefix[:-window]) / window
        ).astype(float)
    else:
        values_float = values.astype(float)
        mean = float(values_float.mean())
        variance = float(values_float.var())
        moving_average = np.convolve(
            values_float, np.ones(window) / window, mode="valid"
        )

    # Среднее единственного элемента равно самому элементу.
    if window == 1:
        moving_average = values.copy()

    return TimeSeriesStatistics(
        mean=mean,
        variance=variance,
        std=float(np.sqrt(variance)),
        local_maxima_indices=np.flatnonzero(maxima) + 1,
        local_minima_indices=np.flatnonzero(minima) + 1,
        moving_average=moving_average,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels = np.asarray(data.labels)

    if labels.ndim != 1:
        raise ValueError("Метки должны быть одномерным вектором")
    if labels.size and not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Метки должны быть целыми числами")
    if np.any(labels < 0):
        raise ValueError("Метки не могут быть отрицательными")

    labels = labels.astype(np.intp)
    class_count = data.class_count

    if class_count is None:
        class_count = int(labels.max()) + 1 if labels.size else 0
    if class_count < 0 or np.any(labels >= class_count):
        raise ValueError("Количество классов не соответствует меткам")

    result = np.zeros((labels.size, class_count), dtype=int)
    result[np.arange(labels.size), labels] = 1
    return result


def _run_tests() -> None:
    equal = np.testing.assert_array_equal
    close = np.testing.assert_allclose

    for p, n in [(1, 1), (2, 3), (0, 2)]:
        result = sum_prod(MatrixVectorBatchInput(
            np.ones((p, n, n)), np.ones((p, n, 1)),
        ))
        equal(result, np.full((n, 1), p * n))

    for threshold, expected in [
        (0.5, [[0, 0, 1]]),
        (-1, [[1, 1, 1]]),
        (1, [[0, 0, 0]]),
    ]:
        matrix = np.array([[0, 0.5, 1]])
        equal(binarize(BinarizeInput(matrix, threshold)), expected)
        equal(matrix, [[0, 0.5, 1]])

    for matrix in [
        [[3, 1, 3], [2, 2, 1]],
        [[7]],
        [[1, 1], [1, 1]],
    ]:
        array = np.asarray(matrix)
        assert unique_rows(MatrixInput(array)) == [
            sorted(set(row)) for row in matrix
        ]
        assert unique_columns(MatrixInput(array)) == [
            sorted(set(column)) for column in array.T
        ]

    for rows, columns in [(1, 1), (2, 3), (3, 2)]:
        result = matrix_statistics(
            RandomMatrixInput(rows, columns, mean=5, std=0)
        )
        equal(result.matrix, np.full((rows, columns), 5))
        equal(result.row_means, np.full(rows, 5))
        equal(result.column_means, np.full(columns, 5))
        equal(result.row_variances, np.zeros(rows))
        equal(result.column_variances, np.zeros(columns))

    equal(chess(ChessInput(3, 4, 0, 1)), [
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ])
    equal(chess(ChessInput(1, 1, 5, 9)), [[5]])
    equal(chess(ChessInput(1, 3, 2, 4)), [[2, 4, 2]])

    red, black = (255, 0, 0), (0, 0, 0)

    for width, height, pixels in [
        (1, 1, 1), (3, 2, 6), (0, 2, 0),
    ]:
        image = draw_rectangle(RectangleInput(
            width, height, 5, 5, red, black,
        ))
        assert image.shape == (5, 5, 3)
        assert np.count_nonzero(image[:, :, 0]) == pixels

    for a, b, pixels in [
        (1, 1, 5), (2, 1, 7), (10, 10, 25),
    ]:
        image = draw_ellipse(EllipseInput(a, b, 5, 5, red, black))
        assert image.shape == (5, 5, 3)
        assert np.count_nonzero(image[:, :, 0]) == pixels

    # Точка (5, 12) лежит на окружности радиуса 13.
    image = draw_ellipse(EllipseInput(13, 13, 27, 27, red, black))
    equal(image[1, 8], red)

    for values, window, maxima, minima, average in [
        ([1, 3, 1, 0, 2], 2, [1], [3], [2, 2, 0.5, 1]),
        ([5], 1, [], [], [5]),
        ([2, 2, 2], 3, [], [], [2]),
    ]:
        result = analyze_time_series(TimeSeriesInput(values, window))
        equal(result.local_maxima_indices, maxima)
        equal(result.local_minima_indices, minima)
        close(result.moving_average, average)

    big = 2 ** 53
    result = analyze_time_series(
        TimeSeriesInput([big, big + 1, big], 1)
    )
    equal(result.local_maxima_indices, [1])
    close(result.variance, 2 / 9)

    equal(one_hot(OneHotInput([0, 2, 3, 0])), [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ])
    equal(one_hot(OneHotInput([1], 3)), [[0, 1, 0]])
    assert one_hot(OneHotInput([], 3)).shape == (0, 3)

    print("NumPy Tasks: все тесты пройдены!")


if __name__ == "__main__":
    _run_tests()
    plot_histograms(matrix_statistics(RandomMatrixInput(4, 5)))