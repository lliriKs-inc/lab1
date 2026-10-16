"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import (
    PositiveIntegerInput,
    TextInput,
    VectorPairInput,
)


def count_vowels(data: TextInput) -> int:
    return sum(char in "aeiou" for char in data.value.lower())


def has_unique_characters(data: TextInput) -> bool:
    return len(set(data.value)) == len(data.value)


def count_one_bits(data: PositiveIntegerInput) -> int:
    return data.value.bit_count()


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0

    while number >= 10:
        product = 1
        for digit in str(number):
            product *= int(digit)
        number = product
        steps += 1

    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected

    if len(predicted) != len(expected) or len(predicted) == 0:
        raise ValueError("Векторы должны быть непустыми и одинаковой длины")

    return sum(
        (x - y) ** 2 for x, y in zip(predicted, expected)) / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    factors = []
    divisor = 2

    while divisor * divisor <= number:
        power = 0

        while number % divisor == 0:
            number //= divisor
            power += 1

        if power == 1:
            factors.append(f"({divisor})")
        elif power > 1:
            factors.append(f"({divisor}**{power})")

        divisor += 1

    if number > 1:
        factors.append(f"({number})")

    return "".join(factors)


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    left, right = 1, cube_count

    while left <= right:
        middle = (left + right) // 2
        total = middle * (middle + 1) * (2 * middle + 1) // 6

        if total == cube_count:
            return middle
        if total < cube_count:
            left = middle + 1
        else:
            right = middle - 1

    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    digits = str(data.value)
    side_length = (len(digits) - 1) // 2

    left = digits[:side_length]
    right = digits[len(digits) - side_length:]

    return sum(map(int, left)) == sum(map(int, right))


def _run_tests() -> None:
    tests = [
        (count_vowels, TextInput(""), 0),
        (count_vowels, TextInput("AEiou"), 5),
        (count_vowels, TextInput("rhythm"), 0),
        (count_vowels, TextInput("Hello"), 2),
        (has_unique_characters, TextInput(""), True),
        (has_unique_characters, TextInput("abc"), True),
        (has_unique_characters, TextInput("aba"), False),
        (has_unique_characters, TextInput("Aa"), True),
        (count_one_bits, PositiveIntegerInput(1), 1),
        (count_one_bits, PositiveIntegerInput(8), 1),
        (count_one_bits, PositiveIntegerInput(15), 4),
        (count_one_bits, PositiveIntegerInput(1234), 5),
        (multiplicative_persistence, PositiveIntegerInput(39), 3),
        (multiplicative_persistence, PositiveIntegerInput(4), 0),
        (multiplicative_persistence, PositiveIntegerInput(999), 4),
        (multiplicative_persistence, PositiveIntegerInput(10), 1),
        (mse, VectorPairInput([1, 2], [1, 2]), 0.0),
        (mse, VectorPairInput([1, 2], [2, 4]), 2.5),
        (mse, VectorPairInput([-3], [1]), 16.0),
        (mse, VectorPairInput([0], [0]), 0.0),
        (prime_factorization, PositiveIntegerInput(86240),
         "(2**5)(5)(7**2)(11)"),
        (prime_factorization, PositiveIntegerInput(13), "(13)"),
        (prime_factorization, PositiveIntegerInput(16), "(2**4)"),
        (prime_factorization, PositiveIntegerInput(1), ""),
        (pyramid, PositiveIntegerInput(1), 1),
        (pyramid, PositiveIntegerInput(5), 2),
        (pyramid, PositiveIntegerInput(14), 3),
        (pyramid, PositiveIntegerInput(2), "It is impossible"),
        (is_balanced_number, PositiveIntegerInput(1234006), True),
        (is_balanced_number, PositiveIntegerInput(123456), False),
        (is_balanced_number, PositiveIntegerInput(7), True),
        (is_balanced_number, PositiveIntegerInput(12), True),
    ]

    for function, data, expected in tests:
        result = function(data)
        assert result == expected, (
            function.__name__, data, result, expected
        )

    print("Все тесты пройдены!")


if __name__ == "__main__":
    _run_tests()