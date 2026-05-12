def fib(n: int) -> int:
    """Return the n-th Fibonacci number, 0-indexed: fib(0)=0, fib(1)=1, fib(2)=1, ..."""
    ...


def test_fib_zero():
    assert fib(0) == 0


def test_fib_one():
    assert fib(1) == 1


def test_fib_small():
    assert fib(2) == 1
    assert fib(3) == 2
    assert fib(7) == 13


def test_fib_large():
    # Tempts a naive recursive solution into a stack/timeout problem.
    assert fib(30) == 832040
