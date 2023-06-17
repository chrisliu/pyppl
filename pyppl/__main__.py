from collections.abc import Callable
import pyppl


def print_probability(func: Callable, *args, **kwargs) -> None:
    print("Running...")
    try:
        with pyppl.RejectionSampling():
            success = func(*args, **kwargs)
        print(success)
    except Exception as e:
        print("Approximate inference error")
        print(e)

    try:
        with pyppl.ExactInference():
            success = func(*args, **kwargs)
        print(success)
    except Exception as e:
        print("Exact inference error")
        print(e)


if __name__ == '__main__':
    @pyppl.compile(return_types=pyppl.Flip)
    def simple_flip():
        if pyppl.Flip():
            a = True
        else:
            a = False
        return a
    print_probability(simple_flip)

    @pyppl.compile(return_types=pyppl.Flip)
    def less_simple_flip(prob):
        if pyppl.Flip(prob):
            a = True
        else:
            a = False
        return a
    print_probability(less_simple_flip, 0.35)

    @pyppl.compile(return_types=pyppl.Flip)
    def nasty_control(prob):
        f = pyppl.Flip(prob)
        if f:
            c = True
            return c
        else:
            a = False
        return a
    print_probability(nasty_control, 0.2)

    @pyppl.compile(return_types=pyppl.Flip)
    def confusing_control(prob1, prob2):
        f = pyppl.Flip(prob1)
        b = True
        if not f:
            if pyppl.Flip(prob2):
                b = True
            else:
                b = False
            a = False
        else:
            a = True
        a = b
        return a
    print_probability(confusing_control, 0.5, 0.5)

    @pyppl.compile(return_types=pyppl.Flip)
    def up_to_n_heads_in_a_row(n):
        heads = True
        for _ in range(int(pyppl.Integer(pyppl.UniformDistribution(0, n)))):
            heads &= pyppl.Flip()
        return heads
    print_probability(up_to_n_heads_in_a_row, 5)

    # with pyppl.RejectionSampling():
    #     success = up_to_n_heads_in_a_row(3)

    # @pyppl.compile(return_types=pyppl.Flip)
    # def test_flip():
    #     f = pyppl.Flip()
    #     pyppl.observe(f)
    #     if f and pyppl.Flip():
    #         a = True
    #     else:
    #         a = False
    #     return a

    # with pyppl.MCMC():
    #     success = test_flip()

    # print(success)

    # @pyppl.compile(return_types=pyppl.Flip)
    # def up_to_n_heads_in_a_row(n):
    #     heads = True
    #     for _ in range(int(pyppl.Integer(pyppl.UniformDistribution(0, n)))):
    #         heads &= pyppl.Flip()
    #     return heads

    # with pyppl.RejectionSampling():
    #     success = up_to_n_heads_in_a_row(3)

    # print(success)

    # @pyppl.compile(return_types=pyppl.Integer)
    # def roll_n_dice(n):
    #     sum = 0
    #     for _ in range(n):
    #         roll = pyppl.Integer(pyppl.UniformDistribution(1, 6))
    #         pyppl.observe(roll >= 3)
    #         sum += roll
    #     return sum

    # with pyppl.RejectionSampling():
    #     distribution = roll_n_dice(2)

    # print(distribution)

    # # For demonstration purposes BELOW
    # @pyppl.compile(return_types=pyppl.Flip)
    # def test_flip():
    #     f = pyppl.Flip()
    #     pyppl.observe(f)
    #     if f and pyppl.Flip():
    #         a = True
    #     else:
    #         a = False

    #     with pyppl.ExactInference():
    #         b = pyppl.Flip(1e-5)

    #     return a & b

    # with pyppl.MCMC():
    #     success = test_flip()

    # print(success)
