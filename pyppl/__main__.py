import pyppl

if __name__ == '__main__':
    @pyppl.compile(return_types=pyppl.Flip)
    def test_flip(prob1, prob2):
        f = pyppl.Flip(prob1)
        b = True
        if f:
            if pyppl.Flip():
                b = True
            else:
                b = False
            a = False
        else:
            a = True
        a = b

        # pyppl.observe(bool(f))
        # b = True
        # if f and not pyppl.Flip(prob2):
        #     if pyppl.Flip():
        #         b = False
        #         c = True
        #     else:
        #         b = True
        #         c = False
        #     a = c
        # else:
        #     a = False
        # a = b
        return a

    with pyppl.RejectionSampling():
        success = test_flip(0.5, 0.5)

    print(success)

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
