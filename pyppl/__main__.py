import pyppl

if __name__ == '__main__':
    @pyppl.compile(return_types=pyppl.Flip)
    def test_flip():
        f = pyppl.Flip()
        pyppl.observe(f)
        if f and pyppl.Flip():
            a = True
        else:
            a = False
        return a

    with pyppl.RejectionSampling():
        success = test_flip()
    print(success)
