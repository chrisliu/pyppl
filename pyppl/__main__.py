import pyppl
import pyppl.lang

if __name__ == '__main__':
    @pyppl.compile
    def test_flip():
        f = pyppl.Flip()
        pyppl.lang.observe(f)
        if f and pyppl.Flip():
            a = True
            b = False
        else:
            a = False
            b = True
        return a, b

    for _ in range(10):
        sample = test_flip()
        if sample is pyppl.lang.NotObservable:
            print("Not observed")
        else:
            print(sample)
