class Studennt(object):

    @property
    def score(self):
        return self.score #实例的变量名不能与属性的方法名相同，否则会造成无限递归

    @score.setter
    def score(self, value):
        if not isinstance(value, int):
            raise ValueError('score must be an integer!')

        if value < 0 or value > 100:
            raise ValueError('score must between 0~100!')
        self.score = value

s = Studennt()

s.score = 60
s.score

#s.score = 999