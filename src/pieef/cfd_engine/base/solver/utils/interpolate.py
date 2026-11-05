import numpy as np

'''
linear interpolation
'''
def linearInterpolate(pos: [float], data) -> np.ndarray:
  value = pos[0]
  [index, frac] = [int(np.floor(value)), value % 1]
  p = lambda x: np.asarray(data(x)) if len(pos) < 2 \
    else linearInterpolate(pos[1:], lambda *values: data(x, *values))
  return p(index) * (1 - frac) + p(index + 1) * frac

if __name__=='__main__':
  print(linearInterpolate([0.5, 0.5], lambda x, y: [x * y]))