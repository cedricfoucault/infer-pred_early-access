//
// Mathematical constants
//

export const TWOPI = 2*Math.PI;
export const DEG_TO_RAD = Math.PI / 180;
export const RAD_TO_DEG = 180 / Math.PI;

//
// Mathematical/Numerical/Random Functions
//

export function mod(a, n) {
  return (a % n + n) % n;
}

export function subtractAngle(theta2, theta1) {
  return mod((theta2 - theta1 + Math.PI), TWOPI) - Math.PI;
}

// Functions adapted from underscore.js
// ref: https://github.com/jashkenas/underscore/blob/ffabcd443fd784e4bc743fff1d25456f7282d531/underscore.js

export function range(start, stop, step) {
  if (stop == null) {
    stop = start || 0;
    start = 0;
  }
  if (!step) {
    step = stop < start ? -1 : 1;
  }

  let length = Math.max(Math.ceil((stop - start) / step), 0);
  let range = Array(length);

  for (var idx = 0; idx < length; idx++, start += step) {
    range[idx] = start;
  }

  return range;
}

export function randomSample(array, n) {
  if (n == null) {
    return array[randomInt(array.length - 1)];
  }
  let sample = array.slice();
  let length = sample.length;
  n = Math.max(Math.min(n, length), 0);
  let last = length - 1;
  for (let index = 0; index < n; index++) {
    let rand = randomInt(index, last);
    let temp = sample[index];
    sample[index] = sample[rand];
    sample[rand] = temp;
  }
  return sample.slice(0, n);
}

export function randomInt(min, max) {
  if (max == null) {
    max = min;
    min = 0;
  }
  return min + Math.floor(Math.random() * (max - min + 1));
}
