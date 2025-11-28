import {TWOPI, DEG_TO_RAD} from './utils/num.js';
import {version, VERSION_CHANGE_POINT, VERSION_RANDOM_WALK} from './version.js';

//
// Task parameters
//

export const BET_WIDTHS_RAD = [60*DEG_TO_RAD, 120*DEG_TO_RAD];

export const SCORE_GAIN_TABLE = {
  win: { precise: 10, imprecise: 5 },
  lose: { precise: -10, imprecise: -10 }
};

export const INITIAL_VALUES = {
  betLoc: 0,
  betWidIdx: 0,
  score: 110
};

export const N_POINTS_PER_COIN = 50;

export function nCoinsWithPoints(score) {
  return Math.floor(score / N_POINTS_PER_COIN);
}

export const N_BLOCKS = 15;

export const N_POINTS_PER_POUND = (version === VERSION_RANDOM_WALK ? 2000 : 2500);

export const MAX_BONUS_POUNDS = (version === VERSION_RANDOM_WALK ? 5.0 : 4.20);